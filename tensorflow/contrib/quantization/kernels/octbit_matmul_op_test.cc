/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class OctbitMatMulTest : public OpsTestBase {
 protected:
};

// Runs two small matrices through the operator, and leaves all the parameters
// at their default values.
TEST_F(OctbitMatMulTest, Small_NoParams) {
  Tensor a1 = Tensor(DT_FLOAT, {2});
  auto b1 = a1.vec<float>();
  for (int i = 0; i < 2; i++) {
    b1(i) = 1.;
  }
  TF_ASSERT_OK(NodeDefBuilder("octbit_mat_mul_op", "OctbitMatMul")
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("bias", a1)
                   .Attr("T", DataTypeToEnum<float>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // A matrix is:
  // |  1 |  2 |  3 |
  // |  4 |  5 |  6 |

  std::vector<qint8> v;
  for (int i = 0; i < 64*2; i++) {
      v.push_back(qint8(1));
  }
  AddInputFromArray<qint8>(TensorShape({2, 64}), v);

  // B matrix is:
  // |  7 |  8 |  9 | 10 |
  // | 11 | 12 | 13 | 14 |
  // | 15 | 16 | 17 | 18 |
  std::vector<float> v2;
  for (int i = 0; i < 64 * 4; i++) {
      v2.push_back(1.);
  }
  AddInputFromArray<float>(TensorShape({4, 64}), v2);


  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 2}));
  test::FillValues<float>(&expected, {64., 64., 64., 64., 64., 64., 64., 64.});
  auto bb = *GetOutput(0);
  std::cout << bb.matrix<float>()(0, 0);
  test::ExpectTensorEqual<float>(expected, bb);
}
}  // namespace tensorflow

