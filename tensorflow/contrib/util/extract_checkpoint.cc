/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/tensor_slice_reader.h"

namespace tensorflow {

int InspectCheckpoint(const string& in) {
  tensorflow::checkpoint::TensorSliceReader reader(
       in, tensorflow::checkpoint::OpenTableTensorSliceReader);
  Status s = reader.status();
  if (!s.ok()) {
    fprintf(stderr, "Unable to open the checkpoint file\n");
    return -1;
  }
  for (auto e : reader.Tensors()) {
    fprintf(stdout, "%s %s\n", e.first.c_str(),
            e.second->shape().DebugString().c_str());
    std::unique_ptr<tensorflow::Tensor> tensor;
    Status s = reader.GetTensor(e.first.c_str(), &tensor);
    // we need know a priori the tensor_datatype and tensor_shape
    // although we can got them by calling tensor's member function
    // we need constants to instantialize template function call
    // TODO: add some Polymorphism

    // this program demo reads one dimension float tensor
    auto data = tensor->tensor<float, 1>();
    int dims = tensor->dims();
    fprintf(stdout, "Dims %d\n", dims);
    tensorflow::int64 dim0_size = tensor->dim_size(0);
    fprintf(stdout, "Dim size of dim %d : %lld\n", 0, dim0_size);
    for (tensorflow::int64 d = 0; d < dim0_size; ++d)
      fprintf(stdout, "%f%c", data(d), (d + 1 == dim0_size)?'\n':' ');
  }
  return 0;
}
}
int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc != 2) {
    fprintf(stderr, "Usage: %s checkpoint_file\n", argv[0]);
    exit(1);
  }
  return tensorflow::InspectCheckpoint(argv[1]);
}
