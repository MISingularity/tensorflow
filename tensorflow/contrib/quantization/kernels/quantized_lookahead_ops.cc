// Copyright by Naturali. 2016
// Author LIU Jiahua
// All rights reserved.

#include <memory>
#include <string>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/contrib/quantization/kernels/quantization_utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template<class T1, class T2, class Toutput>
class QuantizedLookaheadCpuOp : public OpKernel {
 public:
  explicit QuantizedLookaheadCpuOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T1, 3>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T2>();

    const float min_input = context->input(2).flat<float>()(0);
    const float max_input = context->input(3).flat<float>()(0);
    const float min_filter = context->input(4).flat<float>()(0);
    const float max_filter = context->input(5).flat<float>()(0);

    OP_REQUIRES(context, (max_input > min_input),
                errors::InvalidArgument("max_input must be larger than min_input."));
    OP_REQUIRES(context, (max_filter > min_filter),
                errors::InvalidArgument("max_filter must be larger than min_filter."));

    // Check that dimension is equal
    OP_REQUIRES(
        context, input_tensor.dim_size(2) == filter_tensor.dim_size(1),
        errors::InvalidArgument("f is not equal in filter and input"));

    auto TS = input_tensor.dim_size(0);
    auto B = input_tensor.dim_size(1);
    auto F = input_tensor.dim_size(2);
    auto W = filter_tensor.dim_size(0);

    // Create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<Toutput, 3>();

    for (int t = 0; t < TS; t++) {
      for (int b = 0; b < B; b++) {
        for (int f = 0; f < F; f++) {
          output(t, b, f) = 0;
        }
        for(int tau = 0; tau < W && t + tau < TS; tau++) {
          for (int f = 0; f < F; f++) {
            output(t, b, f) += input(t + tau, b, f) * filter(tau, f);
          }
        }
      }
    }

    float min_output_value;
    float max_output_value;
    QuantizationRangeForMultiplication<T1, T2, Toutput>(
        min_input, max_input, min_filter, max_filter,
        &min_output_value, &max_output_value);
    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &output_min));
    output_min->flat<float>()(0) = min_output_value;

    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_max));
    output_max->flat<float>()(0) = max_output_value;
  }
};

REGISTER_KERNEL_BUILDER(Name("QuantizedLookahead")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T1")
                            .TypeConstraint<quint8>("T2")
                            .TypeConstraint<qint32>("Toutput"),
                        QuantizedLookaheadCpuOp<quint8, quint8, qint32>);

} // namespace tensorflow

