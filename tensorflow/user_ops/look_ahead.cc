

#include "tensorflow/user_ops/look_ahead.h"

REGISTER_OP("Lookaheadcpu")
    .Input("input: float")
    .Input("filter: float")
    .Output("output: float");

using namespace tensorflow;

template<typename T>
class LookaheadOp<T, 0> : public OpKernel {
 public:
  explicit LookaheadOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.matrix<T>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    // Check that preserve_index is in range

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template matrix<T>();

    for (int input_x = 0; input_x < input_tensor.dim_size(0); input_x++) {
      for (int input_y = 0; input_y < input_tensor.dim_size(1); input_y++) {
        output(input_x, input_y) = 0;
        for(int input_begin = 0; input_begin < filter_tensor.dim_size(0); input_begin++) {
          if(input_y + input_begin < input_tensor.dim_size(1)) output(input_x, input_y) += input(input_x, input_y + input_begin) * filter(input_begin, input_x);
        }
      }
    }
  }

 private:
  int preserve_index_;
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadcpu").Device(DEVICE_CPU), LookaheadOp<float, 0>);
