

#include "tensorflow/core/kernels/lookahead_ops.h"

using namespace tensorflow;

template<typename T>
class LookaheadOp<T, 0> : public OpKernel {
 public:
  explicit LookaheadOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* context) override {
    printf("lookahead cpu\n");
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T, 3>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    // Check that preserve_index is in range

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();

    for (int batch = 0; batch < input_tensor.dim_size(0); batch++) {
      for (int input_x = 0; input_x < input_tensor.dim_size(1); input_x++) {
        for (int input_y = 0; input_y < input_tensor.dim_size(2); input_y++) {
          output(batch, input_x, input_y) = 0;
          for(int input_begin = 0; input_begin < filter_tensor.dim_size(0); input_begin++) {
            if(input_y + input_begin < input_tensor.dim_size(2)) output(batch, input_x, input_y) += input(batch, input_x, input_y + input_begin) * filter(input_begin, input_x);
          }
        }
      }
    }
  }

 private:
  int preserve_index_;
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadcpu").Device(DEVICE_CPU), LookaheadOp<float, 0>);
