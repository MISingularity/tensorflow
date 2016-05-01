

#include "tensorflow/user_ops/look_ahead_grad.h"

REGISTER_OP("Lookaheadgradinputgpu")
    .Input("input: float")
    .Input("filter: float")
    .Input("backprop_output: float")
    .Output("output: float");

REGISTER_OP("Lookaheadgradfiltergpu")
    .Input("input: float")
    .Input("filter: float")
    .Input("backprop_output: float")
    .Output("output: float");

using namespace tensorflow;

template<typename T>
__global__ void kernel_grad_input(int dim_x, int dim_y, int filter_size, const T* filter, const T* back_prop, T* output) {
  int input_y = threadIdx.x;
  int input_x = blockIdx.x;
  output[blockIdx.x* dim_y + threadIdx.x] = 0;
  for(int input_begin = 0; input_begin < filter_size; input_begin++) {
    int index = input_begin + input_y - filter_size + 1;
    if (index >= 0) {
      output[input_x * dim_y + input_y] += back_prop[input_x * dim_y + index] / filter[(filter_size - 1 - input_begin) * dim_x + input_x];
    }
  }
}

template<typename T>
__global__ void kernel_grad_filter(int dim_x, int dim_y, int input_size, const T* input, const T* back_prop, T* output) {
  int input_y = threadIdx.x;
  int input_x = blockIdx.x;
  output[input_x * dim_y + input_y] = 0;
  for (int input_begin = 0; input_begin < input_size - input_x; input_begin++) {
    output[input_x * dim_y + input_y] += back_prop[input_y * input_size + input_begin] / input[input_y * input_size + input_x + input_begin];
  }
}

template<typename T>
class LookaheadGradInputOp<T, 1> : public OpKernel {
 public:
  explicit LookaheadGradInputOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.matrix<T>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    const Tensor& backprop_output_tensor = context->input(2);
    auto backprop_output = backprop_output_tensor.matrix<T>();

    // Check that preserve_index is in range

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template matrix<T>();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int dim_x = input_tensor.dim_size(0);
    int dim_y = input_tensor.dim_size(1);
    int filter_size = filter_tensor.dim_size(0);
    kernel_grad_input<T><<<dim_x, dim_y, 0, stream>>>(dim_x, dim_y, filter_size, &filter(0, 0), &backprop_output(0, 0), &output(0, 0));
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }

 private:
  int preserve_index_;
};

template<typename T>
class LookaheadGradFilterOp<T, 1> : public OpKernel {
 public:
  explicit LookaheadGradFilterOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.matrix<T>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    const Tensor& backprop_output_tensor = context->input(2);
    auto backprop_output = backprop_output_tensor.matrix<T>();

    // Check that preserve_index is in range

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, filter_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template matrix<T>();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int dim_x = filter_tensor.dim_size(0);
    int dim_y = filter_tensor.dim_size(1);
    int input_size = input_tensor.dim_size(1);
    kernel_grad_filter<T><<<dim_x, dim_y, 0, stream>>>(dim_x, dim_y, input_size, &input(0, 0), &backprop_output(0, 0), &output(0, 0));
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }

 private:
  int preserve_index_;
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadgradinputgpu").Device(DEVICE_GPU), LookaheadGradInputOp<float, 1>);
REGISTER_KERNEL_BUILDER(Name("Lookaheadgradfiltergpu").Device(DEVICE_GPU), LookaheadGradFilterOp<float, 1>);
