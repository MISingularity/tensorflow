

#include "tensorflow/core/kernels/lookahead_grad_op.h"

using namespace tensorflow;

template<typename T>
__global__ void kernel_grad_input(int dim_x, int dim_y, int filter_size, const T* filter, const T* back_prop, T* output) {
  int input_y = threadIdx.x;
  int input_x = blockIdx.x;
  output[blockIdx.x* dim_y + threadIdx.x] = 0;
  for(int input_begin = 0; input_begin < filter_size; input_begin++) {
    int index = input_begin + input_y - filter_size + 1;
    int filter_idx = filter_size - 1 - input_begin;
    if (index >= 0 && filter_idx >= 0 && filter[filter_idx * dim_x + input_x] != 0) {
      output[input_x * dim_y + input_y] += back_prop[input_x * dim_y + index] / filter[filter_idx * dim_x + input_x];
    }
  }
}

template<typename T>
__global__ void kernel_grad_filter(int dim_x, int dim_y, int input_size, const T* input, const T* back_prop, T* output) {
  int input_y = threadIdx.x;
  int input_x = blockIdx.x;
  for (int input_begin = 0; input_begin < input_size - input_x; input_begin++) {
    if (input[input_y * input_size + input_x + input_begin] != 0) {
      output[input_x * dim_y + input_y] += back_prop[input_y * input_size + input_begin] / input[input_y * input_size + input_x + input_begin];
    }
  }
}

template<typename T>
__global__ void kernel_set_zero(int dim_x, int dim_y, T* output) {
  int input_y = threadIdx.x;
  int input_x = blockIdx.x;
  output[input_x * dim_y + input_y] = 0;
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
    auto input = input_tensor.tensor<T, 3>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    const Tensor& backprop_output_tensor = context->input(2);
    auto backprop_output = backprop_output_tensor.tensor<T, 3>();

    // Check that preserve_index is in range

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();

    int batch_size = input_tensor.dim_size(0);
    int dim_x = input_tensor.dim_size(1);
    int dim_y = input_tensor.dim_size(2);
    int filter_size = filter_tensor.dim_size(0);
    cudaStream_t stream[batch_size];
    for (int i = 0; i < batch_size; i++) {
      cudaStreamCreate(&stream[i]);
    }
    for (int i = 0; i < batch_size; i++) {
      kernel_grad_input<T><<<dim_x, dim_y, 0, stream[i]>>>(dim_x, dim_y, filter_size, &filter(0, 0), &backprop_output(i, 0, 0), &output(i, 0, 0));
    }
    for (int i = 0; i < batch_size; i++) {
      cudaStreamSynchronize(stream[i]);
      cudaStreamDestroy(stream[i]);
    }
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
    auto input = input_tensor.tensor<T, 3>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    const Tensor& backprop_output_tensor = context->input(2);
    auto backprop_output = backprop_output_tensor.tensor<T, 3>();

    // Check that preserve_index is in range

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, filter_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template matrix<T>();

    int batch_size = input_tensor.dim_size(0);
    int dim_x = filter_tensor.dim_size(0);
    int dim_y = filter_tensor.dim_size(1);
    int input_size = input_tensor.dim_size(2);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemset(&output(0, 0), 0, dim_x * dim_y * sizeof(T));
    for (int i = 0; i < batch_size; i++) {
      kernel_grad_filter<T><<<dim_x, dim_y, 0, stream>>>(dim_x, dim_y, input_size, &input(i, 0, 0), &backprop_output(i, 0, 0), &output(0, 0));
      cudaStreamSynchronize(stream);
    }
    cudaStreamDestroy(stream);
  }

 private:
  int preserve_index_;
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadgradinputgpu").Device(DEVICE_GPU), LookaheadGradInputOp<float, 1>);
REGISTER_KERNEL_BUILDER(Name("Lookaheadgradfiltergpu").Device(DEVICE_GPU), LookaheadGradFilterOp<float, 1>);
