#include "tensorflow/core/kernels/lookahead_ops.h"

using namespace tensorflow;

template<typename T>
__global__ void kernel(int dim_x, int dim_y, int filter_size, const T* input, const T* filter, T* output) {
  output[blockIdx.x* dim_y + threadIdx.x] = 0;
  for(int input_begin = 0; input_begin < filter_size; input_begin++) {
    if(threadIdx.x + input_begin < dim_y) output[blockIdx.x* dim_y + threadIdx.x] += input[blockIdx.x * dim_y + threadIdx.x + input_begin] * filter[input_begin * dim_x + blockIdx.x];
  }
}

template<typename T>
class LookaheadOp<T, 1> : public OpKernel {
 public:
  explicit LookaheadOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* context) override {
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
    int batch_size = input_tensor.dim_size(0);
    cudaStream_t stream[batch_size];
    int dim_x = input_tensor.dim_size(1);
    int dim_y = input_tensor.dim_size(2);
    int filter_size = filter_tensor.dim_size(0);
    for(int i = 0; i < batch_size; i++) {
      cudaStreamCreate(&stream[i]);
    }
    for(int i = 0; i < batch_size; i++) {
      kernel<T><<<dim_x, dim_y, 0, stream[i]>>>(dim_x, dim_y, filter_size, &input(i, 0, 0), &filter(0, 0), &output(i, 0, 0));
    }
    for(int i = 0; i < batch_size; i++) {
      cudaStreamSynchronize(stream[i]);
      cudaStreamDestroy(stream[i]);
    }
  }

 private:
  int preserve_index_;
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadgpu").Device(DEVICE_GPU), LookaheadOp<float, 1>);
