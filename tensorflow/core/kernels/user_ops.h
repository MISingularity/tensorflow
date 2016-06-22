// check equation 11 on http://arxiv.org/pdf/1512.02595v1.pdf (page 11th)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/numeric_op.h"

using namespace tensorflow;

template<typename T, int K>
class LookaheadOp : public OpKernel {
 public:
  explicit LookaheadOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
  }
};

template<typename T, int K>
class LookaheadGPUOp : public OpKernel {
 public:
  explicit LookaheadGPUOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
  }
};
