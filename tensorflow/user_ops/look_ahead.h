

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/numeric_op.h"
#include <cuda_runtime.h>

using namespace tensorflow;

template<typename T, int K>
class LookaheadOp : public OpKernel {
 public:
  explicit LookaheadOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
  }
};

