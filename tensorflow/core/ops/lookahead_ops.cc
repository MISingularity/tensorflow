#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("Lookaheadcpu")
    .Input("input: float")
    .Input("filter: float")
    .Output("output: float");

REGISTER_OP("Lookaheadgpu")
    .Input("input: float")
    .Input("filter: float")
    .Output("output: float");

REGISTER_OP("Lookaheadgradinputcpu")
    .Input("input: float")
    .Input("filter: float")
    .Input("backprop_output: float")
    .Output("output: float");

REGISTER_OP("Lookaheadgradfiltercpu")
    .Input("input: float")
    .Input("filter: float")
    .Input("backprop_output: float")
    .Output("output: float");

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

}
