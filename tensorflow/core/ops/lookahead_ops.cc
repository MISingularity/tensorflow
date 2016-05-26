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

REGISTER_OP("Lookaheadgradcpu")
    .Input("input: float")
    .Input("filter: float")
    .Input("backprop_output: float")
    .Output("output1: float")
    .Output("output2: float");

REGISTER_OP("Lookaheadgradgpu")
    .Input("input: float")
    .Input("filter: float")
    .Input("backprop_output: float")
    .Output("output1: float")
    .Output("output2: float");

}
