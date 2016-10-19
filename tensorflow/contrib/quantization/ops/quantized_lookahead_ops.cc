
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("QuantizedLookahead")
    .Input("input: T1")
    .Input("filter: T2")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Output("output: Toutput")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("Toutput: quantizedtype = DT_QINT32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle a;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));
        ShapeHandle b;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));

        c->set_output(0, c->MakeShape({c->Dim(a, 0), c->Dim(a, 1), c->Dim(a, 2)}));

        ShapeHandle unused;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));

        c->set_output(1, c->Scalar());
        c->set_output(2, c->Scalar());
        return Status::OK();
    });

}  // namespace tensorflow

