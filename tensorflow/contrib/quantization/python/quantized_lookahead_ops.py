
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.quantization.ops import gen_quantized_lookahead_ops
from tensorflow.contrib.quantization.ops.gen_quantized_lookahead_ops import *
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops


ops.RegisterShape("QuantizedLookahead")(common_shapes.call_cpp_shape_fn)

