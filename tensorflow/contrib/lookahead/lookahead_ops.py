# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=unused-import
"""CTC (Connectionist Temporal Classification) Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util

from tensorflow.python.ops import gen_lookahead_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul


# NOTE(ebrevdo): We redefine CTCLoss from gen_ctc_ops to only return
# the first output. The second output is only used for the gradient.
# pylint: disable=protected-access, invalid-name

def lookaheadcpu(x1, x2):
  return gen_lookahead_ops._lookaheadcpu(x1, x2)

def lookaheadgpu(x1, x2):
  return gen_lookahead_ops._lookaheadgpu(x1, x2)

def lookaheadgradcpu(x1, x2, x3):
  return gen_lookahead_ops._lookaheadgradcpu(x1, x2, x3)

def lookaheadgradgpu(x1, x2, x3):
  return gen_lookahead_ops._lookaheadgradgpu(x1, x2, x3)

@ops.RegisterShape("Lookaheadcpu")
def _Lookaheadcpu(op):
  inputs_shape = op.inputs[0].get_shape().with_rank(3)
  return [inputs_shape]

@ops.RegisterGradient("Lookaheadcpu")
def _Lookaheadcpu_grad(op, grad):
  """The derivatives for deconvolution.

  Args:
    op: the Deconvolution op.
    grad: the tensor representing the gradient w.r.t. the output

  Returns:
    the gradients w.r.t. the input and the filter
  """
  return tf.contrib.lookahead.lookaheadgradcpu(
              op.inputs[0],op.inputs[1],grad)

@ops.RegisterGradient("Lookaheadgpu")
def _Lookaheadgpu_grad(op, grad):
  """The derivatives for deconvolution.

  Args:
    op: the Deconvolution op.
    grad: the tensor representing the gradient w.r.t. the output

  Returns:
    the gradients w.r.t. the input and the filter
  """
  return tf.contrib.lookahead.lookaheadgradgpu(
              op.inputs[0],op.inputs[1],grad)

@ops.RegisterShape("Lookaheadgpu")
def _Lookaheadgpu(op):
  inputs_shape = op.inputs[0].get_shape().with_rank(3)
  return [inputs_shape]

@ops.RegisterShape("Lookaheadgradcpu")
def _Lookaheadgradcpu(op):
  inputs_shape1 = op.inputs[0].get_shape().with_rank(3)
  inputs_shape2 = op.inputs[1].get_shape().with_rank(2)
  return [inputs_shape1, inputs_shape2]

@ops.RegisterShape("Lookaheadgradgpu")
def _Lookaheadgradinputgpu(op):
  inputs_shape1 = op.inputs[0].get_shape().with_rank(3)
  inputs_shape2 = op.inputs[1].get_shape().with_rank(2)
  return [inputs_shape1, inputs_shape2]
