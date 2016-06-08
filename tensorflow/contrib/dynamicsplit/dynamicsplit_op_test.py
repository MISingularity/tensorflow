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

"""Tests for tensorflow.dynamicsplit_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class DynamicSplitTest(tf.test.TestCase):

  def _test(self, args, expected_out=None, expected_err_re=None):

    with self.test_session(use_gpu=False) as sess:
      #print("before define ops")
      dy_split = tf.contrib.dynamicsplit.dynamicsplit(**args)
      #print("after define ops")     

      if expected_err_re is None:
        #print("before run")
        out = sess.run(dy_split)
        #print("after run")
        if out.dtype == np.float32:
          self.assertAllClose(out, expected_out)
        else:
          self.assertAllEqual(out, expected_out)
   
      else:
        with self.assertRaisesOpError(expected_err_re):
          sess.run(dy_split)

  def testString(self):
    args = {"record": "0,1,2,3", "output_type": ["1"],}
    expected_out = ["0","1","2","3"]
    self._test(args, expected_out)

  def testInt(self):
    args = {"record": "0,1,2,3", "output_type": [1],}
    expected_out = [0,1,2,3]
    self._test(args, expected_out)

  def testFloat(self):
    args = {"record": "0,1,2,3", "output_type": [1.0],}
    expected_out = [0.0,1.0,2.0,3.0]
    self._test(args, expected_out)

  def testSpaceDelim(self):
    args = {"record": "0 1 2 3", "output_type": [1], "field_delim": " "}
    expected_out = [0,1,2,3]
    self._test(args, expected_out)

if __name__ == "__main__":
  tf.test.main()
