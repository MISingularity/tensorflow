
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LookaheadTest(tf.test.TestCase):

  def test(self):
    with self.test_session():
      x1 = [[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]], [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]]
      x2 = [[1.0,2.0,3.0],[4.0,5.0,6.0]]
      result = tf.contrib.lookahead.lookaheadcpu(x1,x2)
      self.assertAllEqual(result.eval(), [[[9.0,14.0,3.0],[33.0,40.0,12.0],[69.0,78.0,27.0]], [[9.0,14.0,3.0],[33.0,40.0,12.0],[69.0,78.0,27.0]]])
      result_2 = tf.contrib.lookahead.lookaheadgpu(x1,x2)
      self.assertAllEqual(result_2.eval(), [[[9.0,14.0,3.0],[33.0,40.0,12.0],[69.0,78.0,27.0]], [[9.0,14.0,3.0],[33.0,40.0,12.0],[69.0,78.0,27.0]]])

if __name__ == '__main__':
  tf.test.main()
