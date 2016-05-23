
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class LookaheadGradTest(tf.test.TestCase):

  def test(self):
    with self.test_session():
      x1 = [[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]], [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]]
      x2 = [[1.0,2.0,4.0],[4.0,8.0,16.0]]
      x3 = [[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]], [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]]
      result = tf.contrib.lookahead.lookaheadgradcpu(x1,x2,x3)
      self.assertAllEqual(result[0].eval(), [[[1.0,2.25,3.5],[2.0,3.0,3.625],[1.75,2.4375,2.75]], [[1.0,2.25,3.5],[2.0,3.0,3.625],[1.75,2.4375,2.75]]])
      result_2 = tf.contrib.lookahead.lookaheadgradgpu(x1,x2,x3)
      self.assertAllEqual(result_2[0].eval(), [[[1.0,2.25,3.5],[2.0,3.0,3.625],[1.75,2.4375,2.75]], [[1.0,2.25,3.5],[2.0,3.0,3.625],[1.75,2.4375,2.75]]])
      y1 = [[[1.0,2.0,4.0,8.0],[4.0,8.0,16.0,32.0],[8.0,16.0,32.0,64.0]], [[1.0,2.0,4.0,8.0],[4.0,8.0,16.0,32.0],[8.0,16.0,32.0,64.0]]]
      y2 = [[1.0,2.0,4.0],[4.0,8.0,16.0]]
      y3 = [[[1.0,2.0,4.0,8.0],[4.0,8.0,16.0,32.0],[8.0,16.0,32.0,64.0]], [[1.0,2.0,4.0,8.0],[4.0,8.0,16.0,32.0],[8.0,16.0,32.0,64.0]]]
      result_filter = tf.contrib.lookahead.lookaheadgradcpu(y1,y2,y3)
      self.assertAllEqual(result_filter[1].eval(), [[8.0,8.0,8.0],[3.0,3.0,3.0]])
      result_filter_2 = tf.contrib.lookahead.lookaheadgradgpu(y1,y2,y3)
      self.assertAllEqual(result_filter_2[1].eval(), [[8.0,8.0,8.0],[3.0,3.0,3.0]])
      xw = tf.contrib.lookahead.lookaheadcpu(x1,x2)
      w_grad1 = tf.gradients(xw, [x1, x2])
      print(w_grad1[0], w_grad1[1])

if __name__ == '__main__':
  tf.test.main()
