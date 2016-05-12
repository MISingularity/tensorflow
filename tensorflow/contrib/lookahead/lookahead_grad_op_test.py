
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
      result = tf.contrib.lookahead.lookaheadgradinputcpu(x1,x2,x3)
      self.assertAllEqual(result.eval(), [[[1.0,2.25,3.5],[2.0,3.0,3.625],[1.75,2.4375,2.75]], [[1.0,2.25,3.5],[2.0,3.0,3.625],[1.75,2.4375,2.75]]])
      result_2 = tf.contrib.lookahead.lookaheadgradinputgpu(x1,x2,x3)
      self.assertAllEqual(result_2.eval(), [[[1.0,2.25,3.5],[2.0,3.0,3.625],[1.75,2.4375,2.75]], [[1.0,2.25,3.5],[2.0,3.0,3.625],[1.75,2.4375,2.75]]])
      y1 = [[[1.0,2.0,4.0,8.0],[4.0,8.0,16.0,32.0],[8.0,16.0,32.0,64.0]], [[1.0,2.0,4.0,8.0],[4.0,8.0,16.0,32.0],[8.0,16.0,32.0,64.0]]]
      y2 = [[1.0,2.0,4.0],[4.0,8.0,16.0]]
      y3 = [[[1.0,2.0,4.0,8.0],[4.0,8.0,16.0,32.0],[8.0,16.0,32.0,64.0]], [[1.0,2.0,4.0,8.0],[4.0,8.0,16.0,32.0],[8.0,16.0,32.0,64.0]]]
      result_filter = tf.contrib.lookahead.lookaheadgradfiltercpu(y1,y2,y3)
      self.assertAllEqual(result_filter.eval(), [[8.0,8.0,8.0],[3.0,3.0,3.0]])
      result_filter_2 = tf.contrib.lookahead.lookaheadgradfiltergpu(y1,y2,y3)
      self.assertAllEqual(result_filter_2.eval(), [[8.0,8.0,8.0],[3.0,3.0,3.0]])
      print(result.eval())
      print(result_2.eval())
      print(result_filter.eval())
      print(result_filter_2.eval())
      xw = tf.contrib.lookahead.lookaheadgradinputcpu(x1,x2,x3)
      w_grad = tf.gradients(xw, x2)
      print(w_grad)

if __name__ == '__main__':
  tf.test.main()
