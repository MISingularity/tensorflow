
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf


class LookaheadGradTest(tf.test.TestCase):

  def test(self):
    library_filename = os.path.join(tf.resource_loader.get_data_files_path(),
                                    'look_ahead_grad.so')
    look_ahead_module = tf.load_op_library(library_filename)

    with self.test_session():
      x1 = [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]
      x2 = [[1.0,2.0,4.0],[4.0,8.0,16.0]]
      x3 = [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]
      result = look_ahead_module.lookaheadgradinputcpu(x1,x2,x3)
      self.assertAllEqual(result.eval(), [[1.0,2.25,3.5],[2.0,3.0,3.625],[1.75,2.4375,2.75]])
      result_2 = look_ahead_module.lookaheadgradinputgpu(x1,x2,x3)
      self.assertAllEqual(result_2.eval(), [[1.0,2.25,3.5],[2.0,3.0,3.625],[1.75,2.4375,2.75]])
      y1 = [[1.0,2.0,4.0,8.0],[4.0,8.0,16.0,32.0],[8.0,16.0,32.0,64.0]]
      y2 = [[1.0,2.0,4.0],[4.0,8.0,16.0]]
      y3 = [[1.0,2.0,4.0,8.0],[4.0,8.0,16.0,32.0],[8.0,16.0,32.0,64.0]]
      result_filter = look_ahead_module.lookaheadgradfiltercpu(y1,y2,y3)
      self.assertAllEqual(result_filter.eval(), [[4.0,4.0,4.0],[1.5,1.5,1.5]])
      result_filter_2 = look_ahead_module.lookaheadgradfiltergpu(y1,y2,y3)
      self.assertAllEqual(result_filter_2.eval(), [[4.0,4.0,4.0],[1.5,1.5,1.5]])
      #print(result.eval())
      #print(result_2.eval())

if __name__ == '__main__':
  tf.test.main()
