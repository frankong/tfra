import os
import numpy as np
import tensorflow as tf
from tfra import read_ra_tf, write_ra


class Test_tfio(tf.test.TestCase):
    
    def test_ra(self):
        
        with self.test_session():

            shape = [3, 4]

            for dtype in ['int32', 'int64',
                          'float32', 'float64',
                          'complex64', 'complex128']:
                
                output = np.random.standard_normal(shape).astype(dtype)
                write_ra('test.ra', output)
                input = read_ra_tf('test.ra').eval()

                os.remove('test.ra')

            self.assertAllClose(input, output)
