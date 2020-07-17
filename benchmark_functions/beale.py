from .benchmark import Benchmark
import tensorflow as tf


class Beale(Benchmark):
    def __init__(self):
        super().__init__()
    
        self.domain = [
            [-4.5, 4.5],
            [-4.5, 4.5],
            ]
    def calculate(self, x_var, y_var):
        x_var = tf.Variable(0.0) if x_var is None else x_var
        y_var = tf.Variable(0.0) if y_var is None else y_var

        return tf.square(1.5 - x_var + x_var * y_var) + tf.square(2.25 - x_var + x_var * tf.square(y_var)) + tf.square(2.625 - x_var + x_var * tf.pow(y_var, 3))