from .benchmark import Benchmark
class Sphere(Benchmark):
    def __init__(self, domain):
        super().__init__()
        self.domain = [
            [float('-inf'), float('inf')],
            [float('-inf'), float('inf')],
            ]
    def calculate(self, x_var, y_var):
        x_var = tf.Variable(0.0) if x_var is None else x_var
        y_var = tf.Variable(0.0) if y_var is None else y_var

        return x_var * x_var + y_var * y_var

