from .benchmark import Benchmark
class Rastringin(Benchmark):
    def __init__(self,):
        super().__init__()
        self.domain = [
            [-5.12, 5.12],
            [-5.12, 5.12],
            ]
    
    def calculate(self, x_var, y_var):
        x_var = tf.Variable(0.0) if x_var is None else x_var
        y_var = tf.Variable(0.0) if y_var is None else y_var

        return 10 * 2 + (x_var * x_var - 10 * tf.cos(2 * m.pi * x_var)) + (y_var * y_var - 10 * tf.cos(2 * m.pi * y_var) )
