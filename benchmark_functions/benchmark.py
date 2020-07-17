class Benchmark():
    def __init__(self):
        super().__init__()
        #Domain should be a list of lists where each list yields the minimum and maximum value for one of the variables.
        pass
    
    def calculate(self, x_var, y_var):
        pass
    
    def clip_to_domain(self, x_var, y_var):
        if x_var < self.domain[0][0]:
            x_var = tf.Variable(self.domain[0][0])
        if x_var > self.domain[0][1]:
            x_var = tf.Variable(self.domain[0][1])
        
        if y_var < self.domain[1][0]:
            y_var = tf.Variable(self.domain[1][0])
        if y_var > self.domain[1][1]:
            y_var = tf.Variable(self.domain[1][1])

        return x_var, y_var
