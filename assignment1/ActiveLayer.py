class ActiveLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        self.output = None

    def forward_prop(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_prop(self, output_error):
        return self.activation_prime(self.input) * output_error
