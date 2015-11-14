import math
import tensorflow as tf

class Layer(object):
    def __init__(self, input_sizes, output_size, name='Layer'):
        """Cretes a neural network layer."""
        if type(input_sizes) != list:
            input_sizes = [input_sizes]

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.name = name

        with tf.name_scope(self.name):
            self.Ws = []
            for input_idx, input_size in enumerate(input_sizes):
                tensor_W = tf.random_uniform((input_size, output_size),
                                             -1.0 / math.sqrt(input_size),
                                             1.0 / math.sqrt(input_size))
                self.Ws.append(tf.Variable(tensor_W, name="W_%d" % (input_idx,)))

            tensor_b = tf.zeros((output_size,))
            self.b = tf.Variable(tensor_b, name="b")

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        assert len(xs) == len(self.Ws), \
                "Expected %d input vectors, got %d" % (len(self.Ws), len(x))
        with tf.name_scope(self.name):
            return sum([tf.matmul(x, W) for x, W in zip(xs, self.Ws)]) + self.b


class MLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities, name="MLP"):
        self.input_sizes = input_sizes
        self.hiddens = hiddens
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.name = name

        assert len(hiddens) == len(nonlinearities), \
                "Number of hiddens must be equal to number of nonlinearities"

        with tf.name_scope(self.name):
            self.input_layer = Layer(input_sizes, hiddens[0], name="input_layer")
            self.layers = []

            for l_idx, (h_from, h_to) in enumerate(zip(hiddens[:-1], hiddens[1:])):
                self.layers.append(Layer(h_from, h_to, name="hidden_layer_%d" % (l_idx,)))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        with tf.name_scope(self.name):
            hidden = self.input_nonlinearity(self.input_layer(xs))
            for layer, nonlinearity in zip(self.layers, self.layer_nonlinearities):
                hidden = nonlinearity(layer(hidden))
            return hidden
