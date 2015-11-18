import math
import tensorflow as tf

from .utils import copy_variables


class Layer(object):
    def __init__(self, input_sizes, output_size, name='Layer', given_variables={}):
        """Cretes a neural network layer."""
        if type(input_sizes) != list:
            input_sizes = [input_sizes]

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.name = name

        with tf.name_scope(self.name):
            self.Ws = []
            for input_idx, input_size in enumerate(input_sizes):
                W_name = "W_%d" % (input_idx,)
                if W_name in given_variables:
                    self.Ws.append(given_variables[W_name])
                else:
                    W_tensor = tf.random_uniform((input_size, output_size),
                                                 -1.0 / math.sqrt(input_size),
                                                 1.0 / math.sqrt(input_size))
                    self.Ws.append(tf.Variable(W_tensor, name=W_name))
            if "b" in given_variables:
                self.b = given_variables["b"]
            else:
                b_tensor = tf.zeros((output_size,))
                self.b = tf.Variable(b_tensor, name="b")

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        assert len(xs) == len(self.Ws), \
                "Expected %d input vectors, got %d" % (len(self.Ws), len(x))
        with tf.name_scope(self.name):
            return sum([tf.matmul(x, W) for x, W in zip(xs, self.Ws)]) + self.b

    def variables(self):
        return [self.b] + self.Ws

    def copy(self, name=None):
        name = name or self.name
        return Layer(self.input_sizes, self.output_size, name,
                given_variables=copy_variables(self.variables()))

class MLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities, name="MLP", given_layers=None):
        self.input_sizes = input_sizes
        self.hiddens = hiddens
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.name = name

        assert len(hiddens) == len(nonlinearities), \
                "Number of hiddens must be equal to number of nonlinearities"

        with tf.name_scope(self.name):
            if given_layers is not None:
                self.input_layer = given_layers[0]
                self.layers      = given_layers[1:]
            else:
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

    def variables(self):
        res = self.input_layer.variables()
        for layer in self.layers:
            res.extend(layer.variables())
        return res

    def copy(self, name=None):
        name = name or self.name
        nonlinearities = [self.input_nonlinearity] + self.layer_nonlinearities
        given_layers = [self.input_layer.copy()] + [layer.copy() for layer in self.layers]
        return MLP(self.input_sizes, self.hiddens, nonlinearities, name=name,
                given_layers=given_layers)
