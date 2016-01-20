import math
import tensorflow as tf

from types import FunctionType

from .utils import base_name


class Layer(object):
    def __init__(self, input_sizes, output_size, scope):
        """Cretes a neural network layer."""
        if type(input_sizes) != list:
            input_sizes = [input_sizes]

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.scope       = scope or "Layer"

        with tf.variable_scope(self.scope):
            self.Ws = []
            for input_idx, input_size in enumerate(input_sizes):
                W_name = "W_%d" % (input_idx,)
                W_initializer =  tf.random_uniform_initializer(
                        -1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
                W_var = tf.get_variable(W_name, (input_size, output_size), initializer=W_initializer)
                self.Ws.append(W_var)
            self.b = tf.get_variable("b", (output_size,), initializer=tf.constant_initializer(0))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        assert len(xs) == len(self.Ws), \
                "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))
        with tf.variable_scope(self.scope):
            return sum([tf.matmul(x, W) for x, W in zip(xs, self.Ws)]) + self.b

    def variables(self):
        return [self.b] + self.Ws

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"

        with tf.variable_scope(scope) as sc:
            for v in self.variables():
                tf.get_variable(base_name(v), v.get_shape(),
                        initializer=lambda x,dtype=tf.float32: v.initialized_value())
            sc.reuse_variables()
            return Layer(self.input_sizes, self.output_size, scope=sc)

class ConcatLayer(object):
    def __init__(self, input_sizes, output_size, scope):
        """Cretes a neural network layer."""
        if type(input_sizes) != list:
            input_sizes = [input_sizes]

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.scope       = scope or "ConcatLayer"

        total_input_size = sum(input_sizes)

        with tf.variable_scope(self.scope):
            W_initializer =  tf.random_uniform_initializer(
                        -1.0 / math.sqrt(total_input_size), 1.0 / math.sqrt(total_input_size))
            self.W = tf.get_variable("W", (total_input_size, output_size), initializer=W_initializer)
            self.b = tf.get_variable("b", (output_size,), initializer=tf.constant_initializer(0))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        assert len(xs) == len(self.input_sizes), \
                "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))
        with tf.variable_scope(self.scope):
            print("MATMUL", [x.get_shape() for x in xs])
            if len(xs) == 1:
                return tf.matmul(xs[0], self.W) + self.b
            else:
                return tf.matmul(tf.concat(1, xs,name="layer_concat"), self.W) + self.b

    def variables(self):
        return [self.W,  self.b]

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"

        with tf.variable_scope(scope) as sc:
            for v in self.variables():
                tf.get_variable(base_name(v), v.get_shape(),
                        initializer=lambda x,dtype=tf.float32: v.initialized_value())
            sc.reuse_variables()
            return Layer(self.input_sizes, self.output_size, scope=sc)

class MLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities, scope=None, given_layers=None):
        self.input_sizes = input_sizes
        self.hiddens = hiddens
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.scope = scope or "MLP"

        assert len(hiddens) == len(nonlinearities), \
                "Number of hiddens must be equal to number of nonlinearities"

        with tf.variable_scope(self.scope):
            if given_layers is not None:
                self.input_layer = given_layers[0]
                self.layers      = given_layers[1:]
            else:
                self.input_layer = Layer(input_sizes, hiddens[0], scope="input_layer")
                self.layers = []

                for l_idx, (h_from, h_to) in enumerate(zip(hiddens[:-1], hiddens[1:])):
                    self.layers.append(Layer(h_from, h_to, scope="hidden_layer_%d" % (l_idx,)))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        with tf.variable_scope(self.scope):
            hidden = self.input_nonlinearity(self.input_layer(xs))
            for layer, nonlinearity in zip(self.layers, self.layer_nonlinearities):
                hidden = nonlinearity(layer(hidden))
            return hidden

    def variables(self):
        res = self.input_layer.variables()
        for layer in self.layers:
            res.extend(layer.variables())
        return res

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        nonlinearities = [self.input_nonlinearity] + self.layer_nonlinearities
        with tf.variable_scope(scope):
            given_layers = [self.input_layer.copy()] + [layer.copy() for layer in self.layers]
            return MLP(self.input_sizes, self.hiddens, nonlinearities, scope=scope,
                    given_layers=given_layers)

class ConvLayer(object):
    def __init__(self, filter_H, filter_W, in_C, out_C, nonlinearity=tf.nn.relu, stride=(1,1), scope="Convolution"):
        self.filter_H, self.filter_W, self.in_C, self.out_C = filter_H, filter_W, in_C, out_C
        self.stride       = stride
        self.nonlinearity = nonlinearity
        self.scope        = scope

        with tf.variable_scope(self.scope):
            input_size = filter_H * filter_W * in_C
            W_initializer =  tf.random_uniform_initializer(
                        -1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
            self.W = tf.get_variable('W', (filter_H, filter_W, in_C, out_C), initializer=W_initializer)
            self.b = tf.get_variable('b', (out_C), initializer=tf.constant_initializer(0))

    def __call__(self, X):
        with tf.variable_scope(self.scope):
            return self.nonlinearity(tf.nn.conv2d(X, self.W,  strides=[1] + list(self.stride) + [1], padding='SAME')
                                     + self.b)

    def variables(self):
        return [self.b, self.W]

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"

        with tf.variable_scope(scope) as sc:
            for v in self.variables():
                tf.get_variable(base_name(v), v.get_shape(),
                        initializer=lambda x,dtype=tf.float32: v.initialized_value())
            sc.reuse_variables()
            return ConvLayer(self.filter_H, self.filter_W, self.in_C, self.out_C,
                             nonlinearity=self.nonlinearity, stride=self.stride, scope=sc)


class SequenceWrapper(object):
    def __init__(self, seq, scope=None):
        self.seq   = seq
        self.scope = scope or "Seq"

    def __call__(self, x):
        with tf.variable_scope(self.scope):
            for el in self.seq:
                x = el(x)
            return x

    def variables(self):
        res = []
        for el in self.seq:
            if hasattr(el, 'variables'):
                res.extend(el.variables())
        return res

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            new_seq = [el if type(el) is FunctionType else el.copy() for el in self.seq ]
            return SequenceWrapper(new_seq)

class LSTMLayer(object):
    def __init__(self, input_size, hidden_size, forget_bias=1.0, scope="LSTMLayer", given_layer=None):
        """Long short-term memory cell (LSTM)."""
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias

        self.scope = scope
        with tf.variable_scope(self.scope):
            if given_layer is not None:
                self.layer = given_layer
            else:
                self.layer = Layer([input_size, hidden_size], 4 * hidden_size, scope="lstm_layer")

    def __call__(self, inpt, state, scope=None):
        with tf.variable_scope(self.scope):
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = tf.split(1, 2, state)

            concat = self.layer([inpt, h])

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(1, 4, concat)

            new_c = c * tf.nn.sigmoid(f + self.forget_bias) + tf.nn.sigmoid(i) * tf.nn.tanh(j)
            new_h = tf.nn.tanh(new_c) * tf.nn.sigmoid(o)

            return new_h, tf.concat(1, [new_c, new_h])

    def variables(self):
        return self.layer.variables()

    def initial_state(self, batch_size, dtype=tf.float32):
        """Return state tensor (shape [batch_size x state_size]) filled with 0.
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          A 2D Tensor of shape [batch_size x state_size] filled with zeros.
        """
        zeros = tf.zeros(
            tf.pack([batch_size, 2 * self.hidden_size]), dtype=dtype)
        zeros.set_shape([None, 2 * self.hidden_size])
        return zeros

    def copy(self, scope=None):
        if scope is None:
            scope = self.scope + "_copy"
        with tf.variable_scope(scope):
            return LSTMLayer(self.input_size, self.hidden_size, self.forget_bias, scope=scope, given_layer=self.layer.copy())

class MultiLSTMLayer(object):
    def __init__(self, input_size, hidden_sizes, scope="MultiLSTMLayer", initialize=True):
        self.input_size   = input_size
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes,(tuple,list)) else [hidden_sizes]
        self.scope = scope

        self.lstms = None
        if initialize:
            with tf.variable_scope(scope):
                self.lstms = []
                prev_input = input_size
                for i, hidden_size in enumerate(hidden_sizes):
                    self.lstms.append(LSTMLayer(prev_input, hidden_size, scope='lstm_%d' % (i,)))
                    prev_input = hidden_size

    def __call__(self, inpt, state):
        assert len(state) == len(self.lstms)
        outputs   = []
        new_state = []
        lower_inpt = inpt
        for lstm, lstm_prev_state in zip(self.lstms, state):
            o, s = lstm(lower_inpt, lstm_prev_state)
            outputs.append(o)
            new_state.append(s)
            lower_inpt = o
        return outputs, new_state

    def initial_state(self, batch_size, dtype=tf.float32):
        return [lstm.initial_state(batch_size, dtype=dtype) for lstm in self.lstms]

    def variables(self):
        res = []
        for lstm in self.lstms:
            res.extend(lstm.variables())
        return res

    def copy(self, scope=None):
        if scope is None:
            scope = self.scope + "_copy"
        with tf.variable_scope(scope):
            res = MultiLSTMLayer(self.input_size, self.hidden_sizes, scope=scope, initialize=False)
            res.lstms = [lstm.copy() for lstm in self.lstms]
            return res


class NLPLSTM(object):
    def __init__(self, embedding_size, nsymbols, lstm_hiddens, scope="NLPLSTM", initialize=True):
        self.embedding_size, self.nsymbols, self.lstm_hiddens = embedding_size, nsymbols, lstm_hiddens
        self.scope = scope
        self.embedding, self.lstm = None, None
        if initialize:
            with tf.variable_scope(self.scope):
                embedding_i =  tf.random_uniform_initializer(- 1.0 / math.sqrt(embedding_size),
                                                             1.0 / math.sqrt(embedding_size))
                self.embedding = tf.get_variable('embedding', (nsymbols, embedding_size), initializer=embedding_i)
                self.lstm = MultiLSTMLayer(embedding_size, lstm_hiddens)

    def __call__(self, words):
        embedded = tf.nn.embedding_lookup(self.embedding, words, name="embedded")
        lstm_inputs  = [ embedded[i,:,:] for i in range(embedded.get_shape().as_list()[0])]

        rnn_outputs = []
        rnn_states = []
        batch_size = tf.shape(lstm_inputs[0])[0]
        state = self.lstm.initial_state(batch_size)
        for input_ in lstm_inputs:
            output, state = self.lstm(input_, state)
            rnn_outputs.append(output)
            rnn_states.append(state)

        return rnn_outputs, rnn_states

    def variables(self):
        return [self.embedding] + self.lstm.variables()

    def copy(self, scope=None):
        if scope is None:
            scope = self.scope + "_copy"
        res = NLPLSTM(self.embedding_size, self.nsymbols, self.lstm_hiddens, scope=scope, initialize=False)
        with tf.variable_scope(scope):
            res.embedding = tf.get_variable(base_name(self.embedding), self.embedding.get_shape(),
                        initializer=lambda x, dtype=tf.float32: self.embedding.initialized_value())
            res.lstm     = self.lstm.copy()
        return res
