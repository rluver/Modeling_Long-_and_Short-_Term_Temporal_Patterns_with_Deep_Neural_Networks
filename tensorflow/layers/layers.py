from ...config.config import ModelConfig

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, GRU, Conv2D, Dropout, RepeatVector, Permute

class SkipGRU(GRU):
    def __init__(self, **kwargs):
        super(SkipGRU, self).__init__(**kwargs)
        self.initial_state = None
        self.gru = GRU(units=kwargs.get('units'), return_state=True, activation=kwargs.get('activation'))
    
    def call(self, x, training=False):
        hidden_states = []
        outputs = []

        for p in range(ModelConfig.p, 0, -1):
            input_t = x[:, -(p-1), :]
            input_t = tf.expand_dims(input_t, axis=1)
            hidden_state, cell_state = self.gru(input_t, initial_state=self.initial_state, training=training)
            self.initial_state = cell_state

            hidden_states.append(hidden_state)
            outputs.append(cell_state)
        
        outputs = tf.reduce_sum(outputs, axis=0)
        hidden_states = tf.reduce_sum(hidden_states, axis=0)

        return outputs, hidden_states


class LSTLayer(Layer):
    '''
    h: desirable horizon
    n: the variable dimension
    X: n X T
    w: time dimension kernel size
    d_c: the number of filters
    p: the number of hidden cells skipped through
    h_t^R: hidden state at t
    h_(t-p+1): hiddenstate of recurrent-skip component from t - p + 1
    q: window for highway
    '''
    def __init__(
        self, 
        **kwargs):
        super(LSTLayer, self).__init__(self, config=kwargs)
        self.config = ModelConfig
        self.n = None

    def call(self, x):
        # expand dim
        # change axis (time, feature) -> (feature, time)
        x = Permute((2, 1))(x)
        x_expanded_dim = tf.expand_dims(x, -1)
        self.n = x.shape[1]

        # Convolutional Layer
        conv_output = Conv2D(
            filters=self.config.num_filters, 
            kernel_size=(self.n, self.config.window),
            padding=self.config.padding,
            activation=self.config.activation
            )(x_expanded_dim)  # 1 x T x d_c
        conv_output = tf.reshape(conv_output, shape=(-1, conv_output.shape[-2], conv_output.shape[-1]))  # T x d_c
        conv_output = Dropout(rate=self.config.dropout_rate)(conv_output)

        # Recurrent
        recurrent_output = GRU(units=self.config.recurrent_units, activation='relu')(conv_output)
        recurrent_output = Dropout(rate=self.config.dropout_rate)(recurrent_output)

        # Recurrent-skip layer
        assert self.config.skip_length > 0
        conv_output_sliced = conv_output[:, -(self.config.skip_length-1):, :]
        recurrent_skip_output = GRU(units=self.config.recurrent_skip_units, activation='relu')(conv_output_sliced)
        recurrent_skip_output = Dropout(rate=self.config.dropout_rate)(recurrent_skip_output)

        # dense layer to combine outputs
        concat_layer = tf.concat([recurrent_output, recurrent_skip_output], axis=1)
        dense_layer = Dense(units=self.n)(concat_layer)

        # Autoregressive Componenet
        assert self.config.highway_window > 0
        y = x[:, :, -self.config.highway_window:]
        highway_layer = Dense(units=1)(y)
        ar_output = tf.reshape(highway_layer, shape=(-1, self.n))

        # output
        output = dense_layer + ar_output
        
        return output
        