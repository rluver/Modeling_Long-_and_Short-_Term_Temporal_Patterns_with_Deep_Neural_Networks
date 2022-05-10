from ...config.config import ModelConfig

import tensorflow
from tensorflow.keras.layers import Layer, Dense, GRU, Conv1D, Dropout

    '''
    h: desirable horizon
    n: the variable dimension
    X: n X T
    w: time dimension kernel size
    d_c: the number of filters
    p: the number of hidden cells skipped through
    h_t^R: hidden state at t
    h_(t-p+1): hiddenstate of recurrent-skip component from t - p + 1
    '''
    def __init__(
        self, 
        **kwargs):
        self.config = ModelConfig
        super(Layer, self).__init__(self, config=kwargs)
        self.n = None

    def call(self, x):
        # Autoregressive
        ar_output = GRU(units=self.config.ar_units)(x)
        ar_output = Dropout(rates=self.config.dropout_rate)(ar_output)

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
        conv_output = tf.reshape(conv_output, shape=(-1, conv_output.shape[-2], conv_output.shape[-1]))  # T_c x d_c
        conv_output = Permute((2, 1))(conv_output)   # d_c x T
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
        