from ...config.config import ModelConfig

import tensorflow
from tensorflow.keras.layers import Layer, Dense, GRU, Conv1D, Dropout


class Layer(Layer):
    def __init__(
        self, 
        config=ModelConfig,
        **kwargs):
        self.config = ModelConfig
        super(Layer, self).__init__(self, config=kwargs)

    def call(self, x):
        # Autoregressive
        ar_output = GRU(units=self.config.ar_units)(x)
        ar_output = Dropout(rates=self.config.dropout_rate)(ar_output)

        # Convolutional Layer
        conv_output = Conv1D(
            filters=self.config.num_filters, 
            kernel_size=self.config.kernel_size,
            padding=self.config.padding,
            activation=self.config.activation
            ) # num_filters x T
        conv_output = Dropout(rates=self.config.dropout_rate)(conv_output)

        # Recurrent and Recurrent-skip layer
        recurrent_output = GRU(units=self.config.recurrent_units)(conv_output)
        recurrent_output = Dropout(rates=self.config.dropout_rate)(recurrent_output)

        recurrent_skip_output = GRU(units=self.config.recurrent_units)(conv_output)


        recurrent_skip_output = GRU(units=self.config.recurrent_skip_units)

        # Fully connected and elemen-wise sum output

        
        