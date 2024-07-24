import tensorflow as tf
import numpy as np

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model: int, d_ff: int, dropout: float, **kwargs):
        """
        Initializes the FeedForward layer.

        :param d_model: The size of the input and output embeddings.
        :param d_ff: The hidden layer size of the feed-forward network.
        :param dropout_rate: The dropout rate to apply to the output of the feed-forward network.
        """
        super(FeedForward, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        """
        Forward pass for the FeedForward layer.

        :param x: Input tensor.
        :return: Output tensor after applying the feed-forward network and dropout.
        """
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
