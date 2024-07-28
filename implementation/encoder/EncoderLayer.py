import tensorflow as tf
import numpy as np
from implementation.sub_blocks.FeedForward import FeedForward
from implementation.sub_blocks.InputEmbeddings import InputEmbeddings
from implementation.sub_blocks.MultiHeadAttention import MultiHeadAttention
from implementation.sub_blocks.PositionalEncoding import PositionalEncoding
from implementation.sub_blocks.AddAndNorm import AddAndNorm
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff, dropout)

        self.add_norm1 = AddAndNorm(epsilon=0.01)
        self.add_norm2 = AddAndNorm(epsilon=0.01)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask):
        # Multi-head attention sublayer
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout(attention_output)
        x = self.add_norm1(x, attention_output)

        # Feed-forward network sublayer
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.add_norm2(x, ffn_output)

        return x

