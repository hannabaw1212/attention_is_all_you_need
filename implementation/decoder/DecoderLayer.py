import tensorflow as tf
import numpy as np
from implementation.sub_blocks.MultiHeadAttention import MultiHeadAttention
from implementation.sub_blocks.FeedForward import FeedForward
from implementation.sub_blocks.AddAndNorm import AddAndNorm

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_mha = MultiHeadAttention(d_model, num_heads)

        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.add_norm1 = AddAndNorm()
        self.add_norm2 = AddAndNorm()
        self.add_norm3 = AddAndNorm()

        self.dropout1 = tf.keras.layers.Dropout(dropout)

    def call(self, x, encoder_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.masked_mha(x, x, x, look_ahead_mask)
        attn1 = self.dropout(attn1)
        out1 = self.add_norm1(x, attn1)

        # Multi-head attention over encoder output
        attn2, attn_weights_block2 = self.enc_dec_mha(out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout(attn2)
        out2 = self.add_norm2(out1, attn2)

        # Feed forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout(ffn_output)
        out3 = self.add_norm3(out2, ffn_output)

        return out3, attn_weights_block1, attn_weights_block2