import numpy as np
import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_into_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result to shape (batch_size, num_heads, seq_len, depth).
        """
        x = tf.reshape(x,
                       (batch_size, -1, self.num_heads, self.depth))  # shape: (batch_size, seq_len_q, num_heads, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # shape: (batch_size, num_heads, seq_len_q, depth)

    def call(self, v, k, q, mask):
        batch_size = tf.shape(v)[0]

        q = self.wq(q)
        v = self.wv(v)
        k = self.wk(k)

        q_splitted = self.split_into_heads(q, batch_size)
        v_splitted = self.split_into_heads(v, batch_size)
        k_splitted = self.split_into_heads(k, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q_splitted, k_splitted, v_splitted,
                                                                                mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        """
        Calculate the attention weights.
            q, k, v must have matching leading dimensions.
            k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
            The mask has different shapes depending on its type(padding or look ahead) but it must be
            broadcastable for addition.
        """
        q_k_dot = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_q_k_dot = q_k_dot / np.sqrt(dk)

        if mask is not None:
            scaled_q_k_dot += (mask * -1e9)  # Add the mask to the scaled tensor.

        attention_weights = tf.nn.softmax(scaled_q_k_dot, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights
