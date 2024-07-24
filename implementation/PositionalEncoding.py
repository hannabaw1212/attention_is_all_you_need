import numpy as np
import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model: int, max_len: int, dropout: float, **kwargs):
        """
        Initializes the PositionalEncoding layer.

        :param d_model: The size of each embedding vector.
        :param max_len: The maximum number of positions for which embeddings will be created.
        :param droupout: The dropout rate to apply to the output of this layer.
        :param kwargs: pass
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.positional_encoding = self._get_positional_encoding()

    def _get_positional_encoding(self):
        """
        Generates the positional encodings using sinusoidal patterns.

        Returns:
        Tensor: A tensor containing positional encodings of shape (1, max_len, d_model).
        """
        positions = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))  # Shape (d_model/2,)

        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)

        pe = pe[np.newaxis, ...]  # Add a new batch dimension (1, max_len, d_model)
        return tf.cast(pe, tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        assert seq_len <= self.max_len, "Input sequence length exceeds the maximum length"

        x = x + self.positional_encoding[:, :seq_len]
        return self.dropout(x)

