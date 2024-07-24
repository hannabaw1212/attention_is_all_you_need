import tensorflow as tf
import numpy as np

class InputEmbeddings(tf.keras.layers.Layer):
    def __init__(self, d_model, vocab_size, **kwargs):
        """
        :param d_model: The size of each embedding vector.
        :param vocab_size: The size of the vocabulary, defining the number of unique tokens.
        :param kwargs: pass
        """
        super(InputEmbeddings, self).__init__(**kwargs)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = tf.keras.layers.Embedding(vocab_size, d_model)

    def call(self, x):
        return self.embeddings(x) * np.sqrt(self.d_model)

