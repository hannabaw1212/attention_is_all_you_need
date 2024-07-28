import tensorflow as tf
import numpy as np


class AddAndNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        """
        Initializes the AddAndNorm layer.

        :param epsilon: Small float added to variance to avoid dividing by zero.
        :param kwargs: Additional arguments for the layer.
        """
        super(AddAndNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon)

    def call(self, x, sub_layer_output):
        return self.layer_norm(x + sub_layer_output)


