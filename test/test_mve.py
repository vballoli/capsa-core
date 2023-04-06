import unittest
import numpy as np
import tensorflow as tf

from tensorflow import keras
import tensorflow_probability as tfp

from capsa.utils import generate_moon_data_classification

from capsa import MVEWrapper


class test_mve(unittest.TestCase):

    def test_points(self):
        
        (x_train, y_train) = generate_moon_data_classification()

        max_points = np.load("data/mve_max_points.npy")
        min_points = np.load("data/mve_min_points.npy")

        model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,)),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(2,"softmax"),
        ]
        )

        wrapped_model = MVEWrapper(model,is_classification=True)


        wrapped_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.Accuracy()],
        )

        wrapped_model.fit(x_train, tf.one_hot(y_train,2), epochs=20)


        max_results = tf.reduce_mean(wrapped_model(max_points).aleatoric,axis=-1)
        min_results = tf.reduce_mean(wrapped_model(min_points).aleatoric,axis=-1)

        self.assertGreater(tf.reduce_mean(max_results),tf.reduce_mean(min_results))


if __name__ == "__main__":
    unittest.main()
