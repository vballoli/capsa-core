import unittest
import numpy as np
import tensorflow as tf

from tensorflow import keras
import tensorflow_probability as tfp

from capsa.utils import generate_moon_data_classification
from capsa import EnsembleWrapper

class test_ensemble(unittest.TestCase):

    def test_points(self):
        
        (x_train, y_train) = generate_moon_data_classification()

        max_points = np.load("data/ensemble_max_points.npy")
        min_points = np.load("data/ensemble_min_points.npy")

        model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(None,2)),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(1,"sigmoid"),
        ]
        )

        wrapped_model = EnsembleWrapper(model,num_members=5)

        wrapped_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )

        wrapped_model.fit(x_train, y_train, epochs=2)


        max_results = wrapped_model(max_points).epistemic
        min_results = wrapped_model(min_points).epistemic

        self.assertGreater(tf.reduce_mean(max_results),tf.reduce_mean(min_results))


if __name__ == "__main__":
    unittest.main()
