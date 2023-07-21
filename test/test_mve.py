import unittest
import numpy as np
import keras_core as keras

import tensorflow_probability as tfp

from capsa.utils import generate_moon_data_classification

from capsa import MVEWrapper


class test_mve(unittest.TestCase):

    def test_points(self):
        
        (x_train, y_train) = generate_moon_data_classification()

        max_points = np.load("data/mve_max_points.npy")
        min_points = np.load("data/mve_min_points.npy")

        model = keras.Sequential(
        [
            keras.Input(shape=(2,)),
            keras.layers.Dense(8, "relu"),
            keras.layers.Dense(8, "relu"),
            keras.layers.Dense(8, "relu"),
            keras.layers.Dense(8, "relu"),
            keras.layers.Dense(2,"softmax"),
        ]
        )

        wrapped_model = MVEWrapper(model,is_classification=True)


        wrapped_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=2e-3),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.Accuracy()],
        )

        wrapped_model.fit(x_train, keras.ops.one_hot(y_train,2), epochs=20)


        max_results = keras.ops.mean(wrapped_model(max_points, return_risk=True)[1],axis=-1)
        min_results = keras.ops.mean(wrapped_model(min_points, return_risk=True)[1],axis=-1)

        self.assertGreater(keras.ops.mean(max_results),keras.ops.mean(min_results))


if __name__ == "__main__":
    unittest.main()
