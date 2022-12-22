import unittest
import numpy as np
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import datasets
import tensorflow_probability as tfp

from capsa import MVEWrapper


def generate_moon_data_classification(noise=True):
    x, y = datasets.make_moons(n_samples=60000, noise=0.1)

    if noise:
        dstr = tfp.distributions.MultivariateNormalDiag(loc=[-0.7, 0.7], scale_diag=[0.03,0.05])
        p_flip = dstr.prob(x)
        result = tf.math.top_k(p_flip,k=5000,sorted=True)
        indices_to_flip = result.indices[0::5]
        print(x[result.indices[0:20]])
        
        y[indices_to_flip] = 1 - y[indices_to_flip]


    x = x.astype(float)
    y = y.astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    return (x_train, y_train), (x_test, y_test)


class test_mve(unittest.TestCase):

    def test_points(self):
        
        (x_train, y_train), (x_test, y_test) = generate_moon_data_classification()

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



