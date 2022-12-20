import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from capsa import MVEWrapper  # , HistogramWrapper, HistogramCallback
from capsa.utils import get_user_model, plot_loss, get_preds_names, plot_risk_2d
from data import get_data_v2


def test_regression():

    user_model = get_user_model()
    ds_train, ds_val, _, _, x_val, y_val = get_data_v2(batch_size=256, is_show=False)

    model = MVEWrapper(user_model, is_classification=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss=tf.keras.losses.MeanSquaredError(),
        # optionally, metrics could also be specified
        metrics=tf.keras.metrics.CosineSimilarity(name="cos"),
    )
    history = model.fit(ds_train, epochs=10, validation_data=(x_val, y_val))
    plot_loss(history)

    risk_tensor = model(x_val)


# def test_bias():

#     user_model = get_user_model()
#     ds_train, ds_val, _, _, x_val, y_val = get_data_v2(batch_size=256, is_show=False)

#     model = HistogramWrapper(user_model)

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
#         loss=tf.keras.losses.MeanSquaredError(),
#     )

#     history = model.fit(ds_train, epochs=10, callbacks=[HistogramCallback()])
#     plot_loss(history)

#     risk_tensor = model(x_val)
#     plot_risk_2d(x_val, y_val, risk_tensor, "histogram")


test_regression()
