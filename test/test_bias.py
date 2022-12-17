import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from capsa import (
    HistogramWrapper,
    VAEWrapper,
    HistogramCallback,
)
from capsa.utils import get_user_model, plt_vspan, plot_loss
from data import get_data_v1, get_data_v2


def test_bias():

    user_model = get_user_model()
    ds_train, ds_val, x, y, x_val, y_val = get_data_v2(batch_size=256)

    model = HistogramWrapper(user_model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    history = model.fit(ds_train, epochs=30, callbacks=[HistogramCallback()])
    plt.plot(history.history["histogram_loss"])
    plt.show()

    y_pred, bias = model(x_val)


def test_bias_chained():
    user_model = get_user_model()
    ds_train, ds_val, x, y, x_val, y_val = get_data_v2(batch_size=256)

    model = HistogramWrapper(user_model, metric_wrapper=VAEWrapper)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    history = model.fit(ds_train, epochs=30, callbacks=[HistogramCallback()])

    y_pred, bias = model(x_val)
    y_pred, recon_loss = model.metric_wrapper(x_val)
    fig, axs = plt.subplots(3)
    axs[0].scatter(x_val, y_val, s=0.5, label="gt")
    axs[0].scatter(x_val, y_pred, s=0.5, label="yhat")
    plt_vspan()
    axs[1].scatter(x_val, bias, s=0.5, label="bias")
    axs[2].scatter(x_val, recon_loss, s=0.5, label="recon loss")
    plt_vspan()
    plt.legend()
    plt.show()


test_bias()
test_bias_chained()
