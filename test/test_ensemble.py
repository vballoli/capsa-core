import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from capsa import EnsembleWrapper, MVEWrapper, VAEWrapper
from capsa.utils import (
    get_user_model,
    plot_loss,
    get_preds_names,
    plot_risk_2d,
    plot_epistemic_2d,
)
from data import get_data_v1, get_data_v2


def test_ensemble(use_case):

    user_model = get_user_model()
    ds_train, ds_val, x, y, x_val, y_val = get_data_v2(batch_size=256, is_show=False)

    if use_case == 1:

        model = EnsembleWrapper(user_model, num_members=3)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss=tf.keras.losses.MeanSquaredError(),
            # optionally, metrics could also be specified
            metrics=tf.keras.metrics.CosineSimilarity(name="cos"),
        )

        history = model.fit(x, y, epochs=30, validation_data=(x_val, y_val))
        plot_loss(history)

        risk_tensor = model(x_val)
        plot_risk_2d(x_val, y_val, risk_tensor, model.metric_name)

    elif use_case == 2:

        model = EnsembleWrapper(user_model, metric_wrapper=MVEWrapper, num_members=5)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        history = model.fit(ds_train, epochs=30, validation_data=(x_val, y_val))
        plot_loss(history)

        risk_tensor = model(x_val)
        plot_risk_2d(x_val, y_val, risk_tensor, model.metric_name)


test_ensemble(1)
test_ensemble(2)
