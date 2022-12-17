import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from capsa import VAEWrapper
from capsa.utils import (
    get_user_model,
    plot_loss,
    get_preds_names,
    plot_risk_2d,
    plot_epistemic_2d,
)
from data import get_data_v2


def test_vae():

    # NOTE: the code below is intended to demonstrate how could the VAEWrapper
    # be initialized and used to wrap a user model. In practice, this wrapper
    # should not be used with 1-dim inputs (see VAEWrapper's documentation).
    user_model = get_user_model()
    ds_train, ds_val, _, _, x_val, y_val = get_data_v2(batch_size=256, is_show=False)

    model = VAEWrapper(user_model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss=tf.keras.losses.MeanSquaredError(),
        # optionally, metrics could also be specified
        metrics=tf.keras.metrics.CosineSimilarity(name="cos"),
    )

    history = model.fit(ds_train, epochs=30, validation_data=(x_val, y_val))
    plot_loss(history)

    risk_tensor = model(x_val)

    preds_names = get_preds_names(history)
    plot_risk_2d(x_val, y_val, risk_tensor, preds_names[0])
    # plot_epistemic_2d(x, y, x_val, y_val, risk_tensor)


test_vae()
