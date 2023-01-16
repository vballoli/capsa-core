import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt


def get_user_model():
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            layers.Dense(16, "relu"),
            layers.Dense(32, "relu"),
            layers.Dense(64, "relu"),
            layers.Dense(32, "relu"),
            layers.Dense(16, "relu"),
            layers.Dense(1, None),
        ]
    )


def get_decoder():
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32,)),
            layers.Dense(16, "relu"),
            layers.Dense(32, "relu"),
            layers.Dense(64, "relu"),
            layers.Dense(32, "relu"),
            layers.Dense(16, "relu"),
            layers.Dense(1, None),
        ]
    )


def plot_loss(history):
    for k, v in history.history.items():
        plt.plot(v, label=k)
    plt.legend(loc="upper right")
    plt.show()


def get_preds_names(history):
    l = list(history.history.keys())
    # cut of keras metric names -- e.g., 'MVEW_0_loss' -> 'MVEW_0'
    l_split = [i.rsplit("_", 1)[0] for i in l]
    # remove duplicates (if specified multiple keras metrics)
    return list(dict.fromkeys(l_split))


def plt_vspan():
    plt.axvspan(-6, -4, ec="black", color="grey", linestyle="--", alpha=0.3, zorder=3)
    plt.axvspan(4, 6, ec="black", color="grey", linestyle="--", alpha=0.3, zorder=3)
    plt.xlim([-6, 6])


def plot_epistemic_2d(x, y, x_val, y_val, y_pred, risk, k=3):
    # x, y = x[:, 0], y[:, 0] # x (20480, ), y (20480, )
    # x_val, y_val = x_val[:, 0], y_val[:, 0] # x_val (2304, ), y_val (2304, )
    # risk = risk[:, None]
    plt.scatter(x, y, label="train data")
    plt.plot(x_val, y_val, "g-", label="ground truth")
    plt.plot(x_val, y_pred, "r-", label="pred")
    plt.fill_between(
        x_val[:, 0],
        (y_pred - k * risk)[:, 0],
        (y_pred + k * risk)[:, 0],
        alpha=0.2,
        color="r",
        linestyle="-",
        linewidth=2,
        label="epistemic",
    )
    plt.legend()
    plt.show()


def plot_risk_2d(x_val, y_val, risk_tens, label):
    if risk_tens.aleatoric != None and risk_tens.epistemic != None:
        Exception(
            "RiskTensor has both aleatoric and epistemic uncertainties, please specify which one you would like to plot."
        )
    elif risk_tens.aleatoric != None:
        risk = risk_tens.aleatoric
    elif risk_tens.epistemic != None:
        risk = risk_tens.epistemic
    fig, axs = plt.subplots(2)
    axs[0].scatter(x_val, y_val, s=0.5, label="gt")
    axs[0].scatter(x_val, risk_tens.y_hat, s=0.5, label="yhat")
    plt_vspan()
    axs[1].scatter(x_val, risk, s=0.5, label=label)
    plt_vspan()
    plt.legend()
    plt.show()


def _get_out_dim(model):
    return model.layers[-1].output_shape[1:]


def copy_layer(layer, override_activation=None):
    # if no_activation is False, layer might or
    # might not have activation (depending on the config)
    layer_conf = layer.get_config()
    if override_activation:
        layer_conf["activation"] = override_activation
    # works for any serializable layer
    return type(layer).from_config(layer_conf)


def gen_calibration_plot(model, ds, path=None):
    mu_ = []
    std_ = []
    y_test_ = []

    # x_test_batch, y_test_batch = iter(ds).get_next()
    for (x_test_batch, y_test_batch) in ds:
        out = model(x_test_batch)
        mu_batch, std_batch = out.y_hat, unpack_risk_tensor(out, model.metric_name)

        mu_.append(mu_batch)
        std_.append(std_batch)
        y_test_.append(y_test_batch)

        B = x_test_batch.shape[0]
        # assert mu_batch.shape == (B, 128, 160, 1)
        # assert std_batch.shape == (B, 128, 160, 1)

    mu = np.concatenate(mu_)  # (3029, 128, 160, 1)
    std = np.concatenate(std_)  # (3029, 128, 160, 1)
    y_test = np.concatenate(y_test_)  # (3029, 128, 160, 1)

    # todo-high: need to do it for ensemble of mves as well
    if model.metric_name == "mve":
        std = np.sqrt(std)

    vals = []
    percentiles = np.arange(41) / 40
    for percentile in percentiles:
        # returns the value at the n% percentile e.g., stats.norm.ppf(0.5, 0, 1) == 0.0
        # in other words, if have a normal distrib. with mean 0 and std 1,
        # 50% of data falls below and 50% falls above 0.0.
        # (3029, 128, 160, 1)
        ppf_for_this_percentile = tfp.distributions.Normal(mu, std).quantile(percentile)
        vals.append(
            (y_test <= ppf_for_this_percentile.numpy()).mean()
        )  # (3029, 128, 160, 1) -> scalar

    plt.plot(percentiles, vals)
    plt.plot(percentiles, percentiles)
    plt.title(str(np.mean(abs(percentiles - vals))))

    if path != None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def unpack_risk_tensor(t, model_name):
    if model_name in ["mve"]:
        risk = t.aleatoric
    else:
        risk = t.epistemic
    return risk


def make_moons(n_samples=100, noise=None):
    """
    Generate two interleaving half circles
    """

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out


    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )

    if noise is not None:
        X += np.random.normal(scale=noise, size=X.shape)

    x = tf.convert_to_tensor(X,dtype=tf.float32)
    y = tf.convert_to_tensor(y,dtype=tf.float32)

    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_x = tf.gather(x, shuffled_indices).numpy()
    shuffled_y = tf.gather(y, shuffled_indices).numpy()

    return shuffled_x, shuffled_y

def generate_moon_data_classification(noise=True):

    x, y = make_moons(n_samples=60000, noise=0.1)

    if noise:
        dstr = tfp.distributions.MultivariateNormalDiag(loc=[-0.7, 0.7], scale_diag=[0.03,0.05])
        p_flip = dstr.prob(x)
        result = tf.math.top_k(p_flip,k=5000,sorted=True)
        indices_to_flip = result.indices[0::5]
        
        y[indices_to_flip] = 1 - y[indices_to_flip]

    x = x.astype(float)
    y = y.astype(float)
    return (x, y)