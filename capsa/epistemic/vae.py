from random import sample

# import tensorflow as tf
import keras_core as keras
from keras_core import layers

from ..utils import copy_layer, _get_out_dim
from ..base_wrapper import BaseWrapper
from ..risk_tensor import RiskTensor


def kl(mu, log_std):
    return -0.5 * keras.ops.mean(
        1 + log_std - keras.ops.square(mu) - keras.ops.square(keras.ops.exp(log_std)),
        axis=-1,
    )


def mse(y, y_hat, reduce=True):
    ax = list(range(1, len(y.shape)))
    mse = keras.ops.sum(
        (y - y_hat) ** 2,
        axis=ax,
        keepdims=(False if reduce else True),
    )
    return keras.ops.sum(mse) if reduce else mse


class VAEWrapper(keras.Model):
    """Uses Variational autoencoders (VAEs) (Kingma & Welling, 2013) to estimate
    epistemic uncertainty.

    VAEs are typically used to learn a robust, low-dimensional representation
    of the latent space. They can be used as a method of estimating epistemic
    uncertainty by using the reconstruction loss MSE(x, x_hat) - in cases of
    out-of-distribution data, samples that are hard to learn, or underrepresented
    samples, we expect that the VAE will have high reconstruction loss, since the
    mapping to the latent space will be less accurate. Conversely, when the model
    is very familiar with the features being fed in, or the data is in distribution,
    we expect the latent space mapping to be robust and the reconstruction loss to be low.

    We're making a restrictive assumption about the prior. Our prior over latent space is a
    standard unit diagonal gaussian. In other words, the encoder doesn't output a full covariance
    matrix over all dimensions (doesn't output a high dim Gaussian).

    NOTE: in the VAEWrapper we bottleneck the representation inside the model,
    reconstruct the input from that low dimensional representation, and use the MSE
    between the input and its reconstruction as a measure of epistemic uncertainty.
    However, if the input is already very low dimensional, it's unreasonable to
    talk about bottlenecking this representation even further -- thus it doesn't
    make sense to use the VAEWrapper with e.g. 1-dim inputs.

    Example of usage:
        >>> # initialize a keras model
        >>> user_model = Unet()
        >>> # wrap the model to transform it into a risk-aware variant
        >>> model = VAEWrapper(user_model)
        >>> # compile and fit as a regular tf.keras model
        >>> model.compile(...)
        >>> model.fit(...)
    """

    def __init__(self, base_model, decoder=None):
        """
        Parameters
        ----------
        base_model : tf.keras.Model
            A model to be transformed into a risk-aware variant.
        decoder : tf.keras.Model, default None
            To construct the VAE for any given model in capsa, we use the feature extractor as the encoder,
            and reverse the feature extractor automatically when possible to create a decoder.

        Attributes
        ----------
        metric_name : str
            Represents the name of the metric wrapper.
        mean_layer : tf.keras.layers.Layer
            Used to predict mean of the diagonal gaussian representing the latent space.
        log_std_layer : tf.keras.layers.Layer
            Used to predict variance of the diagonal gaussian representing the latent space.
        feature_extractor : tf.keras.Model
            Creates a ``feature_extractor`` by removing last layer from the ``base_model``.
        """
        super(VAEWrapper, self).__init__()
        self.base_model = base_model
        self.feature_extractor = keras.Model(
            base_model.inputs, base_model.layers[-2].output
        )
        self.out_layer = base_model.layers[-1]
        self.out_dim = _get_out_dim(base_model)

        self.metric_name = "vae"
        latent_dim = self.out_dim[-1]
        self.mean_layer = keras.layers.Dense(latent_dim)
        self.log_std_layer = keras.layers.Dense(latent_dim)

        if decoder != None:
            self.decoder = decoder
        # reverse model if we can, accept user decoder if we cannot
        elif hasattr(self.feature_extractor, "layers"):
            self.decoder = reverse_model(self.feature_extractor, latent_dim)
        else:
            raise ValueError(
                "If you provide a subclassed model, \
                the decoder must also be specified"
            )

    @staticmethod
    def sampling(z_mean, z_log_var):
        """
        Samples from the latent space defied by ``z_mean`` and ``z_log_var``.
        Uses the reparameterization trick to allow to backpropagate through
        the stochastic node.

        Parameters
        ----------
        z_mean : tf.Tensor
            Mean of the diagonal gaussian representing the latent space.
        z_log_var : tf.Tensor
            Log variance of the diagonal gaussian representing the latent space.

        Returns
        -------
        sampled_vector : tf.Tensor
            Vector sampled from the latent space according to the predicted parameters
            of the normal distribution.
        """
        epsilon = keras.random_normal(shape=keras.ops.shape(z_mean))
        return z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon

    def loss_fn(self, x, _):
        """
        Calculates the VAE loss by sampling and then feeding the latent vector
        through the decoder.

        Parameters
        ----------
        x : tf.Tensor
            Input.

        Returns
        -------
        loss : tf.Tensor
            Float, reflects how well does the algorithm perform given the ground truth label,
            predicted label and the metric specific loss function.
        y_hat : tf.Tensor
            Predicted label.
        """
        features = self.feature_extractor(x, True)

        y_hat = self.out_layer(features)
        mu = self.mean_layer(features)
        log_std = self.log_std_layer(features)

        sampled_latent = self.sampling(mu, log_std)
        rec = self.decoder(sampled_latent)
        loss = kl(mu, log_std) + mse(x, rec)
        return loss, y_hat

    def call(self, x, training=False, return_risk=False, T=1):
        """
        Forward pass of the model. The epistemic risk estimate could be calculated differently:
        by running either (1) deterministic or (2) stochastic forward pass.

        Parameters
        ----------
        x : tf.Tensor
            Input.
        training : bool, default False
            Can be used to specify a different behavior in training and inference.
        return_risk : bool, default True
            Indicates whether or not to output a risk estimate in addition to the model's prediction.
        T : int, default 1
            Defines will the model be run deterministically or stochastically, and the number of times
            to sample from the latent space (if run stochastically).

        Returns
        -------
        out : capsa.RiskTensor
            Risk aware tensor, contains both the predicted label y_hat (tf.Tensor) and the epistemic
            uncertainty estimate (tf.Tensor).
        """
        features = self.feature_extractor(x, training=training)
        y_hat = self.out_layer(features)

        if not return_risk:
            return y_hat
        else:
            mu = self.mean_layer(features)
            log_std = self.log_std_layer(features)

            # deterministic
            if T == 1:
                rec = self.decoder(mu, training=training)
                epistemic = mse(x, rec, reduce=False)
                epistemic = keras.ops.repeat(x=epistemic,repeats=y_hat.shape[-1],axis=-1)
                return y_hat, epistemic

            # stochastic
            else:
                recs = []
                for _ in T:
                    sampled_latent = self.sampling(mu, log_std)
                    recs.append(self.decoder(sampled_latent))
                std = keras.ops.std(recs)
                std = keras.ops.repeat(x=std,repeats=y_hat.shape[-1],axis=-1)
                return y_hat, std

    def input_to_histogram(self, x, training=False, features=None):
        # needed to interface with the Histogram metric
        features = self.feature_extractor(x, training)
        mu = self.mean_layer(features)
        return mu


def reverse_model(model, latent_dim):
    inputs = keras.Input(shape=(latent_dim, ))
    i = len(model.layers) - 1
    while type(model.layers[i]) != layers.InputLayer and i >= 0:
        if i == len(model.layers) - 1:
            x = reverse_layer(model.layers[i])(inputs)
        else:
            if type(model.layers[i - 1]) == layers.InputLayer:
                # breakpoint()
                # original_input = model.layers[i - 1].shape
                x = reverse_layer(model.layers[i], (latent_dim, ))(x)
            else:
                x = reverse_layer(model.layers[i])(x)
        i -= 1
    return keras.Model(inputs, x)


def reverse_layer(layer, output_shape=None):
    config = layer.get_config()
    layer_type = type(layer)
    unchanged_layers = [layers.Activation, layers.BatchNormalization, layers.Dropout]
    # TODO: handle global pooling separately
    pooling_1D = [
        layers.MaxPooling1D,
        layers.AveragePooling1D,
        layers.GlobalMaxPooling1D,
    ]
    pooling_2D = [
        layers.MaxPooling2D,
        layers.AveragePooling2D,
        layers.GlobalMaxPooling2D,
    ]
    pooling_3D = [
        layers.MaxPooling3D,
        layers.AveragePooling3D,
        layers.GlobalMaxPooling3D,
    ]
    conv = [layers.Conv1D, layers.Conv2D, layers.Conv3D]

    if layer_type == layers.Dense:
        config["units"] = layer.input.shape[-1]
        return layers.Dense.from_config(config)
    elif layer_type in unchanged_layers:
        return type(layer).from_config(config)
    elif layer_type in pooling_1D:
        return layers.UpSampling1D(size=config["pool_size"])
    elif layer_type in pooling_2D:
        return layers.UpSampling2D(
            size=config["pool_size"],
            data_format=config["data_format"],
            interpolation="bilinear",
        )
    elif layer_type in pooling_3D:
        return layers.UpSampling3D(
            size=config["pool_size"],
            data_format=config["data_format"],
            interpolation="bilinear",
        )
    elif layer_type in conv:
        if output_shape != None:
            config["filters"] = output_shape[0][-1]

        if layer_type == layers.Conv1D:
            return layers.Conv1DTranspose.from_config(config)
        elif layer_type == layers.Conv2D:
            return layers.Conv2DTranspose.from_config(config)
        elif layer_type == layers.Conv3D:
            return layers.Conv3DTranspose.from_config(config)
    else:
        raise NotImplementedError()
