from random import sample

import tensorflow as tf
from tensorflow.keras import layers

from ..utils import copy_layer, _get_out_dim
from ..base_wrapper import BaseWrapper
import tensorflow_probability as tfp
from ..risk_tensor import RiskTensor



def kl(mu, log_std):
    return -0.5 * tf.reduce_mean(
        1 + log_std - tf.math.square(mu) - tf.math.square(tf.math.exp(log_std)),
        axis=-1,
    )

def mse(y, y_hat, reduce=True):
    ax = list(range(1, len(y.shape)))
    mse = tf.reduce_sum(
        (y - y_hat) ** 2,
        axis=ax,
        keepdims=(False if reduce else True),
    )
    return tf.expand_dims(mse,axis=-1) if reduce else mse

class HistogramVAEWrapper(BaseWrapper):
    """
    Combines the functionalities of both HistogramWrapper and VAEWrapper. This is done by converting a given base_model into Variational AutoEncoder architecture, with latent dimension defined by the given parameter ``latent_dim``.

    VAEs are typically used to learn a robust, low-dimensional representation
    of the latent space. They can be used as a method of estimating epistemic
    uncertainty by using the reconstruction loss MSE(x, x_hat) - in cases of
    out-of-distribution data, samples that are hard to learn, or underrepresented
    samples, we expect that the VAE will have high reconstruction loss, since the
    mapping to the latent space will be less accurate. Conversely, when the model
    is very familiar with the features being fed in, or the data is in distribution,
    we expect the latent space mapping to be robust and the reconstruction loss to be low.

    A histogram distribution is constructed from the mean layer of the VAE architecture. This histogram is used to estimate a    
    """

    def __init__(self, base_model, latent_dim,queue_size,num_bins=5,decoder=None):
        """
        Parameters
        ----------
        base_model : tf.keras.Model
            A model to be transformed into a risk-aware variant.
        decoder : tf.keras.Model, default None
            To construct the VAE for any given model in capsa, we use the feature extractor as the encoder,
            and reverse the feature extractor automatically when possible to create a decoder.
        latent_dim : int
            Defines the dimension of latent-space. Making this very small can cause the bottleneck to lose too much information, resulting in high reconstruction loss. On the contrary, making it too big can cause the passing information to not become compressed enough, resulting in less-meaningful latent features.
        queue_size : int
            The size of the internal queue data-structure to use for the histogram
        num_bins : int
            How many bins to use in the histogram

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
        super(HistogramVAEWrapper, self).__init__(base_model)

        self.metric_name = "vae"
        self.mean_layer = tf.keras.layers.Dense(latent_dim)
        self.log_std_layer = tf.keras.layers.Dense(latent_dim)
        self.queue_built = False
        self.queue_size = queue_size
        self.num_bins = num_bins

        if decoder != None:
            self.decoder = decoder
        # reverse model if we can, accept user decoder if we cannot
        elif hasattr(self.feature_extractor, "layers"):
            self.decoder = self.reverse_model(self.feature_extractor, latent_dim)
        else:
            raise ValueError(
                "If you provide a subclassed model, \
                the decoder must also be specified"
            )

    def call(self, x, training=False, return_risk=True, T=1):
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
        features = self.feature_extractor(x, training)
        y_hat = self.out_layer(features)

        if not return_risk:
            return y_hat
        else:
            mu = self.mean_layer(features)
            log_std = self.log_std_layer(features)

            #Keras Model.build doesn't work due to train_step calling loss_fn instead of the model, we need to find a solution to that in the future to remove the horrible next 3 lines of code
            if self.queue_built == False:

                self.build_queue(mu)
                self.queue_built = True

            bias = self.get_histogram_probability(mu)

            # deterministic
            if T == 1:
                rec = self.decoder(mu, training)
                epistemic = mse(x, rec, reduce=True)

                epistemic = tf.repeat(input=epistemic,repeats=y_hat.shape[-1],axis=-1)
                bias = tf.repeat(input=tf.expand_dims(bias,axis=-1),repeats=y_hat.shape[-1],axis=-1)
                if training:
                    return y_hat, epistemic, bias
                else:
                    return RiskTensor(y_hat,epistemic=epistemic,bias=bias)

            # stochastic
            else:
                recs = []
                for _ in T:
                    sampled_latent = self.sampling(mu, log_std)
                    recs.append(self.decoder(sampled_latent))
                std = tf.reduce_std(recs)
                return y_hat,std,bias



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


        #Keras Model.build doesn't work due to train_step calling loss_fn instead of the model, we need to find a solution to that in the future to remove the horrible next 3 lines of code
        if self.queue_built == False:    
            self.build_queue(mu)
            self.queue_built = True

        self.add_queue(mu)
        bias = self.get_histogram_probability(mu)

        sampled_latent = self.sampling(mu, log_std)
        rec = self.decoder(sampled_latent)
        loss = kl(mu, log_std) + mse(x, rec)
        return loss, y_hat, bias
    

    #@tf.function
    def train_step(self, data, prefix=None):
        """
        The logic for one training step.

        Adds the compiled loss such that the models that subclass this class don't need to explicitly add it.
        Thus the ``metric_loss`` returned from such a model is not expected to reflect the compiled
        (user specified) loss -- because it is added here.

        Note: This method could be overwritten in subclasses, but the rule of thumb is to try to avoid
        overwriting it unless it's absolutely necessary, as e.g. in the ``EnsembleWrapper`` (functionality
        of that wrapper cannot be achieved without overwriting ``BaseWrapper``'s ``train_step``). But in general,
        try to only overwrite ``BaseWrapper``'s ``loss_fn`` and ``call`` methods -- in most of the cases it
        should be enough.

        Parameters
        ----------
        data : tuple
            (x, y) pairs, as in the regular Keras ``train_step``.
        prefix : str, default None
            Used to modify entries in the dict of `keras metrics <https://keras.io/api/metrics/>`_
            such that they reflect the name of the metric wrapper that produced them (e.g., mve_loss: 2.6763).
            Note, keras metrics dict contains e.g. loss values for the current epoch/iteration
            not to be confused with what we call "metric wrappers".

        Returns
        -------
        keras_metrics : dict
            `Keras metrics <https://keras.io/api/metrics/>`_.
        """
        x, y = data

        with tf.GradientTape() as t:
            metric_loss, y_hat,bias = self.loss_fn(x, y)
            compiled_loss = self.compiled_loss(
                y, y_hat, regularization_losses=self.losses
            )
            loss = metric_loss + compiled_loss

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_hat)
        prefix = self.metric_name if prefix == None else prefix
        keras_metrics = {
            f"{prefix}_compiled_{m.name}": m.result() for m in self.metrics
        }
        keras_metrics[f"{prefix}_wrapper_loss"] = loss


        #Possible area to add DB-VAE additions: the variable ``bias`` is available in this scope


        return keras_metrics


        
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
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    # Defining a Tensor Queue that saves the last ``queue_size`` values
    def build_queue(self, features):
        # Get the shape of the features
        feature_shape = tf.shape(features)

        # Create a queue with the shape of the features and an index to keep track of how many values are in the queue
        self.queue = tf.Variable(
            tf.zeros([self.queue_size, feature_shape[-1]]), trainable=False
        )
        self.queue_index = tf.Variable(0, trainable=False)
        
    # def build(self,input_shape):
    #     # Get the shape of the features
    #     feature_shape = self.mean_layer.output_shape

    #     # Create a queue with the shape of the features and an index to keep track of how many values are in the queue
    #     self.queue = tf.Variable(
    #         tf.zeros([self.queue_size, feature_shape[-1]]), trainable=False
    #     )
    #     self.queue_index = tf.Variable(0, trainable=False)


    def get_histogram_probability(self,features):

        """
        Get the probability of each feature in the histogram. This utilizes the internal queue data-structure to calculate the probability.

        Parameters
        ----------
        features : tf.Tensor
            Features to calculate probability for.

        Returns
        -------
        logits : tf.Tensor
            Calculated probabilities for each feature.
        """
        
        # DON'T NEED TO CALCULATE EVERY TIME!---------------------

        edges = self.get_histogram_edges()

        frequencies = tfp.stats.histogram(
            self.queue.value(),
            edges,
            axis=0,
            extend_lower_interval=True,
            extend_upper_interval=True,
        )

        epsilon = 1e-8

        # Normalize histograms
        hist_probs = tf.divide(frequencies, tf.reduce_sum(frequencies, axis=0)) + epsilon

        # DON'T NEED TO CALCULATE EVERY TIME!---------------------

        # Get the corresponding bins of the features
        bin_indices = tf.cast(
            tfp.stats.find_bins(
                features,
                edges,
                extend_lower_interval=True,
                extend_upper_interval=True,
            ),
            tf.dtypes.int32,
        )

        # Multiply probabilities together to compute bias
        second_element = tf.repeat(
            [tf.range(tf.shape(features)[1])], repeats=[tf.shape(features)[0]], axis=0
        )
        indices = tf.stack([bin_indices, second_element], axis=2)

        probabilities = tf.gather_nd(hist_probs, indices)

        logits = tf.reduce_sum(tf.math.log(probabilities), axis=1)

        # logits = logits - tf.math.reduce_mean(
        #     logits
        # )  # log probabilities are the wrong sign if we don't subtract the mean

        #return tf.math.softmax(logits)
        
        return tf.math.exp(logits)


    def add_queue(self, features):

        # Get the index of the queue
        index = self.get_queue_index(features)

        # Add the features to the queue
        queue_state = self.queue.value()
        updated_queue_state = tf.tensor_scatter_nd_update(
            queue_state, updates=features, indices=index
        )
        self.queue.assign(updated_queue_state)


    # Get the indices of where to insert new features and increment current-index by the length of indices
    def get_queue_index(self, features):

        # Get the index of the queue
        index = self.queue_index.value()

        batch_size = tf.shape(features)[0]

        # Get the index of the queue
        indices = tf.range(start=index, limit=(index + batch_size), name="range")

        # Increment the index by one and assign it to the class variable
        indices = tf.math.floormod(indices, self.queue_size)
        self.queue_index.assign(tf.math.add(indices[-1], 1))

        # Return the old index
        return tf.expand_dims(indices, axis=1)

    def get_histogram_edges(self):

        # Get queue values
        queue_state = self.queue.value()

        queue_minimums = tf.math.reduce_min(queue_state, axis=0)
        queue_maximums = tf.math.reduce_max(queue_state, axis=0)

        edges = tf.linspace(queue_minimums, queue_maximums, self.num_bins + 1)

        return edges

    def reverse_model(self,model, latent_dim):
        inputs = tf.keras.Input(shape=latent_dim)
        i = len(model.layers) - 1
        while type(model.layers[i]) != layers.InputLayer and i >= 0:
            if i == len(model.layers) - 1:
                x = self.reverse_layer(model.layers[i])(inputs)
            else:
                if type(model.layers[i - 1]) == layers.InputLayer:
                    original_input = model.layers[i - 1].input_shape
                    x = self.reverse_layer(model.layers[i], original_input)(x)
                else:
                    x = self.reverse_layer(model.layers[i])(x)
            i -= 1
        return tf.keras.Model(inputs, x)


    def reverse_layer(self,layer, output_shape=None):
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
            config["units"] = layer.input_shape[-1]
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
