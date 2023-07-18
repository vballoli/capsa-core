#import tensorflow as tf
import keras_core as keras

from .utils import _get_out_dim, copy_layer


class BaseWrapper(keras.Model):
    """Base class for a metric wrapper, all of our individual metric wrappers
    (``MVEWrapper``, ``HistogramWrapper``, ``DropoutWrapper``, etc.) subclass it.

    Serves two purposes:
        - abstracts away methods that are similar between different metric wrappers
          to reduce code duplication;
        - represents a "template class" that indicates which other methods users need
          to overwrite when creating their own metric wrappers.

    Transforms a model, into a risk-aware variant. Wrappers are given an arbitrary neural
    network and, while preserving the structure and function of the network, add and modify
    the relevant components of the model in order to be a drop-in replacement while being
    able to estimate the risk metric.

    In order to wrap an arbitrary neural network model, there are few distinct steps that
    every wrapper needs to follow:
        - extracting the feature extractor;
        - modifying the child;
        - adding new layers;
        - changing the loss.
    """

    def __init__(self, base_model):
        """
        We add a few instance variables in the ``init`` of the base class to make it
        available by default to the metric wrappers that subclass it.

        Parameters
        ----------
        base_model : tf.keras.Model
            A model to be transformed into a risk-aware variant.

        Attributes
        ----------
        feature_extractor : tf.keras.Model
            Creates a ``feature_extractor`` from the provided model.
        out_layer : tf.keras.layers.Layer
            A duplicate of the last layer of the base_model which is used to predict ``y_hat``
            (same output as before the wrapping).
        out_dim : int
            Number of units in the last layer.
        """
        super(BaseWrapper, self).__init__()

        # if base_model is not None:
        #     self.base_model = base_model
        #     self.feature_extractor = keras.Model(
        #         base_model.inputs, base_model.layers[-2].output
        #     )

        #     last_layer = base_model.layers[-1]
        #     self.out_layer = copy_layer(last_layer)
        #     self.out_dim = _get_out_dim(base_model)
    

    # def train_step(self, data, prefix=None):
    #     """
    #     The logic for one training step.

    #     Adds the compiled loss such that the models that subclass this class don't need to explicitly add it.
    #     Thus the ``metric_loss`` returned from such a model is not expected to reflect the compiled
    #     (user specified) loss -- because it is added here.

    #     Note: This method could be overwritten in subclasses, but the rule of thumb is to try to avoid
    #     overwriting it unless it's absolutely necessary, as e.g. in the ``EnsembleWrapper`` (functionality
    #     of that wrapper cannot be achieved without overwriting ``BaseWrapper``'s ``train_step``). But in general,
    #     try to only overwrite ``BaseWrapper``'s ``loss_fn`` and ``call`` methods -- in most of the cases it
    #     should be enough.

    #     Parameters
    #     ----------
    #     data : tuple
    #         (x, y) pairs, as in the regular Keras ``train_step``.
    #     prefix : str, default None
    #         Used to modify entries in the dict of `keras metrics <https://keras.io/api/metrics/>`_
    #         such that they reflect the name of the metric wrapper that produced them (e.g., mve_loss: 2.6763).
    #         Note, keras metrics dict contains e.g. loss values for the current epoch/iteration
    #         not to be confused with what we call "metric wrappers".

    #     Returns
    #     -------
    #     keras_metrics : dict
    #         `Keras metrics <https://keras.io/api/metrics/>`_.
    #     """
    #     x, y = data

    #     with tf.GradientTape() as t:
    #         metric_loss, y_hat = self.loss_fn(x, y)
    #         compiled_loss = self.compiled_loss(
    #             y, y_hat, regularization_losses=self.losses
    #         )
    #         loss = metric_loss + compiled_loss

    #     trainable_vars = self.trainable_variables
    #     gradients = t.gradient(loss, trainable_vars)
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    #     self.compiled_metrics.update_state(y, y_hat)
    #     prefix = self.metric_name if prefix == None else prefix
    #     keras_metrics = {
    #         f"{prefix}_compiled_{m.name}": m.result() for m in self.metrics
    #     }
    #     keras_metrics[f"{prefix}_wrapper_loss"] = loss

    #     return keras_metrics
    
    # @tf.function
    # def test_step(self, data, prefix=None):
    #     """
    #     The logic for one evaluation step.

    #     Note: This method could be overwritten in subclasses, but the rule of thumb is to try to avoid
    #     overwriting it unless it's absolutely necessary, as e.g. in the ``EnsembleWrapper`` (functionality
    #     of that wrapper cannot be achieved without overwriting ``BaseWrapper``'s ``test_step``). But in general,
    #     try to only overwrite ``BaseWrapper``'s ``loss_fn`` and ``call`` methods -- in most of the cases it
    #     should be enough.

    #     Parameters
    #     ----------
    #     data : tuple
    #         (x, y) pairs, as in the regular Keras ``test_step``.
    #     prefix : str, default None
    #         Used to modify entries in the dict of `keras metrics <https://keras.io/api/metrics/>`_
    #         such that they reflect the name of the metric wrapper that produced them (e.g., mve_loss: 2.6763).
    #         Note, keras metrics dict contains e.g. loss values for the current epoch/iteration
    #         not to be confused with what we call "metric wrappers".

    #     Returns
    #     -------
    #     keras_metrics : dict
    #         `Keras metrics <https://keras.io/api/metrics/>`_`.
    #     """
    #     x, y = data

    #     metric_loss, y_hat = self.loss_fn(x, y)
    #     compiled_loss = self.compiled_loss(y, y_hat, regularization_losses=self.losses)
    #     loss = metric_loss + compiled_loss

    #     self.compiled_metrics.update_state(y, y_hat)
    #     prefix = self.metric_name if prefix == None else prefix
    #     # prefix 'val' is added by tf.keras automatically, so no need to add it here
    #     keras_metrics = {
    #         f"{prefix}_compiled_{m.name}": m.result() for m in self.metrics
    #     }
    #     keras_metrics[f"{prefix}_wrapper_loss"] = loss

    #     return keras_metrics

    def loss_fn(self, x, y):
        """
        An empty method, raises exception to indicate that this method requires derived classes to override it.

        Note: This method is used in the "train_step" and the "test_step" methods, thus this method is not
        required to be overwritten if both the "train_step" and the "test_step" methods themselves are overwritten.

        Parameters
        ----------
        x : tf.Tensor
            Input.
        y : tf.Tensor
            Ground truth label.

        Raises
        ------
        AttributeError
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    def call(self, x, training=False, return_risk=True):
        """
        An empty method, raises exception to indicate that this method requires derived classes to override it.

        Parameters
        ----------
        x : tf.Tensor
            Input.
        training : bool, default False
            Can be used to specify a different behavior in training and inference.
        return_risk : bool, default True
            Indicates whether or not to output a risk estimate in addition to the model's prediction.

        Raises
        ------
        AttributeError
        """
        raise NotImplementedError("Must be implemented in subclasses.")
