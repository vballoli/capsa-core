import tensorflow as tf
from tensorflow.keras import optimizers as optim

from ..base_wrapper import BaseWrapper
from ..risk_tensor import RiskTensor


class EnsembleWrapper(BaseWrapper):
    """Uses an ensemble of N models (each one is randomly initialized) to accurately
    estimate epistemic uncertainty Lakshminarayanan et al. (2017).

    This approach presents the gold-standard of estimating epistemic uncertainty.
    However, it comes with significant computational costs.

    Example of usage:
        >>> # initialize a keras model
        >>> user_model = Unet()
        >>> # wrap the model to transform it into a risk-aware variant
        >>> model = EnsembleWrapper(user_model, metric_wrapper=MVEWrapper, num_members=3)
        >>> # compile and fit as a regular tf.keras model
        >>> model.compile(...)
        >>> model.fit(...)
    """

    def __init__(
        self,
        base_model,
        num_members=3,
        metric_wrapper=None,
        kwargs={},
    ):
        """
        Parameters
        ----------
        base_model : tf.keras.Model
            A model to be transformed into a risk-aware variant.
        num_members : int, default 3
            Number of members in the deep ensemble.
        metric_wrapper : capsa.BaseWrapper, default None
            Class object of an individual metric wrapper (which subclass ``capsa.BaseWrapper``) that
            user wants to ensemble, if it's ``None`` then this wrapper ensembles the ``base_model``.
        kwargs : dict
            Keyword args used to initialize metric wrappers, used only if ``metric_wrapper`` is provided.
            The kwargs are metric wrapper specific, they could be different depending on the wrapper to
            be ensembled. But they should not include the ``base_model`` keyword.


        Attributes
        ----------
        metric_name : str
            Represents the name of the metric wrapper.
        metrics_compiled : dict
            An empty dict, will be used to map ``metric_name``s (string identifiers) of the wrappers that
            a user wants to ensemble to their respective compiled models.
        """
        super(EnsembleWrapper, self).__init__(base_model)

        self.metric_name = "ensemble"
        self.metric_wrapper = metric_wrapper
        self.num_members = num_members
        self.metrics_compiled = {}
        self.kwargs = kwargs

    def compile(self, optimizer, loss, metrics=None):
        """
        Compiles every member in the deep ensemble. Overrides ``tf.keras.Model.compile()``.

        If user passes only 1 ``optimizer`` and ``loss_fn`` yet they specified e.g. ``num_members``=3,
        duplicate that one ``optimizer`` and ``loss_fn`` for all members in the ensemble.

        Parameters
        ----------
        optimizer : tf.keras.optimizer or list
        loss : tf.keras.losses or list
        metrics : tf.keras.metrics or list, default None
        """
        super(EnsembleWrapper, self).compile()

        optimizer = [optimizer] if not isinstance(optimizer, list) else optimizer
        loss = [loss] if not isinstance(loss, list) else loss
        metrics = [metrics] if not isinstance(metrics, list) else metrics

        # if user passes only 1 optimizer or loss_fn yet they specified e.g. num_members=3,
        # duplicate that one optimizer and loss_fn for all members in the ensemble
        if len(optimizer) < self.num_members:
            optim_conf = optim.serialize(optimizer[0])
            optimizer = [optim.deserialize(optim_conf) for _ in range(self.num_members)]
        # losses and *most* keras metrics are stateless, no need to serialize as above
        if len(loss) < self.num_members:
            loss = [loss[0] for _ in range(self.num_members)]
        if len(metrics) < self.num_members:
            metrics = [metrics[0] for _ in range(self.num_members)]

        base_model_config = self.base_model.get_config()
        assert base_model_config != {}, "Please implement get_config()."

        for i in range(self.num_members):

            if isinstance(self.base_model, tf.keras.Sequential):
                m = tf.keras.Sequential.from_config(base_model_config)
            elif isinstance(self.base_model, tf.keras.Model):
                m = tf.keras.Model.from_config(base_model_config)
            else:
                raise Exception(
                    "Please provide a Sequential, Functional or subclassed model."
                )

            m = (
                m
                if self.metric_wrapper == None
                else self.metric_wrapper(m, **self.kwargs)
            )
            m_name = (
                f"usermodel_{i}"
                if self.metric_wrapper == None
                else f"{m.metric_name}_{i}"
            )
            m.compile(optimizer[i], loss[i], metrics[i])
            self.metrics_compiled[m_name] = m
    
    @tf.function
    def train_step(self, data):
        """
        The logic for one training step.

        Parameters
        ----------
        data : tuple
            (x, y) pairs, as in the regular Keras ``train_step``.

        Returns
        -------
        keras_metrics : dict
            `Keras metrics <https://keras.io/api/metrics/>`_.
        """
        keras_metrics = {}

        for name, wrapper in self.metrics_compiled.items():

            # ensembling user model
            if self.metric_wrapper == None:
                _ = wrapper.train_step(data)
                for m in wrapper.metrics:
                    keras_metrics[f"{name}_compiled_{m.name}"] = m.result()

            # ensembling one of our metric wrappers
            else:
                keras_metric = wrapper.train_step(data, prefix=name)
                keras_metrics.update(keras_metric)

        # # If user utilizes a callback, which saves weights by monitoring loss,
        # # but in this model there's no single loss that we can monitor -- each member
        # # has its own loss. So add another entry to the keras metric dict called
        # # "average loss" which is an average of all member's losses.
        # # Account for the case of metrics containing two different losses, or even non loss items.
        # if self.metric_wrapper == None:
        #     keras_metrics["loss"] = tf.reduce_mean(
        #         [v for k, v in keras_metrics.items() if "loss" in k]
        #     )
        # else:
        #     keras_metrics["wrapper_loss"] = tf.reduce_mean(
        #         [v for k, v in keras_metrics.items() if "wrapper_loss" in k]
        #     )

        return keras_metrics

    def test_step(self, data):
        """
        The logic for one evaluation step.

        Parameters
        ----------
        data : tuple
            (x, y) pairs, as in the regular Keras ``test_step``.

        Returns
        -------
        keras_metrics : dict
            `Keras metrics <https://keras.io/api/metrics/>`_.
        """
        keras_metrics = {}

        for name, wrapper in self.metrics_compiled.items():

            # ensembling user model
            if self.metric_wrapper == None:
                _ = wrapper.test_step(data)
                for m in wrapper.metrics:
                    keras_metrics[f"{name}_compiled_{m.name}"] = m.result()

            # ensembling one of our metric wrappers
            else:
                keras_metric = wrapper.test_step(data, prefix=name)
                keras_metrics.update(keras_metric)

        # if self.metric_wrapper == None:
        #     keras_metrics["loss"] = tf.reduce_mean(
        #         [v for k, v in keras_metrics.items() if "val_loss" in k]
        #     )
        # else:
        #     keras_metrics["wrapper_loss"] = tf.reduce_mean(
        #         [v for k, v in keras_metrics.items() if "val_wrapper_loss" in k]
        #     )

        return keras_metrics

    def call(self, x, training=False, return_risk=True):
        """
        Forward pass of the model

        Parameters
        ----------
        x : tf.Tensor
            Input.
        training : bool, default False
            Can be used to specify a different behavior in training and inference.
        return_risk : bool, default True
            Indicates whether or not to output a risk estimate in addition to the model's prediction.

        Returns
        -------
        out : capsa.RiskTensor
            Risk aware tensor, contains both the predicted label y_hat (tf.Tensor) and the epistemic
            uncertainty estimate (tf.Tensor).
        """
        T = 1 if return_risk == False else self.num_members

        outs = []
        for wrapper in list(self.metrics_compiled.values())[:T]:
            # ensembling the user model
            if self.metric_wrapper == None:
                out = wrapper(x)
            # ensembling one of our own metrics
            else:
                out = wrapper(x, training, return_risk)
            outs.append(out)

        if not return_risk:
            return out
        else:
            outs = tf.stack(outs)
            # ensembling the user model
            if self.metric_wrapper == None:
                mean, std = tf.reduce_mean(outs, 0), tf.math.reduce_std(outs, 0)
                return RiskTensor(mean, epistemic=std)
            # ensembling one of our own metrics
            else:
                return tf.reduce_mean(outs, 0)
