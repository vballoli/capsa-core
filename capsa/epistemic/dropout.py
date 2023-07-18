



#import tensorflow as tf
import keras_core as keras

from keras_core.layers import (
    Dense,
    Conv1D,
    Conv2D,
    Conv3D,
    Dropout,
    SpatialDropout1D,
    SpatialDropout2D,
    SpatialDropout3D,
)

#from ..base_wrapper import BaseWrapper
#from ..risk_tensor import RiskTensor


class DropoutWrapper(keras.Model):
    """Adds dropout layers (Srivastava et al., 2014) to capture epistemic
    uncertainty Gal & Ghahramani (2016).

    Calculates epistemic uncertainty by adding dropout layers after dense layers
    (or spatial dropout layers after conv layers).

    To calculate the uncertainty, we run ``T`` forward passes, which is equivalent to
    Monte Carlo sampling. Computing the first and second moments from the ``T`` stochastic
    samples yields a prediction and uncertainty estimate, respectively.

    Example of usage:
        >>> # initialize a keras model
        >>> user_model = Unet()
        >>> # wrap the model to transform it into a risk-aware variant
        >>> model = DropoutWrapper(user_model)
        >>> # compile and fit as a regular tf.keras model
        >>> model.compile(...)
        >>> model.fit(...)
    """

    def __init__(self, base_model, p=0.1):
        """
        Parameters
        ----------
        base_model : tf.keras.Model
            A model to be transformed into a risk-aware variant.
        p : float, default 0.1
            Float between 0 and 1. Fraction of the units to drop.

        Attributes
        ----------
        metric_name : str
            Represents the name of the metric wrapper.
        model : tf.keras.Model
            ``base_model`` with added dropout layers.
        """
        super().__init__()

        self.metric_name = "dropout"

        if p == 0.0:
            assert_has_dropout(base_model)
            self.model = base_model
        else:
            self.model = add_dropout(base_model, p)


    


    def loss_fn(self, x, y):
        """
        Parameters
        ----------
        x : tf.Tensor
            Input.
        y : tf.Tensor
            Ground truth label.

        Returns
        -------
        loss : tf.Tensor
            Float, reflects how well does the algorithm perform given the ground truth label,
            predicted label and the metric specific loss function. In this case it is
            0 because ``DropoutWrapper`` does not introduce an additional loss function,
            and the compiled loss is already added in the parent class ``BaseWrapper.train_step()``.
        y_hat : tf.Tensor
            Predicted label.
        """
        y_hat = self.model(x, True)
        metric_loss = 0
        return metric_loss, y_hat

    def call(self, x, training=False, return_risk=True, T=20):
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
        T : int, default 20
            Number of forward passes with different dropout masks.

        Returns
        -------
        out : capsa.RiskTensor
            Risk aware tensor, contains both the predicted label y_hat (tf.Tensor) and the epistemic
            uncertainty estimate (tf.Tensor).
        """

        if training:
            y_hat = self.model(x, training=True)
            return y_hat


        if not return_risk:
            y_hat = self.model(x, training=training)
            return y_hat
        else:
            # user model
            outs = []
            for _ in range(T):
                # we need training=True so that dropout is applied
                outs.append(self.model(x, training=True))
            outs = keras.ops.stack(outs)  # (T, N, 1)
            mean, std = keras.ops.mean(outs, 0), keras.ops.std(outs, 0)  # (N, 1)x2
            return mean, std


def add_dropout(model, p):
    """Adds dropout layers after dense layers (or spatial dropout
    layers after conv layers)."""

    inputs = model.layers[0].input

    for i in range(len(model.layers)):
        cur_layer = model.layers[i]
        # we do not add dropouts after the input or final layers to preserve stability
        if i == 0:
            x = cur_layer(inputs)
        elif i == len(model.layers) - 1:
            x = model.layers[i](x)
        else:
            next_layer = model.layers[i + 1]
            x = cur_layer(x)
            # we do not repeat dropout layers if they're already added
            if type(cur_layer) == Dense and type(next_layer) != Dropout:
                x = Dropout(rate=p)(x)
            elif type(cur_layer) == Conv1D and type(next_layer) != SpatialDropout1D:
                x = SpatialDropout1D(rate=p)(x)
            elif type(cur_layer) == Conv2D and type(next_layer) != SpatialDropout2D:
                x = SpatialDropout2D(rate=p)(x)
            elif type(cur_layer) == Conv3D and type(next_layer) != SpatialDropout3D:
                x = SpatialDropout1D(rate=p)(x)

    new_model = keras.Model(inputs, x)
    return new_model


def assert_has_dropout(base_model):
    """check if dropout is inside the base_model, otherwise throw an error"""

    is_dropout = False
    for l in base_model.layers:
        # returns True for both regular and spatial dropout layers
        if issubclass(type(l), Dropout):
            is_dropout = True
    assert is_dropout == True, "p=0.0 but the base_model has no dropout layers."
