"""
Provides a class for keras model evaluation.
"""

from __future__ import absolute_import

import warnings

import foolbox
import keras


class KerasModel:
    """ Class for keras model evaluation.

    Provide predict, intermediate_layer_outputs and adversarial_attack
    methods for model evaluation. Set callback functions for each method
    to process the results.

    Parameters
    ----------
    model : instance of keras.Model
        Keras model to evaluate.

    Notes
    ----------
    All operations will be done using GPU if the environment is available
    and set properly.

    """

    def __init__(self, model):
        assert isinstance(model, keras.Model)
        self._model = model

    def predict(self, x, y, callbacks, batch_size=16):
        """Predict with the model.

        The method will use the model to do prediction batch by batch. For
        every batch, callback functions will be invoked. Labels and predictions
        will be passed to the callback functions to do further process.

        Parameters
        ----------
        x : array
            Array of inputs which will be used to do prediction.
        y : array
            Labels of inputs.
        callbacks : list of functions
            Callback functions, each of which will be invoked when a batch is done.
        batch_size : integer
            Batch size for prediction

        See Also
        --------
        :class:`metrics.accuracy.Accuracy`

        """
        index = 0
        while index < len(x):
            data = x[index:index + batch_size]
            labels = y[index:index + batch_size]
            y_mini_batch_pred = self._model.predict(data)
            for callback in callbacks:
                callback(labels, y_mini_batch_pred)
            index += batch_size

    def intermediate_layer_outputs(self, x, callbacks, batch_size=8):
        """Get the intermediate layer outputs of the model.

        The method will use the model to do prediction batch by batch. For
        every batch, the the intermediate layer outputs will be captured and
        callback functions will be invoked. all intermediate layer output
        will be passed to the callback functions to do further process.

        Parameters
        ----------
        x : array
            Array of inputs which will be used to do prediction.
        callbacks : list of functions
            Callback functions, each of which will be invoked when a batch is done.
        batch_size : integer
            Batch size for getting intermediate layer outputs.

        See Also
        --------
        :class:`metrics.neuron_coverage.NeuronCoverage`

        """
        layer_names = self._intermediate_layer_names()
        intermediate_layer_model = keras.Model(inputs=self._model.input, outputs=[self._model.get_layer(layer_name).output for layer_name in layer_names])
        index = 0
        while index < len(x):
            data = x[index:index + batch_size]
            y_mini_batch_outputs = intermediate_layer_model.predict(data)
            for callback in callbacks:
                callback(y_mini_batch_outputs, -1)
            index += batch_size

    def adversarial_attack(self,
                           x,
                           y,
                           bounds,
                           callbacks,
                           batch_size=1,
                           preprocessing=(0, 1),
                           attack=foolbox.attacks.FGSM,
                           criterion=foolbox.criteria.Misclassification(),
                           distance=foolbox.distances.Linf,
                           threshold=None):
        """Do adversarial attack over the model.

        The method will use foolbox adversarial attack over the model batch by
        batch. For every batch, callback functions will be invoked. Origin and
        adversary pairs and their corresponding labels and predictions will be
        passed to the callback functions to do further process.

        Parameters
        ----------
        x : array
            Array of inputs which will be used to do prediction.
        y : array
            Labels of inputs.
        bounds : tuple of length 2
            The bounds for the pixel values.
        callbacks : list of functions
            Callback functions, each of which will be invoked when a batch is done.
        batch_size : integer
            Batch size for adversarial attack.
        preprocessing : dict or tuple
            Can be a tuple with two elements representing mean and standard
            deviation or a dict with keys "mean" and "std". The two elements
            should be floats or numpy arrays. "mean" is subtracted from the input,
            the result is then divided by "std". If "mean" and "std" are
            1-dimensional arrays, an additional (negative) "axis" key can be
            given such that "mean" and "std" will be broadcasted to that axis
            (typically -1 for "channels_last" and -3 for "channels_first", but
            might be different when using e.g. 1D convolutions). Finally,
            a (negative) "flip_axis" can be specified. This axis will be flipped
            (before "mean" is subtracted), e.g. to convert RGB to BGR.
        attack : a :class:`Attack` class
            The adversarial attack will be conducted with this attack method.
        criterion : instance of criterion
            The criterion that determines which inputs are adversarial.
        distance : a :class:`Distance` class
            The measure used to quantify how close inputs are.
        threshold : float or :class:`Distance`
                If not None, the attack will stop as soon as the adversarial
                perturbation has a size smaller than this threshold. Can be
                an instance of the :class:`Distance` class passed to the distance
                argument, or a float assumed to have the same unit as the
                the given distance. If None, the attack will simply minimize
                the distance as good as possible. Note that the threshold only
                influences early stopping of the attack; the returned adversarial
                does not necessarily have smaller perturbation size than this
                threshold; the `reached_threshold()` method can be used to check
                if the threshold has been reached.

        See Also
        --------
        :class:`metrics.robustness.Robustness`

        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            foolbox_model = foolbox.models.KerasModel(self._model, bounds, preprocessing=preprocessing)
        attack = attack(foolbox_model, criterion, distance, threshold)
        index = 0
        while index < len(x):
            data = x[index:index + batch_size]
            labels = y[index:index + batch_size]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                adversaries = attack(data, labels)
            indexes = [i for i in range(index, index + batch_size)]
            y_corrects = foolbox_model.forward(data).argmax(-1)
            y_incorrects = foolbox_model.forward(adversaries).argmax(-1)
            for callback in callbacks:
                callback(indexes, data, adversaries, y_corrects, y_incorrects)
            index += batch_size

    def _intermediate_layer_names(self):
        """Get the intermediate layer names of the model.

        The method will get all intermediate layers of the model except
        the flatten and the input layer.

        Returns
        -------
        list of str
            Intermediate layer names of the model.

        """
        layer_names = []
        for layer in self._model.layers:
            name = layer.name.lower()
            if 'flatten' in name:
                continue
            if 'input' in name:
                continue
            layer_names.append(layer.name)
        return layer_names
