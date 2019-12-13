"""
Provides a class for tensorflow model evaluation.
"""

from __future__ import absolute_import

import warnings

import foolbox


class TensorFlowModel:
    """ Class for tensorflow model evaluation.

    Provide predict, intermediate_layer_outputs and adversarial_attack
    methods for model evaluation. Set callback functions for each method
    to process the results.

    Parameters
    ----------
    session : `tensorflow.session`
        The session with which the graph will be computed.
    logits : `tensorflow.Tensor`
        The predictions of the model.
    input : `tensorflow.Tensor`
        The input to the model, usually a `tensorflow.placeholder`.

    Notes
    ----------
    All operations will be done using GPU if the environment is available
    and set properly.

    """

    def __init__(self, session, logits, input):
        self._session = session
        self._logits = logits
        self._input = input

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
            y_mini_batch_pred = self._session.run(self._logits, feed_dict={self._input: data})
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
        intermediate_layers = self._intermediate_layers()
        index = 0
        while index < len(x):
            data = x[index:index + batch_size]
            y_mini_batch_outputs = self._session.run(intermediate_layers, feed_dict={self._input: data})
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
            foolbox_model = foolbox.models.TensorFlowModel(self._input, self._logits, bounds, preprocessing=preprocessing)
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

    def _intermediate_layers(self):
        """Get the intermediate layers of the model.

        The method will get some intermediate layers of the model which might
        be useful for neuron coverage computation. Some layers such as reshape
        layers, squeeze layers and etc are excluded empirically.

        Returns
        -------
        list of `tensorflow.Tensor`
            Intermediate layers of the model.

        """
        ordered_tensors = self._ordered_tensors_in_graph(self._logits)
        tensors = []
        for tensor in ordered_tensors:
            if not len(tensor.shape) > 0:
                continue
            if tensor.shape[0] is not None and str(tensor.shape[0]) != '?':
                continue
            if not len(tensor.op.inputs._inputs) > 0:
                continue
            if 'Reshape' == tensor.op.type:
                continue
            if 'Squeeze' == tensor.op.type:
                continue
            if 'MatMul' in tensor.op.type:
                continue
            if 'Identity' in tensor.op.type:
                continue
            if 'BiasAdd:' in tensor.name:
                continue
            if 'add:' in tensor.name and 'fc' not in tensor.name:
                continue
            tensors.append(tensor)
        return tensors

    def _ordered_tensors_in_graph(self, tensor, tensors_in_order=None, entrypoint=True):
        """Get ordered tensors in the graph.

        Starting from the tensor which is passed as the parameter, this method
        will search all tensors in the graph and sort them accordingly.

        Parameters
        ----------
        tensor : `tensorflow.Tensor`
            The searching start point.
        tensors_in_order : list of `tensorflow.Tensor`
            Tensors which have been explored so far.
        entrypoint : bool
            Indicate whether or not the invocation is made by the user instead
            of by recursion.

        Returns
        -------
        list of `tensorflow.Tensor`
            Tensors in the graph.

        Notes
        -------
        To get all tensors in the graph, when a user is trying to invoke this
        method, the tensor which represents the predictions of the model should
        be passed as the parameter. The other two parameters can be omitted.

        """
        if tensors_in_order is None:
            tensors_in_order = []
        if tensor in tensors_in_order:
            return tensors_in_order
        tensors_in_order.append(tensor)
        for input_tensor in tensor.op.inputs._inputs:
            self._ordered_tensors_in_graph(input_tensor, tensors_in_order, False)
        if entrypoint:
            list.reverse(tensors_in_order)
        return tensors_in_order
