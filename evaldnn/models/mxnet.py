"""
Provides a class for mxnet model evaluation.
"""

from __future__ import absolute_import

import warnings

import foolbox
import mxnet

from evaldnn.utils import common


class MXNetModel:
    """ Class for mxnet model evaluation.

    Provide predict, intermediate_layer_outputs and adversarial_attack
    methods for model evaluation. Set callback functions for each method
    to process the results.

    Parameters
    ----------
    model : instance of mxnet.gluon.nn.Block
        MXNet model to evaluate.

    Notes
    ----------
    All operations will be done using GPU if the environment is available
    and set properly.

    """

    def __init__(self, model):
        assert isinstance(model, mxnet.gluon.nn.Block)
        self._model = model
        self._best_ctx = mxnet.gpu() if mxnet.test_utils.list_gpus() else mxnet.cpu()
        self._model.collect_params().reset_ctx(self._best_ctx)

    def predict(self, dataset, callbacks, batch_size=16):
        """Predict with the model.

        The method will use the model to do prediction batch by batch. For
        every batch, callback functions will be invoked. Labels and predictions
        will be passed to the callback functions to do further process.

        Parameters
        ----------
        dataset : instance of mxnet.gluon.data.Dataset
            Source dataset. Note that numpy and mxnet arrays can be directly used
            as a Dataset.
        callbacks : list of functions
            Callback functions, each of which will be invoked when a batch is done.
        batch_size : integer
            Batch size for prediction

        See Also
        --------
        :class:`metrics.accuracy.Accuracy`

        """
        dataloader = mxnet.gluon.data.DataLoader(dataset, batch_size=batch_size)
        for data, labels in dataloader:
            data = data.as_in_context(self._best_ctx)
            labels = labels.as_in_context(self._best_ctx)
            y_mini_batch_pred = self._model(data)
            for callback in callbacks:
                callback(labels, y_mini_batch_pred)

    def intermediate_layer_outputs(self, dataset, callbacks, batch_size=8):
        """Get the intermediate layer outputs of the model.

        The method will use the model to do prediction batch by batch. For
        every batch, the the intermediate layer outputs will be captured and
        callback functions will be invoked. all intermediate layer output
        will be passed to the callback functions to do further process.

        Parameters
        ----------
        dataset : instance of mxnet.gluon.data.Dataset
            Source dataset. Note that numpy and mxnet arrays can be directly used
            as a Dataset.
        callbacks : list of functions
            Callback functions, each of which will be invoked when a batch is done.
        batch_size : integer
            Batch size for getting intermediate layer outputs.

        See Also
        --------
        :class:`metrics.neuron_coverage.NeuronCoverage`

        """
        dataloader = mxnet.gluon.data.DataLoader(dataset, batch_size=batch_size)
        inputs = mxnet.sym.var('data')
        outputs = self._intermediate_layers()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            feat_model = mxnet.gluon.SymbolBlock(outputs, inputs, params=self._model.collect_params())
        for data in dataloader:
            if isinstance(data, list):
                data = data[0]
            data = data.as_in_context(self._best_ctx)
            y_mini_batch_outputs = feat_model(data)
            for y_mini_batch_output in y_mini_batch_outputs:
                y_mini_batch_output.wait_to_read()
            for callback in callbacks:
                callback(y_mini_batch_outputs, 0)

    def adversarial_attack(self,
                           dataset,
                           bounds,
                           num_classes,
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
        dataset : instance of mxnet.gluon.data.Dataset
            Source dataset. Note that numpy and mxnet arrays can be directly used
            as a Dataset.
        bounds : tuple of length 2
            The bounds for the pixel values.
        num_classes : int
            The number of classes.
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
            foolbox_model = foolbox.models.MXNetGluonModel(self._model, bounds, num_classes, preprocessing=preprocessing, ctx=self._best_ctx)
        attack = attack(foolbox_model, criterion, distance, threshold)
        dataloader = mxnet.gluon.data.DataLoader(dataset, batch_size=batch_size)
        index = 0
        for data, labels in dataloader:
            data = common.to_numpy(data)
            labels = common.to_numpy(labels)
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
        be useful for neuron coverage computation. Some layers such as dropout
        layers and non-fwd layers are excluded empirically.

        Returns
        -------
        list of mxnet symbols
            Intermediate layers of the model.

        """
        inputs = mxnet.sym.var('data')
        outputs = [output for output in self._model(inputs).get_internals() if output.name.endswith('_fwd') and 'dropout' not in output.name]
        return outputs
