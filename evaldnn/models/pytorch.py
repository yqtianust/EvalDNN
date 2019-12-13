"""
Provides a class for torch model evaluation.
"""

from __future__ import absolute_import

import warnings

import foolbox
import torch

from evaldnn.utils import common


class PyTorchModel:
    """ Class for torch model evaluation.

    Provide predict, intermediate_layer_outputs and adversarial_attack
    methods for model evaluation. Set callback functions for each method
    to process the results.

    Parameters
    ----------
    model : instance of torch.nn.Module
        torch model to evaluate.

    Notes
    ----------
    All operations will be done using GPU if the environment is available
    and set properly.

    """

    def __init__(self, model):
        assert isinstance(model, torch.nn.Module)
        self._model = model
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model.eval()
        self._model.to(self._device)

    def predict(self, dataset, callbacks, batch_size=16):
        """Predict with the model.

        The method will use the model to do prediction batch by batch. For
        every batch, callback functions will be invoked. Labels and predictions
        will be passed to the callback functions to do further process.

        Parameters
        ----------
        dataset : instance of torch.utils.data.Dataset
            Dataset from which to load the data.
        callbacks : list of functions
            Callback functions, each of which will be invoked when a batch is done.
        batch_size : integer
            Batch size for prediction

        See Also
        --------
        :class:`metrics.accuracy.Accuracy`

        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self._device)
                labels = labels.to(self._device)
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
        dataset : instance of torch.utils.data.Dataset
            Dataset from which to load the data.
        callbacks : list of functions
            Callback functions, each of which will be invoked when a batch is done.
        batch_size : integer
            Batch size for getting intermediate layer outputs.

        See Also
        --------
        :class:`metrics.neuron_coverage.NeuronCoverage`

        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        y_mini_batch_outputs = []
        hook_handles = []
        intermediate_layers = self._intermediate_layers(self._model)
        for intermediate_layer in intermediate_layers:
            def hook(module, input, output):
                y_mini_batch_outputs.append(output)

            handle = intermediate_layer.register_forward_hook(hook)
            hook_handles.append(handle)
        with torch.no_grad():
            for data in dataloader:
                if isinstance(data, list):
                    data = data[0]
                y_mini_batch_outputs.clear()
                data = data.to(self._device)
                self._model(data)
                for callback in callbacks:
                    callback(y_mini_batch_outputs, 0)
        for handle in hook_handles:
            handle.remove()

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
        dataset : instance of torch.utils.data.Dataset
            Dataset from which to load the data.
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
            foolbox_model = foolbox.models.PyTorchModel(self._model, bounds, num_classes, preprocessing=preprocessing, device=self._device)
        attack = attack(foolbox_model, criterion, distance, threshold)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
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

    def _intermediate_layers(self, module):
        """Get the intermediate layers of the model.

        The method will get some intermediate layers of the model which might
        be useful for neuron coverage computation. Some layers such as dropout
        layers are excluded empirically.

        Returns
        -------
        list of torch.nn.modules
            Intermediate layers of the model.

        """
        intermediate_layers = []
        for submodule in module.children():
            if len(submodule._modules) > 0:
                intermediate_layers += self._intermediate_layers(submodule)
            else:
                if 'Dropout' in str(submodule.type):
                    continue
                intermediate_layers.append(submodule)
        return intermediate_layers
