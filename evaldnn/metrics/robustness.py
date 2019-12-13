"""
Provides a class for model robustness evaluation.
"""

from __future__ import absolute_import

import math
import os

import foolbox
import matplotlib.pyplot as plt
import numpy as np

from evaldnn.utils import common


class Robustness:
    """ Class for model neuron coverage evaluation.

    Compare the origins and adversaries, update and report the model
    robustness accordingly.

    Parameters
    ----------
    bounds : tuple of length 2
        The bounds for the pixel values.

    """

    def __init__(self, bounds):
        self._bounds = bounds
        self._num_correct = 0
        self._num_incorrect = 0
        self._num_success = 0
        self._total_mean_squared_distance = 0
        self._total_mean_absolute_distance = 0
        self._total_linfinity = 0
        self._total_l0 = 0
        self._save_dir = '.'
        self._save_prefix = 'robustness'
        self._save_count = 0
        self._descriptions = None

    def update(self, indexes, origins, adversaries, y_corrects, y_incorrects):
        """Update model robustness accordingly.

        Based on the origins and adversaries, the robustness info will be
        re-calculated and updated accordingly. The info includes success rate,
        MSE, MAE, LInf, L0, etc.

        Parameters
        ----------
        indexes : list of integers
            The index of each input.
        origins : array
            Original inputs.
        adversaries: array
            Perturbed inputs.
        y_corrects : array
            Labels of original inputs.
        y_incorrects : array
            Predictions of perturbed inputs.

        Notes
        -------
        This method can be invoked for many times in one instance which means
        that once the adversaries of a batch of inputs are made, this method
        can be invoked to update the status. The robustness of the model will
        be updated for every invocation.
        The MSE, MAE, LInf and L0 are only updated for those inputs on which
        the adversarial attack is successfully performed.

        """
        for i in range(len(origins)):
            origin = origins[i]
            adversary = adversaries[i]
            y_correct = y_corrects[i]
            y_incorrect = y_incorrects[i]
            if y_correct == y_incorrect:
                self._num_incorrect += 1
                continue
            self._num_correct += 1
            if math.isnan(adversary.max()):
                continue
            self._num_success += 1
            self._total_mean_squared_distance += foolbox.distances.MeanSquaredDistance(origin, adversary, self._bounds).value
            self._total_mean_absolute_distance += foolbox.distances.MeanAbsoluteDistance(origin, adversary, self._bounds).value
            self._total_linfinity += foolbox.distances.Linfinity(origin, adversary, self._bounds).value
            self._total_l0 += foolbox.distances.L0(origin, adversary, self._bounds).value

    def draw(self, indexes, origins, adversaries, y_corrects, y_incorrects):
        """Draw out the origins and adversaries with corresponding descriptions.

        Every origin and adversary pair and the difference between them will ba
        painted. Relevant info such as index, label, prediction, etc are also showed.

        Parameters
        ----------
        indexes : list of integers
            The index of each input.
        origins : array
            Original inputs.
        adversaries: array
            Perturbed inputs.
        y_corrects : array
            Labels of original inputs.
        y_incorrects : array
            Predictions of perturbed inputs.

        Notes
        -------
        When an adversarial attack on an input fails, the prediction for the adversary
        will be showed as failed.

        """
        for i in range(len(origins)):
            index = indexes[i]
            origin = origins[i]
            adversary = adversaries[i]
            y_correct = y_corrects[i]
            y_incorrect = y_incorrects[i]
            if y_correct == y_incorrect:
                continue
            if math.isnan(adversary.max()):
                y_incorrect = 'failed'
            origin = self._to_image(origin)
            adversary = self._to_image(adversary)
            plt.subplot(1, 3, 1)
            plt.imshow(origin)
            plt.title('correct: ' + str(y_correct))
            plt.subplot(1, 3, 2)
            plt.imshow(adversary)
            plt.title('incorrect: ' + str(y_incorrect))
            plt.subplot(1, 3, 3)
            plt.imshow(np.abs(adversary - origin))
            plt.title('difference')
            if self._descriptions is None:
                plt.suptitle('index: ' + str(index))
            else:
                plt.suptitle('index: ' + str(index) + ', desc: ' + self._descriptions[index])
            plt.show()

    def save(self, indexes, origins, adversaries, y_corrects, y_incorrects):
        """Save origin and adversary pairs to specified directory.

        All origin and adversary pairs will be saved to the disk in numpy
        array format. The filenames will contain some relevant info such as
        labels and predictions.

        Parameters
        ----------
        indexes : list of integers
            The index of each input.
        origins : array
            Original inputs.
        adversaries: array
            Perturbed inputs.
        y_corrects : array
            Labels of original inputs.
        y_incorrects : array
            Predictions of perturbed inputs.

        Notes
        -------
        To set the output directory, one can invoke set_save_dir.
        To set the output file prefix, one can invoke set_save_prefix.
        To set the output file description, one can invoke set_descriptions.

        """
        for i in range(len(origins)):
            index = indexes[i]
            origin = origins[i]
            adversary = adversaries[i]
            y_correct = y_corrects[i]
            y_incorrect = y_incorrects[i]
            if y_correct == y_incorrect:
                continue
            if math.isnan(adversary.max()):
                y_incorrect = 'failed'
            if self._descriptions is None:
                path_prefix = self._save_dir + '/' + self._save_prefix + '_' + str(self._save_count) + '_' + str(index) + '_' + str(y_correct) + '_' + str(y_incorrect) + '_'
            else:
                path_prefix = self._save_dir + '/' + self._save_prefix + '_' + str(self._save_count) + '_' + str(index) + '_' + self._descriptions[index] + '_' + str(y_correct) + '_' + str(y_incorrect) + '_'
            self._save_count += 1
            np.save(path_prefix + 'origin', origin)
            np.save(path_prefix + 'adversary', adversary)

    def report(self, *args):
        """Report model robustness.

        The robustness info will be reported if this method is invoked.
        Reported info includes report time, number of inputs, number of
        correct predictions, number of incorrect predictions, success rate,
        avg MSE, avg MAE, avg LInf, avg L0, etc.

        """
        print('[Robustness] Time: {:s}, Total Num: {:d}, Correct Num: {:d}, Incorrect Num: {:d}, Success Rate: {:.10f}({:d}/{:d}), Avg Mean Squared Distance: {:.10f}, Avg Mean Absolute Distance: {:.10f}, Avg Linfinity: {:.10f}, Avg L0: {:.10f}'.format(common.readable_time_str(), self._num_correct + self._num_incorrect, self._num_correct, self._num_incorrect, self.success_rate, self._num_success, self._num_correct, self.avg_mean_squared_distance, self.avg_mean_absolute_distance, self.avg_linfinity, self.avg_l0))

    @property
    def success_rate(self):
        """Get adversarial attack success rate.

        Returns
        -------
        float
            Adversarial attack success rate.

        """
        return self._num_success / self._num_correct if self._num_correct != 0 else 0

    @property
    def avg_mean_squared_distance(self):
        """Get adversarial attack avg mean squared distance over all origin
        and adversary pairs passed to update.

        Returns
        -------
        float
            Adversarial attack avg mean squared distance (avg MSE).

        """
        return self._total_mean_squared_distance / self._num_correct if self._num_correct != 0 else 0

    @property
    def avg_mean_absolute_distance(self):
        """Get adversarial attack avg mean absolute distance over all origin
        and adversary pairs passed to update.

        Returns
        -------
        float
            Adversarial attack avg mean absolute distance (avg MAE).

        """
        return self._total_mean_absolute_distance / self._num_correct if self._num_correct != 0 else 0

    @property
    def avg_linfinity(self):
        """Get adversarial attack avg LInf distance over all origin
        and adversary pairs passed to update.

        Returns
        -------
        float
            Adversarial attack avg LInf distance.

        """
        return self._total_linfinity / self._num_correct if self._num_correct != 0 else 0

    @property
    def avg_l0(self):
        """Get adversarial attack avg L0 distance over all origin
        and adversary pairs passed to update.

        Returns
        -------
        float
            Adversarial attack avg L0 distance.

        """
        return self._total_l0 / self._num_correct if self._num_correct != 0 else 0

    def set_save_dir(self, save_dir):
        """Set the output directory for saving origin and adversary pairs.

        Parameters
        ----------
        save_dir : str
            The path of output directory.

        Notes
        -------
        If the specified directory does not exist, it will be created automatically.

        """
        self._save_dir = save_dir
        self._save_count = 0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def set_save_prefix(self, save_prefix):
        """Set the output filename prefix for saving origin and adversary pairs.

        Parameters
        ----------
        save_prefix : str
            The output filename prefix.

        """
        self._save_prefix = save_prefix

    def set_descriptions(self, descriptions):
        """Set the descriptions for origin and adversary pairs.

        Parameters
        ----------
        descriptions : list of str
            The descriptions of inputs.

        """
        self._descriptions = descriptions

    def _to_image(self, image):
        """Convert an array to a standard image array format.

        For some origins or adversaries, the shape can be (W, H, color).
        While for some others, the shape can be (color, W, H).
        There are also some cases in which the array of an image is not 2-dimensional.
        This method will convert an array to a standard image array format with
        a shape of (W, H, color)

        Parameters
        ----------
        image : array
            The array of an image, which could be in irregular format.

        Returns
        -------
        array
            Adversarial attack avg mean absolute distance (avg MAE).

        """

        # convert the image to numpy array
        image = common.to_numpy(image)

        # if the array is 1-dimensional, convert it to 3-dimensional
        shape = image.shape
        shape_len = len(shape)
        if shape_len == 1:
            size = int(math.sqrt(shape[0]))
            image = image.reshape(size, size)
            image = np.expand_dims(image, axis=0)
            shape_len = 3
        assert shape_len == 3

        # convert format to (W, H, color)
        transpose = [v for v in range(shape_len)]
        min_index = 0
        for i in range(shape_len):
            if image.shape[i] < image.shape[min_index]:
                min_index = i
        transpose.remove(min_index)
        transpose.append(min_index)
        transpose = tuple(transpose)
        image = np.transpose(image, transpose)

        # if the image is not in RGB format, convert it
        shape = image.shape
        if shape[-1] == 1:
            shape = list(shape)
            shape[-1] = 3
            image_new = np.empty(shape)
            image_new[..., 0] = image_new[..., 1] = image_new[..., 2] = image[..., 0]
            image = image_new

        # if the image is not ranged in [0,1), convert it
        if image.max() > 1:
            image /= 255.0

        return image
