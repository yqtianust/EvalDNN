"""
Provides a class for model accuracy evaluation.
"""

from __future__ import absolute_import

import numpy as np

from evaldnn.utils import common


class Accuracy:
    """ Class for model accuracy evaluation.

    Compare the predictions and the labels, update and report the model
    prediction accuracy accordingly.

    Parameters
    ----------
    ks : list of integers
        For each k in ks, top-k accuracy will be computed separately.

    """

    def __init__(self, ks=(1, 5)):
        self._ks = ks
        self._correct = {}
        self._total = 0
        for k in self._ks:
            self._correct[k] = 0

    def update(self, y_true, y_pred):
        """Update model accuracy accordingly.

        For each k in ks, the correctness and accuracy will be re-calculated
        and updated accordingly.

        Parameters
        ----------
        y_true : array
            Labels for data.
        y_pred : array
            Predictions from model.

        Notes
        -------
        This method can be invoked for many times in one instance which means
        that once a batch prediction is made this method can be invoked to update
        the status. The accuracy will be updated for every invocation.

        """
        y_true = common.to_numpy(y_true)
        y_pred = common.to_numpy(y_pred)
        size = len(y_true)
        self._total += size
        for k in self._ks:
            top_k_predictions = np.argsort(y_pred)[:, -k:].T
            correct_matrix = np.zeros(size, bool)
            for i_th_prediction in top_k_predictions:
                correct_matrix = np.logical_or(correct_matrix, y_true == i_th_prediction)
            self._correct[k] += len([v for v in correct_matrix if v])

    def report(self, *args):
        """Report model accuracy.

        The accuracy info will be reported for each k in ks. Reported info includes
        report time, number of inputs evaluated, k, accuracy, number of correct
        predictions.

        """
        for k in self._ks:
            print('[Accuracy] Time: {:s}, Num: {:d}, topK: {:d}, Accuracy: {:.6f}({:d}/{:d})'.format(common.readable_time_str(), self._total, k, self.get(k), self._correct[k], self._total))

    def get(self, k):
        """Get model top-k accuracy.

        Parameters
        ----------
        k : integer
            Top-k accuracy.

        Returns
        -------
        float
            Model top-k accuracy.

        Notes
        -------
        The parameter k must be one value in the list ks.

        """
        if self._total == 0:
            return 0
        else:
            return self._correct[k] / self._total
