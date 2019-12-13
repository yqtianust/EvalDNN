"""
Provides a class for testing keras model evaluation.
"""

from __future__ import absolute_import

import unittest
import warnings

import foolbox

from evaldnn.metrics.accuracy import Accuracy
from evaldnn.metrics.neuron_coverage import NeuronCoverage
from evaldnn.metrics.robustness import Robustness
from evaldnn.models.keras import KerasModel
from evaldnn.utils.keras import *


class TestKeras(unittest.TestCase):

    def test_cifar10_simple(self):
        print('test_cifar10_simple')

        # load the model and data
        keras.backend.set_learning_phase(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            model = keras.models.load_model(common.user_home_dir() + '/EvalDNN-models/keras/cifar10_simple.h5')
        x, y = cifar10_data()
        x_robustness, y_robustness = cifar10_data(num_max=3)
        bounds = (0, 1)

        # wrap the model with EvalDNN
        measure_model = KerasModel(model)

        # evaluate the top-1 and top-5 accuracy of the model
        accuracy = Accuracy()
        measure_model.predict(x, y, [accuracy.update, accuracy.report])

        # evaluate the neuron coverage of the model
        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update, neuron_coverage.report])

        # evaluate the robustness of the model
        robustness = Robustness(bounds)
        measure_model.adversarial_attack(x_robustness,
                                         y_robustness,
                                         bounds,
                                         [robustness.update, robustness.draw, robustness.report],
                                         attack=foolbox.attacks.FGSM,
                                         distance=foolbox.distances.Linf)

        self.assertAlmostEqual(accuracy.get(1), 0.777400)
        self.assertAlmostEqual(accuracy.get(5), 0.985000)
        self.assertAlmostEqual(neuron_coverage.get(0.8), 0.743902, places=2)

    def test_mnist_simple(self):
        print('test_mnist_simple')

        # load the model and data
        keras.backend.set_learning_phase(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            model = keras.models.load_model(common.user_home_dir() + '/EvalDNN-models/keras/mnist_simple.h5')
        x, y = mnist_data()
        x_robustness, y_robustness = mnist_data(num_max=3)
        bounds = (0, 1)

        # wrap the model with EvalDNN
        measure_model = KerasModel(model)

        # evaluate the top-1 and top-5 accuracy of the model
        accuracy = Accuracy()
        measure_model.predict(x, y, [accuracy.update, accuracy.report])

        # evaluate the neuron coverage of the model
        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update, neuron_coverage.report])

        # evaluate the robustness of the model
        robustness = Robustness(bounds)
        measure_model.adversarial_attack(x_robustness,
                                         y_robustness,
                                         bounds,
                                         [robustness.update, robustness.draw, robustness.report],
                                         attack=foolbox.attacks.FGSM,
                                         distance=foolbox.distances.Linf)

        self.assertAlmostEqual(accuracy.get(1), 0.991000)
        self.assertAlmostEqual(accuracy.get(5), 1.000000)
        self.assertAlmostEqual(neuron_coverage.get(0.6), 0.544898, places=2)

    def test_imagenet_vgg16(self):
        print('test_imagenet_vgg16')

        # load the model and data
        model, data_normalized, data_original, mean, std, flip_axis, bounds = imagenet_benchmark_zoo('vgg16', data_original_num_max=3)

        # wrap the model with EvalDNN
        measure_model = KerasModel(model)

        # evaluate the top-1 and top-5 accuracy of the model
        accuracy = Accuracy()
        measure_model.predict(data_normalized.x, data_normalized.y, [accuracy.update, accuracy.report])

        # evaluate the neuron coverage of the model
        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(data_normalized.x, [neuron_coverage.update, neuron_coverage.report])

        # evaluate the robustness of the model
        robustness = Robustness(bounds)
        robustness.set_descriptions(data_original.filenames)
        measure_model.adversarial_attack(data_original.x,
                                         data_original.y,
                                         bounds,
                                         [robustness.update, robustness.draw, robustness.report],
                                         attack=foolbox.attacks.FGSM,
                                         distance=foolbox.distances.Linf,
                                         preprocessing=dict(flip_axis=flip_axis, mean=mean, std=std))

        self.assertAlmostEqual(accuracy.get(1), 0.650000)
        self.assertAlmostEqual(accuracy.get(5), 0.950000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.506985, places=2)


if __name__ == '__main__':
    unittest.main()
