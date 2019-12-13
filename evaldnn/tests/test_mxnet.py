"""
Provides a class for testing mxnet model evaluation.
"""

from __future__ import absolute_import

import unittest

import foolbox

from evaldnn.metrics.accuracy import Accuracy
from evaldnn.metrics.neuron_coverage import NeuronCoverage
from evaldnn.metrics.robustness import Robustness
from evaldnn.models.mxnet import MXNetModel
from evaldnn.utils.mxnet import *


class TestMXNet(unittest.TestCase):

    def test_cifar10_simple(self):
        print('test_cifar10_simple')

        # load the model and data
        model = mxnet.gluon.nn.Sequential()
        with model.name_scope():
            model.add(mxnet.gluon.nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
            model.add(mxnet.gluon.nn.MaxPool2D(pool_size=2))
            model.add(mxnet.gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
            model.add(mxnet.gluon.nn.MaxPool2D(pool_size=2))
            model.add(mxnet.gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
            model.add(mxnet.gluon.nn.Flatten())
            model.add(mxnet.gluon.nn.Dense(64, activation='relu'))
            model.add(mxnet.gluon.nn.Dense(10))
        model.load_parameters(common.user_home_dir() + '/EvalDNN-models/mxnet/cifar10_simple.params')
        dataset = cifar10_dataset()
        dataset_robustness = cifar10_dataset(num_max=3)
        bounds = (0, 1)
        num_classes = 10

        # wrap the model with EvalDNN
        measure_model = MXNetModel(model)

        # evaluate the top-1 and top-5 accuracy of the model
        accuracy = Accuracy()
        measure_model.predict(dataset, [accuracy.update, accuracy.report])

        # evaluate the neuron coverage of the model
        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset, [neuron_coverage.update, neuron_coverage.report])

        # evaluate the robustness of the model
        robustness = Robustness(bounds)
        measure_model.adversarial_attack(dataset_robustness,
                                         bounds,
                                         num_classes,
                                         [robustness.update, robustness.draw, robustness.report],
                                         attack=foolbox.attacks.FGSM,
                                         distance=foolbox.distances.Linf)

        self.assertAlmostEqual(accuracy.get(1), 0.687000)
        self.assertAlmostEqual(accuracy.get(5), 0.966600)
        self.assertAlmostEqual(neuron_coverage.get(0.6), 0.631769, places=2)

    def test_mnist_simple(self):
        print('test_mnist_simple')

        # load the model and data
        model = mxnet.gluon.nn.Sequential()
        with model.name_scope():
            model.add(mxnet.gluon.nn.Dense(128, activation='relu'))
            model.add(mxnet.gluon.nn.Dense(64, activation='relu'))
            model.add(mxnet.gluon.nn.Dense(10))
        model.load_parameters(common.user_home_dir() + '/EvalDNN-models/mxnet/mnist_simple.params')
        dataset = mnist_dataset()
        dataset_robustness = mnist_dataset(num_max=3)
        bounds = (0, 1)
        num_classes = 10

        # wrap the model with EvalDNN
        measure_model = MXNetModel(model)

        # evaluate the top-1 and top-5 accuracy of the model
        accuracy = Accuracy()
        measure_model.predict(dataset, [accuracy.update, accuracy.report])

        # evaluate the neuron coverage of the model
        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset, [neuron_coverage.update, neuron_coverage.report])

        # evaluate the robustness of the model
        robustness = Robustness(bounds)
        measure_model.adversarial_attack(dataset_robustness,
                                         bounds,
                                         num_classes,
                                         [robustness.update, robustness.draw, robustness.report],
                                         attack=foolbox.attacks.FGSM,
                                         distance=foolbox.distances.Linf)

        self.assertAlmostEqual(accuracy.get(1), 0.978700)
        self.assertAlmostEqual(accuracy.get(5), 0.999700)
        self.assertAlmostEqual(neuron_coverage.get(0.8), 0.893401, places=2)

    def test_imagenet_vgg16(self):
        print('test_imagenet_vgg16')

        # load the model and data
        model, dataset_normalized, dataset_original, preprocessing, num_classes, bounds = imagenet_benchmark_zoo('vgg16', data_original_num_max=3)

        # wrap the model with EvalDNN
        measure_model = MXNetModel(model)

        # evaluate the top-1 and top-5 accuracy of the model
        accuracy = Accuracy()
        measure_model.predict(dataset_normalized, [accuracy.update, accuracy.report])

        # evaluate the neuron coverage of the model
        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset_normalized, [neuron_coverage.update, neuron_coverage.report])

        # evaluate the robustness of the model
        robustness = Robustness(bounds)
        robustness.set_descriptions(dataset_original.filenames)
        measure_model.adversarial_attack(dataset_original,
                                         bounds,
                                         num_classes,
                                         [robustness.update, robustness.draw, robustness.report],
                                         attack=foolbox.attacks.FGSM,
                                         distance=foolbox.distances.Linf,
                                         preprocessing=preprocessing)

        self.assertAlmostEqual(accuracy.get(1), 0.725000)
        self.assertAlmostEqual(accuracy.get(5), 0.950000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.750842, places=2)


if __name__ == '__main__':
    unittest.main()
