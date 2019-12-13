"""
Provides a class for testing torch model evaluation.
"""

from __future__ import absolute_import

import unittest

import foolbox

from evaldnn.metrics.accuracy import Accuracy
from evaldnn.metrics.neuron_coverage import NeuronCoverage
from evaldnn.metrics.robustness import Robustness
from evaldnn.models.pytorch import PyTorchModel
from evaldnn.utils.pytorch import *


class TestPyTorch(unittest.TestCase):

    def test_cifar10_simple(self):
        print('test_cifar10_simple')

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, 3)
                self.pool1 = torch.nn.MaxPool2d(2)
                self.conv2 = torch.nn.Conv2d(32, 64, 3)
                self.pool2 = torch.nn.MaxPool2d(2)
                self.conv3 = torch.nn.Conv2d(64, 64, 3)
                self.fc1 = torch.nn.Linear(64 * 4 * 4, 64)
                self.fc2 = torch.nn.Linear(64, 10)

            def forward(self, data):
                data = torch.nn.functional.relu(self.conv1(data))
                data = self.pool1(data)
                data = torch.nn.functional.relu(self.conv2(data))
                data = self.pool2(data)
                data = torch.nn.functional.relu(self.conv3(data))
                data = data.view(-1, 64 * 4 * 4)
                data = torch.nn.functional.relu(self.fc1(data))
                data = self.fc2(data)
                return data

        # load the model and data
        model = Model()
        model.load_state_dict(torch.load(common.user_home_dir() + '/EvalDNN-models/pytorch/cifar10_simple.pth'))
        dataset = cifar10_dataset()
        dataset_robustness = cifar10_dataset(num_max=3)
        bounds = (0, 1)
        num_classes = 10

        # wrap the model with EvalDNN
        measure_model = PyTorchModel(model)

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

        self.assertAlmostEqual(accuracy.get(1), 0.427500)
        self.assertAlmostEqual(accuracy.get(5), 0.906400)
        self.assertAlmostEqual(neuron_coverage.get(0.6), 0.409091, places=2)

    def test_mnist_simple(self):
        print('test_mnist_simple')

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(1 * 28 * 28, 128)
                self.fc2 = torch.nn.Linear(128, 64)
                self.fc3 = torch.nn.Linear(64, 10)

            def forward(self, data):
                data = data.view(-1, 1 * 28 * 28)
                data = torch.nn.functional.relu(self.fc1(data))
                data = torch.nn.functional.relu(self.fc2(data))
                data = self.fc3(data)
                return data

        # load the model and data
        model = Model()
        model.load_state_dict(torch.load(common.user_home_dir() + '/EvalDNN-models/pytorch/mnist_simple.pth'))
        dataset = mnist_dataset()
        dataset_robustness = mnist_dataset(num_max=3)
        bounds = (0, 1)
        num_classes = 10

        # wrap the model with EvalDNN
        measure_model = PyTorchModel(model)

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

        self.assertAlmostEqual(accuracy.get(1), 0.962200)
        self.assertAlmostEqual(accuracy.get(5), 0.998900)
        self.assertAlmostEqual(neuron_coverage.get(0.7), 0.876238, places=2)

    def test_imagenet_vgg16(self):
        print('test_imagenet_vgg16')

        # load the model and data
        model, dataset_normalized, dataset_original, preprocessing, num_classes, bounds = imagenet_benchmark_zoo('vgg16', data_original_num_max=3)

        # wrap the model with EvalDNN
        measure_model = PyTorchModel(model)

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

        self.assertAlmostEqual(accuracy.get(1), 0.650000)
        self.assertAlmostEqual(accuracy.get(5), 0.925000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.540804, places=2)


if __name__ == '__main__':
    unittest.main()
