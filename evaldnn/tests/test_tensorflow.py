"""
Provides a class for testing tensorflow model evaluation.
"""

from __future__ import absolute_import

import unittest

import foolbox

from evaldnn.metrics.accuracy import Accuracy
from evaldnn.metrics.neuron_coverage import NeuronCoverage
from evaldnn.metrics.robustness import Robustness
from evaldnn.models.tensorflow import TensorFlowModel
from evaldnn.utils.tensorflow import *


class TestTensorFlow(unittest.TestCase):
    def test_cifar10_simple(self):
        print('test_cifar10_simple')

        # load the model and data
        tf.get_logger().setLevel('ERROR')
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        restorer = tf.compat.v1.train.import_meta_graph(common.user_home_dir() + '/EvalDNN-models/tensorflow/cifar10_simple/tensorflow_cifar10_simple.meta')
        restorer.restore(session, tf.train.latest_checkpoint(common.user_home_dir() + '/EvalDNN-models/tensorflow/cifar10_simple/'))
        input = session.graph.get_tensor_by_name('Placeholder:0')
        logits = session.graph.get_tensor_by_name('fc2/add:0')
        x, y = cifar10_test_data()
        x_robustness, y_robustness = cifar10_test_data(num_max=3)
        bounds = (0, 1)

        # wrap the model with EvalDNN
        measure_model = TensorFlowModel(session, logits, input)

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

        self.assertAlmostEqual(accuracy.get(1), 0.327100)
        self.assertAlmostEqual(accuracy.get(5), 0.820200)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.747292, places=2)

        session.close()

    def test_mnist_simple(self):
        print('test_mnist_simple')

        # load the model and data
        tf.get_logger().setLevel('ERROR')
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        restorer = tf.compat.v1.train.import_meta_graph(common.user_home_dir() + '/EvalDNN-models/tensorflow/mnist_simple/tensorflow_mnist_simple.meta')
        restorer.restore(session, tf.train.latest_checkpoint(common.user_home_dir() + '/EvalDNN-models/tensorflow/mnist_simple/'))
        input = session.graph.get_tensor_by_name('Placeholder:0')
        logits = session.graph.get_tensor_by_name('fc2/add:0')
        x, y = mnist_test_data()
        x_robustness, y_robustness = mnist_test_data(num_max=3)
        bounds = (0, 1)

        # wrap the model with EvalDNN
        measure_model = TensorFlowModel(session, logits, input)

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

        self.assertAlmostEqual(accuracy.get(1), 0.937700)
        self.assertAlmostEqual(accuracy.get(5), 0.997200)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.769821, places=2)

        session.close()

    def test_imagenet_vgg16(self):
        print('test_imagenet_vgg16')

        # load the model and data
        session, logits, input, data_normalized, data_original, mean, std, bounds = imagenet_benchmark_zoo('vgg16', data_original_num_max=3)

        # wrap the model with EvalDNN
        measure_model = TensorFlowModel(session, logits, input)

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
                                         preprocessing=(mean, std))

        self.assertAlmostEqual(accuracy.get(1), 0.600000)
        self.assertAlmostEqual(accuracy.get(5), 0.925000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.764870, places=2)

        session.close()


if __name__ == '__main__':
    unittest.main()
