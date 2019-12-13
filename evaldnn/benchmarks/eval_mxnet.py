"""
Provides code for mxnet pretrained models evaluation
"""

from __future__ import absolute_import

import foolbox

from evaldnn.metrics.accuracy import Accuracy
from evaldnn.metrics.neuron_coverage import NeuronCoverage
from evaldnn.metrics.robustness import Robustness
from evaldnn.models.mxnet import MXNetModel
from evaldnn.utils.mxnet import *


def eval():
    """Evaluate accuracy, neuron coverage and robustness of all pretrained
    models.

    Top-1 accuracy and top-5 accuracy are both measured.
    For neuron coverage, measured thresholds include 0.0, 0.1, 0.2, ..., 0.9.
    To measure robustness, three adversarial attack methods, FGSM, BIM and
    DeepFoolAttack, are used.

    """
    for model_name in imagenet_benchmark_zoo_model_names():
        print('Model: ' + model_name)

        # load the model and data
        model, dataset_normalized, dataset_original, preprocessing, num_classes, bounds = imagenet_benchmark_zoo(model_name)

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
        robustness.set_save_dir(common.user_home_dir() + '/EvalDNN-adversarial-attack/mxnet/imagenet/' + model_name + '/FGSM')
        measure_model.adversarial_attack(dataset_original,
                                         bounds,
                                         num_classes,
                                         [robustness.update, robustness.save, robustness.report],
                                         attack=foolbox.attacks.FGSM,
                                         distance=foolbox.distances.Linf,
                                         preprocessing=preprocessing)

        # evaluate the robustness of the model
        robustness = Robustness(bounds)
        robustness.set_descriptions(dataset_original.filenames)
        robustness.set_save_dir(common.user_home_dir() + '/EvalDNN-adversarial-attack/mxnet/imagenet/' + model_name + '/BIM')
        measure_model.adversarial_attack(dataset_original,
                                         bounds,
                                         num_classes,
                                         [robustness.update, robustness.save, robustness.report],
                                         attack=foolbox.attacks.BIM,
                                         distance=foolbox.distances.Linf,
                                         preprocessing=preprocessing)

        # evaluate the robustness of the model
        robustness = Robustness(bounds)
        robustness.set_descriptions(dataset_original.filenames)
        robustness.set_save_dir(common.user_home_dir() + '/EvalDNN-adversarial-attack/mxnet/imagenet/' + model_name + '/DeepFoolAttack')
        measure_model.adversarial_attack(dataset_original,
                                         bounds,
                                         num_classes,
                                         [robustness.update, robustness.save, robustness.report],
                                         attack=foolbox.attacks.DeepFoolAttack,
                                         distance=foolbox.distances.MSE,
                                         preprocessing=preprocessing)


if __name__ == '__main__':
    eval()
