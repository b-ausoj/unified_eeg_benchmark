# Description: Benchmarking script for the unified EEG benchmark.
from unified_eeg_benchmark.enums.split import Split
from unified_eeg_benchmark.tasks.bci import (
    AbstractBCITask,
    LeftHandvRightHandMITask,
    RightHandvFeetMITask,
    LeftHandvRightHandvFeetvTongueMITask,
    FleExtSupProCloOpnMITask,
    FlexionvExtensionMITask,
    PronationvSupinationMITask,
    HandOpenvCloseMITask,
)   
from unified_eeg_benchmark.models.abstract_model import AbstractModel
from unified_eeg_benchmark.models.bci import (
    BENDRModel,
    CSPSVMModel,
    CSPLDAModel,
    CSPriemannLDAModel,
    FgMDMModel,
    LaBraMModel,
    NeuroGPTModel,
    TSLRModel,
)
from unified_eeg_benchmark.enums.classes import Classes
from unified_eeg_benchmark.utils.evaluate_and_plot import print_classification_results, generate_classification_plots
from typing import Sequence
from tqdm import tqdm
import logging
import torch
import numpy as np

logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark(tasks: Sequence[AbstractBCITask], models: Sequence[AbstractModel]):
    for task in tasks:
        logger.info(f"Running benchmark for task {task}")
        print(f"Running benchmark for task {task}")
        (X_train, y_train, meta_train) = task.get_data(Split.TRAIN)
        def make_multiple_of_64(data):
            num_samples = data.shape[0]
            remainder = num_samples % 64
            if remainder != 0:
                data = data[:num_samples - remainder]
            return data

        #X_train = [make_multiple_of_64(x) for x in X_train]
        #y_train = [make_multiple_of_64(y) for y in y_train]
        # y_train = [y - 1 for y in y_train]
        (X_test, y_test, meta_test) = task.get_data(Split.TEST)
        #X_test = [make_multiple_of_64(x) for x in X_test]
        #y_test = [make_multiple_of_64(y) for y in y_test]
        # y_test = [y - 1 for y in y_test]

        # X_train = [x[:1260] for x in X_train]
        # y_train = [y[:1260] for y in y_train]
        # X_test = [x[:320] for x in X_test]
        # y_test = [y[:320] for y in y_test]

        # X is a list of numpy arrays, for each dataset one numpy array
        # each numpy array has dimensions (n_samples, n_channels, n_timepoints)
        # y is a list of numpy arrays, for each dataset one numpy array
        # each numpy array has dimensions (n_samples, )
        # meta is a list of dictionaries, for each dataset one dictionary
        # each dictionary contains meta information about the samples
        # such as the sampling frequency, the channel names, the labels mapping, etc.

        scoring = task.get_scoring()
        dataset_names = [m["name"] for m in meta_train]
        models_names = []
        results = []
        for model in tqdm(models):

            model.fit(X_train, y_train, meta_train)
            y_pred = []
            for x, m in zip(X_test, meta_test):
                if len(x) > 0:
                    y_pred.append(model.predict([x], [m]))
                else:
                    y_pred.append([])

            models_names.append(str(model))
            results.append(y_pred)
        
        for model_name, y_pred_model in zip(models_names, results):
            unique, counts = np.unique(np.concatenate(y_pred_model), return_counts=True)
            distribution = dict(zip(unique, counts))
            print(f"Distribution for model {model_name}: {distribution}")
        
        print_classification_results(
            y_train, y_test, models_names, results, dataset_names, task.name
        )
        generate_classification_plots(y_train, y_test, models_names, results, dataset_names, task.name)


if __name__ == "__main__":
    tasks = [
        #LeftHandvRightHandMITask(),
        #RightHandvFeetMITask(),
        LeftHandvRightHandvFeetvTongueMITask(),
        #FleExtSupProCloOpnMITask(),
        #FlexionvExtensionMITask(),
        #HandOpenvCloseMITask(),
        #PronationvSupinationMITask(),
    ]
    models = [
        CSPSVMModel(),
        CSPLDAModel(),
        CSPriemannLDAModel(),
        FgMDMModel(),
        TSLRModel(),
        LaBraMModel(),
        NeuroGPTModel(),
        BENDRModel(),
    ]
    benchmark(tasks, models)


# this should also be possible to be defined in a separate file json/yaml so that it can be easily extended
# and shared with others, thus only strings and then somewhere else mapping to the actual classes

# tasks = [
# add tasks here
# ]
# models_alt = {
#    "lr_hand_mi": {
#        # add models here
#        "csp+svm": CSPSVMModel()
#    }
# }

"""
CustomMITask(
            classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI],
            subjects_split={
                BCICompIV2aMDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7, 8],
                    Split.TEST: [9],
                },
                BCICompIV2bMDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7, 8],
                    Split.TEST: [9],
                },
                GrosseWentrup2009MDataset: {
                    Split.TRAIN: [1],
                    Split.TEST: [2, 3, 4, 5, 6],
                },
            },
        ),
"""