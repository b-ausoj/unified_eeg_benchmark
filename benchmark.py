# Description: Benchmarking script for the unified EEG benchmark.
from unified_eeg_benchmark.enums.split import Split
from unified_eeg_benchmark.datasets.abstract_dataset import AbstractDataset
from unified_eeg_benchmark.tasks.left_hand_right_hand_task import LeftHandRightHandTask
from models.csp_svm_model import CSPSVMModel
from models.abstract_model import AbstractModel
from typing import List
import numpy as np

def benchmark(tasks: List[AbstractDataset], models: List[AbstractModel]):
    for task in tasks:
        (X_train, y_train, meta_train) = task.get_data(Split.TRAIN)
        y_train = [y - 1 for y in y_train]
        (X_test, y_test, meta_test) = task.get_data(Split.TEST)
        y_test = np.concatenate(y_test, axis=0)
        y_test = y_test - 1

        # X is a list of numpy arrays, for each dataset one numpy array
        # each numpy array has dimensions (n_samples, n_channels, n_timepoints)
        # y is a list of numpy arrays, for each dataset one numpy array
        # each numpy array has dimensions (n_samples, )
        # meta is a list of dictionaries, for each dataset one dictionary
        # each dictionary contains meta information about the samples
        # such as the sampling frequency, the channel names, the labels mapping, etc.

        scoring = task.get_scoring()
        # something like scoring = (lambda y, y_pred : np.sqrt(mean_squared_error(y, y_pred.ravel())))

        for model in models:
            # now train a model by calling model.fit(X_train, y_train, meta_train)
            # and evaluate it by calling model.predict(X_test, meta_test)
            # and then evaluate the predictions by calling scoring(y_test, y_pred)
            model.fit(X_train, y_train, meta_train)
            y_pred = model.predict(X_test, meta_test)
            score = scoring(y_test, y_pred)
            print(f"Task: {task}, Model: {model}, Score: {score}")
            # print number of training samples, number of training samples for each class, 
            # number of test samples, number of test samples for each class
            print(f"Train samples: {len(np.concatenate(y_train, axis=0))}, Test samples: {len(y_test)}")
            print(f"Train samples per class: {np.bincount(np.concatenate(y_train, axis=0).ravel())}")
            print(f"Test samples per class: {np.bincount(y_test.ravel())}")
            

# this should also be possible to be defined in a separate file json/yaml so that it can be easily extended
# and shared with others, thus only strings and then somewhere else mapping to the actual classes
tasks = [
    # add tasks here
    LeftHandRightHandTask()
]

models_alt = {
    "lr_hand_mi": {
        # add models here
        "csp+svm": CSPSVMModel()
    }
}

models = [
    CSPSVMModel()
]

if __name__ == "__main__":
    benchmark(tasks, models)