# Description: Benchmarking script for the unified EEG benchmark.
from unified_eeg_benchmark.enums.split import Split
from unified_eeg_benchmark.tasks.abstract_task import AbstractTask
from unified_eeg_benchmark.tasks.left_hand_right_hand_task import LeftHandRightHandTask
from models.csp_svm_model import CSPSVMModel
from models.csp_lda_model import CSPLDAModel
from models.abstract_model import AbstractModel
from utils import print_classification_results
from typing import List


def benchmark(tasks: List[AbstractTask], models: List[AbstractModel]):
    for task in tasks:
        (X_train, y_train, meta_train) = task.get_data(Split.TRAIN)
        # y_train = [y - 1 for y in y_train]
        (X_test, y_test, meta_test) = task.get_data(Split.TEST)
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
        for model in models:

            model.fit(X_train, y_train, meta_train)
            y_pred = []
            for x, m in zip(X_test, meta_test):
                y_pred.append(model.predict([x], [m]))

            models_names.append(str(model))
            results.append(y_pred)

        print_classification_results(
            y_train, y_test, models_names, results, dataset_names
        )


if __name__ == "__main__":
    tasks = [LeftHandRightHandTask()]
    models = [CSPSVMModel(), CSPLDAModel()]

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
