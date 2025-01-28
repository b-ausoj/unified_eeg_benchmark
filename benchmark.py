# Description: Benchmarking script for the unified EEG benchmark.
from unified_eeg_benchmark.enums.split import Split
from unified_eeg_benchmark.datasets.abstract_dataset import AbstractDataset
from unified_eeg_benchmark.tasks.left_hand_right_hand_task import LeftHandRightHandTask
from models.csp_svm_model import CSPSVMModel
from models.csp_lda_model import CSPLDAModel
from models.abstract_model import AbstractModel
from sklearn.preprocessing import LabelEncoder
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def print_classification_results(y_train, y_test, model_names, y_preds):
    # Assuming y_train and y_test are lists of numpy arrays

    # Gather basic statistics
    train_samples = [len(y) for y in y_train]
    test_samples = [len(y) for y in y_test]

    def class_distribution(y_list):
        encoder = LabelEncoder()
        distributions = []
        for y in y_list:
            encoded_y = encoder.fit_transform(y)  # Encode string labels to integers
            distributions.append(np.bincount(encoded_y).tolist())
        return distributions

    train_distribution = class_distribution(y_train)
    test_distribution = class_distribution(y_test)

    # Create a DataFrame for better formatting
    task_data = {
        "Dataset": [f"Dataset {i+1}" for i in range(len(y_train))],
        "Train Samples": train_samples,
        "Test Samples": test_samples,
        "Train Class Distribution": [str(dist) for dist in train_distribution],
        "Test Class Distribution": [str(dist) for dist in test_distribution],
    }
    task_table = pd.DataFrame(task_data)

    # Display the task overview
    print("\n" + "=" * 24 + " Task Overview " + "=" * 24 + "\n")
    print(task_table.to_string(index=False))
    print()

    # Function to calculate metrics
    def calculate_metrics(y_true, y_pred):
        encoder = LabelEncoder()
        y_true = encoder.fit_transform(y_true)
        y_pred = encoder.transform(y_pred)
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted"),
            "Recall": recall_score(y_true, y_pred, average="weighted"),
            "F1 Score": f1_score(y_true, y_pred, average="weighted"),
            "AUC": (
                roc_auc_score(y_true, y_pred, multi_class="ovr")
                if len(np.unique(y_true)) > 2
                else roc_auc_score(y_true, y_pred)
            ),
        }

    # Iterate over models and create tables
    for model_name, y_pred in zip(model_names, y_preds):
        print("-" * 25 + f" {model_name} " + "-" * 25)

        # Overall metrics
        combined_y_test = np.concatenate(y_test)
        combined_y_pred = np.concatenate(y_pred)
        combined_metrics = calculate_metrics(combined_y_test, combined_y_pred)

        # Create a table for combined and per-dataset metrics
        results = []

        # Add overall metrics to the table
        results.append(["Combined"] + list(combined_metrics.values()))

        # Add per-dataset metrics
        for i, (y_true, y_pred) in enumerate(zip(y_test, y_pred)):
            dataset_metrics = calculate_metrics(y_true, y_pred)
            results.append([f"Dataset {i+1}"] + list(dataset_metrics.values()))

        # Create a DataFrame for tabular formatting
        metrics_table = pd.DataFrame(
            results,
            columns=["Dataset", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
        )

        # Display the table
        print(metrics_table.to_string(index=False))
        print()


def benchmark(tasks: List[AbstractDataset], models: List[AbstractModel]):
    for task in tasks:
        (X_train, y_train, meta_train) = task.get_data(Split.TRAIN)
        # y_train = [y - 1 for y in y_train]
        (X_test, y_test, meta_test) = task.get_data(Split.TEST)
        # y_test = [y - 1 for y in y_test]

        # X is a list of numpy arrays, for each dataset one numpy array
        # each numpy array has dimensions (n_samples, n_channels, n_timepoints)
        # y is a list of numpy arrays, for each dataset one numpy array
        # each numpy array has dimensions (n_samples, )
        # meta is a list of dictionaries, for each dataset one dictionary
        # each dictionary contains meta information about the samples
        # such as the sampling frequency, the channel names, the labels mapping, etc.

        scoring = task.get_scoring()

        models_names = []
        results = []
        for model in models:

            model.fit(X_train, y_train, meta_train)
            y_pred = []
            for x, m in zip(X_test, meta_test):
                y_pred.append(model.predict([x], [m]))

            models_names.append(str(model))
            results.append(y_pred)

        print_classification_results(y_train, y_test, models_names, results)


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

models = [CSPSVMModel(), CSPLDAModel()]

if __name__ == "__main__":
    benchmark(tasks, models)
