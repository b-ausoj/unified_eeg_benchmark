from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def print_classification_results(y_train, y_test, model_names, y_preds, dataset_names):
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
        "Dataset": [f"{dataset_names[i]}" for i in range(len(y_train))],
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
            results.append([f"{dataset_names[i]}"] + list(dataset_metrics.values()))

        # Create a DataFrame for tabular formatting
        metrics_table = pd.DataFrame(
            results,
            columns=["Dataset", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
        )

        # Display the table
        print(metrics_table.to_string(index=False))
        print()
