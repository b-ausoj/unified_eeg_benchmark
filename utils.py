from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

def one_hot_encode(y):
        encoder = OneHotEncoder(sparse_output=False)
        y_reshaped = np.array(y).reshape(-1, 1)
        return encoder.fit_transform(y_reshaped)

def print_classification_results(y_train, y_test, model_names, y_preds, dataset_names, task_name):
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
        "Dataset": [f"{dataset_names[i]}" for i in range(len(dataset_names))],
        "Train Samples": train_samples,
        "Test Samples": test_samples,
        "Train Class Distribution": [str(dist) for dist in train_distribution],
        "Test Class Distribution": [str(dist) for dist in test_distribution],
    }
    task_table = pd.DataFrame(task_data)

    # Display the task overview
    print("\n" + "=" * 24 + " Task Overview " + "=" * 24 + "\n")
    print(f"{'Task: ' + task_name:^60}\n")
    print(task_table.to_string(index=False))
    print()

    # Function to calculate metrics
    def calculate_metrics(y_true, y_pred):
        encoder = LabelEncoder()
        y_true = encoder.fit_transform(y_true)
        y_pred = encoder.transform(y_pred)
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted"),
            "Recall": recall_score(y_true, y_pred, average="weighted"),
            "F1 Score": f1_score(y_true, y_pred, average="weighted"),
            "AUC": (
                roc_auc_score(one_hot_encode(y_true), one_hot_encode(y_pred), multi_class="ovr")
                if len(np.unique(y_true)) > 2
                else (roc_auc_score(y_true, y_pred) if len(y_true) > 0 else 0)
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
            columns=["Dataset", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
        )

        # Display the table
        print(metrics_table.to_string(index=False))
        print()

def save_class_distribution_plots(dataset_names, train_distribution, test_distribution, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    for i, dataset in enumerate(dataset_names):
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(train_distribution[i])), train_distribution[i], alpha=0.7, label="Train")
        plt.bar(range(len(test_distribution[i])), test_distribution[i], alpha=0.7, label="Test")
        plt.xlabel("Class Label")
        plt.ylabel("Count")
        plt.title(f"Class Distribution - {dataset}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"class_distribution_{dataset}.png"))
        plt.close()

def save_evaluation_plots(model_names, dataset_names, all_metrics, task_name, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, metrics_table in zip(model_names, all_metrics):
        plt.figure(figsize=(10, 6))
        metrics_table.set_index("Dataset").plot(kind="bar", figsize=(10, 6))
        plt.title(f"Evaluation Metrics - {model_name} - {task_name}")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend(loc="lower right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"metrics_{model_name}_{task_name}.png"))
        plt.close()

def generate_classification_plots(y_train, y_test, model_names, y_preds, dataset_names, task_name, output_dir="plots"):
    train_samples = [len(y) for y in y_train]
    test_samples = [len(y) for y in y_test]
    
    def class_distribution(y_list):
        encoder = LabelEncoder()
        distributions = []
        for y in y_list:
            encoded_y = encoder.fit_transform(y)
            distributions.append(np.bincount(encoded_y).tolist())
        return distributions
    
    train_distribution = class_distribution(y_train)
    test_distribution = class_distribution(y_test)
    
    # Save class distribution plots
    save_class_distribution_plots(dataset_names, train_distribution, test_distribution, output_dir)
    
    def calculate_metrics(y_true, y_pred):
        encoder = LabelEncoder()
        y_true = encoder.fit_transform(y_true)
        y_pred = encoder.transform(y_pred)
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted"),
            "Recall": recall_score(y_true, y_pred, average="weighted"),
            "F1 Score": f1_score(y_true, y_pred, average="weighted"),
            "AUC": (
                roc_auc_score(one_hot_encode(y_true), one_hot_encode(y_pred), multi_class="ovr")
                if len(np.unique(y_true)) > 2
                else roc_auc_score(y_true, y_pred)
            ),
        }
    
    all_metrics = []
    
    for model_name, y_pred in zip(model_names, y_preds):
        combined_y_test = np.concatenate(y_test)
        combined_y_pred = np.concatenate(y_pred)
        combined_metrics = calculate_metrics(combined_y_test, combined_y_pred)
        
        results = [["Combined"] + list(combined_metrics.values())]
        for i, (y_true, y_pred) in enumerate(zip(y_test, y_pred)):
            dataset_metrics = calculate_metrics(y_true, y_pred)
            results.append([f"{dataset_names[i]}"] + list(dataset_metrics.values()))
        
        metrics_table = pd.DataFrame(
            results,
            columns=["Dataset", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
        )
        all_metrics.append(metrics_table)
    
    # Save evaluation metric plots
    save_evaluation_plots(model_names, dataset_names, all_metrics, task_name, output_dir)
