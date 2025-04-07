import random
import numpy as np
import torch
import os
import json
from datetime import datetime

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_results(
    y_train,
    y_test,
    models_names,
    results,
    dataset_names,
    task_name,
):

    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_names_unique = list(set(models_names))
    models_str = "_".join(models_names_unique) if models_names_unique else "models"

    # Build the filename with task name, models, and timestamp
    filename = os.path.join("results", "raw", f"{task_name}_{models_str}_{timestamp}.json")

    # Prepare the data to be saved
    data_to_save = {
        "y_train": y_train,
        "y_test": y_test,
        "models_names": models_names,
        "results": results,
        "dataset_names": dataset_names,
        "task_name": task_name,
        "timestamp": timestamp
    }

    # Save the results to the file
    with open(filename, "w") as f:
        json.dump(data_to_save, f)
    print(f"Results saved to {filename}")