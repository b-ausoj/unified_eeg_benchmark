# Description: Benchmarking script for the Unified EEG Benchmark.
from unified_eeg_benchmark.enums.split import Split
from unified_eeg_benchmark.tasks.clinical import (
    AbstractClinicalTask,
    AbnormalClinicalTask,
    SchizophreniaClinicalTask,
    MTBIClinicalTask,
    OCDClinicalTask,
    DepressionClinicalTask,
    EpilepsyClinicalTask,
    ParkinsonsClinicalTask,
    MedClinicalTask,
    BDIClinicalTask,
    AgeClinicalTask,
    SexClinicalTask,
)
from unified_eeg_benchmark.models.clinical import (
    BrainfeaturesLDAModel,
    BrainfeaturesSVMModel,
    BrainfeaturesDTModel,
    BrainfeaturesRFModel,
    BrainfeaturesSGDModel,
    LaBraMModel,
    BENDRModel,
    NeuroGPTModel,
    MaximsModel,
)
from unified_eeg_benchmark.models.abstract_model import AbstractModel
from unified_eeg_benchmark.utils.evaluate_and_plot import print_classification_results, generate_classification_plots
from unified_eeg_benchmark.utils.utils import set_seed, save_results
from typing import Sequence
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.ERROR,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark(tasks: Sequence[AbstractClinicalTask], models: Sequence[AbstractModel], seed: int):
    for task in tasks:
        logger.info(f"Running benchmark for task {task}")
        print(f"Running benchmark for task {task}")
        (X_train, y_train, meta_train) = task.get_data(Split.TRAIN)
        (X_test, y_test, meta_test) = task.get_data(Split.TEST)

        metrics = task.get_metrics()
        dataset_names = [m["name"] for m in meta_train]
        models_names = []
        results = []
        for model in tqdm(models):
            set_seed(seed) # set seed for reproducibility,

            model.fit(X_train, y_train, meta_train)
            y_pred = []
            for x, m in zip(X_test, meta_test):
                y_pred.append(model.predict([x], [m]))

            models_names.append(str(model))
            results.append(y_pred)
            seed += 1

        save_results(y_train, y_test, models_names, results, dataset_names, task.name)
        print_classification_results(
            y_train, y_test, models_names, results, dataset_names, task.name, metrics
        )
        generate_classification_plots(y_train, y_test, models_names, results, dataset_names, task.name, metrics)


if __name__ == "__main__":
    seed = 100
    print(f"Seed: {seed}")
    tasks = [
        # ParkinsonsClinicalTask(),
        # SchizophreniaClinicalTask(),
        # MTBIClinicalTask(),
        # OCDClinicalTask(),
        # EpilepsyClinicalTask(),
        AbnormalClinicalTask(),
    ]
    models = [
        MaximsModel(),
        # BrainfeaturesLDAModel(),
        # BrainfeaturesSVMModel(seed=seed),
        # LaBraMModel(),
        # BENDRModel(),
        # NeuroGPTModel(seed=seed),
    ]

    benchmark(tasks, models, seed)

