import argparse
import logging
from tqdm import tqdm
from unified_eeg_benchmark.enums.split import Split
from unified_eeg_benchmark.tasks.clinical import (
    AbnormalClinicalTask,
    SchizophreniaClinicalTask,
    MTBIClinicalTask,
    OCDClinicalTask,
    EpilepsyClinicalTask,
    ParkinsonsClinicalTask,
)
from unified_eeg_benchmark.tasks.bci import (
    LeftHandvRightHandMITask,
    RightHandvFeetMITask,
    LeftHandvRightHandvFeetvTongueMITask,
    FiveFingersMITask,
)
from unified_eeg_benchmark.models.clinical import (
    BrainfeaturesLDAModel as BrainfeaturesLDA,
    BrainfeaturesSVMModel as BrainfeaturesSVM,
    LaBraMModel as LaBraMClinical,
    BENDRModel as BENDRClinical,
    NeuroGPTModel as NeuroGPTClinical,
)
from unified_eeg_benchmark.models.bci import (
    CSPLDAModel as CSPLDA,
    CSPSVMModel as CSPSVM,
    LaBraMModel as LaBraMBci,
    BENDRModel as BENDRBci,
    NeuroGPTModel as NeuroGPTBci,
)
from unified_eeg_benchmark.utils.evaluate_and_plot import print_classification_results, generate_classification_plots
from unified_eeg_benchmark.utils.utils import set_seed, save_results

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def benchmark(tasks, models, seed, reps):
    for task in tasks:
        logger.info(f"Running benchmark for task {task}")
        X_train, y_train, meta_train = task.get_data(Split.TRAIN)
        X_test, y_test, meta_test = task.get_data(Split.TEST)

        metrics = task.get_metrics()
        dataset_names = [m["name"] for m in meta_train]
        models_names = []
        results = []
        for model in tqdm(models):
            for i in range(reps):
                set_seed(seed + i)  # set seed for reproducibility

                model.fit(X_train, y_train, meta_train)
                y_pred = []
                for x, m in zip(X_test, meta_test):
                    y_pred.append(model.predict([x], [m]))

                models_names.append(str(model))
                results.append(y_pred)

        save_results(y_train, y_test, models_names, results, dataset_names, task.name)
        print_classification_results(
            y_train, y_test, models_names, results, dataset_names, task.name, metrics
        )
        generate_classification_plots(y_train, y_test, models_names, results, dataset_names, task.name, metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Run Unified EEG Benchmark for a specific task and model."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to run. Options: parkinsons, schizophrenia, mtbi, ocd, epilepsy, abnormal, depression, med, bdi, age, sex"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use. Options: brainfeatures_lda, brainfeatures_svm, brainfeatures_dt, brainfeatures_rf, brainfeatures_sgd, labram, bendr, neurogpt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Number of repetitions with different seeds for variability assessment"
    )
    args = parser.parse_args()

    # Mapping command-line strings to task classes
    tasks_map = {
        "pd": ParkinsonsClinicalTask,
        "sz": SchizophreniaClinicalTask,
        "mtbi": MTBIClinicalTask,
        "ocd": OCDClinicalTask,
        "ep": EpilepsyClinicalTask,
        "ab": AbnormalClinicalTask,
        "lr": LeftHandvRightHandMITask,
        "rf": RightHandvFeetMITask,
        "lrft": LeftHandvRightHandvFeetvTongueMITask,
        "5f": FiveFingersMITask,
    }

    # Mapping command-line strings to model classes
    clinical_models_map = {
        "lda": BrainfeaturesLDA,
        "svm": BrainfeaturesSVM,
        "labram": LaBraMClinical,
        "bendr": BENDRClinical,
        "neurogpt": NeuroGPTClinical,
    }
    bci_models_map = {
        "lda": CSPLDA,
        "svm": CSPSVM,
        "labram": LaBraMBci,
        "bendr": BENDRBci,
        "neurogpt": NeuroGPTBci,
    }

    task_key = args.task.lower()
    model_key = args.model.lower()

    if task_key not in tasks_map:
        parser.error(f"Invalid task specified. Choose from: {', '.join(tasks_map.keys())}")
    task_instance = tasks_map[task_key]()
    
    if task_key in ["pd", "sz", "mtbi", "ocd", "ep", "ab"]:
        models_map = clinical_models_map
    elif task_key in ["lr", "rf", "lrft", "5f"]:
        models_map = bci_models_map
    else:
        models_map = {}
        parser.error(f"Invalid task specified. Choose from: {', '.join(tasks_map.keys())}")
    
    if model_key not in models_map:
        parser.error(f"Invalid model specified. Choose from: {', '.join(models_map.keys())}")
    model_instance = models_map[model_key]()

    # Run the benchmark with the specified task, model, and seed
    benchmark([task_instance], [model_instance], args.seed, args.reps)

if __name__ == "__main__":
    main()
