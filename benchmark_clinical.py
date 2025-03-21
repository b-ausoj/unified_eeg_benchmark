# Description: Benchmarking script for the unified EEG benchmark.
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
from models.csp_lda_cli_unm_model import CSPLDACliUnmModel
from models.csp_lda_epilepsy_model import CSPLDAEpilepsyModel
from models.csp_lda_abnormal_model import CSPLDAAbnormalModel
#from models.LaBraM.labram_model_old import LaBraMModel
from models.clinical import (
    BrainfeaturesModel,
)
from models.abstract_model import AbstractModel
from .utils.evaluate_and_plot import print_classification_results, generate_classification_plots
from typing import Sequence
from tqdm import tqdm
import logging
import mne

logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)
mne.set_log_level("ERROR")

def benchmark(tasks: Sequence[AbstractClinicalTask], models: Sequence[AbstractModel]):
    for task in tasks:
        logger.info(f"Running benchmark for task {task}")
        X_train, y_train, meta_train = task.get_data(Split.TRAIN)
        X_test, y_test, meta_test = task.get_data(Split.TEST)


        # X is a list of numpy arrays, for each dataset one numpy array
        # each numpy array has dimensions (n_samples, n_channels, n_timepoints)
        # y is a list of numpy arrays, for each dataset one numpy array
        # each numpy array has dimensions (n_samples, )
        # meta is a list of dictionaries, for each dataset one dictionary
        # each dictionary contains meta information about the samples
        # such as the sampling frequency, the channel names, the labels mapping, etc.

        scoring = task.get_scoring() # TODO rename to metrics and then return the metrics I should use in print and generate the plots
        dataset_names = [m["name"] for m in meta_train]
        models_names = []
        results = []
        for model in tqdm(models):

            model.fit(X_train, y_train, meta_train)
            y_pred = []
            for x, m in zip(X_test, meta_test):
                y_pred.append(model.predict([x], [m]))

            models_names.append(str(model))
            results.append(y_pred)

        print_classification_results(
            y_train, y_test, models_names, results, dataset_names, task.name
        )
        generate_classification_plots(y_train, y_test, models_names, results, dataset_names, task.name)


if __name__ == "__main__":
    tasks = [
        ParkinsonsClinicalTask(),
        #DepressionClinicalTask(),
        #SchizophreniaClinicalTask(),
        #MTBIClinicalTask(),
        #OCDClinicalTask(), # error in some data
        #EpilepsyClinicalTask(),
        #AbnormalClinicalTask(),
        #MedClinicalTask, # difficult
        #BDIClinicalTask, # regression
        #AgeClinicalTask, # regression
        #SexClinicalTask, # labels unsure
    ]
    models = [
        #CSPLDACliUnmModel(),
        #CSPLDAEpilepsyModel(),
        #CSPLDAAbnormalModel(),
        #LaBraMModel(),
        BrainfeaturesModel(),
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
