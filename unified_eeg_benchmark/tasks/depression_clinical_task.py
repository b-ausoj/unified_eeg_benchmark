from .abstract_clinical_task import AbstractClinicalTask
from ..enums.clinical_classes import ClinicalClasses
from ..datasets.cavanagh2019a import Cavanagh2019ADataset
from ..datasets.cavanagh2019b import Cavanagh2019BDataset
from sklearn.metrics import f1_score
from ..enums.split import Split

base_path = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/"


class DepressionClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="depression_clinical",
            clinical_class = ClinicalClasses.DEPRESSION,
            datasets = [
                #Cavanagh2019ADataset,
                Cavanagh2019BDataset,
            ],
            subjects_split={
                Cavanagh2019ADataset: {
                    Split.TRAIN: list(range(1, 26)) + list(range(51, 76)),
                    Split.TEST: list(range(26, 51)) + list(range(76, 101)),
                },
                Cavanagh2019BDataset: {
                    Split.TRAIN: list(range(1, 26)) + list(range(51, 76)),
                    Split.TEST: list(range(26, 51)) + list(range(76, 101)),
                },
            },
        )

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
