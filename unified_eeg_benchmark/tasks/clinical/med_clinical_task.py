from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import (
    ParkinsonsRestD002Dataset,
    SchizophreniaConflictD004Dataset,
    ParkinsonsConflictTaskD005Dataset,
    ParkinsonsRLTaskD007Dataset,
)
from sklearn.metrics import f1_score
from ...enums.split import Split


class MedClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="ON/OFF Medication",
            clinical_class = ClinicalClasses.MEDICATION,
            datasets = [
                ParkinsonsRestD002Dataset,
                SchizophreniaConflictD004Dataset,
                ParkinsonsConflictTaskD005Dataset,
                ParkinsonsRLTaskD007Dataset,
            ],
            subjects_split={
                ParkinsonsRestD002Dataset: {
                    Split.TRAIN: list(range(1, 15)) + list(range(29, 43)),
                    Split.TEST: list(range(15, 29)) + list(range(43, 57)),
                },
                SchizophreniaConflictD004Dataset: {
                    Split.TRAIN: list(range(1, 13)) + list(range(26, 60)),
                    Split.TEST: list(range(13, 26)) + list(range(60, 76)),
                },
                ParkinsonsConflictTaskD005Dataset: {
                    Split.TRAIN: list(range(1, 15)) + list(range(29, 43)),
                    Split.TEST: list(range(15, 29)) + list(range(43, 57)),
                },
                ParkinsonsRLTaskD007Dataset: {
                    Split.TRAIN: list(range(1, 15)) + list(range(29, 43)),
                    Split.TEST: list(range(15, 29)) + list(range(43, 57)),
                },
            }
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
