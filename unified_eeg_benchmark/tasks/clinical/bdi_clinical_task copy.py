from abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import (
    ParkinsonsRestD002Dataset,
    DepressionRestD003Dataset,
    ParkinsonsConflictTaskD005Dataset,
    DepressionRLD006Dataset,
    ParkinsonsRLTaskD007Dataset,
    OCDFlankersD008Dataset,
    MTBIOddballD009Dataset,
    MTBIRestD012Dataset,
)
from sklearn.metrics import f1_score
from ...enums.split import Split


class BDIClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="BDI Score",
            clinical_class = ClinicalClasses.BDI,
            datasets = [
                ParkinsonsRestD002Dataset,
                DepressionRestD003Dataset,
                ParkinsonsConflictTaskD005Dataset,
                DepressionRLD006Dataset,
                ParkinsonsRLTaskD007Dataset,
                OCDFlankersD008Dataset,
                MTBIOddballD009Dataset,
                MTBIRestD012Dataset,
            ],
            subjects_split={
                ParkinsonsRestD002Dataset: {
                    Split.TRAIN: list(range(1, 15)) + list(range(29, 43)),
                    Split.TEST: list(range(15, 29)) + list(range(43, 57)),
                },
                DepressionRestD003Dataset: {
                    Split.TRAIN: list(range(1, 26)) + list(range(51, 76)),
                    Split.TEST: list(range(26, 51)) + list(range(76, 101)),
                },
                ParkinsonsConflictTaskD005Dataset: {
                    Split.TRAIN: list(range(1, 15)) + list(range(29, 43)),
                    Split.TEST: list(range(15, 29)) + list(range(43, 57)),
                },
                DepressionRLD006Dataset: {
                    Split.TRAIN: list(range(1, 26)) + list(range(51, 76)),
                    Split.TEST: list(range(26, 51)) + list(range(76, 101)),
                },
                ParkinsonsRLTaskD007Dataset: {
                    Split.TRAIN: list(range(1, 15)) + list(range(29, 43)),
                    Split.TEST: list(range(15, 29)) + list(range(43, 57)),
                },
                OCDFlankersD008Dataset: {
                    Split.TRAIN: list(range(1, 13)) + list(range(26, 37)),
                    Split.TEST: list(range(13, 26)) + list(range(37, 46)),
                },
                MTBIOddballD009Dataset: {
                    Split.TRAIN: list(range(1, 13)) + list(range(26, 70)),
                    Split.TEST: list(range(13, 26)) + list(range(70, 88)),
                },
                MTBIRestD012Dataset: {
                    Split.TRAIN: list(range(1, 13)) + list(range(26, 70)),
                    Split.TEST: list(range(13, 26)) + list(range(70, 88)),
                },
            }
        )

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
