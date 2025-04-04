from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import (
    ParkinsonsRestD002Dataset,
    DepressionRestD003Dataset,
    SchizophreniaConflictD004Dataset,
    ParkinsonsConflictTaskD005Dataset,
    DepressionRLD006Dataset,
    OCDFlankersD008Dataset,
    MTBIOddballD009Dataset,
    PDGaitD011Dataset,
    MTBIRestD012Dataset,
    PDLPCRestD013Dataset,
    PDIntervalTimingD014Dataset,
)
from sklearn.metrics import f1_score
from ...enums.split import Split


class AgeClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="Age Clinical Task",
            clinical_class = ClinicalClasses.AGE,
            datasets = [
                ParkinsonsRestD002Dataset,
                DepressionRestD003Dataset,
                SchizophreniaConflictD004Dataset,
                ParkinsonsConflictTaskD005Dataset,
                DepressionRLD006Dataset,
                OCDFlankersD008Dataset,
                MTBIOddballD009Dataset,
                PDGaitD011Dataset,
                MTBIRestD012Dataset,
                PDLPCRestD013Dataset,
                PDIntervalTimingD014Dataset,
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
                OCDFlankersD008Dataset: {
                    Split.TRAIN: list(range(1, 13)) + list(range(26, 37)),
                    Split.TEST: list(range(13, 26)) + list(range(37, 46)),
                },
                MTBIOddballD009Dataset: {
                    Split.TRAIN: list(range(1, 13)) + list(range(26, 70)),
                    Split.TEST: list(range(13, 26)) + list(range(70, 88)),
                },
                PDGaitD011Dataset: {
                    Split.TRAIN: list(range(1, 11)) + list(range(14, 35)),
                    Split.TEST: list(range(11, 14)) + list(range(35, 40)),
                },
                MTBIRestD012Dataset: {
                    Split.TRAIN: list(range(1, 13)) + list(range(26, 70)),
                    Split.TEST: list(range(13, 26)) + list(range(70, 88)),
                },
                PDLPCRestD013Dataset: {
                    Split.TRAIN: list(range(1, 12)) + list(range(15, 26)),
                    Split.TEST: list(range(12, 15)) + list(range(26, 29)),
                },
                PDIntervalTimingD014Dataset: {
                    Split.TRAIN: list(range(1, 11)) + list(range(38, 48)),
                    Split.TEST: list(range(31, 38)) + list(range(101, 108)),
                },
            }
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
