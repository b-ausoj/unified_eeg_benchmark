from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import (
    ParkinsonsOddballD001Dataset,
    ParkinsonsRestD002Dataset,
    ParkinsonsConflictTaskD005Dataset,
    ParkinsonsRLTaskD007Dataset,
    PDGaitD011Dataset,
    PDLPCRestD013Dataset,
    PDIntervalTimingD014Dataset,
)
from sklearn.metrics import f1_score
from ...enums.split import Split


class ParkinsonsClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="parkinsons_clinical",
            clinical_class = ClinicalClasses.PARKINSONS,
            datasets = [
                ParkinsonsOddballD001Dataset, # good results
                ParkinsonsRestD002Dataset, # okay results
                ParkinsonsConflictTaskD005Dataset, # bad results
                ParkinsonsRLTaskD007Dataset, # good results
                PDGaitD011Dataset, # good results
                PDLPCRestD013Dataset, # bad results
                PDIntervalTimingD014Dataset, # bad results
            ],
            subjects_split={
                ParkinsonsOddballD001Dataset: {
                    Split.TRAIN: list(range(1, 13)) + list(range(26, 40)),
                    Split.TEST: list(range(13, 26)) + list(range(40, 53)),
                },
                ParkinsonsRestD002Dataset: {
                    Split.TRAIN: list(range(1, 15)) + list(range(29, 43)),
                    Split.TEST: list(range(15, 29)) + list(range(43, 57)),
                },
                ParkinsonsConflictTaskD005Dataset: {
                    Split.TRAIN: list(range(1, 15)) + list(range(29, 43)),
                    Split.TEST: list(range(15, 29)) + list(range(43, 57)),
                },
                ParkinsonsRLTaskD007Dataset: {
                    Split.TRAIN: list(range(1, 15)) + list(range(29, 43)),
                    Split.TEST: list(range(15, 29)) + list(range(43, 57)),
                },
                PDGaitD011Dataset: {
                    Split.TRAIN: list(range(1, 11)) + list(range(14, 35)),
                    Split.TEST: list(range(11, 14)) + list(range(35, 40)),
                },
                PDLPCRestD013Dataset: {
                    Split.TRAIN: list(range(1, 12)) + list(range(15, 26)),
                    Split.TEST: list(range(12, 15)) + list(range(26, 29)),
                },
                PDIntervalTimingD014Dataset: {
                    #Split.TRAIN: list(range(1, 31)) + list(range(38, 101)),
                    #Split.TEST: list(range(31, 38)) + list(range(101, 112)),
                    Split.TRAIN: list(range(1, 11)) + list(range(38, 48)),
                    Split.TEST: list(range(31, 38)) + list(range(101, 108)),
                },
            }
        )

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
