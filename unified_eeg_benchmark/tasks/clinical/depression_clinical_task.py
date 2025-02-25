from abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import (
    DepressionRestD003Dataset,
    DepressionRLD006Dataset,
)
from sklearn.metrics import f1_score
from ...enums.split import Split


class DepressionClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="depression_clinical",
            clinical_class = ClinicalClasses.DEPRESSION,
            datasets = [
                DepressionRestD003Dataset,
                DepressionRLD006Dataset,
            ],
            subjects_split={
                DepressionRestD003Dataset: {
                    Split.TRAIN: list(range(1, 26)) + list(range(51, 76)),
                    Split.TEST: list(range(26, 51)) + list(range(76, 101)),
                },
                DepressionRLD006Dataset: {
                    Split.TRAIN: list(range(1, 26)) + list(range(51, 76)),
                    Split.TEST: list(range(26, 51)) + list(range(76, 101)),
                },
            },
        )

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
