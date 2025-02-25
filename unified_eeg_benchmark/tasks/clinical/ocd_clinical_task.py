from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import OCDFlankersD008Dataset
from sklearn.metrics import f1_score
from ...enums.split import Split


class OCDClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="ocd_clinical",
            clinical_class = ClinicalClasses.OCD,
            datasets = [
                OCDFlankersD008Dataset, 
            ],
            subjects_split={
                OCDFlankersD008Dataset: {
                    Split.TRAIN: list(range(1, 13)) + list(range(26, 37)),
                    Split.TEST: list(range(13, 26)) + list(range(37, 46)),
                },
            }
        )

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
