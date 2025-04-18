from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import (
    MTBIOddballD009Dataset,
    MTBIRestD012Dataset,
)
from sklearn.metrics import f1_score
from ...enums.split import Split


class MTBIClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="mtbi_clinical",
            clinical_class = ClinicalClasses.MTBI,
            datasets = [
                MTBIOddballD009Dataset, 
                #MTBIRestD012Dataset,
            ],
            subjects_split={
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

    def get_data(
        self, split: Split
    ):
        data = [
            dataset(
                target_class=self.clinical_class,
                subjects=self.subjects_split[dataset][split],
            ).get_data(split)
            for dataset in self.datasets
        ]

        X, y, meta = map(list, zip(*data))
        for m in meta:
            m["task_name"] = self.name
        return X, y, meta

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
