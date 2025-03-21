from .abstract_bci_task import AbstractBCITask
from ...enums.classes import Classes
from sklearn.metrics import f1_score
from ...enums.split import Split


class CustomMITask(AbstractBCITask):
    def __init__(self, classes, subjects_split):
        super().__init__(
            name="custom_mi_task",
            classes=classes,
            datasets=subjects_split.keys(),
            subjects_split=subjects_split,
        )
    
    def get_data(
        self, split: Split
    ):
        data = []
        for dataset in self.datasets:
                data.append(
                    dataset(
                        target_classes=self.classes,
                        subjects=self.subjects_split[dataset][split],
                    ).get_data()
                )

        X, y, meta = map(list, zip(*data))
        return X, y, meta

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
