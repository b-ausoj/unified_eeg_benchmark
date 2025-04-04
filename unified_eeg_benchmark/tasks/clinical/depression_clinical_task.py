from .abstract_clinical_task import AbstractClinicalTask
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
                    # Split.TRAIN: list(range(1, 2)) + list(range(51, 52)),
                    # Split.TEST: list(range(26, 27)) + list(range(76, 77)),
                    Split.TRAIN: [-1],
                    Split.TEST: [-1],
                },
                DepressionRLD006Dataset: {
                    # Split.TRAIN: list(range(1, 2)) + list(range(51, 52)),
                    # Split.TEST: list(range(26, 27)) + list(range(76, 77)),
                    Split.TRAIN: [-1],
                    Split.TEST: [-1],
                },
            },
        )

    def get_data(
        self, split: Split
    ):
        """Get the data of the TUEG Abnormal dataset for the given split.
        
        The dataset contains EEG recordings of subjects with normal or abnormal EEGs.
        The subjects are randomly mapped to the corresponding files in the dataset. (Improve me!)

        Args:
            split (Split): The split for which to load the data.
    
        Returns:
            Tuple:
                - List[BaseRaw]: A list of `RawEDF` EEG recordings.
                - List[str]: A list of labels (`"abnormal"` or `"normal"`).
                - Dict: Metadata containing a list of montage types for each EEG signal.
        """
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
