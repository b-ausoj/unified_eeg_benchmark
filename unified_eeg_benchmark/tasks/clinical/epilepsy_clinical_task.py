from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import TUEGEpilepsyDataset
from sklearn.metrics import f1_score
from ...enums.split import Split
from typing import List, Tuple, Dict
from mne.io import Raw


class EpilepsyClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="epilepsy_clinical",
            clinical_class = ClinicalClasses.EPILEPSY,
            datasets = [
                TUEGEpilepsyDataset,
            ],
            subjects_split={
                TUEGEpilepsyDataset: {
                    #Split.TRAIN: list(range(1, 101)),
                    #Split.TEST: list(range(101, 201)),
                    Split.TRAIN: list(range(1, 11)),
                    Split.TEST: list(range(11, 21)),
                },
            },
        )

    def get_data(
        self, split: Split
    ) -> Tuple[List[List[List[Raw]]], List[List[str]], List[Dict]]:
        """Get the data of the TUEG Epilepsy dataset for the given split.

        The dataset contains EEG recordings of subjects with and without epilepsy.
        Subjects are mapped to the corresponding files in the dataset using the 
        following rule: even-numbered subjects have epilepsy, while odd-numbered 
        subjects do not.

        
        Args:
            split: The split for which to get the data.
    
        Returns:
            Tuple with the follwoing elements:
                - A list of nested lists where each outer list corresponds to a subject, 
                and each inner list contains `Raw` EEG recordings for that subject.
                - A list of lists containing labels (`"epilepsy"` or `"no_epilepsy"`) for each subject.
                - A list of Metadata containing montage types for each EEG signal.
        """
        return super().get_data(split) # type: ignore

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
