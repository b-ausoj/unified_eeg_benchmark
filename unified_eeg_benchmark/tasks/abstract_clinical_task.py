from abc import ABC, abstractmethod
from ..datasets.abstract_dataset import AbstractDataset
from ..enums.split import Split
from ..enums.clinical_classes import ClinicalClasses
from typing import List, Tuple, Dict, Type, Sequence
import numpy as np


class AbstractClinicalTask(ABC):
    def __init__(
        self,
        name: str,
        clinical_class: ClinicalClasses,
        datasets: Sequence[Type[AbstractDataset]],
        subjects_split: Dict[Type[AbstractDataset], Dict[Split, Sequence[int]]],
    ):
        assert len(datasets) > 0, "At least one dataset is required"
        assert set(datasets).issubset(subjects_split.keys()), "Subjects split must match datasets"
        assert all(
            subjects_split[dataset].keys() == {Split.TRAIN, Split.TEST}
            for dataset in datasets
        ), "Subjects split must contain train and test splits"
        self.name = name
        self.clinical_class = clinical_class
        self.datasets = datasets
        self.subjects_split = subjects_split

    def get_data(
        self, split: Split
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Get the data for the given split.

        Parameters
        ----------
        split : Split
            The split for which to get the data.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]
            The data, labels and meta information for the given split.

            X is a list of numpy arrays, for each dataset one numpy array.
                Each numpy array has dimensions (n_samples, n_channels, n_timepoints).
            y is alist of numpy arrays, for each dataset one numpy array.
                Each numpy array has dimensions (n_samples, ).
            meta is a list of dictionaries, for each dataset one dictionary.
                Each dictionary contains meta information about the samples.
                Such as the sampling frequency, the channel names, the labels mapping, etc.
        """

        data = [
            dataset(
                target_class=self.clinical_class,
                subjects=self.subjects_split[dataset][split],
            ).get_data()
            for dataset in self.datasets
        ]

        X, y, meta = map(list, zip(*data))
        return X, y, meta

    def __str__(self):
        return self.name

    @abstractmethod
    def get_scoring(self):
        """
        Retrieve the scoring function associated with this task.

        Returns
        -------
        function : callable
            A function that accepts two parameters, y_true and y_pred, and returns a float
            representing the score. This function is used to evaluate the performance of
            predictions against the true labels.
        """
        pass
