from abc import ABC, abstractmethod
from ..datasets.abstract_dataset import AbstractDataset
from ..enums.split import Split
from typing import List


class AbstractTask(ABC):
    def __init__(self, name: str, classes: List[str], datasets: List[AbstractDataset]):
        self._name = name
        self.classes = classes
        self.datasets = datasets

    def get_dataset(
        self, split: Split, channels=None, target_frequency=None, preload=False
    ):
        result = []
        for dataset in self.datasets:
            result.append(
                dataset(
                    task=self,
                    split=split,
                    target_channels=channels,
                    target_frequency=target_frequency,
                    preload=preload,
                )
            )
        return result

    @property
    def name(self):
        return self._name

    def __str__(self):
        return self._name
