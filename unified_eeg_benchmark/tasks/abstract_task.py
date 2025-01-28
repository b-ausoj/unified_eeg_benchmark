from abc import ABC, abstractmethod
from ..datasets.abstract_dataset import AbstractDataset
from ..enums.split import Split
from ..enums.classes import Classes
from typing import List, Tuple, Dict, Type
import numpy as np


class AbstractTask(ABC):
    def __init__(
        self, name: str, classes: List[Classes], datasets: List[Type[AbstractDataset]]
    ):
        self.name = name
        self.classes = classes
        self.datasets = datasets

    def get_data(
        self, split: Split, channels=None, target_frequency=None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        # todo: noch schöner machen
        Xs = []
        ys = []
        metas = []
        for dataset in self.datasets:
            X, y, meta = dataset(
                classes=self.classes,
                split=split,
                target_channels=channels,
                target_frequency=target_frequency,
                preload=True,
            ).get_data()
            Xs.append(X)
            ys.append(y)
            metas.append(meta)
        return Xs, ys, metas

    def __str__(self):
        return self.name

    @abstractmethod
    def get_scoring(self):
        pass
