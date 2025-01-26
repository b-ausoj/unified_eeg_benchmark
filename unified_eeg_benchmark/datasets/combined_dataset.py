from typing import List
from .abstract_dataset import AbstractDataset
import numpy as np


class CombinedDataset:
    def __init__(self, datasets: List[AbstractDataset]):
        # check if the datasets are compatible
        # i.e. they have the same sampling frequency, same channels, same interval length, same task, same split
        self.datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)
        raise IndexError("Index out of bounds")
