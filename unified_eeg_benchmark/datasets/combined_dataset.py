from typing import List
from .abstract_dataset import AbstractDataset
import numpy as np


class CombinedDataset:
    def __init__(self, datasets: List[AbstractDataset]):
        # check if the datasets are compatible
        # i.e. they have the same sampling frequency, same channels, same interval length, same task, same split
        # if not, raise an error
        # if they are compatible, store the datasets
        if len({dataset.target_frequency for dataset in datasets}) != 1:
            raise ValueError("Sampling frequencies are not the same")
        # if len({dataset.target_channels for dataset in datasets}) != 1:
        #    raise ValueError("Channels are not the same")
        if len({dataset.interval_length for dataset in datasets}) != 1:
            raise ValueError("Interval lengths are not the same")

        self.datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)
        raise IndexError("Index out of bounds")
