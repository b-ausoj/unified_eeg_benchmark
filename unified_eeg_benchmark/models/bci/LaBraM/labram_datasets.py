from torch.utils.data import Dataset
from torch import FloatTensor
import numpy as np
from typing import List
import os
import pickle
import random


class LaBraMBCIDataset(Dataset):
    """
    instance of torch.utils.data.Dataset for LaBraM dataset
    data is already preprocessed and ready to be used
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray|None, sampling_rate: int, ch_names: List[str]):
        self.data = data
        print("data shape2 ", self.data.shape)
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.ch_names = ch_names

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index):
        X = FloatTensor(self.data[index]) 
        #print("X shape ", X.shape) X shape  torch.Size([22, 200])
        #print("Maximum value:", np.max(self.data[index]))
        #print("Minimum value:", np.min(self.data[index]))
        if self.labels is not None:
            y = self.labels[index]
            return X, y
        else:
            return X
