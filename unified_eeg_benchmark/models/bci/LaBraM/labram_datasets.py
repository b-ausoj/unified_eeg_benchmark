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
            return X, random.randint(0, 1)

class LaBraMPdDataset(Dataset):
    """
    instance of torch.utils.data.Dataset for LaBraM dataset
    data is already preprocessed and ready to be used
    """

    def __init__(self, files: List[str], sampling_rate: int, ch_names: List[str], train: bool = True):
        seed = 12345
        np.random.seed(seed)

        self.sampling_rate = sampling_rate
        self.ch_names = ch_names
        self.files = files
        if (train):
            np.random.shuffle(self.files)

    def __len__(self):
        return len(self.files)
#    def __len__(self) -> int:
#        return self.data.shape[0]

    def __getitem__(self, index):
        sample = pickle.load(open(self.files[index], "rb"))
        X = sample["X"]
        #print("Maximum value:", np.max(X))
        #print("Minimum value:", np.min(X))
        Y = sample["y"]
        if Y is None:
            Y = 0
        X = FloatTensor(X)
        # print("X shape ", X.shape) X shape  torch.Size([23, 2000])
        return X, Y

class LaBraMAbnormalDataset(Dataset):
    """
    instance of torch.utils.data.Dataset for LaBraM dataset
    data is already preprocessed and ready to be used
    """

    def __init__(self, files: List[str], sampling_rate: int, ch_names: List[str], train: bool = True):
        seed = 12345
        np.random.seed(seed)

        self.sampling_rate = sampling_rate
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        self.ch_names = ch_names
        self.files = files
        if (train):
            np.random.shuffle(self.files)

    def __len__(self):
        return len(self.files)
#    def __len__(self) -> int:
#        return self.data.shape[0]

    def __getitem__(self, index):
        sample = pickle.load(open(self.files[index], "rb"))
        X = sample["X"]
        #print("Maximum value:", np.max(X))
        #print("Minimum value:", np.min(X))
        Y = sample["y"]
        X = FloatTensor(X)
        # print("X shape ", X.shape) X shape  torch.Size([23, 2000])
        if Y is None:
            Y = 0
        return X, Y


#    def __getitem__(self, index):
#        X = FloatTensor(self.data[index])
#        if self.labels is not None:
        #     y = self.labels[index]
        #     return X, y
        # else:
        #     return X