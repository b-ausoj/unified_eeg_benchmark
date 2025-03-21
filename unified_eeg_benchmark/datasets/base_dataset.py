from abc import ABC, abstractmethod
import os
from joblib import Memory
from ..enums.classes import Classes
from ..enums.clinical_classes import ClinicalClasses
from ..enums.paradigm import Paradigm
from typing import Dict, Sequence, Tuple, Optional, Set, List
import numpy as np
from mne.io import BaseRaw
from ...utils.config import get_config_value


ClassesType = Classes | ClinicalClasses
DataType = np.ndarray | List[BaseRaw]
LabelsType = np.ndarray | List[str]

class BaseDataset(ABC):
    def __init__(
        self,
        name: str,
        paradigm: Paradigm,
        subjects: Sequence[int],
        available_classes: Set[ClassesType],
        target_classes: Set[ClassesType],
        channel_names: Sequence[str],
        sampling_frequency: int,
        preload: bool = False,
        interval: Optional[Tuple[float, float]] = None,
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = None,
    ):
        self.name = name
        self.subjects = subjects
        self.paradigm = paradigm
        if paradigm == Paradigm.MI:
            assert isinstance(interval, Tuple), "Interval must be a tuple"
            self.interval = interval
        elif paradigm == Paradigm.CLINICAL:
            assert interval is None, "Interval must be None"
        else:
            raise ValueError("Paradigm not recognized")

        self.available_classes = available_classes
        assert target_classes.issubset(available_classes), "Target classes must be a subset of the available classes"
        self.target_classes = target_classes
        self.channel_names = channel_names
        self.sampling_frequency = sampling_frequency
        if target_channels is not None:
            assert all([channel in self.channel_names for channel in target_channels]), "Target channels must be a subset of the available channels"
        self.target_channels = target_channels
        self.target_frequency = target_frequency
        self.preload = preload

        self.data: DataType
        self.labels: LabelsType
        self.meta: Dict = {
            "name": self.name,
        }

        self.cache = Memory(location=get_config_value("cache"), verbose=0)

    @abstractmethod
    def load_data(self) -> None:
        pass

    @abstractmethod
    def _download(self, subject: int) -> None:
        pass


    def get_data(self) -> Tuple[DataType, LabelsType, Dict]:
        """
        Returns the data, labels and meta information of the dataset.
        For BCI it looks mostly like this:
        X is a list of numpy arrays, for each dataset one numpy array.
            Each numpy array has dimensions (n_samples, n_channels, n_timepoints).
        y is alist of numpy arrays, for each dataset one numpy array.
            Each numpy array has dimensions (n_samples, ).
        meta is a list of dictionaries, for each dataset one dictionary.
            Each dictionary contains meta information about the samples.
            Such as the sampling frequency, the channel names, the labels mapping, etc.
        """
        if not self._check_data_loaded():
            self.load_data()
        return self.data, self.labels, self.meta

    def __getitem__(self, index) -> Tuple[DataType, LabelsType]:
        if not self._check_data_loaded():
            self.load_data()
        return self.data[index], self.labels[index]
    
    def __len__(self) -> int:
        if not self._check_data_loaded():
            self.load_data()
        return len(self.data)
    
    def __str__(self) -> str:
        return self.name

    def _check_data_loaded(self) -> bool:
        if not hasattr(self, "data") or self.data is None:
            return False
        if not hasattr(self, "labels") or self.labels is None:
            return False
        return True