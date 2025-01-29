from abc import ABC, abstractmethod
import os
from joblib import Memory
from ..enums.classes import Classes
from typing import Dict, Sequence, Tuple
import numpy as np

base_path = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/"


class AbstractDataset(ABC):
    def __init__(
        self,
        target_classes: Sequence[Classes],
        subjects: Sequence[int],
    ):
        self.data: np.ndarray
        self.labels: np.ndarray
        self.meta: Dict
        self.target_classes = target_classes
        self.subjects = subjects
        self.cache = Memory(location=os.path.join(base_path, "cache"), verbose=0)

    @abstractmethod
    def load_data(self):
        pass

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        if self.data is None or self.labels is None:
            self.load_data()
        if self.meta is None:
            raise ValueError("Meta information not implemented in {self}")
        return self.data, self.labels, self.meta
