from ..abstract_model import AbstractModel
from typing import List, Dict, cast, Tuple
from sklearn.svm import SVC
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import os
import pandas as pd
from sklearn.utils import shuffle
import logging
from joblib import Memory
from ...utils.config import get_config_value
from .brainfeatures.feature_extraction import _prepare_data_cached
from .brainfeatures.feature_generation.generate_features import default_feature_generation_params


class BrainfeaturesSVMModel(AbstractModel):
    def __init__(
        self,
        resample_rate: int = 100,
        channels: List[str] = ['C4', 'FC3', 'P6', 'O1', 'CP4', 'C5', 'PO7', 'TP7', 'F4', 'P3', 'CP6', 'C3', 'FC4', 'F5', 'FC5', 'CP2', 'F2', 'P2', 'P5', 'F8', 'CP1', 'FC1', 'C6', 'F7', 'C2', 'T7', 'FCZ', 'CZ', 'AF3', 'FC6', 'F6', 'TP8', 'CP5', 'P7', 'O2', 'F1', 'FC2', 'FZ', 'F3', 'P8', 'C1', 'P4', 'POZ', 'T8', 'PO8', 'AF4', 'P1', 'OZ', 'CP3'], # len:  49
    ):
        super().__init__('Brainfeatures-SVM')
        self.svm = SVC(kernel='linear', C=1)
        self.resample_rate = resample_rate
        self.channels = channels
        self.cache = Memory(location=get_config_value("cache"), verbose=0)

    def fit(self, X: List[List[np.ndarray]], y: List[np.ndarray], meta: List[Dict]) -> None:
        [self.validate_meta(m) for m in meta]
        # bring data into the right shape, so resample if needed and only take the C3, Cz, C4 channels
        # TODO handle case if no channel names are provided
        X_prepared = self._prepare_data(X, meta)
        del X, meta
        print(X_prepared.shape)
        y_prepared = np.concatenate(y, axis=0)
        print(y_prepared.shape)

        X_prepared, y_prepared = shuffle(X_prepared, y_prepared, random_state=42)
        # should be done by the benchmark and not by models

        # Fit SVM on extracted featues
        self.svm.fit(X_prepared, y_prepared)

    def predict(self, X: List[List[np.ndarray]], meta: List[Dict]) -> np.ndarray:
        [self.validate_meta(m) for m in meta]

        X_prepared = self._prepare_data(X, meta)
        return self.svm.predict(X_prepared)

    def _prepare_data(self, X: List[List[np.ndarray]], meta: List[Dict]) -> np.ndarray:
        domains = ["cwt", "dwt", "dft"] # ["cwt", "dwt", "dft"] or "all"
        default_feature_generation_params["agg_mode"] = "mean"
        n_jobs = 16 # 16
        all_data = self.cache.cache(_prepare_data_cached)(X, meta, self.channels, domains, default_feature_generation_params, n_jobs) # type: ignore
        return all_data