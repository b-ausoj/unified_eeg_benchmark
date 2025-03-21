from ..abstract_model import AbstractModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from typing import List, Dict, cast, Tuple
import numpy as np
from mne.decoding import CSP
from resampy import resample
from multiprocessing import Pool
from tqdm import tqdm
import os
import pandas as pd
from sklearn.utils import shuffle
from .brainfeatures.feature_generation.generate_features import (
    generate_features_of_one_file, default_feature_generation_params)
import logging
from joblib import Memory
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from ...utils.config import get_config_value


def process_one_file_wrapper(args):
    signals, sfreq, domains, feature_generation_params = args
    return process_one_file(signals, sfreq, domains, **feature_generation_params)

def process_one_file(signals, sfreq, domains, epoch_duration_s,
                     max_abs_val, window_name, band_limits, agg_mode,
                     discrete_wavelet, continuous_wavelet, band_overlap):
    
    features_df = generate_features_of_one_file(
        signals, sfreq, epoch_duration_s, max_abs_val, window_name,
        band_limits, agg_mode, discrete_wavelet,
        continuous_wavelet, band_overlap, domains)
    # features has shape (1, n_features)
    if features_df is None:
        logging.error("feature generation failed")
        return
    return features_df.to_numpy()
    

def _prepare_data_cached(X, meta, channels, domains, feature_generation_params, n_jobs):
    all_data = []
    for data, m in zip(X, meta):
        dataset_name = m['name']
        print(f"Processing dataset: {dataset_name}")
        ch_names = m['channel_names']
        ch_names = [ch.upper() for ch in ch_names]
        channel_indices = [ch_names.index(ch.upper()) for ch in channels]
        new_channels = [ch_names[i] for i in channel_indices]
        data = [pd.DataFrame(d[channel_indices, :], index=new_channels) for d in data]
        print("Data length before processing: ", len(data))
        print("Data shape before processing: ", data[0].shape)

        with Pool(n_jobs) as pool:
            sfreq = m['sampling_frequency']
            parameters = [(signals, sfreq, domains, feature_generation_params) for signals in data]

            processed_signals = list(tqdm(pool.imap(process_one_file_wrapper, parameters), total=len(parameters), desc=f"Processing signals of {dataset_name}"))
        processed_signals = np.vstack(processed_signals)
        all_data.append(processed_signals)
        print("Data shape after processing: ", processed_signals.shape)
    all_data = np.vstack(all_data)
    print("All data shape: ", all_data.shape)
    return all_data


class BrainfeaturesModel(AbstractModel):
    def __init__(
        self,
        resample_rate: int = 100,
        channels: List[str] = ['C4', 'FC3', 'P6', 'O1', 'CP4', 'C5', 'PO7', 'TP7', 'F4', 'P3', 'CP6', 'C3', 'FC4', 'F5', 'FC5', 'CP2', 'F2', 'P2', 'P5', 'F8', 'CP1', 'FC1', 'C6', 'F7', 'C2', 'T7', 'FCZ', 'CZ', 'AF3', 'FC6', 'F6', 'TP8', 'CP5', 'P7', 'O2', 'F1', 'FC2', 'FZ', 'F3', 'P8', 'C1', 'P4', 'POZ', 'T8', 'PO8', 'AF4', 'P1', 'OZ', 'CP3'], # len:  49
    ):
        super().__init__('Brainfeatures')
        self.lda = LDA()
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.dt = DecisionTreeClassifier(random_state=42)
        self.svm = SVC(random_state=42)
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

        # Fit LDA on CSP-transformed training data
        self.lda.fit(X_prepared, y_prepared)

    def predict(self, X: List[List[np.ndarray]], meta: List[Dict]) -> np.ndarray:
        [self.validate_meta(m) for m in meta]

        X_prepared = self._prepare_data(X, meta)
        return self.lda.predict(X_prepared)

    def _prepare_data(self, X: List[List[np.ndarray]], meta: List[Dict]) -> np.ndarray:
        domains = ["cwt", "dwt", "dft"] # ["cwt", "dwt", "dft"] or "all"
        default_feature_generation_params["agg_mode"] = "mean"
        n_jobs = 16 # 16
        all_data = self.cache.cache(_prepare_data_cached)(X, meta, self.channels, domains, default_feature_generation_params, n_jobs) # type: ignore
        return all_data