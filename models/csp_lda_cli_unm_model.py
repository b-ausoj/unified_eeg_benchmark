from .abstract_model import AbstractModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from typing import List, Dict
import numpy as np
from mne.decoding import CSP
from resampy import resample
from sklearn.utils import shuffle


class CSPLDACliUnmModel(AbstractModel):
    def __init__(
        self,
        n_components: int = 4,
        reg=None,
        log=True,
        resample_rate: int = 100,
        channels: List[str] = ['C4', 'FC3', 'P6', 'O1', 'CP4', 'C5', 'PO7', 'TP7', 'F4', 'P3', 'CP6', 'C3', 'FC4', 'F5', 'FC5', 'CP2', 'F2', 'P2', 'P5', 'F8', 'CP1', 'FC1', 'C6', 'F7', 'C2', 'T7', 'FCZ', 'CZ', 'AF3', 'FC6', 'F6', 'TP8', 'CP5', 'P7', 'O2', 'F1', 'FC2', 'FZ', 'F3', 'P8', 'C1', 'P4', 'POZ', 'T8', 'PO8', 'AF4', 'P1', 'OZ', 'CP3'],
    ):
        super().__init__('CSP-LDA')
        self.lda = LDA()
        self.csp = CSP(n_components=n_components, reg=reg, log=log)
        self.resample_rate = resample_rate
        self.channels = channels

    def fit(self, X: List[List[np.ndarray]], y: List[np.ndarray], meta: List[Dict]) -> None:
        [self.validate_meta(m) for m in meta]
        # bring data into the right shape, so resample if needed and only take the C3, Cz, C4 channels
        # TODO handle case if no channel names are provided
        X_prepared = self._prepare_data(X, meta)
        print(X_prepared.shape)
        y_prepared = np.concatenate(y, axis=0)
        print(y_prepared.shape)

        X_prepared, y_prepared = shuffle(X_prepared, y_prepared, random_state=42)  # type: ignore
        # should be done by the benchmark and not by models

        # Transform both training and test data using the learned CSP filters
        X_csp = self.csp.fit_transform(X_prepared, y_prepared)

        # Fit LDA on CSP-transformed training data
        self.lda.fit(X_csp, y_prepared)

    def predict(self, X: List[List[np.ndarray]], meta: List[Dict]) -> np.ndarray:
        [self.validate_meta(m) for m in meta]

        X_prepared = self._prepare_data(X, meta)
        X_csp = self.csp.transform(X_prepared)
        return self.lda.predict(X_csp)

    def _prepare_data(self, X: List[List[np.ndarray]], meta: List[Dict]) -> np.ndarray:
        
        # check if all have the same duration i.e. n_timepoints
        durations = []
        for dataset, m in zip(X, meta):
            durations.append(min([data.shape[1] for data in dataset]) / m['sampling_frequency'])
        durations.append(20 * 60)  # max 20 minutes
        X_resampled = []
        if not len(durations) == 1:
            min_duration = min(durations)
            for dataset, m in zip(X, meta):
                ch_names = m['channel_names'] # to upper case
                ch_names = [ch.upper() for ch in ch_names]
                channel_indices = [ch_names.index(ch) for ch in self.channels]
                X_cropped = [data[channel_indices, :int(min_duration * m['sampling_frequency'])] for data in dataset]
                X_cropped = np.array(X_cropped)
                if m['sampling_frequency'] != self.resample_rate:
                    X_cropped = resample(
                        X_cropped,
                        m['sampling_frequency'],
                        self.resample_rate,
                        axis=2,
                        filter='kaiser_fast',
                    )
                X_resampled.append(X_cropped)
        
        X_resampled = np.concatenate(X_resampled, axis=0)
        X_resampled = X_resampled.astype(np.float64)

        return X_resampled
