from ..abstract_model import AbstractModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from typing import List, Dict, cast
import numpy as np
from resampy import resample
from sklearn.utils import shuffle
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import Pipeline
from pyriemann.classification import FgMDM


class FgMDMModel(AbstractModel):
    def __init__(
        self,
        resample_rate: int = 200,
        channels: List[str] = ["C3", "Cz", "C4"]#['AFZ', 'F3', 'F1', 'FZ', 'F2', 'F4', 'FFC5h', 'FFC3h', 'FFC1h', 'FFC2h', 'FFC4h', 'FFC6h', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FCC5h', 'FCC3h', 'FCC1h', 'FCC2h', 'FCC4h', 'FCC6h', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CCP5h', 'CCP3h', 'CCP1h', 'CCP2h', 'CCP4h', 'CCP6h', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'CPP5h', 'CPP3h', 'CPP1h', 'CPP2h', 'CPP4h', 'CPP6h', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'PPO1h', 'PPO2h',  'POz'], #["C3", "Cz", "C4"], # 'FC3', "FCz", 'FC4', 
    ):
        super().__init__("FgMDM")
        self.pipeline = Pipeline(
            [
                (
                    "Covariances",
                    Covariances(estimator="oas"),
                ),  # Estimate covariance matrices
                (
                    "FgMDM",
                    FgMDM(metric="riemann"),
                ),  # Use FgMDM classifier with Riemannian metric
            ]
        )
        self.resample_rate = resample_rate
        self.channels = channels

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        [self.validate_meta(m) for m in meta]
        self._set_channels(meta[0]["task_name"])

        # bring data into the right shape, so resample if needed and only take the C3, Cz, C4 channels
        # TODO handle case if no channel names are provided
        X_prepared = self._prepare_data(X, meta)
        y_prepared = np.concatenate(y, axis=0)

        X_prepared, y_prepared = cast(tuple[np.ndarray, np.ndarray], shuffle(X_prepared, y_prepared, random_state=42))
        # should be done by the benchmark and not by models

        self.pipeline.fit(X_prepared, y_prepared)

    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        [self.validate_meta(m) for m in meta]

        X_prepared = self._prepare_data(X, meta)

        return self.pipeline.predict(X_prepared) # type: ignore

    def _prepare_data(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:

        X_resampled = []
        for data, m in zip(X, meta):
            if data.size == 0:
                continue
            # resample if needed
            # only take the C3, Cz, C4 channels
            channel_indices = [m["channel_names"].index(ch) for ch in self.channels]
            data = data[:, channel_indices, :]
            if m["sampling_frequency"] != self.resample_rate:
                data = resample(
                    data,
                    m["sampling_frequency"],
                    self.resample_rate,
                    axis=2,
                    filter="kaiser_best",
                )
            X_resampled.append(data)

        # check if all have the same duration i.e. n_timepoints
        durations = set([d.shape[2] for d in X_resampled])
        if not len(durations) == 1:
            min_duration = min(durations)
            X_resampled = [d[:, :, :min_duration] for d in X_resampled]

        X_resampled = np.concatenate(X_resampled, axis=0)
        X_resampled = X_resampled.astype(np.float64)

        return X_resampled
