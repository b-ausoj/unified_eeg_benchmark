from ..abstract_model import AbstractModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from typing import List, Dict
import numpy as np
from resampy import resample
from sklearn.utils import shuffle
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import Pipeline
from pyriemann.classification import FgMDM
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class TSSVMGridModel(AbstractModel):
    def __init__(
        self,
        resample_rate: int = 200,
        channels: List[str] = ["C3", "Cz", "C4"],
    ):
        super().__init__("TS+SVM (Grid)")
        self.pipeline = Pipeline(
            [
                (
                    "Covariances",
                    Covariances(estimator="oas"),
                ),  # Estimate covariance matrices
                (
                    "TangentSpace",
                    TangentSpace(metric="riemann"),
                ),  # Project into Tangent Space
                ("SVC", SVC(kernel="linear")),  # Support Vector Classifier
            ]
        )
        self.param_grid = {"SVC__C": [0.5, 1, 1.5], "SVC__kernel": ["rbf", "linear"]}
        self.grid_search = GridSearchCV(
            self.pipeline, self.param_grid, cv=5, scoring="accuracy", n_jobs=-1
        )

        self.resample_rate = resample_rate
        self.channels = channels

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        [self.validate_meta(m) for m in meta]
        # bring data into the right shape, so resample if needed and only take the C3, Cz, C4 channels
        # TODO handle case if no channel names are provided
        X_prepared = self._prepare_data(X, meta)
        y_prepared = np.concatenate(y, axis=0)

        X_prepared, y_prepared = shuffle(X_prepared, y_prepared, random_state=42)  # type: ignore
        # should be done by the benchmark and not by models

        self.grid_search.fit(X_prepared, y_prepared)

    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        [self.validate_meta(m) for m in meta]

        X_prepared = self._prepare_data(X, meta)

        best_model = self.grid_search.best_estimator_

        return best_model.predict(X_prepared)

    def _prepare_data(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:

        X_resampled = []
        for data, m in zip(X, meta):
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
