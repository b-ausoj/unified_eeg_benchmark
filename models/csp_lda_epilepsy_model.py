from .abstract_model import AbstractModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from typing import List, Dict, Sequence, Optional, Tuple
import numpy as np
from mne.decoding import CSP
from sklearn.utils import shuffle
from mne.io import Raw
import logging
from sklearn.preprocessing import LabelEncoder


class CSPLDAEpilepsyModel(AbstractModel):
    def __init__(
        self,
        n_components: int = 4,
        reg=None,
        log=True,
        resample_rate: int = 200,
        channels: Sequence[str] = ["FP1", "FP2", "F3", "F4", "FZ", "C3", "C4", "CZ", "P3", "P4", "PZ", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"], # "FPZ", "OZ", "A1", "A2", 
    ):
        super().__init__('CSP-LDA')
        self.lda = LDA()
        self.csp = CSP(n_components=n_components, reg=reg, log=log)
        self.resample_rate = resample_rate
        self.channels = channels

    def fit(self, X: List[List[List[Raw]]], y: List[List[str]], meta: List[Dict]) -> None:
        
        X_, y_, meta_ = X[0], y[0], meta[0] # In epilepsy task, we only have one dataset
        
        X_prepared, y_prepared = self._prepare_data(X_, y_, meta_)
        print(X_prepared.shape)
        print(y_prepared.shape)

        X_prepared, y_prepared = shuffle(X_prepared, y_prepared, random_state=42)  # type: ignore
        # should be done by the benchmark and not by models

        # Transform both training and test data using the learned CSP filters
        X_csp = self.csp.fit_transform(X_prepared, y_prepared)

        # Fit LDA on CSP-transformed training data
        self.lda.fit(X_csp, y_prepared)

    def predict(self, X: List[List[List[Raw]]], meta: List[Dict]) -> np.ndarray:

        X_, meta_ = X[0], meta[0] # In epilepsy task, we only have one dataset

        X_prepared = self._prepare_data(X_, None, meta_)
        X_csp = self.csp.transform(X_prepared)
        y_pred = self.lda.predict(X_csp)
        return y_pred

        """ only take the first recording for now
        encoder = LabelEncoder()
        y_pred_encoded = encoder.fit_transform(y_pred)
        print(encoder.classes_)
        print(f"y_pred_encoded {y_pred_encoded}")

        # have to modify the output to match the expected format (one guess per subject)
        # for each subject s there are len(X[s]) recordings
        # we predict one class for each recording
        # we have to aggregate the predictions to get one prediction per subject
        y_pred_prepared = np.array([np.bincount(y_pred_encoded[i * len(X_[i]):(i + 1) * len(X_[i])], minlength=2).argmax() for i in range(len(X_))])
        print(f"y_pred {y_pred_prepared}")
        y_pred_decoded = encoder.inverse_transform(y_pred_prepared)
        return y_pred_decoded
        """

    def _prepare_data(self, X: List[List[Raw]], y: Optional[List[str]], meta: Dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        
        X_prepared = []
        y_prepared = [] if y is not None else None

        for i, subject in enumerate(X):
            X_subject_prepared = []
            y_subject_prepared = [] if y is not None else None
            for raw in subject:
                
                # drop if too short or too long
                if raw.times[-1] < 610:
                    # logging.warning(f"Skipping {raw.filenames[0]} because it is {'too short' if raw.times[-1] < 130 else 'too long'} with {raw.times[-1]} seconds")
                    raw.close()
                    continue

                # load data
                raw.load_data(verbose="error")

                # crop to 10 minutes and remove 10 seconds from the beginning
                raw.crop(tmin=10, tmax=610, include_tmax=False)

                # standardize channel names
                new_ch_names = {ch: ch.replace('-REF', '').replace('-LE', '').replace('EEG ', '').upper() for ch in raw.ch_names}
                raw.rename_channels(new_ch_names)

                # apply common average reference
                raw.set_eeg_reference('average', verbose='error')

                # select and reorder channels
                missing_channels = [ch for ch in self.channels if ch not in raw.ch_names]
                if missing_channels:
                    raise ValueError(f"Missing channels in {raw.filenames[0]}: {missing_channels}")
                raw.pick(self.channels)
                raw.reorder_channels(self.channels)

                # apply bandpass filter
                raw.filter(0.5, 40, verbose='error')

                # resample
                raw.resample(self.resample_rate)
                
                # append to list
                X_subject_prepared.append(raw.get_data())
                if y_subject_prepared is not None and y is not None:
                    y_subject_prepared.append(y[i])
                raw.close()

                # only take the first recording for now
                break

            X_prepared.append(np.array(X_subject_prepared))
            if y_prepared is not None:
                y_prepared.append(np.array(y_subject_prepared))
                #print(f"Shape of y prepared{np.array(y_subject_prepared).shape}")
            #print(f"Shape of subject prepared{np.array(X_subject_prepared).shape}")
        
        X_prepared = np.concatenate(X_prepared, axis=0)
        if y_prepared is not None:
            y_prepared = np.concatenate(y_prepared, axis=0)
        
        return X_prepared if y is None else (X_prepared, y_prepared)
