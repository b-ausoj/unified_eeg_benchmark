from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence, Tuple
from resampy import resample
import logging
from scipy.io import loadmat
import numpy as np
import pandas as pd
import glob
from mne.io import read_raw_brainvision
import warnings
from tqdm import tqdm


DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/d011_singh2019/data/PD Gait/"


def _load_data_singh2019(subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    ctr_subjects = [1129, 1139, 1149, 1159, 1169, 1179, 1199, 1209, 1229, 1239, 1249, 1359, 1369]
    pd_subjects = [1099, 1159, 1169, 1199, 1209, 1219, 1239, 1299, 1329, 1339, 1349, 1369, 1389, 1089, 1129, 1149, 1229, 1249, 1259, 1279, 1309, 1319, 1359, 1469, 1539, 1559]
    df_vars = pd.read_csv(DATA_PATH + 'ALL_data_Modeling.csv', sep='\t')
    df_vars['id_unique'] = 'PD' + df_vars['Pedal_ID'].astype(str)
    df_vars.loc[df_vars['Group']=='Control', ['id_unique']] = 'Control' + df_vars['Pedal_ID'].astype(str)

    data = []
    labels = []
    for subject in tqdm(subjects, desc="Loading data from Singh2019"):
        if subject < len(ctr_subjects) + 1:
            subject_id = ctr_subjects[subject - 1]
            file_path = f"{DATA_PATH}ALL_DATA/RAW_DATA/Control{subject_id}.vhdr"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = read_raw_brainvision(file_path, preload=True)
            raw.pick(['eeg'])
            data.append(raw.get_data())
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("no_parkinsons")
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['id_unique']==subject_id, ['Age']].values[0][0])
        else:
            subject_id = pd_subjects[subject - len(ctr_subjects) - 1]
            file_path = f"{DATA_PATH}ALL_DATA/RAW_DATA/PD{subject_id}.vhdr"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = read_raw_brainvision(file_path, preload=True)
            raw.pick(['eeg'])
            data.append(raw.get_data())
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("parkinsons")
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['id_unique']==subject_id, ['Age']].values[0][0])
    labels = np.array(labels)
    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class PDGaitD011Dataset(BaseClinicalDataset):
    def __init__(
        self,
        target_class: ClinicalClasses,
        subjects: Sequence[int],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = 200,
        preload: bool = True,
    ):
        # fmt: off
        super().__init__(
            name="D011Singh2019Dataset", # d011
            target_class=target_class,
            available_classes=[ClinicalClasses.PARKINSONS, ClinicalClasses.AGE],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['Fp1', 'Fz', 'F3', 'F7', 'Aa', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'Bb', 'POz', 'Cc', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'FCz'],
            preload=preload,
        )
        # fmt: on
        logging.info("in D011SinghDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "name": self.name,
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        
        self.data, self.labels = self.cache.cache(_load_data_singh2019)(self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency
