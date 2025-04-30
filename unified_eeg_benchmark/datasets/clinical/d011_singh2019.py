from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from ...enums.split import Split
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
from ...utils.config import get_config_value
import os
import random

DATA_PATH = get_config_value("d011")


def _load_data_singh2019(split: Split, subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    ctr_subjects = ['Control1179', 'Control1199', 'Control1159', 'Control1369', 'Control1229', 'Control1249', 'Control1139', 'Control1239', 'Control1149', 'Control1129', 'Control1209', 'Control1359', 'Control1169']
    pd_subjects = ['PD1169', 'PD1329', 'PD1149', 'PD1389', 'PD1469', 'PD1339', 'PD1359', 'PD1219', 'PD1259', 'PD1349', 'PD1279', 'PD1309', 'PD1299', 'PD1539', 'PD1199', 'PD1559', 'PD1129', 'PD1089', 'PD1159', 'PD1319', 'PD1099', 'PD1229', 'PD1249', 'PD1369', 'PD1209', 'PD1239']

    df_vars = pd.read_csv(DATA_PATH + 'ALL_data_Modeling.csv', sep='\t')
    df_vars['id_unique'] = 'PD' + df_vars['Pedal_ID'].astype(str)
    df_vars.loc[df_vars['Group']=='Control', ['id_unique']] = 'Control' + df_vars['Pedal_ID'].astype(str)

    rng = random.Random(42)

    if split == Split.TRAIN:
        subjects = []
    else:
        subjects = ctr_subjects + pd_subjects

    rng.shuffle(subjects)

    print("Loading data from Singh2019")
    data = []
    labels = []
    for subject in tqdm(subjects, desc="Loading data from Singh2019"):
        if subject in ctr_subjects:
            file_path = f"{DATA_PATH}ALL_DATA/RAW_DATA/{subject}.vhdr"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = read_raw_brainvision(file_path, preload=True)
            raw.pick(['eeg'])
            signals = raw.get_data()*1e5
            if np.max(signals) > 1000 or np.min(signals) < -1000:
                    print(f"Large values in subject {subject}: signal range out of bounds (min={np.min(signals)}, max={np.max(signals)})")
            data.append(signals)
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("no_parkinsons")
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['id_unique']==subject, ['Age']].values[0][0])
        else:
            file_path = f"{DATA_PATH}ALL_DATA/RAW_DATA/{subject}.vhdr"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = read_raw_brainvision(file_path, preload=True)
            raw.pick(['eeg'])
            signals = raw.get_data()*1e5
            if np.max(signals) > 1000 or np.min(signals) < -1000:
                    print(f"Large values in subject {subject}: signal range out of bounds (min={np.min(signals)}, max={np.max(signals)})")
            data.append(signals)
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("parkinsons")
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['id_unique']==subject, ['Age']].values[0][0])
    labels = np.array(labels)
    print("Number of subjects: ", len(subjects))
    print("Number of data: ", len(data))
    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class PDGaitD011Dataset(BaseClinicalDataset):
    def __init__(
        self,
        target_class: ClinicalClasses,
        subjects: Sequence[int],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = 250,
        preload: bool = False,
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
            self.load_data(split=Split.TRAIN)

    def _download(self, subject: int):
        pass

    def load_data(self, split) -> None:
        
        self.data, self.labels = self.cache.cache(_load_data_singh2019)(split, self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency

        """
        print("data shape ", self.data[0].shape)
        print(f"Data range: min={np.min(self.data[0])}, max={np.max(self.data[0])}")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        data_to_plot =  self.data #[self.data[i] for i in [1,11,21]] 
        plt.hist([d.flatten() for d in data_to_plot], bins=100, alpha=0.75, label=[f'Subject {i+1}' for i in range(len(data_to_plot))])
        plt.yscale('log')
        plt.xlabel('EEG Signal Value (uV)')
        plt.ylabel('Frequency')
        plt.title('Distribution of EEG Signal Values')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig('eeg_signal_distribution_d011.png')
        plt.close()
        """
    
    def get_data(self, split: Split):
        self.load_data(split)
        return self.data, self.labels, self.meta