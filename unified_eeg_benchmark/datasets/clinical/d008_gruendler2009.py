from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from ...enums.split import Split
from typing import Optional, Sequence, Tuple
from resampy import resample
import logging
from scipy.io import loadmat
import numpy as np
import pandas as pd
from mne.io import read_raw_cnt
import warnings
from tqdm import tqdm


DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/d008_gruendler2009/data/OCI Flankers/"


def _load_data_gruendler2009(split: Split, subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    all_subjects = [i for i in range(901, 961) if i != 902 and i != 910 and i != 913 and i != 918 and i != 923 and i != 928 and i != 942 and i != 943 and i != 944 and i != 947 and i != 949 and i != 951 and i != 954 and i != 955]
    #intersection = ['C4', 'FC3', 'P6', 'O1', 'CP4', 'C5', 'PO7', 'TP7', 'F4', 'P3', 'CP6', 'C3', 'FC4', 'F5', 'FC5', 'CP2', 'F2', 'P2', 'P5', 'F8', 'CP1', 'FC1', 'C6', 'F7', 'C2', 'T7', 'FCZ', 'CZ', 'AF3', 'FC6', 'F6', 'TP8', 'CP5', 'P7', 'O2', 'F1', 'FC2', 'FZ', 'F3', 'P8', 'C1', 'P4', 'POZ', 'T8', 'PO8', 'AF4', 'P1', 'OZ', 'CP3']
    channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    df_vars = pd.read_excel(DATA_PATH + 'Info.xlsx', sheet_name='SELECT', skiprows=[47, 48, 49, 50])
    ocd_list = df_vars.loc[df_vars['OCI'] >= 21.0, ['ID']].values.flatten().astype(int)
    selected_columns = ['ID', 'OCI', 'Sex', 'Age', 'BDI']
    values_df = df_vars[selected_columns]

    train_subjects = ['901', '904', '905', '906', '907', '908', '911', '912', '915', '916', '917', '919', '920', '921', '922', '924', '926', '927', '929', '933', '934', '935', '936', '937', '939', '940', '941', '945', '946', '948', '952', '953', '958', '959', '960']
    test_subjects = ['903', '909', '914', '925', '930', '931', '932', '938', '950', '956', '957']

    if split == Split.TRAIN:
        subjects = train_subjects
    elif split == Split.TEST:
        subjects = test_subjects

    data = []
    labels = []
    for subject in tqdm(subjects, desc="Loading data from Gründler2009"):
        subject_id = int(subject)
        file_path = f"{DATA_PATH}EEG Data/{subject_id}flankers{'' if subject_id < 945 else '_ready'}.cnt"
        targets_df = values_df.loc[values_df['ID'] == subject_id]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = read_raw_cnt(file_path, preload=True)
        cols = [ch['ch_name'] for ch in raw.info['chs'] if ch['ch_name'].upper() in channels]
        raw = raw.reorder_channels(cols)
        raw.set_eeg_reference("average")
        signals = raw.get_data(units='uV')
        if np.max(np.abs(signals)) > 10000000:
            print(f"Subject {subject_id} has signals with max value {np.max(np.abs(signals))} uV")
            signals = signals / 1000000
        if np.max(np.abs(signals)) > 10000:
            print(f"Subject {subject_id} has signals with max value {np.max(np.abs(signals))} uV")
            signals = signals / 1000
        data.append(signals)
        if target_class == ClinicalClasses.OCD:
            labels.append("ocd" if subject_id in ocd_list else "no_ocd")
        elif target_class == ClinicalClasses.OCI:
            labels.append(targets_df['OCI'].values[0])
        elif target_class == ClinicalClasses.BDI:
            labels.append(targets_df['BDI'].values[0])
        elif target_class == ClinicalClasses.AGE:
            labels.append(targets_df['Age'].values[0])
        elif target_class == ClinicalClasses.SEX:
            # TODO have to check whether 0, 1 or 2 is male or female
            labels.append(targets_df['Sex'].values[0])

    labels = np.array(labels)
    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class OCDFlankersD008Dataset(BaseClinicalDataset):
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
            name="OCDFlankers", # d008
            target_class=target_class,
            available_classes=[ClinicalClasses.OCD, ClinicalClasses.OCI, ClinicalClasses.BDI, ClinicalClasses.AGE, ClinicalClasses.SEX],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'],
            #channel_names=['C4', 'FC3', 'P6', 'O1', 'CP4', 'C5', 'PO7', 'TP7', 'F4', 'P3', 'CP6', 'C3', 'FC4', 'F5', 'FC5', 'CP2', 'F2', 'P2', 'P5', 'F8', 'CP1', 'FC1', 'C6', 'F7', 'C2', 'T7', 'FCZ', 'CZ', 'AF3', 'FC6', 'F6', 'TP8', 'CP5', 'P7', 'O2', 'F1', 'FC2', 'FZ', 'F3', 'P8', 'C1', 'P4', 'POZ', 'T8', 'PO8', 'AF4', 'P1', 'OZ', 'CP3'],
            preload=preload,
        )
        # fmt: on
        logging.info("in OCDFlankersD008Dataset.__init__")
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
        
        self.data, self.labels = self.cache.cache(_load_data_gruendler2009)(split, self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency
        """
        for i, d in enumerate(self.data):
            print(f"S {i} Value range: {np.min(d)} to {np.max(d)} with std {np.std(d)}")
                
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
        plt.savefig('eeg_signal_distribution_d008.png')
        plt.close()
        """
        
    def get_data(self, split: Split):
        self.load_data(split)
        return self.data, self.labels, self.meta