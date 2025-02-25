from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence
import logging
from scipy.io import loadmat
import numpy as np
import pandas as pd
from mne.io import read_raw_cnt
import warnings

DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/d008_gruendler2009/data/OCI Flankers/"


def _load_data_gruendler2009(subjects: Sequence[int], target_class: ClinicalClasses):
    all_subjects = [i for i in range(901, 961) if i != 902 and i != 910 and i != 913 and i != 918 and i != 920 and i != 923 and i != 928 and i != 942 and i != 943 and i != 944 and i != 947 and i != 949 and i != 951 and i != 954 and i != 955]
    df_vars = pd.read_excel(DATA_PATH + 'Info.xlsx', sheet_name='SELECT', skiprows=[47, 48, 49, 50])
    ocd_list = df_vars.loc[df_vars['OCI'] >= 21.0, ['ID']].values.flatten().astype(int)
    selected_columns = ['ID', 'OCI', 'Sex', 'Age', 'BDI']
    values_df = df_vars[selected_columns]

    data = []
    labels = []
    for subject in subjects:
        subject_id = all_subjects[subject - 1]
        file_path = f"{DATA_PATH}EEG Data/{subject_id}flankers{'' if subject_id < 945 else '_ready'}.cnt"
        targets_df = values_df.loc[values_df['ID'] == subject_id]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = read_raw_cnt(file_path, preload=False, data_format='int16')
        if raw.times[-1] > 1500:
            print(f"Subject {subject_id} has {raw.times[-1]} seconds of data. Cropping to 1500 seconds.")
            raw.crop(tmax=1500)
        raw.pick(['eeg'])
        raw.load_data()
        data.append(raw.get_data()[:64, :] * 1e6)
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
    return data, labels


class OCDFlankersD008Dataset(BaseClinicalDataset):
    def __init__(
        self,
        target_class: ClinicalClasses,
        subjects: Sequence[int],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = None,
        preload: bool = True,
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
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        
        self.data, self.labels = self.cache.cache(_load_data_gruendler2009)(self.subjects, self.target_classes[0]) # type: ignore
        print("data shape: ", self.data[0].shape)