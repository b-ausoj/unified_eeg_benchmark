from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence, Tuple
import logging
from scipy.io import loadmat
import numpy as np
import pandas as pd
from tqdm import tqdm
from resampy import resample


DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/d004_albrecht2017/data/Cost Conflict in Schizophrenia/"


def _load_data_albrecht2017(subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    all_subjects = [i for i in range(101, 181) if i != 106 and i != 118 and i != 148 and i != 175]
    df_vars = pd.read_csv(DATA_PATH + 'DeID_Dems.csv')
    sz_ids = df_vars.loc[df_vars['group'] == 'SZ', ['subno']].values

    # fmt: off
    channels = np.array(['Fp1', 'Fz', 'F3', 'F7', 'AFp9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'AFp10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'VEOGL', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'VEOGU', 'F6', 'F2', 'AF4', 'AF8'])
    non_eeg_channels = {'VEOGL', 'VEOGU'}
    eeg_indices = [i for i, ch in enumerate(channels) if ch not in non_eeg_channels]
    # fmt: on

    data = []
    labels = []
    for subject in tqdm(subjects, desc="Loading data from Albrecht2017"):
        subject_id = all_subjects[subject - 1]
        file_path = f"{DATA_PATH}CCunproc/CC_EEG_s{subject_id}_{'P' if subject_id < 149 else 'N'}.mat"
        mat = loadmat(file_path, simplify_cells=True)
        data.append(mat["EEG"]["data"][eeg_indices, :])
        if target_class == ClinicalClasses.SCHIZOPHRENIA:
            labels.append("schizophrenia" if subject_id in sz_ids else "no_schizophrenia")
        elif target_class == ClinicalClasses.MEDICATION:
            assert subject_id in sz_ids, f"Subject {subject_id} is not in the schizophrenia group, so it has no medication label."
            labels.append(df_vars.loc[df_vars['id']==subject_id, ['Cloz']].values[0][0])
        elif target_class == ClinicalClasses.AGE:
            labels.append(df_vars.loc[df_vars['id']==subject_id, ['Age']].values[0][0])
        elif target_class == ClinicalClasses.SEX:
            # TODO have to check whether 0, 1 or 2 is male or female
            labels.append(df_vars.loc[df_vars['id']==subject_id, ['Sex']].values[0][0])

    labels = np.array(labels)
    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class SchizophreniaConflictD004Dataset(BaseClinicalDataset):
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
            name="SchizophreniaConflict", # d004
            target_class=target_class,
            available_classes=[ClinicalClasses.SCHIZOPHRENIA, ClinicalClasses.MEDICATION, ClinicalClasses.AGE, ClinicalClasses.SEX],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=1000,
            channel_names=['Fp1', 'Fz', 'F3', 'F7', 'AFp9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'AFp10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'F6', 'F2', 'AF4', 'AF8'],
            preload=preload,
        )
        # fmt: on
        logging.info("in D004Albrecht2017Dataset.__init__")
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
        
        self.data, self.labels = self.cache.cache(_load_data_albrecht2017)(self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency