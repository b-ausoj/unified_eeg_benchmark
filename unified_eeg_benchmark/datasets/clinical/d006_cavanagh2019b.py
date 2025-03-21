from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence, Tuple
from resampy import resample
import logging
from scipy.io import loadmat
import numpy as np
import pandas as pd
from tqdm import tqdm


DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/d006_cavanagh2019b/data/Depression PS Task/"


def _load_data_cavanagh2019b(subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    all_subjects = [str(i) for i in range(507, 629) if i != 544 and i != 599 and i != 600]
    df_vars = pd.read_excel(DATA_PATH + "Scripts from Manuscript/Data_4_Import.xlsx")
    dep_ids = df_vars.loc[df_vars['BDI']>=7.0, ['id']].values

    data = []
    labels = []
    for subject in tqdm(subjects, desc="Loading data from Cavanagh2019b"):
        subject_id = all_subjects[subject - 1]
        file_path = f"{DATA_PATH}Data/{subject_id}.mat"
        mat = loadmat(file_path, simplify_cells=True)
        data.append(mat["EEG"]["data"][:64, :])
        if target_class == ClinicalClasses.DEPRESSION:
            labels.append("depression" if int(subject_id) in dep_ids else "no_depression")
        elif target_class == ClinicalClasses.BDI:
            labels.append(df_vars.loc[df_vars['id']==int(subject_id), ['BDI']].values[0][0])
        elif target_class == ClinicalClasses.AGE:
            labels.append(df_vars.loc[df_vars['id']==int(subject_id), ['age']].values[0][0])
        elif target_class == ClinicalClasses.SEX:
            # TODO have to check whether 0, 1 or 2 is male or female
            labels.append(df_vars.loc[df_vars['id']==int(subject_id), ['sex']].values[0][0])

    labels = np.array(labels)
    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class DepressionRLD006Dataset(BaseClinicalDataset):
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
            name="Cavanagh2019b", # d006
            target_class=target_class,
            available_classes=[ClinicalClasses.DEPRESSION, ClinicalClasses.BDI, ClinicalClasses.AGE, ClinicalClasses.SEX],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Cavanagh2019BDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "name": "Cavanagh2019b",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        
        self.data, self.labels = self.cache.cache(_load_data_cavanagh2019b)(self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency
        