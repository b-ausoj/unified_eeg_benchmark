from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence
import logging
from scipy.io import loadmat
import numpy as np
import pandas as pd


DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/d003_cavanagh2019a/data/Depression Rest/"


def _load_data_cavanagh2019a(subjects: Sequence[int], target_class: ClinicalClasses):
    all_subjects = [str(i) for i in range(507, 629) if i != 544 and i != 571 and i != 572]
    df_vars = pd.read_excel(DATA_PATH + "Data_4_Import_REST.xlsx")
    dep_ids = df_vars.loc[df_vars['BDI']>=13.0, ['id']].values

    data = []
    labels = []
    for subject in subjects:
        subject_id = all_subjects[subject - 1]
        file_path = f"{DATA_PATH}Matlab Files/{subject_id}_Depression_REST.mat"
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
    return data, labels


class DepressionRestD003Dataset(BaseClinicalDataset):
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
            name="Cavanagh2019a", # d003
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
        logging.info("in Cavanagh2019ADataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "name": "Cavanagh2019a",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        
        self.data, self.labels = self.cache.cache(_load_data_cavanagh2019a)(self.subjects, self.target_classes[0])
