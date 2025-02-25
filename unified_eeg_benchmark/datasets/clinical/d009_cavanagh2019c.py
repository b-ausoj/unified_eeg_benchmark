from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence
import logging
from scipy.io import loadmat
import numpy as np
import pandas as pd
import glob

DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/d009_cavanagh2019c/data/"


def _load_data_cavanagh2019c(subjects: Sequence[int], target_class: ClinicalClasses):
    all_subjects = [i for i in range(3001, 3092) if i != 3053 and i != 3055 and i != 3057]
    mat_raw = loadmat(DATA_PATH + "Scripts/BigAgg_Data.mat", simplify_cells=True)
    df_vars = pd.DataFrame()
    df_vars_q = pd.DataFrame()

    df_vars['SubID'] = mat_raw['DEMO']['ID'][:, 0]
    df_vars['ControlEquals1'] = mat_raw['DEMO']['Group_CTL1'][:, 0]
    df_vars['FemaleEquals1'] = mat_raw['DEMO']['Sex_F1']
    df_vars['Age'] = mat_raw['DEMO']['Age']
    df_vars['BDItotal'] = mat_raw['QUEX']['BDI'][:, 0]

    df_vars_q['SubID'] = mat_raw['Q_DEMO']['URSI']
    df_vars_q['ControlEquals1'] = 0
    df_vars_q['FemaleEquals1'] = mat_raw['Q_DEMO']['Sex_F1']
    df_vars_q['Age'] = mat_raw['Q_DEMO']['Age']
    df_vars_q['BDItotal'] = mat_raw['Q_QUEX']['BDI']

    df_vars = pd.concat([df_vars, df_vars_q], ignore_index=True)
    mtbi_list = df_vars.loc[df_vars['ControlEquals1'] != 1, ['SubID']].values.astype(int).flatten()

    data = []
    labels = []
    for subject in subjects:
        subject_id = all_subjects[subject - 1]
        files = glob.glob(f"{DATA_PATH}{subject_id}_*_3AOB.mat")
        for file_path in files:
            mat = loadmat(file_path, simplify_cells=True)
            data.append(mat["EEG"]["data"].reshape(60, -1))
            if target_class == ClinicalClasses.MTBI:
                labels.append(subject_id in mtbi_list)
            elif target_class == ClinicalClasses.BDI:
                labels.append(df_vars.loc[df_vars['SubID'] == subject_id, 'BDItotal'].values[0])
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['SubID'] == subject_id, 'Age'].values[0])
            elif target_class == ClinicalClasses.SEX:
                # TODO have to check whether 0, 1 or 2 is male or female
                labels.append(df_vars.loc[df_vars['SubID'] == subject_id, 'FemaleEquals1'].values[0])

    labels = np.array(labels)
    return data, labels


class MTBIOddballD009Dataset(BaseClinicalDataset):
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
            name="MTBIOddball", # d009
            target_class=target_class,
            available_classes=[ClinicalClasses.MTBI, ClinicalClasses.BDI, ClinicalClasses.AGE, ClinicalClasses.SEX],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8', 'CPz'],
            preload=preload,
        )
        # fmt: on
        logging.info("in MTBIOddballD009Dataset.__init__")
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
        
        self.data, self.labels = self.cache.cache(_load_data_cavanagh2019c)(self.subjects, self.target_classes[0]) # type: ignore
