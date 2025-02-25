from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence
import logging
from scipy.io import loadmat
import numpy as np
import pandas as pd
import warnings


DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/d007_brown2020/data/PD RewP/"


def _load_data_brown2020(subjects: Sequence[int], target_class: ClinicalClasses):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        df_vars = pd.read_excel(DATA_PATH + 'MEASURES.xlsx', sheet_name='Sheet1', header=None)

    mat_on_off_med = loadmat(DATA_PATH + 'ONOFF.mat', simplify_cells=True)
    df_on_off_med = pd.DataFrame(data=mat_on_off_med['ONOFF'])

    pd_subjects = ["801", "802", "803", "804", "805", "806", "807", "808", "809", "810", "811", "813", "814", "815", "816", "817", "818", "819", "820", "821", "822", "823", "824", "825", "826", "827", "828", "829"]
    no_pd_subjects = ["890", "891", "892", "893", "894", "895", "896", "897", "898", "899", "900", "901", "902", "903", "904", "905", "906", "907", "908", "909", "910", "911", "912", "913", "914", "8010", "8060", "8070"]

    data = []
    labels = []
    for subject in subjects:
        if subject <= 28:
            subject_id = pd_subjects[subject - 1]
            subj_int = int(subject_id)
            med_list = df_on_off_med.loc[df_on_off_med[0]==subj_int, :].values[0]
            file_name_1 = DATA_PATH + "PROCESSED EEG DATA/" + subject_id + "_Session_1_PDDys_VV_withcueinfo.mat"
            mat_1 = loadmat(file_name_1, simplify_cells=True)
            data.append(mat_1['EEG']['data'].reshape(60, -1))
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("parkinsons")
            elif target_class == ClinicalClasses.BDI:
                labels.append(df_vars.loc[df_vars[0]==subj_int, [2]].values[0][0])
            elif target_class == ClinicalClasses.MEDICATION:
                labels.append("ON" if med_list[2] == 1 else "OFF")
        
            file_name_2 = DATA_PATH + "PROCESSED EEG DATA/" + subject_id + "_Session_2_PDDys_VV_withcueinfo.mat"
            mat_2 = loadmat(file_name_2, simplify_cells=True)
            data.append(mat_2['EEG']['data'].reshape(60, -1))
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("parkinsons")
            elif target_class == ClinicalClasses.BDI:
                labels.append(df_vars.loc[df_vars[0]==subj_int, [2]].values[0][0])
            elif target_class == ClinicalClasses.MEDICATION:
                labels.append("OFF" if med_list[2] == 1 else "ON")
        else:
            assert not target_class in [ClinicalClasses.MEDICATION, ClinicalClasses.BDI], "no medication or bdi data for subjects without parkinsons"
            subject_id = no_pd_subjects[subject - 29]
            file_name = DATA_PATH + "PROCESSED EEG DATA/" + subject_id + "_Session_1_PDDys_VV_withcueinfo.mat"
            mat = loadmat(file_name, simplify_cells=True)
            data.append(mat['EEG']['data'].reshape(60, -1))
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("no_parkinsons")
            
    labels = np.array(labels)
    return data, labels


class ParkinsonsRLTaskD007Dataset(BaseClinicalDataset):
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
            name="Brown2020", # d007
            target_class=target_class,
            available_classes=[ClinicalClasses.PARKINSONS, ClinicalClasses.MEDICATION, ClinicalClasses.BDI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8', 'CPz'],
            preload=preload,
        )
        # fmt: on
        logging.info("in ParkinsonsRLTaskD007Dataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "name": "Brown2020",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        
        self.data, self.labels = self.cache.cache(_load_data_brown2020)(self.subjects, self.target_classes[0]) # type: ignore