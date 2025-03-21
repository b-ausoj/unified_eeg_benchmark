from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence, Tuple
import logging
from scipy.io import loadmat
import numpy as np
from mne.filter import filter_data, notch_filter
from resampy import resample
import pandas as pd
from tqdm import tqdm


DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/d002_cavanagh2017b/data/PDREST/"


def _load_data_cavanagh2017b(subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    # TODO make the caching more efficient by moving the target_class out of here and returning all the class labels
    # subjects 1-28 have parkinsons, 29-56 don't have parkinsons
    # the ones with parkinsons have two recordings
    pd_subjects = ["801", "802", "803", "804", "805", "806", "807", "808", "809", "810", "811", "813", "814", "815", "816", "817", "818", "819", "820", "821", "822", "823", "824", "825", "826", "827", "828", "829"]
    no_pd_subjects = ["890", "891", "892", "893", "894", "895", "896", "897", "898", "899", "900", "901", "902", "903", "904", "905", "906", "907", "908", "909", "910", "911", "912", "913", "914", "8010", "8060", "8070"]
    df_vars = pd.read_excel(DATA_PATH + "IMPORT_ME_REST.xlsx")

    data = []
    labels = []
    for subject in tqdm(subjects, desc="Loading data from Cavanagh2017b"):
        if subject <= 28:
            subject_id = pd_subjects[subject - 1]
            file_name_1 = DATA_PATH + subject_id + "_1_PD_REST.mat"
            if df_vars.loc[df_vars['PD_ID']==int(subject_id), ['1st Visit Meds Status']].values[0][0] == "OFF":
                mat_1 = loadmat(file_name_1, simplify_cells=True)
                data.append(mat_1['EEG']['data'][:63, :])
                if target_class == ClinicalClasses.PARKINSONS:
                    labels.append("parkinsons")
                elif target_class == ClinicalClasses.AGE:
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject_id), ['PD_Age']].values[0][0])
                elif target_class == ClinicalClasses.SEX:
                    # have to check whether 0, 1 or 2 is men or women
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject_id), ['PD_Sex']].values[0][0])
                elif target_class == ClinicalClasses.BDI:
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject_id), ['BDI']].values[0][0])
                elif target_class == ClinicalClasses.MEDICATION:
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject_id), ['1st Visit Meds Status']].values[0][0])
            else:
                file_name_2 = DATA_PATH + subject_id + "_2_PD_REST.mat"
                mat_2 = loadmat(file_name_2, simplify_cells=True)
                data.append(mat_2['EEG']['data'][:63, :])
                if target_class == ClinicalClasses.PARKINSONS:
                    labels.append("parkinsons")
                elif target_class == ClinicalClasses.AGE:
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject_id), ['PD_Age']].values[0][0])
                elif target_class == ClinicalClasses.SEX:
                    # have to check whether 0, 1 or 2 is men or women
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject_id), ['PD_Sex']].values[0][0])
                elif target_class == ClinicalClasses.BDI:
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject_id), ['BDI']].values[0][0])
                elif target_class == ClinicalClasses.MEDICATION:
                    labels.append("ON" if df_vars.loc[df_vars['PD_ID']==int(subject_id), ['1st Visit Meds Status']].values[0][0] == "OFF" else "OFF")
        
        else:
            assert not target_class in [ClinicalClasses.MEDICATION, ClinicalClasses.BDI], "no medication or bdi data for subjects without parkinsons"
            subject_id = no_pd_subjects[subject - 29]
            file_name = DATA_PATH + subject_id + "_1_PD_REST.mat"
            mat = loadmat(file_name, simplify_cells=True)
            data.append(mat['EEG']['data'][:63, :])
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("no_parkinsons")
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['MATCH CTL_ID']==int(subject_id), ['MATCH CTL_Age']].values[0][0])
            elif target_class == ClinicalClasses.SEX:
                # have to check whether 0, 1 or 2 is men or women
                labels.append(df_vars.loc[df_vars['MATCH CTL_ID']==int(subject_id), ['MATCH CTL_Sex']].values[0][0])
    data = [d * 1e-6 for d in data]
    labels = np.array(labels)

    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class ParkinsonsRestD002Dataset(BaseClinicalDataset):
    """
    - self.data: List of length n_subjects, where each element is a numpy array of shape (n_channels, n_samples)
    """
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
            name="Cavanagh2017b", # d002
            target_class=target_class,
            available_classes=[ClinicalClasses.PARKINSONS, ClinicalClasses.MEDICATION, ClinicalClasses.BDI, ClinicalClasses.AGE, ClinicalClasses.SEX],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8'],
            preload=preload,
        )
        # fmt: on
        logging.info("in ParkinsonsRestD002Dataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "name": "Cavanagh2017b",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        
        self.data, self.labels = self.cache.cache(_load_data_cavanagh2017b)(self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency