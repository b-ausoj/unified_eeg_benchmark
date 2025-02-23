from .base_clinical_dataset import BaseClinicalDataset
from ..enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence
import logging
from scipy.io import loadmat
import numpy as np


DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/d001_cavanagh2017a/data/"


def _load_data_cavanagh2017a(subjects: Sequence[int]):
    # subjects 1-25 have parkinsons, 26-53 don't have parkinsons
    # the ones with parkinsons have two recordings
    pd_subjects = ["804", "805", "806", "807", "808", "809", "810", "811", "813", "814", "815", "816", "817", "818", "819", "820", "821", "822", "823", "824", "825", "826", "827", "828", "829"]
    no_pd_subjects = ["890", "891", "892", "893", "894", "895", "896", "897", "898", "899", "900", "901", "902", "903", "904", "905", "906", "907", "908", "909", "910", "911", "912", "913", "914", "8010", "8060", "8070"]

    data = []
    labels = []
    for subject in subjects:
        if subject <= 25:
            subject_id = pd_subjects[subject - 1]
            
            if (subject_id != "810"):
                file_name_1 = DATA_PATH + subject_id + "_1_PDDys_ODDBALL.mat"
                mat_1 = loadmat(file_name_1, simplify_cells=True)
                data.append(mat_1['EEG']['data'])
                labels.append("parkinsons")

            file_name_2 = DATA_PATH + subject_id + "_2_PDDys_ODDBALL.mat"
            mat_2 = loadmat(file_name_2, simplify_cells=True)
            data.append(mat_2['EEG']['data'])
            labels.append("parkinsons")
        
        else:
            subject_id = no_pd_subjects[subject - 26]
            file_name = DATA_PATH + subject_id + "_1_PDDys_ODDBALL.mat"
            mat = loadmat(file_name, simplify_cells=True)
            data.append(mat['EEG']['data'])
            labels.append("no_parkinsons")

    labels = np.array(labels)
    return data, labels


class Cavanagh2017ADataset(BaseClinicalDataset):
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
            name="Cavanagh2017a", # d001
            target_class=target_class,
            available_classes=[ClinicalClasses.PARKINSONS],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8', 'CPz'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Cavanagh2017ADataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "name": "Cavanagh2017a",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        
        self.data, self.labels = self.cache.cache(_load_data_cavanagh2017a)(self.subjects)
        # have to reshape the data to (n_subjects, n_channels, n_timepoints) right now it is (n_subjects, n_channels, n_trials, n_timepoints)
        self.data = [np.reshape(subject_data, (subject_data.shape[0], -1)) for subject_data in self.data]
        #print("data shape: ", self.data[0].shape)