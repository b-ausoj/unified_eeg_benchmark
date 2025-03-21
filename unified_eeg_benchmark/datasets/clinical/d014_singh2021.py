from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
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


DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/d014_singh2021/data/PD Interval Timing NPJ/"


def _load_data_singh2021(subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    ctr_subjects = ["Control1025", "Control1035", "Control1055", "Control1065", "Control1075", "Control1085", "Control1095", "Control1105", "Control1115", "Control1125", "Control1135", "Control1145", "Control1155", "Control1175", "Control1185", "Control1195", "Control1205", "Control1215", "Control1225", "Control1235", "Control1245", "Control1255", "Control1265", "Control1275", "Control1285", "Control1295", "Control1305", "Control1315", "Control1325", "Control1335", "Control1345", "Control1365", "Control1375", "Control1385", "Control1395", "Control1405", "Control1415"]
    pd_subjects = ["PD1005", "PD1015", "PD1025", "PD1035", "PD1045", "PD1055", "PD1065", "PD1075", "PD1085", "PD1095", "PD1105", "PD1115", "PD1125", "PD1135", "PD1145", "PD1155", "PD1165", "PD1175", "PD1185", "PD1195", "PD1215", "PD1225", "PD1235", "PD1245", "PD1265", "PD1275", "PD1285", "PD1295", "PD1305", "PD1315", "PD1325", "PD1335", "PD1365", "PD1375", "PD1385", "PD1395", "PD1405", "PD1415", "PD1425", "PD1435", "PD1445", "PD1455", "PD1465", "PD1475", "PD1485", "PD1505", "PD1515", "PD1525", "PD1535", "PD1555", "PD1565", "PD1575", "PD1585", "PD1595", "PD1605", "PD1615", "PD1625", "PD1635", "PD1645", "PD1655", "PD1665", "PD1675", "PD1685", "PD1695", "PD1705", "PD1715", "PD1725", "PD1735", "PD1745", "PD1755", "PD1765", "PD1775", "PD1785", "PD1795"]
    df_vars = pd.read_excel(DATA_PATH + 'Copy of IntervalTiming_Subj_Info_AIE.xlsx', sheet_name='MAIN')

    data = []
    labels = []
    for subject in tqdm(subjects, desc="Loading data from Singh2021"):
        if subject < len(ctr_subjects) + 1:
            subject_id = ctr_subjects[subject - 1]
            file_path = f"{DATA_PATH}{subject_id}.vhdr"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = read_raw_brainvision(file_path, preload=True)
                raw.pick(['eeg'])
            data.append(raw.get_data(units='uV')) # type: ignore
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("no_parkinsons")
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['Rest']==subject_id, ['Age']].values[0][0])
            elif target_class == ClinicalClasses.SEX:
                labels.append(df_vars.loc[df_vars['Rest']==subject_id, ['Gender']].values[0][0])
        else:
            subject_id = pd_subjects[subject - len(ctr_subjects) - 1]
            file_path = f"{DATA_PATH}{subject_id}.vhdr"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = read_raw_brainvision(file_path, preload=True)     
                raw.pick(['eeg'])       
            data.append(raw.get_data(units='uV')) # type: ignore
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("parkinsons")
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['Rest']==subject_id, ['Age']].values[0][0])
            elif target_class == ClinicalClasses.SEX:
                labels.append(df_vars.loc[df_vars['Rest']==subject_id, ['Gender']].values[0][0])
            
    labels = np.array(labels)
    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    print("data shape ", data[0].shape)
    print(f"Data range: min={np.min(data[0])}, max={np.max(data[0])}")
    return data, labels


class PDIntervalTimingD014Dataset(BaseClinicalDataset):
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
            name="Singh2021", # d014
            target_class=target_class,
            available_classes=[ClinicalClasses.PARKINSONS, ClinicalClasses.AGE, ClinicalClasses.SEX],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'FCz'],
            preload=preload,
        )
        # fmt: on
        logging.info("in PDIntervalTimingD014Dataset.__init__")
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
        
        self.data, self.labels = self.cache.cache(_load_data_singh2021)(self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency
