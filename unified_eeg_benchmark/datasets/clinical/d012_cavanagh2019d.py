from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence, Tuple
from resampy import resample
import logging
from scipy.io import loadmat
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from ...utils.config import get_config_value
import os


#DATA_PATH = os.path.join(get_config_value("d012"), "data/mTBI Rest/")
DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/d012_cavanagh2019d/data/mTBI Rest/"


def _load_data_cavanagh2019d(subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    all_subjects = [i for i in range(3001, 3092) if i != 3053 and i != 3055 and i != 3057]
    df_vars = pd.read_excel(DATA_PATH + 'Quex.xlsx', sheet_name='S1', skiprows=0)
    mtbi_list = df_vars.loc[df_vars['ControlEquals1']!=1, ['SubID']].values.astype(int).flatten()
    
    data = []
    labels = []
    for subject in tqdm(subjects, desc="Loading data from Cavanagh2019d"):
        subject_id = all_subjects[subject - 1]
        files = glob.glob(f"{DATA_PATH}Data/{subject_id}_*_Rest.mat")
        for file_path in files:
            mat = loadmat(file_path, simplify_cells=True)
            signals = mat["EEG"]["data"]
            signals = signals.reshape(60, -1)
            signals = signals[:, sampling_frequency * 30:-sampling_frequency * 30]
            signals = np.clip(signals, -800, 800)
            data.append(signals)
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
    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class MTBIRestD012Dataset(BaseClinicalDataset):
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
            name="MTBIRest", # d012
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
        logging.info("in MTBIRestD012Dataset.__init__")
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
        
        self.data, self.labels = self.cache.cache(_load_data_cavanagh2019d)(self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency
        
        """
        print("data shape ", self.data[0].shape)
        print(f"Data range: min={np.min(self.data[0])}, max={np.max(self.data[0])}")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        self.data = [np.clip(d, -1000, 1000) for d in self.data]
        data_to_plot =  self.data #[self.data[i] for i in [1,11,21]] 
        plt.hist([d.flatten() for d in data_to_plot], bins=100, alpha=0.75, label=[f'Subject {i+1}' for i in range(len(data_to_plot))])
        plt.yscale('log')
        plt.xlabel('EEG Signal Value (uV)')
        plt.ylabel('Frequency')
        plt.title('Distribution of EEG Signal Values')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig('eeg_signal_distribution_d012.png')
        plt.close()
        """