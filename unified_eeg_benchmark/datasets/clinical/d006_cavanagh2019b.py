from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from ...enums.split import Split
from typing import Optional, Sequence, Tuple
from resampy import resample
import logging
from scipy.io import loadmat
import numpy as np
import pandas as pd
from tqdm import tqdm
from ...utils.config import get_config_value


DATA_PATH = get_config_value("d006")


def _load_data_cavanagh2019b(subjects: Sequence[int], split: Split, target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    all_subjects = [str(i) for i in range(507, 629) if i != 599 and i != 600]
    df_vars = pd.read_excel(DATA_PATH + "Scripts from Manuscript/Data_4_Import.xlsx")
    #dep_ids = df_vars.loc[df_vars['BDI']>=13.0, ['id']].values

    dep_ids = ["558", "559", "561", "564", "565", "566", "567", "569", "571", "572", "586", "587", "590", "591", "592", "594", "595", "597", "598", "602", "603", "604", "605", "606", "607", "608", "609", "610", "611", "612", "613", "614", "615", "616", "617", "618", "619", "620", "621", "622", "623", "624", "625", "626", "627", "628"]
    ctr_ids = ["507", "508", "509", "510", "511", "512", "513", "514", "515", "516", "517", "518", "519", "520", "521", "522", "523", "524", "525", "526", "527", "528", "529", "530", "531", "532", "533", "534", "535", "536", "537", "538", "539", "540", "541", "542", "543", "545", "546", "547", "548", "549", "550", "551", "552", "553", "554", "555", "556", "557", "560", "562", "563", "568", "570", "573", "574", "575", "576", "577", "578", "579", "580", "581", "582", "583", "584", "585", "588", "589", "593", "596", "601"]

    rng = np.random.RandomState(seed=42)
    
    rng.shuffle(dep_ids)
    rng.shuffle(ctr_ids)
    
    # Compute 70/30 split for each group
    n_train_dep = int(len(dep_ids) * 0.7)
    n_train_ctr = int(len(ctr_ids) * 0.7)

    print("dep_ids[:n_train_dep] ", dep_ids[:n_train_dep])
    
    if split == Split.TRAIN:
        subjects = dep_ids[:n_train_dep] + ctr_ids[:n_train_ctr]
    else:
        subjects = dep_ids[n_train_dep:] + ctr_ids[n_train_ctr:]
    
    rng.shuffle(subjects)

    data = []
    labels = []
    for subject in tqdm(subjects, desc="Loading data from Cavanagh2019b"):
        file_path = f"{DATA_PATH}Data/{subject}.mat"
        mat = loadmat(file_path, simplify_cells=True)
        signals = mat["EEG"]["data"]
        signals = signals[:64, :] # only use the first 64 channels
        signals = signals * 1e-3  # convert to microvolts
        data.append(signals)
        if target_class == ClinicalClasses.DEPRESSION:
            labels.append("depression" if subject in dep_ids else "no_depression")
        elif target_class == ClinicalClasses.BDI:
            labels.append(df_vars.loc[df_vars['id']==int(subject), ['BDI']].values[0][0])
        elif target_class == ClinicalClasses.AGE:
            labels.append(df_vars.loc[df_vars['id']==int(subject), ['age']].values[0][0])
        elif target_class == ClinicalClasses.SEX:
            # TODO have to check whether 0, 1 or 2 is male or female
            labels.append(df_vars.loc[df_vars['id']==int(subject), ['sex']].values[0][0])

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
        target_frequency: Optional[int] = None,
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
            self.load_data(split=Split.TRAIN)

    def _download(self, subject: int):
        pass

    def load_data(self, split) -> None:
        
        self.data, self.labels = self.cache.cache(_load_data_cavanagh2019b)(self.subjects, split, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency
        
        """
        print("data shape ", self.data[0].shape)
        print(f"Data range: min={np.min(self.data[0])}, max={np.max(self.data[0])}")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        data_to_plot =  self.data #[self.data[i] for i in [1,11,21]] 
        plt.hist([d.flatten() for d in data_to_plot], bins=100, alpha=0.75, label=[f'Subject {i+1}' for i in range(len(data_to_plot))])
        plt.yscale('log')
        plt.xlabel('EEG Signal Value (uV)')
        plt.ylabel('Frequency')
        plt.title('Distribution of EEG Signal Values')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig('eeg_signal_distribution_d006.png')
        plt.close()
        """
    
    def get_data(self, split: Split):
        """Get the data of the TUEG Abnormal dataset.
        
        The dataset contains EEG recordings of subjects with normal or abnormal EEGs.
        The subjects are randomly mapped to the corresponding files in the dataset. (Improve me!)

        Args:
            split (Split): The split for which to load the data.
    
        Returns:
            Tuple:
                - List[BaseRaw]: A list of `RawEDF` EEG recordings.
                - List[str]: A list of labels (`"abnormal"` or `"normal"`).
                - Dict: Metadata containing a list of montage types for each EEG signal.
        """

        self.load_data(split)
        return self.data, self.labels, self.meta