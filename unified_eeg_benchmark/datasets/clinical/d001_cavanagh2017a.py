from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from ...enums.split import Split
from typing import Optional, Sequence, Tuple
import logging
from scipy.io import loadmat
from mne.filter import filter_data, notch_filter
from resampy import resample
import numpy as np
from tqdm import tqdm
from ...utils.config import get_config_value


DATA_PATH = get_config_value("d001")


def _load_data_cavanagh2017a(split: Split, subjects: Sequence[int], sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    # subjects 1-25 have parkinsons, 26-53 don't have parkinsons
    # the ones with parkinsons have two recordings
    pd_subjects = ["804", "805", "806", "807", "808", "809", "810", "811", "813", "814", "815", "816", "817", "818", "819", "820", "821", "822", "823", "824", "825", "826", "827", "828", "829"]
    no_pd_subjects = ["890", "891", "892", "893", "894", "895", "896", "897", "898", "899", "900", "901", "902", "903", "904", "905", "906", "907", "908", "909", "910", "911", "912", "913", "914", "8010", "8060", "8070"]

    train_subjects = ["804", "805", "806", "807", "808", "809", "810", "811", "813", "814", "815", "816", "817", "818", "819", "820", "821", "890", "891", "893", "894", "895", "896", "899", "900", "903", "905", "906", "908", "909", "910", "911", "912", "913", "914", "8010", "8060"]
    test_subjects = ["822", "823", "824", "825", "826", "827", "828", "829", "892", "897", "898", "901", "902", "904", "907", "8070"]

    if split == Split.TRAIN:
        subjects = train_subjects
    else:
        subjects = test_subjects
    print("Loading data from Cavanagh2017a")
    data = []
    labels = []
    for subject in tqdm(subjects, desc="Loading data from Cavanagh2017a"):
        if subject in pd_subjects:
            if subject != "810":
                file_name_1 = DATA_PATH + subject + "_1_PDDys_ODDBALL.mat"
                mat_1 = loadmat(file_name_1, simplify_cells=True)
                signals = mat_1['EEG']['data']
                if np.max(signals) > 1000 or np.min(signals) < -1000:
                    print(f"Large values in subject {subject}: signal range out of bounds (min={np.min(signals)}, max={np.max(signals)})")
                data.append(signals)
                labels.append("parkinsons")

            file_name_2 = DATA_PATH + subject + "_2_PDDys_ODDBALL.mat"
            mat_2 = loadmat(file_name_2, simplify_cells=True)
            signals = mat_2['EEG']['data']
            if np.max(signals) > 1000 or np.min(signals) < -1000:
                print(f"Large values in subject {subject}: signal range out of bounds (min={np.min(signals)}, max={np.max(signals)})")
            data.append(signals)
            labels.append("parkinsons")
        
        else:
            file_name = DATA_PATH + subject + "_1_PDDys_ODDBALL.mat"
            mat = loadmat(file_name, simplify_cells=True)
            signals = mat['EEG']['data']
            if np.max(signals) > 1000 or np.min(signals) < -1000:
                print(f"Large values in subject {subject}: signal range out of bounds (min={np.min(signals)}, max={np.max(signals)})")
            data.append(signals)
            labels.append("no_parkinsons")

    print("Number of subjects: ", len(subjects))
    print("Number of data: ", len(data))
    data = [np.reshape(subject_data, (subject_data.shape[0], -1)) for subject_data in data]
    labels = np.array(labels)
    
    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class ParkinsonsOddballD001Dataset(BaseClinicalDataset):
    """
    - self.data: List[np.ndarray], list of data samples, one (or two) for each subject, not all samples are of the same shape
    - self.labels: List[str], list of labels
    """

    def __init__(
        self,
        target_class: ClinicalClasses,
        subjects: Sequence[int],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = 250,
        preload: bool = False,
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
        logging.info("in ParkinsonsOddballD001Dataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "name": "Cavanagh2017a",
        }

        if preload:
            self.load_data(split=Split.TRAIN)

    def _download(self, subject: int):
        pass

    def load_data(self, split) -> None:
        
        self.data, self.labels = self.cache.cache(_load_data_cavanagh2017a)(split, self.subjects, self._sampling_frequency, self._target_frequency) # type: ignore
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
        plt.savefig('eeg_signal_distribution_d001.png')
        plt.close()
        """   
        
    def get_data(self, split: Split):
        self.load_data(split)
        return self.data, self.labels, self.meta