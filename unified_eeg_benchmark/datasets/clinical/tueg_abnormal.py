from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from ...enums.split import Split
from typing import Optional, Sequence, Tuple, List, Dict, cast
import logging
import glob
import os
from mne.io import read_raw_edf, BaseRaw
from tqdm import tqdm
from sklearn.utils import shuffle
from pathlib import Path


DATA_PATH = "/itet-stor/jbuerki/net_scratch/data/tueg_abnormal/tuab/edf/"


def _load_data_tueg_abnormal(subjects: Sequence[int], split: Split, preload: bool) -> Tuple[List[BaseRaw], List[str], List[str]]:
    """Loads EEG data from the TUEG Abnormal dataset.
    
    This function retrieves EEG recordings of subjects with normal or abnormal EEGs.
    The subjects are randomly mapped to the corresponding files in the dataset. (Improve me!)

    Args:
        subjects (Sequence[int]): A list of subject indices (1-indexed) to load data for.
        split (Split): The split for which to load the data.
        preload (bool): If True, preloads EEG data into memory.

    Returns:
        Tuple:
            - List[BaseRaw]: A list of `RawEDF` EEG recordings.
            - List[str]: A list of labels (`"abnormal"` or `"normal"`).
            - List[str]: A list of montage types for each EEG signal.
    """
    
    data: List[BaseRaw] = []
    labels: List[str] = []
    montage_type: List[str] = []

    files = glob.glob(os.path.join(DATA_PATH, "train" if split == Split.TRAIN else "eval", "**", "*.edf"), recursive=True)
    files = cast(List[str], shuffle(files, random_state=42))
    files = [files[i] for i in subjects]

    for file in files:
        raw = read_raw_edf(file, verbose="error", preload=preload)
        data.append(raw)
        labels.append(Path(file).parents[1].name)
        montage_type.append(Path(file).parents[0].name)
        
    return data, labels, montage_type


class TUEGAbnormalDataset(BaseClinicalDataset):
    def __init__(
        self,
        target_class: ClinicalClasses,
        subjects: Sequence[int],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = None,
        preload: bool = False,
    ):
        # fmt: off
        super().__init__(
            name="TUEG Abnormal",
            target_class=target_class,
            available_classes=[ClinicalClasses.ABNORMAL],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=0, # individual for each recording
            channel_names=[], # individual for each recording
            preload=preload,
        )
        # fmt: on
        logging.info("in TUEGAbnormalDataset.__init__")
        self.data: List[BaseRaw]
        self.labels: List[str]
        self.meta = {
            "name": self.name,
            "montage_type": []
        }

        if preload:
            self.load_data(Split.TRAIN)

    def _download(self, subject: int):
        pass

    def load_data(self, split: Split) -> None:
        
        self.data, self.labels, montage_type = self.cache.cache(_load_data_tueg_abnormal)(self.subjects, split, self._preload) # type: ignore
        self.meta["montage_type"] = montage_type

    def get_data(self, split: Split) -> Tuple[List[BaseRaw], List[str], Dict]:
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