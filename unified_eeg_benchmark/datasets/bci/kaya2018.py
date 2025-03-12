from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.classes import Classes
import moabb
from typing import Optional, Sequence
import logging
import numpy as np

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_kaya2018(paradigm: str, subjects: Sequence[int]):
    return [], []  # TODO


class Kaya2018Dataset(BaseBCIDataset):
    """
    doi: https://doi.org/10.6084/m9.figshare.c.3917698.v1
    subjects: 13
    hardware: EEG-1200
    so far only CLA (left vs right hand) and HaLT (left hand vs right hand vs legs vs tongue) paradigms
    but could be extended to also include 5F (five fingers) paradigm
    """
    def __init__(
        self,
        target_classes: Sequence[Classes],
        subjects: Sequence[int],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = None,
        preload: bool = True,
    ):
        # fmt: off
        super().__init__(
            name="kaya_2018", # MI SCP
            interval=(2, 6), # TODO check
            target_classes=target_classes,
            available_classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI, Classes.FEET_MI, Classes.TONGUE_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=200, # TODO but not always
            channel_names=["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "A1", "A2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz", "X3"],
            preload=preload,
        )
        # fmt: on
        logging.info("in BCICompIV2aMDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
            "name": "Kaya2018",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        print("Please download the data from https://doi.org/10.6084/m9.figshare.c.3917698.v1")

    def load_data(self) -> None:
        if self.target_classes is None:
            logging.warning("target_classes is None, loading CLA...")
            paradigm = "CLA"
        elif set(self.target_classes) == set(
            [
                Classes.LEFT_HAND_MI,
                Classes.RIGHT_HAND_MI,
                Classes.FEET_MI,
                Classes.TONGUE_MI,
            ]
        ):
            paradigm = "HaLT"
        elif set(self.target_classes) == set(
            [Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI]
        ):
            paradigm = "CLA" # TODO add the ones from HaLT
        elif set(self.target_classes) == set([Classes.RIGHT_HAND_MI, Classes.FEET_MI]):
            paradigm = "HaLT" # TODO adjust code to remove left and tongue
        else:
            raise ValueError("Invalid target classes")

        if (self.subjects is None) or (len(self.subjects) == 0):
            self.data = np.array([])
            self.labels = np.array([])
        else:
            self.data, self.labels = self.cache.cache(_load_data_kaya2018)(
                paradigm, self.subjects
            )  # type: ignore
