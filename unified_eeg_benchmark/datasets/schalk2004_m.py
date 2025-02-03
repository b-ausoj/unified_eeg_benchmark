from .base_dataset import BaseDataset
import warnings
from ..enums.classes import Classes
import moabb
from typing import Optional, Sequence
from moabb.paradigms.base import BaseParadigm
from moabb.datasets import (
    PhysionetMI,
)
import moabb.datasets.base as base
from moabb.paradigms import MotorImagery
import numpy as np
import logging

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_schalk2004(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class Schalk2004MDataset(BaseDataset):
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
            name="schalk2004_m", # EEGMMIDB or PhysionetMI
            interval=(0, 3),
            target_classes=target_classes,
            available_classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI, Classes.FEET_MI, Classes.BOTH_HANDS_MI],  
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=160,
            channel_names=['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'FPz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Schalk2004MDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {
                "left_hand": 2,
                "right_hand": 3,
                "hands": 4,
                "feet": 5,
                "rest": 1,
            },
            "name": "Schalk2004",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        Schalk2004 = PhysionetMI()
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
            paradigm = MotorImagery(
                n_classes=4, events=["left_hand", "right_hand", "feet", "hands"]
            )
        elif set(self.target_classes) == set(
            [Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI]
        ):
            paradigm = MotorImagery(n_classes=2, events=["left_hand", "right_hand"])
        elif set(self.target_classes) == set([Classes.RIGHT_HAND_MI, Classes.FEET_MI]):
            paradigm = MotorImagery(n_classes=2, events=["right_hand", "feet"])
        else:
            raise ValueError("Invalid target classes")

        if (self.subjects is None) or (len(self.subjects) == 0):
            self.data = np.array([])
            self.labels = np.array([])
            return
        self.data, self.labels, _ = self.cache.cache(_load_data_schalk2004)(
            paradigm, Schalk2004, self.subjects
        )  # type: ignore
