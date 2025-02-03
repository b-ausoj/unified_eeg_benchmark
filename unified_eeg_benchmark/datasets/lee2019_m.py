from .base_dataset import BaseDataset
import warnings
from ..enums.classes import Classes
import moabb
from typing import Optional, Sequence
from moabb.paradigms.base import BaseParadigm
from moabb.datasets import (
    Lee2019_MI,
)
import moabb.datasets.base as base
from moabb.paradigms import LeftRightImagery
import numpy as np
import logging

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_lee2019(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class Lee2019MDataset(BaseDataset):
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
            name="lee2019_m", # aka Lee2019_MI 
            interval=(0, 4),
            target_classes=target_classes,
            available_classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI],  
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=1000,
            channel_names=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Lee2019MDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {
                "left_hand": 2,
                "right_hand": 1,
            },
            "name": "Lee2019",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        Lee2019 = Lee2019_MI()  # type: ignore
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
            paradigm = LeftRightImagery()
        elif set(self.target_classes) == set(
            [Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI]
        ):
            paradigm = LeftRightImagery()
        else:
            raise ValueError("Invalid target classes")

        if (self.subjects is None) or (len(self.subjects) == 0):
            self.data = np.array([])
            self.labels = np.array([])
            return
        self.data, self.labels, _ = self.cache.cache(_load_data_lee2019)(
            paradigm, Lee2019, self.subjects
        )  # type: ignore
