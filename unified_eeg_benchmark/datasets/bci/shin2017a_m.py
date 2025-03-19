from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.classes import Classes
from typing import Optional, Sequence
import moabb
from moabb.datasets import (
    Shin2017A,
)
from moabb.paradigms import LeftRightImagery
import moabb.datasets.base as base
from moabb.paradigms.base import BaseParadigm
import logging
import numpy as np

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_shin2017a(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class Shin2017AMDataset(BaseBCIDataset):
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
            name="Shin2017A",
            interval=(3, 7.5),
            target_classes=target_classes,
            available_classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=200,
            # remapped to 10-20/10-10 system (and thus renamed the channels)
            channel_names = ["AF3", "AF4", "AF7", "AF8", "F5", "F6", "F3", "F4", "F7", "F8", "FC3", "FC4", "FC5", "FC6", "T7", "T8", "Cz", "C3", "C4", "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8", "PO7", "PO8", "PO9", "PO10"],
            preload=preload,
        )
        # fmt: on
        logging.info("in Shin2017AMDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {"left_hand": 1, "right_hand": 2},
            "name": "Shin2017A",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        Shin2017AM = Shin2017A(accept=True)
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
        elif self.target_classes == [Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI]:
            paradigm = LeftRightImagery()
        else:
            raise ValueError("Invalid target classes")
        if (self.subjects is None) or (len(self.subjects) == 0):
            self.data = np.array([])
            self.labels = np.array([])
            return
        self.data, self.labels, _ = self.cache.cache(_load_data_shin2017a)(
            paradigm, Shin2017AM, self.subjects
        )  # type: ignore
