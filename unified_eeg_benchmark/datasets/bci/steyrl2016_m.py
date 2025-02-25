from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.classes import Classes
from typing import Optional, Sequence
import moabb
from moabb.datasets import (
    BNCI2014_002,
)
from moabb.paradigms import MotorImagery
import moabb.datasets.base as base
from moabb.paradigms.base import BaseParadigm
import logging

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_steyrl2016(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class Steyrl2016MDataset(BaseBCIDataset):
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
            name="steyrl2016_m", # aka MI TWO or BNCI 2014-002
            interval=(3, 8),
            target_classes=target_classes,
            available_classes=[Classes.FEET_MI, Classes.RIGHT_HAND_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=512,
            channel_names = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8', 'channel9', 'channel10', 'channel11', 'channel12', 'channel13', 'channel14', 'channel15'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Steyrl2016MDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {"right_hand": 1, "feet": 2},
            "name": self.name,
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        Steyrl2016 = BNCI2014_002()
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
            paradigm = MotorImagery()
        elif self.target_classes == [Classes.FEET_MI, Classes.RIGHT_HAND_MI]:
            paradigm = MotorImagery()
        else:
            raise ValueError("Invalid target classes")
        self.data, self.labels, _ = self.cache.cache(_load_data_steyrl2016)(
            paradigm, Steyrl2016, self.subjects
        )  # type: ignore
