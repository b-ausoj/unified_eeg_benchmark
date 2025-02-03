from .base_dataset import BaseDataset
import warnings
from ..enums.classes import Classes
from typing import Optional, Sequence
import moabb
from moabb.datasets import (
    Shin2017A,
)
from moabb.paradigms import LeftRightImagery
import moabb.datasets.base as base
from moabb.paradigms.base import BaseParadigm
import logging

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_shin2017a(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)

############
# doesn't have the channels C3 and C3 so can't use it for now
############

class Shin2017AMDataset(BaseDataset):
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
            channel_names = ["AFp1", "AFp2", "AFF1h", "AFF2h", "AFF5h", "AFF6h", "F3", "F4", "F7", "F8", "FCC3h", "FCC4h", "FCC5h", "FCC6h", "T7", "T8", "Cz", "CCP3h", "CCP4h", "CCP5h", "CCP6h", "Pz", "P3", "P4", "P7", "P8", "PPO1h", "PPO2h", "POO1", "POO2"],
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
        Shin2017AM = Shin2017A()
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
        elif self.target_classes == [Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI]:
            paradigm = LeftRightImagery()
        else:
            raise ValueError("Invalid target classes")
        self.data, self.labels, _ = self.cache.cache(_load_data_shin2017a)(
            paradigm, Shin2017AM, self.subjects
        )  # type: ignore
