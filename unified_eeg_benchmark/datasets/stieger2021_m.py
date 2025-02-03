from .base_dataset import BaseDataset
import warnings
from ..enums.classes import Classes
import moabb
from typing import Optional, Sequence
from moabb.paradigms.base import BaseParadigm
from moabb.datasets import (
    Stieger2021,
)
import moabb.datasets.base as base
from moabb.paradigms import MotorImagery
import logging

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_stieger2021(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)

###########
# The download is broken: https://github.com/NeuroTechX/moabb/issues/677
###########

class Stieger2021MDataset(BaseDataset):
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
            name="stieger2021_m",
            interval=(0, 3),
            target_classes=target_classes,
            available_classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI, Classes.BOTH_HANDS_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=1000,
            channel_names=[], # TODO this
            preload=preload,
        )
        # fmt: on
        logging.info("in Stieger2021MDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {
                "left_hand": 2,
                "right_hand": 1,
                "both_hand": 3,
            },
            "name": "Stieger2021",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        Stieger2021M = Stieger2021()
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
            paradigm = MotorImagery(
                n_classes=3, events=["left_hand", "right_hand", "both_hand"]
            )
        elif set(self.target_classes) == set(
            [Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI]
        ):
            paradigm = MotorImagery(n_classes=2, events=["left_hand", "right_hand"])
        else:
            raise ValueError("Invalid target classes")

        self.data, self.labels, _ = self.cache.cache(_load_data_stieger2021)(
            paradigm, Stieger2021M, self.subjects
        )  # type: ignore
