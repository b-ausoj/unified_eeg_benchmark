from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.classes import Classes
import moabb
from typing import Optional, Sequence
from moabb.paradigms.base import BaseParadigm
from moabb.datasets import (
    Weibo2014,
)
import moabb.datasets.base as base
from moabb.paradigms import MotorImagery
import logging

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_weibo2013(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class Weibo2013MDataset(BaseBCIDataset):
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
            name="weibo2013_m", # MI Limb
            interval=(3, 7),
            target_classes=target_classes,
            available_classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI, Classes.FEET_MI, Classes.BOTH_HANDS_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=200,
            channel_names=["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5","C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "O1", "Oz", "O2"],
            preload=preload,
        )
        # fmt: on
        logging.info("in Weibo2013MDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {
                "left_hand": 1,
                "right_hand": 2,
                "hands": 3,
                "feet": 4,
                "left_hand_right_foot": 5,
                "right_hand_left_foot": 6,
                "rest": 7,
            },
            "name": "Weibo2013",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        MI_Limb = Weibo2014()
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

        self.data, self.labels, _ = self.cache.cache(_load_data_weibo2013)(
            paradigm, MI_Limb, self.subjects
        )  # type: ignore
