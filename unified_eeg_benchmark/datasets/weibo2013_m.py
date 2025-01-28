from .abstract_dataset import AbstractDataset
from ..tasks.abstract_task import AbstractTask
import warnings
from ..enums.split import Split
from ..enums.classes import Classes
from typing import List
import moabb
from moabb.datasets import Weibo2014
from moabb.paradigms import LeftRightImagery


moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_weibo2013(paradigm, dataset, subjects):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class Weibo2013MDataset(AbstractDataset):
    def __init__(
        self,
        classes: List[Classes],
        split: Split,
        target_channels=None,
        target_frequency=None,
        preload=False,
    ):
        # fmt: off
        super().__init__(
            interval=[3, 7],
            name="weibo2013_m", # MI Limb
            target_classes=classes,
            classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI],
            split=split,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=200,
            channel_names=["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5","C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "O1", "Oz", "O2"],
            preload=preload,
        )
        # fmt: on
        print("Weibo2013Dataset.__init__")
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
        self._load_task_split()
        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self):
        MI_Limb = Weibo2014()
        paradigm = LeftRightImagery()
        subjects = self.task_split["left_right"][self._split.value]["subjects"]
        self.data, self.labels, _ = self._cache.cache(_load_data_weibo2013)(
            paradigm, MI_Limb, subjects
        )
