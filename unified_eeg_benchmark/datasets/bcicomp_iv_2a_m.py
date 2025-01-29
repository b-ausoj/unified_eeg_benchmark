from .base_dataset import BaseDataset
import warnings
from ..enums.classes import Classes
import moabb
from typing import Optional, Sequence
from moabb.paradigms.base import BaseParadigm
from moabb.datasets import (
    BNCI2014_001,
)
import moabb.datasets.base as base
from moabb.paradigms import LeftRightImagery


moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_bcicomp_iv_2a(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class BCICompIV2aMDataset(BaseDataset):
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
            name="bcicomp_iv_2a_m", # MI Limb
            interval=(2, 6),
            target_classes=target_classes,
            available_classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI, Classes.FEET_MI, Classes.TONGUE_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=250,
            channel_names=["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"],
            preload=preload,
        )
        # fmt: on
        print("BCICompIV2aMDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
            "name": "BCICompIV2a",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        BCI_IV_2a = BNCI2014_001()
        paradigm = LeftRightImagery()
        self.data, self.labels, _ = self.cache.cache(_load_data_bcicomp_iv_2a)(
            paradigm, BCI_IV_2a, self.subjects
        )  # type: ignore
