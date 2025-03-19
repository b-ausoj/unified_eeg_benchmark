from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.classes import Classes
from typing import Optional, Sequence
import moabb
from moabb.datasets import (
    Ofner2017,
)
from moabb.paradigms import MotorImagery
import moabb.datasets.base as base
from moabb.paradigms.base import BaseParadigm
import logging

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_ofner2017(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)

class Ofner2017MDataset(BaseBCIDataset):
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
            name="Ofner2017",
            interval=(0, 3),
            target_classes=target_classes,
            available_classes=[Classes.RIGHT_ELBOW_EXTENSION_MI, Classes.RIGHT_ELBOW_FLEXION_MI, Classes.RIGHT_HAND_CLOSE_MI, Classes.RIGHT_HAND_OPEN_MI, Classes.RIGHT_PRONATION_MI, Classes.RIGHT_SUPINATION_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=512,
            channel_names = ['F3', 'F1', 'FZ', 'F2', 'F4', 'FFC5h', 'FFC3h', 'FFC1h', 'FFC2h', 'FFC4h', 'FFC6h', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FTT7h', 'FCC5h', 'FCC3h', 'FCC1h', 'FCC2h', 'FCC4h', 'FCC6h', 'FTT8h', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'TTP7h', 'CCP5h', 'CCP3h', 'CCP1h', 'CCP2h', 'CCP4h', 'CCP6h', 'TTP8h', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'CPP5h', 'CPP3h', 'CPP1h', 'CPP2h', 'CPP4h', 'CPP6h', 'P3', 'P1', 'Pz', 'P2', 'P4', 'PPO1h', 'PPO2h'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Ofner2017Dataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {
                "right_elbow_flexion": 1536,
                "right_elbow_extension": 1537,
                "right_supination": 1538,
                "right_pronation": 1539,
                "right_hand_close": 1540,
                "right_hand_open": 1541
            },
            "name": self.name,
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self) -> None:
        Ofner2017M = Ofner2017()
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
            paradigm = MotorImagery()
        elif self.target_classes == [Classes.RIGHT_ELBOW_EXTENSION_MI, Classes.RIGHT_ELBOW_FLEXION_MI, Classes.RIGHT_HAND_CLOSE_MI, Classes.RIGHT_HAND_OPEN_MI, Classes.RIGHT_PRONATION_MI, Classes.RIGHT_SUPINATION_MI]:
            paradigm = MotorImagery(
                n_classes=6, events=["right_elbow_flexion", "right_elbow_extension", "right_supination", "right_pronation", "right_hand_close", "right_hand_open"]
            )
        elif self.target_classes == [Classes.RIGHT_ELBOW_EXTENSION_MI, Classes.RIGHT_ELBOW_FLEXION_MI]:
            paradigm = MotorImagery(
                n_classes=2, events=["right_elbow_flexion", "right_elbow_extension"]
            )
        elif self.target_classes == [Classes.RIGHT_HAND_CLOSE_MI, Classes.RIGHT_HAND_OPEN_MI]:
            paradigm = MotorImagery(
                n_classes=2, events=["right_hand_close", "right_hand_open"]
            )
        elif self.target_classes == [Classes.RIGHT_PRONATION_MI, Classes.RIGHT_SUPINATION_MI]:
            paradigm = MotorImagery(
                n_classes=2, events=["right_supination", "right_pronation"]
            )
        else:
            raise ValueError("Invalid target classes")
        if (self.subjects is None) or (len(self.subjects) == 0):
            self.data = np.array([])
            self.labels = np.array([])
            return
        self.data, self.labels, _ = self.cache.cache(_load_data_ofner2017)(
            paradigm, Ofner2017M, self.subjects
        )  # type: ignore
