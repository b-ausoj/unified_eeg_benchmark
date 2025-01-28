from .abstract_dataset import AbstractDataset
import warnings
from ..enums.split import Split
from ..tasks.abstract_task import AbstractTask
import moabb
from moabb.datasets import (
    BNCI2014_004,
)
from moabb.paradigms import LeftRightImagery


moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_bcicomp_iv_2b(paradigm, dataset, subjects):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class BCICompIV2bMDataset(AbstractDataset):
    def __init__(
        self,
        task: AbstractTask,
        split: Split,
        target_channels=None,
        target_frequency=None,
        preload=False,
    ):
        # fmt: off
        super().__init__(
            interval=[3, 7.5],
            name="bcicomp_iv_2b_m",
            task=task,
            tasks=["left_right"],
            split=split,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=250,
            channel_names = ["C3", "Cz", "C4"],
            preload=preload,
        )
        # fmt: on
        print("BCICompIV2b.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {"left_hand": 1, "right_hand": 2},
        }

        self._load_task_split()
        if preload:
            self.load_data()

    def _download(self, subject: int):
        pass

    def load_data(self):
        BCI_IV_2a = BNCI2014_004()
        paradigm = LeftRightImagery()
        subjects = self.task_split[self._task.name][self._split.value]["subjects"]
        self.data, self.labels, _ = self._cache.cache(_load_data_bcicomp_iv_2b)(
            paradigm, BCI_IV_2a, subjects
        )
