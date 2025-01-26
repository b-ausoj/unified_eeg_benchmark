from .abstract_task import AbstractTask
from ..datasets.bcicomp_iv_2a import BCICompIV2aDataset
from ..datasets.weibo2013 import Weibo2013Dataset


class LeftHandRightHandTask(AbstractTask):
    def __init__(self):
        super().__init__(
            name="left_right",
            classes=["left_hand", "right_hand"],
            datasets=[BCICompIV2aDataset, Weibo2013Dataset],
        )
