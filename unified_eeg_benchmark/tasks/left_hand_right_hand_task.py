from .abstract_task import AbstractTask
from ..datasets.bcicomp_iv_2a_m import BCICompIV2aMDataset
from ..datasets.weibo2013_m import Weibo2013MDataset
from ..datasets.bcicomp_iv_2b_m import BCICompIV2bMDataset
from sklearn.metrics import accuracy_score


class LeftHandRightHandTask(AbstractTask):
    def __init__(self):
        super().__init__(
            name="left_right",
            classes=["left_hand", "right_hand"],
            datasets=[
                BCICompIV2aMDataset,
                BCICompIV2bMDataset,
                Weibo2013MDataset,
            ],
        )

    def get_scoring(self):
        return lambda y, y_pred: accuracy_score(y, y_pred.ravel())
