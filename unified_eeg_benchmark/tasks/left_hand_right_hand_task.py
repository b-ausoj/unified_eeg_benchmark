from .abstract_task import AbstractTask
from ..datasets.bcicomp_iv_2a_m import BCICompIV2aMDataset
from ..datasets.weibo2013_m import Weibo2013MDataset
from ..datasets.bcicomp_iv_2b_m import BCICompIV2bMDataset
from ..enums.classes import Classes
from sklearn.metrics import accuracy_score


class LeftHandRightHandTask(AbstractTask):
    def __init__(self):
        super().__init__(
            name="left_right_hand_motor_imagery",
            classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI],
            datasets=[
                BCICompIV2aMDataset,
                BCICompIV2bMDataset,
                Weibo2013MDataset,
            ],
            # TODO add the split (subjects) here and remove it from the dataset
        )

    def get_scoring(self):
        return lambda y, y_pred: accuracy_score(y, y_pred.ravel())
