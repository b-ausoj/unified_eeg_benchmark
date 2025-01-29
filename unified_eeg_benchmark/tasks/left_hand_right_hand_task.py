from .abstract_task import AbstractTask
from ..datasets.bcicomp_iv_2a_m import BCICompIV2aMDataset
from ..datasets.weibo2013_m import Weibo2013MDataset
from ..datasets.bcicomp_iv_2b_m import BCICompIV2bMDataset
from ..enums.classes import Classes
from sklearn.metrics import accuracy_score
from ..enums.split import Split
import os
import json

base_path = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/"


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
            subjects_split={
                BCICompIV2aMDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7],
                    Split.TEST: [8, 9],
                },
                BCICompIV2bMDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7],
                    Split.TEST: [8, 9],
                },
                Weibo2013MDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7, 8],
                    Split.TEST: [9, 10],
                },
            },
        )

    def get_scoring(self):
        return lambda y, y_pred: accuracy_score(y, y_pred.ravel())
