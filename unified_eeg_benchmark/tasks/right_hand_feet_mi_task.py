from .abstract_task import AbstractTask
from ..datasets.bcicomp_iv_2a_m import BCICompIV2aMDataset
from ..datasets.weibo2013_m import Weibo2013MDataset
from ..datasets.barachant2012_m import Barachant2012MDataset
from ..datasets.schalk2004_m import Schalk2004MDataset
from ..datasets.schirrmeister2017_m import Schirrmeister2017MDataset
from ..enums.classes import Classes
from sklearn.metrics import f1_score
from ..enums.split import Split
import os
import json

base_path = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/"


class RightHandvFeetMITask(AbstractTask):
    def __init__(self):
        super().__init__(
            name="right_hand_vs_feet_mi",
            classes=[Classes.RIGHT_HAND_MI, Classes.FEET_MI],
            datasets=[
                Weibo2013MDataset,
                Schirrmeister2017MDataset,
                Schalk2004MDataset,
                BCICompIV2aMDataset,
                Barachant2012MDataset,
            ],
            subjects_split={
                Weibo2013MDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7, 8],
                    Split.TEST: [9, 10],
                },
                Schirrmeister2017MDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    Split.TEST: [12, 13, 14],
                },
                Schalk2004MDataset: {
                    Split.TRAIN: list(range(1, 90)),
                    Split.TEST: list(range(90, 110)),
                },
                BCICompIV2aMDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7],
                    Split.TEST: [8, 9],
                },
                Barachant2012MDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6],
                    Split.TEST: [7, 8],
                },
            },
        )

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
