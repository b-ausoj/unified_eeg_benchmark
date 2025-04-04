from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    BCICompIV2aMDataset,
    Kaya2018Dataset,
)
from ...enums.classes import Classes
from sklearn.metrics import f1_score
from ...enums.split import Split
import os
import json


class FiveFingersMITask(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Five Fingers MI",
            classes=[
                Classes.FIVE_FINGERS_MI,
                Classes.FIVE_FINGERS_MI,
            ],
            datasets=[
                Kaya2018Dataset,
            ],
            subjects_split={
                Kaya2018Dataset: {
                    Split.TRAIN: ["B", "C", "E", "F", "G", "H", "I", "J", "K"], 
                    Split.TEST: ["A", "L", "M" ], 
                },
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
