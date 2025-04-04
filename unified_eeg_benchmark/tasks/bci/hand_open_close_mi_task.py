from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import Ofner2017MDataset
from ...enums.classes import Classes
from sklearn.metrics import f1_score
from ...enums.split import Split
import os
import json


class HandOpenvCloseMITask(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="hand_open_vs_close_mi",
            classes=[Classes.RIGHT_HAND_CLOSE_MI, Classes.RIGHT_HAND_OPEN_MI],
            datasets=[
                Ofner2017MDataset,
            ],
            subjects_split={
                Ofner2017MDataset: {
                    Split.TRAIN: list(range(1, 12)),
                    Split.TEST: list(range(12, 16)),
                },
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
