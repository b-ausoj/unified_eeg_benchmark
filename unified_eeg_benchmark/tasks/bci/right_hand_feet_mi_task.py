from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    Weibo2013MDataset,
    Schirrmeister2017MDataset,
    Schalk2004MDataset,
    BCICompIV2aMDataset,
    Barachant2012MDataset,
    Zhou2016MDataset,
    Faller2012MDataset,
    Steyrl2016MDataset,
    Scherer2015MDataset,
)
from ...enums.classes import Classes
from sklearn.metrics import f1_score
from ...enums.split import Split
import os
import json

base_path = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/"


class RightHandvFeetMITask(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Right Hand vs Feet MI",
            classes=[Classes.RIGHT_HAND_MI, Classes.FEET_MI],
            datasets=[
                Weibo2013MDataset,
                Schirrmeister2017MDataset,
                Schalk2004MDataset,
                BCICompIV2aMDataset,
                Barachant2012MDataset,
                Zhou2016MDataset,
                Faller2012MDataset,
                Scherer2015MDataset,
                #Steyrl2016MDataset, # no channel names
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
                    Split.TRAIN: list(range(1, 10)),
                    Split.TEST: list(range(10, 15)),
                },
                BCICompIV2aMDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7],
                    Split.TEST: [8, 9],
                },
                Barachant2012MDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6],
                    Split.TEST: [7, 8],
                },
                Zhou2016MDataset: {
                    Split.TRAIN: list(range(1, 4)),
                    Split.TEST: list(range(4, 5)),
                },
                Faller2012MDataset: {
                    Split.TRAIN: list(range(1, 10)),
                    Split.TEST: list(range(10, 12)),
                },
                Scherer2015MDataset: {
                    Split.TRAIN: list(range(1, 7)),
                    Split.TEST: list(range(7, 9)),
                },
                Steyrl2016MDataset: {
                    Split.TRAIN: list(range(1, 11)),
                    Split.TEST: list(range(11, 15)),
                },
            },
        )

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
