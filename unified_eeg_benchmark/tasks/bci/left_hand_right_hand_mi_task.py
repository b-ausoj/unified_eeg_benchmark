from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    BCICompIV2aMDataset,
    BCICompIV2bMDataset,
    Weibo2013MDataset,
    Cho2017MDataset,
    GrosseWentrup2009MDataset,
    Lee2019MDataset,
    Liu2022MDataset,
    Schirrmeister2017MDataset,
    Schalk2004MDataset,
    Zhou2016MDataset,
)
from ...enums.classes import Classes
from sklearn.metrics import f1_score
from ...enums.split import Split

base_path = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/"


class LeftHandvRightHandMITask(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="left_hand_vs_right_hand_motor_imagery",
            classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI],
            datasets=[
                BCICompIV2aMDataset,
                BCICompIV2bMDataset,
                Weibo2013MDataset,
                Cho2017MDataset,
                GrosseWentrup2009MDataset,
                #Lee2019MDataset, # some errors in some subjects
                Liu2022MDataset,
                Schirrmeister2017MDataset,
                Schalk2004MDataset,
                Zhou2016MDataset,
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
                Cho2017MDataset: {
                    Split.TRAIN: list(range(1, 10)),
                    Split.TEST: list(range(10, 15)),
                },
                GrosseWentrup2009MDataset: {
                    Split.TRAIN: list(range(1, 9)),
                    Split.TEST: list(range(9, 11)),
                },
                Lee2019MDataset: {
                    Split.TRAIN: list(range(1, 42)),
                    Split.TEST: list(range(42, 55)),
                },
                Liu2022MDataset: {
                    Split.TRAIN: list(range(1, 10)),
                    Split.TEST: list(range(10, 15)),
                },
                Schirrmeister2017MDataset: {
                    Split.TRAIN: list(range(1, 12)),
                    Split.TEST: list(range(12, 15)),
                },
                Schalk2004MDataset: {
                    Split.TRAIN: list(range(1, 10)),
                    Split.TEST: list(range(10, 15)),
                },
                Zhou2016MDataset: {
                    Split.TRAIN: list(range(1, 4)),
                    Split.TEST: list(range(4, 5)),
                },
            },
        )

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
