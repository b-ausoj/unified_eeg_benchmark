from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    BCICompIV2aMDataset,
    BCICompIV2bMDataset,
    Weibo2014MDataset,
    Cho2017MDataset,
    GrosseWentrup2009MDataset,
    Lee2019MDataset,
    Liu2022MDataset,
    Schirrmeister2017MDataset,
    PhysionetMIMDataset,
    Zhou2016MDataset,
    Kaya2018Dataset,
    Shin2017AMDataset,
)
from ...enums.classes import Classes
from sklearn.metrics import f1_score
from ...enums.split import Split


class LeftHandvRightHandMITask(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Left Hand vs Right Hand MI",
            classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI],
            datasets=[
                BCICompIV2aMDataset,
                BCICompIV2bMDataset,
                Weibo2014MDataset,
                Cho2017MDataset,
                #GrosseWentrup2009MDataset, # no channel names
                #Lee2019MDataset, # some errors in some subjects
                Liu2022MDataset, # gave rather bad results
                Schirrmeister2017MDataset,
                PhysionetMIMDataset,
                Zhou2016MDataset,
                Kaya2018Dataset,
                #Shin2017AMDataset, # not same channels, did some remapping but still not good
            ],
            subjects_split={
                BCICompIV2aMDataset: {
                    Split.TRAIN: [2, 3, 4, 5, 6, 7, 8, 9],
                    Split.TEST: [1],
                },
                BCICompIV2bMDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7],
                    Split.TEST: [8, 9],
                },
                Weibo2014MDataset: {
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
                PhysionetMIMDataset: {
                    Split.TRAIN: list(range(1, 88)),
                    Split.TEST: list(range(89, 92)) + list(range(93, 100)) + list(range(101, 110)),
                },
                Zhou2016MDataset: {
                    Split.TRAIN: [],
                    Split.TEST: list(range(1, 5)),
                },
                Kaya2018Dataset: {
                    Split.TRAIN: ["B", "C", "E", "F", "G", "H", "I", "J", "K"],
                    Split.TEST: ["A", "L", "M"],
                },
                Shin2017AMDataset: {
                    Split.TRAIN: list(range(1, 20)),
                    Split.TEST: list(range(20, 29)),
                },
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
