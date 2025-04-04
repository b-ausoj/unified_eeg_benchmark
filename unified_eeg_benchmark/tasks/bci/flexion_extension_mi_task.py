from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import Ofner2017MDataset
from ...enums.classes import Classes
from sklearn.metrics import f1_score
from ...enums.split import Split
import os
import json


class FlexionvExtensionMITask(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="flexion_vs_extension_mi",
            classes=[Classes.RIGHT_ELBOW_EXTENSION_MI, Classes.RIGHT_ELBOW_FLEXION_MI],
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
