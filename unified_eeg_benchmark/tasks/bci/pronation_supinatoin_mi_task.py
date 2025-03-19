from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    Ofner2017MDataset,
    Ofner2019Dataset,    
)
from ...enums.classes import Classes
from sklearn.metrics import f1_score
from ...enums.split import Split
import os
import json


class PronationvSupinationMITask(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="pronation_vs_supination_mi",
            classes=[Classes.RIGHT_PRONATION_MI, Classes.RIGHT_SUPINATION_MI],
            datasets=[
                #Ofner2017MDataset,
                Ofner2019Dataset,
            ],
            subjects_split={
                Ofner2017MDataset: {
                    Split.TRAIN: list(range(1, 12)),
                    Split.TEST: list(range(12, 16)),
                },
                Ofner2019Dataset: {
                    Split.TRAIN: list(range(1, 9)),
                    Split.TEST: list(range(9, 10)),
                },
            },
        )

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
