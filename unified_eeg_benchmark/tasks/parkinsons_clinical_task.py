from .abstract_clinical_task import AbstractClinicalTask
from ..enums.clinical_classes import ClinicalClasses
from ..datasets.cavanagh2017a import Cavanagh2017ADataset
from ..datasets.cavanagh2017b import Cavanagh2017BDataset
from sklearn.metrics import f1_score
from ..enums.split import Split

base_path = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/"


class ParkinsonsClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="parkinsons_clinical",
            clinical_class = ClinicalClasses.PARKINSONS,
            datasets = [
                #Cavanagh2017ADataset,
                Cavanagh2017BDataset,
            ],
            subjects_split={
                Cavanagh2017ADataset: {
                    Split.TRAIN: list(range(1, 13)) + list(range(26, 40)),
                    Split.TEST: list(range(13, 26)) + list(range(40, 53)),
                },
                Cavanagh2017BDataset: {
                    Split.TRAIN: list(range(1, 13)) + list(range(26, 40)),
                    Split.TEST: list(range(13, 26)) + list(range(40, 53)),
                },
            }
        )

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
