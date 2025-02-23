from .abstract_bci_task import AbstractBCITask
from ..datasets.bcicomp_iv_2a_m import BCICompIV2aMDataset
from ..datasets.weibo2013_m import Weibo2013MDataset
from ..datasets.bcicomp_iv_2b_m import BCICompIV2bMDataset
from ..datasets.schalk2004_m import Schalk2004MDataset
from ..datasets.cho2017_m import Cho2017MDataset
from ..datasets.grossewentrup2009_m import GrosseWentrup2009MDataset
from ..datasets.lee2019_m import Lee2019MDataset
from ..datasets.liu2022_m import Liu2022MDataset
from ..datasets.schirrmeister2017_m import Schirrmeister2017MDataset
from ..enums.classes import Classes
from sklearn.metrics import f1_score
from ..enums.split import Split

base_path = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/"


class CustomMITask(AbstractBCITask):
    def __init__(self, classes, subjects_split):
        super().__init__(
            name="custom_mi_task",
            classes=classes,
            datasets=subjects_split.keys(),
            subjects_split=subjects_split,
        )
    
    def get_data(
        self, split: Split
    ):
        data = []
        for dataset in self.datasets:
                data.append(
                    dataset(
                        target_classes=self.classes,
                        subjects=self.subjects_split[dataset][split],
                    ).get_data()
                )

        X, y, meta = map(list, zip(*data))
        return X, y, meta

    def get_scoring(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
