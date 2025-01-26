from unified_eeg_benchmark.datasets.weibo2013 import Weibo2013Dataset
from unified_eeg_benchmark.datasets.bcicomp_iv_2a import BCICompIV2aDataset
from unified_eeg_benchmark.enums.split import Split
from unified_eeg_benchmark.tasks.left_hand_right_hand_task import LeftHandRightHandTask
from unified_eeg_benchmark.datasets.combined_dataset import CombinedDataset

datasets = LeftHandRightHandTask().get_dataset(Split.ALL, preload=True)

print(datasets[0][:][0].shape)
print(datasets[1][:][0].shape)

datasets = LeftHandRightHandTask().get_dataset(
    Split.ALL, channels=["C3", "Cz", "C4"], target_frequency=200, preload=True
)

combined = CombinedDataset(datasets)

print(datasets[0][:][0].shape)
print(datasets[1][:][0].shape)
