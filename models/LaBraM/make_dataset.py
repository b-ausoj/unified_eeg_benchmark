from .labram_bci_dataset import LaBraMBCIDataset
import numpy as np
from typing import List, Tuple, Optional, cast
from resampy import resample
from mne.filter import filter_data, notch_filter
from mne.io import BaseRaw
from tqdm import tqdm


def make_dataset(data: np.ndarray, labels: np.ndarray|None, sampling_rate: int, 
                 ch_names: List[str], target_rate: int = 200, target_channels: Optional[List[str]] = None,
                 l_freq: float = 0.1, h_freq: float = 75.0) -> LaBraMBCIDataset:
    """
    data: np.ndarray, shape=(n_trials, n_channels, n_samples)
    labels: np.ndarray, shape=(n_trials,)
    ch_names: List[str], list of channel names
    target_channels: List[str], list of target channel names
    sampling_rate: int, sampling rate of the data
    target_rate: int, target sampling rate
    l_freq: int, low cut-off frequency
    h_freq: int, high cut-off frequency
    """

    # filter out the channels that are not in the target_channels
    if target_channels is not None:
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = [ch.upper() for ch in target_channels]
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]
    else:
        target_channels = ch_names
    # bandpass filter
    data = filter_data(data, sfreq=sampling_rate, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
    # notch filter
    data = notch_filter(data, Fs=sampling_rate, freqs=50, verbose=False)
    # resample data
    data = resample(data, sampling_rate, target_rate, axis=2, filter='kaiser_best')
    
    # Crop data to have a whole number of seconds
    n_seconds = data.shape[2] // target_rate
    n_samples = n_seconds * target_rate
    data = data[:, :, :n_samples]

    # One hot encode labels if they are not None
    if labels is not None:
        unique_labels = np.unique(labels)
        unique_labels.sort()  # Ensure the labels are sorted
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        print(label_mapping)
        labels = np.vectorize(label_mapping.get)(labels)
        labels = np.eye(len(unique_labels))[labels]
        print(labels)

    return LaBraMBCIDataset(data, labels, target_rate, target_channels)


drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

standard_channels = [
    "EEG FP1-REF",
    "EEG F7-REF",
    "EEG T3-REF",
    "EEG T5-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F8-REF",
    "EEG T4-REF",
    "EEG T6-REF",
    "EEG O2-REF",
    "EEG FP1-REF",
    "EEG F3-REF",
    "EEG C3-REF",
    "EEG P3-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F4-REF",
    "EEG C4-REF",
    "EEG P4-REF",
    "EEG O2-REF",
]


def make_dataset_abnormal(data: List[BaseRaw], labels: List[str]|None, 
                          target_rate: int = 200, target_channels: Optional[List[str]] = None, 
                          l_freq: float = 0.1, h_freq: float = 75.0) -> LaBraMBCIDataset:
    """
    data: List[BaseRaw], list of raw data
    labels: List[str], list of labels
    target_rate: int, target sampling rate
    target_channels: List[str], list of target channel names
    l_freq: int, low cut-off frequency
    h_freq: int, high cut-off frequency
    """

    processed_data = []
    processed_labels = []
    for raw in tqdm(data):
        try:
            raw.load_data(verbose="error")
            if drop_channels is not None:
                useless_chs = []
                for ch in drop_channels:
                    if ch in raw.ch_names:
                        useless_chs.append(ch)
                raw.drop_channels(useless_chs)
            if chOrder_standard is not None and len(chOrder_standard) == len(raw.ch_names):
                raw.reorder_channels(chOrder_standard)
            if raw.ch_names != chOrder_standard:
                raise Exception("channel order is wrong!")

            raw.filter(l_freq=l_freq, h_freq=h_freq)
            raw.notch_filter(50.0)
            raw.resample(target_rate, n_jobs=5)

            ch_name = raw.ch_names
            raw_data = cast(np.ndarray, raw.get_data(units='uV'))
            channeled_data = raw_data.copy()
        except:
            print(f"Error in {raw}")
            continue
        finally:
            raw.close()
        for i in range(channeled_data.shape[1] // 2000):
                processed_data.append(channeled_data[:, i * 2000 : (i + 1) * 2000])
                processed_labels.append(labels)
    processed_data = np.array(processed_data)
    processed_labels = np.array(processed_labels)

    
    # One hot encode labels if they are not None
    if processed_labels is not None:
        unique_labels = np.unique(processed_labels)
        unique_labels.sort()  # Ensure the labels are sorted
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        print(label_mapping)
        processed_labels = np.vectorize(label_mapping.get)(processed_labels)
        processed_labels = np.eye(len(unique_labels))[processed_labels]
        print(processed_labels)

    return LaBraMBCIDataset(processed_data, processed_labels, target_rate, chOrder_standard)    
   
