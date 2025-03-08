from .labram_datasets import LaBraMAbnormalDataset, LaBraMBCIDataset, LaBraMPdDataset
import numpy as np
from typing import List, Tuple, Optional, cast
from resampy import resample
from mne.filter import filter_data, notch_filter
from mne.io import BaseRaw
from tqdm import tqdm
import gc
import os
import pickle
from multiprocessing import Pool
from sklearn.model_selection import train_test_split


def make_dataset(data: np.ndarray, labels: np.ndarray|None, sampling_rate: int, 
                 ch_names: List[str], target_rate: int = 200, target_channels: Optional[List[str]] = None,
                 l_freq: float = 0.1, h_freq: float = 75.0, train: bool = True) -> LaBraMAbnormalDataset:
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
    print("data shape: ", data.shape)
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

    return LaBraMBCIDataset(data, labels, target_rate, target_channels)


def split_and_dump_pd(params):
        EEG, label, id, dump_folder, target_channels, sampling_rate, target_rate, l_freq, h_freq, ch_names = params
        EEG = EEG.astype(np.float64)

        # filter out the channels that are not in the target_channels
        if target_channels is not None:
            ch_names = [ch.upper() for ch in ch_names]
            target_channels = [ch.upper() for ch in target_channels]
            EEG = EEG[[ch_names.index(ch) for ch in target_channels], :]
        else:
            target_channels = ch_names

        # bandpass filter
        EEG = filter_data(EEG, sfreq=sampling_rate, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
        # notch filter
        EEG = notch_filter(EEG, Fs=sampling_rate, freqs=50, verbose=False)
        # resample data
        EEG = resample(EEG, sampling_rate, target_rate, axis=-1, filter='kaiser_best')

        all_paths = []
        for i in range(EEG.shape[1] // 2000):
            dump_path = os.path.join(
                    dump_folder, id + "_" + str(i) + ".pkl"
                )
            all_paths.append(dump_path)
            if not label is None:
                pickle.dump(
                    {"X": EEG[:, i * 2000 : (i + 1) * 2000], "y": 0 if label == "parkinsons" else 1},
                    open(dump_path, "wb"),
                )
            else:
                pickle.dump(
                    {"X": EEG[:, i * 2000 : (i + 1) * 2000], "y": None},
                    open(dump_path, "wb"),
                )
        
        if label is None:
            return all_paths, EEG.shape[1] // 2000
        
        return all_paths


def make_dataset_pd(data: List[np.ndarray], labels: List[np.ndarray]|None, sampling_rate: int, 
                 ch_names: List[str], dataset_name: str, target_rate: int = 200, target_channels: Optional[List[str]] = None,
                 l_freq: float = 0.1, h_freq: float = 75.0, train: bool = True, val_per: float = 0.2) -> LaBraMPdDataset:
    
    root = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/data/pd/"
    dump_folder = os.path.join(root, dataset_name, "train" if train else "test")
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)
    else:
        files = os.listdir(dump_folder)
        if len(files) > 0:
            print(f"Dataset already exists in {dump_folder}")
            if val_per > 0 and train:
                files_train, files_val = train_test_split(files, test_size=val_per, random_state=42, shuffle=False)
                train_dataset = LaBraMPdDataset([os.path.join(dump_folder, f) for f in files_train], target_rate, ch_names, train)
                val_dataset = LaBraMPdDataset([os.path.join(dump_folder, f) for f in files_val], target_rate, ch_names, train)
                return train_dataset, val_dataset
            else:
                return LaBraMPdDataset([os.path.join(dump_folder, f) for f in files], target_rate, ch_names, train)

    parameters = [(data[i], labels[i] if not labels is None else None, str(i), dump_folder, ch_names, sampling_rate, target_rate, l_freq, h_freq, ch_names) for i in range(len(data))]

    with Pool(24) as pool:
        files = list(tqdm(pool.imap(split_and_dump_pd, parameters), total=len(parameters)))
    
    if labels is None:
        files, number_of_samples_list = zip(*files)
        print("Number of samples per file: ", number_of_samples_list)
    files = [f for sublist in files for f in sublist]

    if val_per > 0 and train:
        files_train, files_val = train_test_split(files, test_size=val_per, random_state=42, shuffle=False)
        train_dataset = LaBraMPdDataset(files, target_rate, ch_names, train)
        val_dataset = LaBraMPdDataset(files, target_rate, ch_names, train)
        return train_dataset, val_dataset
    else:
        return LaBraMPdDataset(files, target_rate, ch_names, train)


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

def split_and_dump_abnormal(params):
        raw, label, id, dump_folder, target_rate, l_freq, h_freq = params

        try:
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

            raw.load_data()
            raw.filter(l_freq=l_freq, h_freq=h_freq)
            raw.notch_filter(50.0)
            raw.resample(target_rate, n_jobs=1)

            channeled_data = cast(np.ndarray, raw.get_data(units='uV'))
        except:
            print(f"Error in {raw}")
            raw.close()
            return []
        all_paths = []
        for i in range(channeled_data.shape[1] // 2000):
            dump_path = os.path.join(
                dump_folder, id + "_" + str(i) + ".pkl"
            )
            all_paths.append(dump_path)
            if not label is None:
                pickle.dump(
                    {"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": 0 if label == "abnormal" else 1},
                    open(dump_path, "wb"),
                )
            else:
                pickle.dump(
                    {"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": None},
                    open(dump_path, "wb"),
                )
                break
        raw.close()
        return all_paths

def make_dataset_abnormal(data: List[BaseRaw], labels: List[str]|None, 
                          target_rate: int = 200, target_channels: Optional[List[str]] = None, 
                          l_freq: float = 0.1, h_freq: float = 75.0, train: bool = True, val_per: float = 0.2) -> LaBraMAbnormalDataset:
    
    root = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/data/tueg_abnormal/"
    dump_folder = os.path.join(root, "train" if train else "test")
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)
    else:
        files = os.listdir(dump_folder)
        if len(files) > 0:
            print(f"Dataset already exists in {dump_folder}")
            if val_per > 0 and train:
                files_train, files_val = train_test_split(files, test_size=val_per, random_state=42)
                train_dataset = LaBraMAbnormalDataset([os.path.join(dump_folder, f) for f in files_train], target_rate, chOrder_standard, train)
                val_dataset = LaBraMAbnormalDataset([os.path.join(dump_folder, f) for f in files_val], target_rate, chOrder_standard, train)
                return train_dataset, val_dataset
            else:
                return LaBraMAbnormalDataset([os.path.join(dump_folder, f) for f in files], target_rate, chOrder_standard, train)

    parameters = [(data[i], labels[i] if not labels is None else None, str(i), dump_folder, target_rate, l_freq, h_freq) for i in range(len(data))]

    with Pool(24) as pool:
        files = list(tqdm(pool.imap(split_and_dump_abnormal, parameters), total=len(parameters)))
    
    files = [f for sublist in files for f in sublist]
    
    if val_per > 0 and train:
        files_train, files_val = train_test_split(files, test_size=val_per, random_state=42)
        train_dataset = LaBraMAbnormalDataset(files, target_rate, chOrder_standard, train)
        val_dataset = LaBraMAbnormalDataset(files, target_rate, chOrder_standard, train)
        return train_dataset, val_dataset
    else:
        return LaBraMAbnormalDataset(files, target_rate, chOrder_standard, train)
