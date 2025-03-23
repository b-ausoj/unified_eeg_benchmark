from .labram_datasets import LaBraMAbnormalDataset, LaBraMClinicalDataset
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
from joblib import Memory
from ....utils.config import get_config_value


standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9H', 'TTP7H', 'TPP9H', 'FTT10H', 'TPP8H', 'TPP10H', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

def make_dataset(data: List[np.ndarray], labels: np.ndarray|None, task_name: str, sampling_rate: int, 
                 ch_names: List[str], target_rate: int = 200, target_channels: Optional[List[str]] = None,
                 l_freq: float = 0.1, h_freq: float = 75.0, train: bool = True) -> LaBraMAbnormalDataset:
    """
    data: List[np.ndarray], shape=(n_trials, n_channels, n_samples)
    labels: np.ndarray, shape=(n_trials,)
    ch_names: List[str], list of channel names
    target_channels: List[str], list of target channel names
    sampling_rate: int, sampling rate of the data
    target_rate: int, target sampling rate
    l_freq: int, low cut-off frequency
    h_freq: int, high cut-off frequency
    """
    print("\ndata len: ", len(data))
    if len(data) == 0:
        return LaBraMClinicalDataset(data, labels, sampling_rate, ch_names)
    # filter out the channels that are not in the target_channels
    if target_channels is not None:
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = [ch.upper() for ch in target_channels]
        data = [d[[ch_names.index(ch) for ch in target_channels], :] for d in data]
    else:
        # target_channels = ch_names
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = list(set([ch.upper() for ch in standard_1020]).intersection(set(ch_names)))
        data = [d[[ch_names.index(ch) for ch in target_channels], :] for d in data]

    # set to real floating (not float32)
    data = [d.astype(np.float64) for d in data]
    # bandpass filter
    data = [filter_data(d, sfreq=sampling_rate, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False) for d in data]
    # notch filter
    data = [notch_filter(d, Fs=sampling_rate, freqs=50, verbose=False) for d in data]
    # resample data
    data = [resample(d, sampling_rate, target_rate, axis=2, filter='kaiser_best') for d in data]
    # set to float32
    data = [d.astype(np.float32) for d in data]

    # One hot encode labels if they are not None
    if labels is not None:
        if task_name == "parkinsons_clinical":
            label_mapping = {'parkinsons': 0, 'no_parkinsons': 1}
        elif task_name == "schizophrenia_clinical":
            label_mapping = {'schizophrenia': 0, 'no_schizophrenia': 1}
        elif task_name == "depression_clinical":
            label_mapping = {'depression': 0, 'no_depression': 1}
        elif task_name == "mtbi_clinical":
            label_mapping = {True: 0, False: 1}
        elif task_name == "ocd_clinical":
            label_mapping = {'ocd': 0, 'no_ocd': 1}
        else:
            raise ValueError("Invalid task name")
        labels = np.vectorize(label_mapping.get)(labels)
        labels = np.eye(len(label_mapping))[labels]
        print("labels shape: ", labels.shape)
    return LaBraMClinicalDataset(data, labels, target_rate, target_channels)

def process_one_file(args):
    raw, target_rate, l_freq, h_freq = args

    # select the channels
    wanted_chs = sorted(['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6'])
    ch_name_pattern="EEG {}-REF"
    if ch_name_pattern is not None:
        chs = [ch_name_pattern.format(ch) for ch in wanted_chs]
    else:
        chs = wanted_chs
    raw = raw.reorder_channels(chs)
    # get the signals
    raw.load_data()
    raw.set_eeg_reference("average")
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw.notch_filter(50.0)
    raw.resample(target_rate, n_jobs=1)

    signals = raw.get_data(units="uV")
    return signals

def prepare_cached_data(X, target_rate, l_freq, h_freq, n_jobs):
    parameters = [(X[i], target_rate, l_freq, h_freq) for i in range(len(X))]
    with Pool(n_jobs) as pool:
        processed_signals = list(tqdm(pool.imap(process_one_file, parameters), total=len(parameters)))
    return processed_signals

def make_dataset_abnormal(data: List[BaseRaw], labels: List[str]|None, task_name: str,
                      target_rate: int = 200, target_channels: Optional[List[str]] = None, 
                      l_freq: float = 0.1, h_freq: float = 75.0, train: bool = True, val_per: float = 0.2) -> LaBraMClinicalDataset:
    
    target_channels = sorted(['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6'])
    
    cache = Memory(location=get_config_value("cache"), verbose=0)
    processed_signals = cache.cache(prepare_cached_data)(data, target_rate, l_freq, h_freq, n_jobs=24)
    gc.collect()

    # One hot encode labels if they are not None
    if labels is not None:
        if task_name == "abnormal_clinical":
            label_mapping = {'abnormal': 0, 'normal': 1}
        elif task_name == "epilepsy_clinical":
            label_mapping = {'epilepsy': 0, 'no_epilepsy': 1}
        else:
            raise ValueError("Invalid task name")
        labels = np.vectorize(label_mapping.get)(labels)
        labels = np.eye(len(label_mapping))[labels]
        print("labels shape: ", labels.shape)
        
    if val_per > 0 and train:
        processed_signals_train, processed_signals_val, labels_train, labels_val = train_test_split(processed_signals, labels, test_size=val_per, random_state=42)
        train_dataset = LaBraMClinicalDataset(processed_signals_train, labels_train, target_rate, target_channels)
        val_dataset = LaBraMClinicalDataset(processed_signals_val, labels_val, target_rate, target_channels)
        return train_dataset, val_dataset
    else:
        return LaBraMClinicalDataset(processed_signals, labels, target_rate, target_channels)




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
