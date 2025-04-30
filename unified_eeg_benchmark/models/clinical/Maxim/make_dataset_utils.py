from torch.utils.data import Dataset
from mne.filter import filter_data, notch_filter
from resampy import resample
import numpy as np
import h5py
import torch
import logging
from ..NeuroGPT.src.batcher.base import EEGDataset
from typing import Dict, List
import time


def writer_task(output_queue, h5_path):
    """
    Dedicated writer process that listens to the queue and writes data to the HDF5 file.
    """
    with h5py.File(h5_path, 'a') as hf:
        recordings_grp = hf.require_group('/recordings')
        while True:
            message = output_queue.get()
            if message is None:
                # Sentinel received: all work is done
                break
            idx, signals, label, sfreq, channels = message
            logging.info(f"Writing recording {idx} with label {label}")
            recording_grp = recordings_grp.create_group(f'recording_{idx:04d}')
            recording_grp.create_dataset('data', data=signals)
            if label is not None:
                recording_grp.create_dataset('label', data=label)
            recording_grp.create_dataset('sfreq', data=sfreq)
            dt = h5py.string_dtype(encoding='utf-8')
            recording_grp.create_dataset('channels', data=np.array(channels, dtype=object), dtype=dt)
            logging.info(f"Finished writing recording {idx}")
            
    print("[Writer] All recordings have been written.")

def process_one_abnormal(parameters, output_queue):
    """
    Preprocess a single recording.
    Instead of writing directly to disk, send the processed result to the output_queue.
    """
    idx, raw, label, model_name = parameters
    if label is not None:
        label = map_label(label)

    if model_name == "MaximsModel":
        t_channels = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']
        ch_name_pattern="EEG {}-REF"
        chs = [ch_name_pattern.format(ch) for ch in t_channels]
        raw = raw.reorder_channels(chs)
        # Limit the raw data to a maximum of 30 minutes
        max_duration_s = 20 * 60  # 20 minutes in seconds
        if raw.times[-1] > max_duration_s:
            raw.crop(tmax=max_duration_s)
        raw.load_data()
        raw.set_eeg_reference("average")
        signals = raw.get_data(units="uV")
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Send the processed data to the writer process
    time.sleep(1)  # Give the writer process some time to start
    output_queue.put((idx, signals, label, raw.info["sfreq"], raw.ch_names))
    logging.info(f"Processed recording {idx} with label {label}")
    return

def process_one_epilepsy(parameters, output_queue):
    """
    Preprocess a single recording.
    Instead of writing directly to disk, send the processed result to the output_queue.
    """
    idx, raw, label, montage, model_name = parameters

    if label is not None:
        label = map_label(label)

    if model_name == "MaximsModel":
        t_channels = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']
        if "le" in montage:
            ch_name_pattern="EEG {}-LE"
        else:
            ch_name_pattern="EEG {}-REF"
        chs = [ch_name_pattern.format(ch) for ch in t_channels]
        raw = raw.reorder_channels(chs)
        # Limit the raw data to a maximum of 30 minutes
        max_duration_s = 20 * 60  # 20 minutes in seconds
        if raw.times[-1] > max_duration_s:
            raw.crop(tmax=max_duration_s)
        raw.load_data()
        raw.set_eeg_reference("average")
        signals = raw.get_data(units="uV")
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Send the processed data to the writer process
    time.sleep(1)  # Give the writer process some time to start
    output_queue.put((idx, signals, label, raw.info["sfreq"], raw.ch_names))
    logging.info(f"Processed recording {idx} with label {label}")
    return

def calc_class_weights(labels: List[np.ndarray]) -> List[float]:
    """
    Calculate class weights for the given labels.
    Args:
        labels (List[np.ndarray]): List of numpy arrays containing the labels.
    Returns:
        List[float]: List of weights for each class.
    """
    # Flatten the list of labels
    all_labels = np.concatenate(labels)

    # Map labels to integers
    all_labels = np.array([map_label(label) for label in all_labels])
    
    # Count the occurrences of each class
    class_counts = np.bincount(all_labels)
    
    # Calculate the total number of samples
    total_samples = len(all_labels)
    
    # Calculate class weights for each class (0 weight if class count is 0)
    n_classes = len(class_counts)
    class_weights = [np.float32(total_samples / (n_classes * count)) if count > 0 else np.float32(0.0) for count in class_counts]
    
    return class_weights

def map_label(label: str) -> int:
    """
    Map the label to a numerical value.
    Args:
        label (str): The label to map.
    Returns:
        int: The mapped numerical value.
    """
    if label is not None:
        if label == "abnormal":
            return 0
        elif label == "normal":
            return 1
        elif label == "epilepsy":
            return 0
        elif label == "no_epilepsy":
            return 1
        elif label == "parkinsons":
            return 0
        elif label == "no_parkinsons":
            return 1
        elif label == "schizophrenia":
            return 0
        elif label == "no_schizophrenia":
            return 1
        elif label == "depression":
            return 0
        elif label == "no_depression":
            return 1
        elif label == "ocd":
            return 0
        elif label == "no_ocd":
            return 1
        elif label == True:
            return 0
        elif label == False:
            return 1
        else:
            raise ValueError("Invalid label: ", label)
    else:
        raise ValueError("Label is None")
        
def map_label_reverse(label: int, task_name: str) -> str:
    """
    Map the label back to its original string value.
    Args:
        label (int): The label to map.
        task_name (str): The name of the task.
    Returns:
        str: The mapped string value.
    """
    if task_name == "abnormal_clinical":
        return "abnormal" if label == 0 else "normal"
    elif task_name == "epilepsy_clinical":
        return "epilepsy" if label == 0 else "no_epilepsy"
    elif task_name == "parkinsons_clinical":
        return "parkinsons" if label == 0 else "no_parkinsons"
    elif task_name == "schizophrenia_clinical":
        return "schizophrenia" if label == 0 else "no_schizophrenia"
    elif task_name == "depression_clinical":
        return "depression" if label == 0 else "no_depression"
    elif task_name == "ocd_clinical":
        return "ocd" if label == 0 else "no_ocd"
    elif task_name == "mtbi_clinical":
        return label == 0
    else:
        raise ValueError("Invalid task name ", task_name)