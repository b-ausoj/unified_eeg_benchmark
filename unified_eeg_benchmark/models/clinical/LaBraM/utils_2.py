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


channel_mapping = { "FP1": ["FP1", "FZ"],
                    "FP2": ["FP2", "FZ"],
                    "F7": ["F7", "FC3"],
                    "F3": ["F3", "FC1"],
                    "FZ": ["FZ", "FCZ"],
                    "F4": ["F4", "FC2"],
                    "F8": ["F8", "FC4"],
                    "T7": ["T7", "FT7", "TP7", "T3", "C5"],
                    "C3": ["C3"],
                    "CZ": ["CZ"],
                    "C4": ["C4"],
                    "T8": ["T8", "FT8", "TP8", "T4", "C6"],
                    "P7": ["P7", "CP3", "T5"],
                    "P3": ["P3", "CP1"],
                    "PZ": ["PZ", "CPZ"],
                    "P4": ["P4", "CP2"],
                    "P8": ["P8", "CP4", "T6"],
                    "O1": ["O1", "P1", "POZ"],
                    "O2": ["O2", "P2", "POZ"]}

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

class LaBraMDataset2(Dataset):
    def __init__(self, h5_path, is_train_set, channels, recording_names=None):
        """
        Args:
            h5_path (string): Path to the HDF5 file.
            is_train_set (bool): Whether this is the training set.
            channels (list): List of channel names.
            recording_names (list, optional): Pre-selected list of recording names.
        """
        self.h5_path = h5_path
        self.is_train_set = is_train_set
        self.ch_names = channels

        if recording_names is None:
            # Get list of all recording names from the HDF5 file
            with h5py.File(h5_path, 'r') as hf:
                self.recording_names = sorted(list(hf['/recordings'].keys()))
        else:
            self.recording_names = recording_names

    def __len__(self):
        return len(self.recording_names)
    
    def __getitem__(self, idx):
        rec_name = self.recording_names[idx]
        
        with h5py.File(self.h5_path, 'r') as hf:
            recording_grp = hf[f'/recordings/{rec_name}']
            data = recording_grp['data'][:]
            label = recording_grp['label'][()] if 'label' in recording_grp else None
        if self.is_train_set:
            label = np.eye(2)[label]
        
        if self.is_train_set:        
            # If the recording is longer than 128 seconds (24000 samples at 200Hz),
            # select a random contiguous subsample of 320 seconds
            required_length = 128 * 200  # 24000 samples
            if data.shape[-1] > required_length:
                max_start = data.shape[-1] - required_length
                start = np.random.randint(0, max_start + 1)
                data = data[..., start:start+required_length]
        
        # Convert to torch tensor
        data = torch.from_numpy(data).float()
        
        return data, label  # Data, label, metadata for train
    
    def split_train_val(self, val_split=0.1):
        """
        Split the dataset into training and validation sets.
        Args:
            val_split (float): Fraction of the dataset to use for validation.
        Returns:
            Tuple[LaBraMDataset2, LaBraMDataset2]: Training and validation dataset instances.
        """
        n = len(self.recording_names)
        indices = list(range(n))
        np.random.shuffle(indices)
        
        split = int(np.floor(val_split * n))
        val_indices = indices[:split]
        train_indices = indices[split:]
        
        train_recordings = [self.recording_names[i] for i in train_indices]
        val_recordings = [self.recording_names[i] for i in val_indices]
        
        train_dataset = LaBraMDataset2(self.h5_path, True, self.ch_names, recording_names=train_recordings)
        val_dataset = LaBraMDataset2(self.h5_path, True, self.ch_names, recording_names=val_recordings)
        
        return train_dataset, val_dataset

class NeuroGPTDataset2(EEGDataset):
    def __init__(self, h5_path, is_train_set, channels, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True, recording_names=None):
        super().__init__([], sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.h5_path = h5_path
        self.is_train_set = is_train_set
        self.ch_names = channels

        # Get list of all recording names
        with h5py.File(h5_path, 'r') as hf:
            self.recording_names = sorted(list(hf['/recordings'].keys()))

    def __len__(self):
        return len(self.recording_names)
    
    def __getitem__(self, idx):
        rec_name = self.recording_names[idx]
        
        with h5py.File(self.h5_path, 'r') as hf:
            recording_grp = hf[f'/recordings/{rec_name}']
            data = recording_grp['data'][:]
            label = recording_grp['label'][()] if 'label' in recording_grp else None
            # if self.is_train_set:
            #     label = np.eye(2)[label]
                    
            # Convert to torch tensor
            #data = torch.from_numpy(data).float()
            if not self.is_train_set:
                return self.preprocess_sample(data, self.num_chunks, None)
            else:
                return self.preprocess_sample(data, self.num_chunks, label)
        
    def split_train_val(self, val_split=0.1):
        """
        Split the dataset into training and validation sets.
        Args:
            val_split (float): Fraction of the dataset to use for validation.
        Returns:
            Tuple[LaBraMDataset2, LaBraMDataset2]: Training and validation dataset instances.
        """
        n = len(self.recording_names)
        indices = list(range(n))
        np.random.shuffle(indices)
        
        split = int(np.floor(val_split * n))
        val_indices = indices[:split]
        train_indices = indices[split:]
        
        train_recordings = [self.recording_names[i] for i in train_indices]
        val_recordings = [self.recording_names[i] for i in val_indices]
        
        train_dataset = NeuroGPTDataset2(self.h5_path, True, self.ch_names, self.sample_keys, chunk_len=self.chunk_len, num_chunks=self.num_chunks, ovlp=self.ovlp, root_path="", gpt_only=self.gpt_only, recording_names=train_recordings)
        val_dataset = NeuroGPTDataset2(self.h5_path, True, self.ch_names, self.sample_keys, chunk_len=self.chunk_len, num_chunks=self.num_chunks, ovlp=self.ovlp, root_path="", gpt_only=self.gpt_only, recording_names=val_recordings)
        
        return train_dataset, val_dataset
    

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
            idx, signals, label, chunk_len_s = message
            if chunk_len_s is None:
                logging.info(f"Writing recording {idx} with label {label}")
                recording_grp = recordings_grp.create_group(f'recording_{idx:04d}')
                recording_grp.create_dataset('data', data=signals)
                if label is not None:
                    recording_grp.create_dataset('label', data=label)
                else:
                    recording_grp.create_dataset('label', data=idx)
                logging.info(f"Finished writing recording {idx}")
            else:
                # Split the signals into chunks
                chunk_len = int(chunk_len_s * 200)
                n_chunks = signals.shape[1] // chunk_len
                signals = signals[:, :n_chunks * chunk_len].reshape(signals.shape[0], n_chunks, chunk_len)
                for i in range(n_chunks):
                    recording_grp = recordings_grp.create_group(f'recording_{idx:04d}_{i:03d}')
                    recording_grp.create_dataset('data', data=signals[:, i, :])
                    if label is not None:
                        recording_grp.create_dataset('label', data=label)
                    else:
                        recording_grp.create_dataset('label', data=idx)
    print("[Writer] All recordings have been written.")

def process_one_abnormal(parameters, output_queue):
    """
    Preprocess a single recording.
    Instead of writing directly to disk, send the processed result to the output_queue.
    """
    idx, raw, label, model_name, chunk_len_s = parameters
    l_freq: float = 0.1
    h_freq: float = 75.0

    if label is not None:
        label = map_label(label)

    if model_name == "LaBraMModel":
        t_channels = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']
        t_channels = list(set(standard_1020).intersection(set(t_channels)))
        ch_name_pattern="EEG {}-REF"
        chs = [ch_name_pattern.format(ch) for ch in t_channels]
        raw = raw.reorder_channels(chs)
        # Limit the raw data to a maximum of 30 minutes
        max_duration_s = 30 * 60  # 30 minutes in seconds
        if raw.times[-1] > max_duration_s:
            raw.crop(tmax=max_duration_s)
        raw.load_data()
        raw.set_eeg_reference("average")
        raw.filter(l_freq=l_freq, h_freq=h_freq)
        raw.notch_filter(50.0)
        raw.resample(200)
        signals = raw.get_data(units="uV")
    elif model_name == "NeuroGPTModel":
        # Limit the raw data to a maximum of 30 minutes
        max_duration_s = 30 * 60  # 30 minutes in seconds
        if raw.times[-1] > max_duration_s:
            raw.crop(tmax=max_duration_s)
        raw.load_data()
        raw.set_eeg_reference("average")
        raw.filter(l_freq=l_freq, h_freq=h_freq)
        raw.notch_filter(50.0)
        raw.resample(250)
        signals = raw.get_data(units="uV")

        required_channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T2', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2']
        ch_names = raw.info["ch_names"]
        ch_names = [ch.upper()[4:].split('-')[0] for ch in ch_names]

        channel_indices = []
        for ch in required_channels:
            if ch in ch_names:
                channel_indices.append(ch_names.index(ch))
            else:
                channel_indices.append(None)

        trial_data = []
        for ch_i, ch in zip(channel_indices, required_channels):
            if ch_i is not None:
                trial_data.append(signals[ch_i, :])  # Select the data for that channel
            else:
                trial_data.append(np.zeros(signals.shape[1]))  # Shape (n_timepoints)

        signals = np.array(trial_data)
    elif model_name == "BENDRModel":
        # Limit the raw data to a maximum of 30 minutes
        max_duration_s = 30 * 60  # 30 minutes in seconds
        if raw.times[-1] > max_duration_s:
            raw.crop(tmax=max_duration_s)
        raw.load_data()
        raw.set_eeg_reference("average")
        raw.filter(l_freq=l_freq, h_freq=h_freq)
        raw.notch_filter(50.0)
        raw.resample(200)
        signals = raw.get_data(units="uV")
        ch_names = raw.info["ch_names"]

        reorder_channels = []
        new_ch_names = []
        ch_names = [ch.upper()[4:].split('-')[0] for ch in ch_names]
        #print("ch_names: ", ch_names)
        for key, value in channel_mapping.items():
            if key in ch_names:
                reorder_channels.append(ch_names.index(key))
                new_ch_names.append(key)
            else:
                found = False
                for v in value:
                    if v in ch_names:
                        reorder_channels.append(ch_names.index(v))
                        new_ch_names.append(v)
                        found = True
                        break
                if not found:
                    reorder_channels.append(len(ch_names))
                    new_ch_names.append("0")
                    signals = np.insert(signals, len(ch_names), 0, axis=1)
                    print(f"Channel {key} not found")
                
        signals = signals[reorder_channels, :]
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Send the processed data to the writer process
    time.sleep(1)  # Give the writer process some time to start
    output_queue.put((idx, signals, label, chunk_len_s))
    logging.info(f"Processed recording {idx} with label {label}")
    return

def process_one_epilepsy(parameters, output_queue):
    """
    Preprocess a single recording.
    Instead of writing directly to disk, send the processed result to the output_queue.
    """
    idx, raw, label, montage, model_name, chunk_len_s = parameters
    l_freq: float = 0.1
    h_freq: float = 75.0

    if label is not None:
        label = map_label(label)

    if model_name == "LaBraMModel":
        t_channels = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']
        t_channels = list(set(standard_1020).intersection(set(t_channels)))
        if "le" in montage:
            ch_name_pattern="EEG {}-LE"
        else:
            ch_name_pattern="EEG {}-REF"
        chs = [ch_name_pattern.format(ch) for ch in t_channels]
        raw = raw.reorder_channels(chs)
        # Limit the raw data to a maximum of 30 minutes
        max_duration_s = 30 * 60  # 30 minutes in seconds
        if raw.times[-1] > max_duration_s:
            raw.crop(tmax=max_duration_s)
        raw.load_data()
        raw.set_eeg_reference("average")
        raw.filter(l_freq=l_freq, h_freq=h_freq)
        raw.notch_filter(50.0)
        raw.resample(200)
        signals = raw.get_data(units="uV")
    elif model_name == "NeuroGPTModel":
        # Limit the raw data to a maximum of 30 minutes
        max_duration_s = 30 * 60  # 30 minutes in seconds
        if raw.times[-1] > max_duration_s:
            raw.crop(tmax=max_duration_s)
        raw.load_data()
        raw.set_eeg_reference("average")
        raw.filter(l_freq=l_freq, h_freq=h_freq)
        raw.notch_filter(50.0)
        raw.resample(250)
        signals = raw.get_data(units="uV")

        required_channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T2', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2']
        ch_names = raw.info["ch_names"]
        ch_names = [ch.upper()[4:].split('-')[0] for ch in ch_names]

        channel_indices = []
        for ch in required_channels:
            if ch in ch_names:
                channel_indices.append(ch_names.index(ch))
            else:
                channel_indices.append(None)

        trial_data = []
        for ch_i, ch in zip(channel_indices, required_channels):
            if ch_i is not None:
                trial_data.append(signals[ch_i, :])  # Select the data for that channel
            else:
                trial_data.append(np.zeros(signals.shape[1]))  # Shape (n_timepoints)

        signals = np.array(trial_data)
    elif model_name == "BENDRModel":
        # Limit the raw data to a maximum of 30 minutes
        max_duration_s = 30 * 60  # 30 minutes in seconds
        if raw.times[-1] > max_duration_s:
            raw.crop(tmax=max_duration_s)
        raw.load_data()
        raw.set_eeg_reference("average")
        raw.filter(l_freq=l_freq, h_freq=h_freq)
        raw.notch_filter(50.0)
        raw.resample(200)
        signals = raw.get_data(units="uV")
        ch_names = raw.info["ch_names"]

        reorder_channels = []
        new_ch_names = []
        ch_names = [ch.upper()[4:].split('-')[0] for ch in ch_names]
        #print("ch_names: ", ch_names)
        for key, value in channel_mapping.items():
            if key in ch_names:
                reorder_channels.append(ch_names.index(key))
                new_ch_names.append(key)
            else:
                found = False
                for v in value:
                    if v in ch_names:
                        reorder_channels.append(ch_names.index(v))
                        new_ch_names.append(v)
                        found = True
                        break
                if not found:
                    reorder_channels.append(len(ch_names))
                    new_ch_names.append("0")
                    signals = np.insert(signals, len(ch_names), 0, axis=1)
                    print(f"Channel {key} not found")
                
        signals = signals[reorder_channels, :]
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Send the processed data to the writer process
    time.sleep(1)  # Give the writer process some time to start
    output_queue.put((idx, signals, label, chunk_len_s))
    logging.info(f"Processed recording {idx} with label {label}")
    return

def process_one_cli_unm(parameters, output_queue):
    """
    Preprocess a single recording.
    Instead of writing directly to disk, send the processed result to the output_queue.
    """

    idx, signals, label, o_channels, sfreq, model_name, task_name, chunk_len_s = parameters
    l_freq: float = 0.1
    h_freq: float = 75.0

    if label is not None:
        label = map_label(label)

    if model_name == "LaBraMModel":
        ch_names = [ch.upper() for ch in o_channels]
        #target_channels = list(set(ch_names).intersection(set([ch.upper() for ch in standard_1020])))
        t_channels = get_channels(task_name)
        t_channels = [c.upper() for c in t_channels]
        target_channels = list(set(ch_names).intersection(set(t_channels)))
        #target_channels = list(set(['P8', 'C2', 'PO8', 'PO7', 'P6', 'P4', 'CP1', 'FT7', 'Fz', 'Fp2', 'F2', 'Cz', 'C4', 'Fp1', 'P7', 'C5', 'TP7', 'P2', 'CP5', 'P1', 'F5', 'C3', 'FC6', 'FC1', 'C1', 'FC5', 'F1', 'FC3', 'O1', 'AF8', 'T7', 'CP2', 'O2', 'FCz', 'AF4', 'F6', 'F8', 'F4', 'CP4', 'CP6', 'P3', 'AFz', 'Oz', 'T8', 'C6', 'FC2', 'CP3', 'FC4', 'POz', 'FT8', 'TP8', 'AF3', 'AF7', 'P5', 'F3', 'F7']).intersection(set([ch.upper() for ch in standard_1020])))
        target_channels = sorted(target_channels)
        #logging.info(f"Target channels: {target_channels}")
        signals = signals[[ch_names.index(ch) for ch in target_channels], :]
        # Limit the raw data to a maximum of 30 minutes
        max_duration_s = 30 * 60  # 30 minutes in seconds
        if signals.shape[1] > max_duration_s * sfreq:
            signals = signals[:, :max_duration_s * sfreq]

        # set eeg reference to average
        #signals = signals - np.mean(signals, axis=0, keepdims=True)
        # bandpass filter
        signals = filter_data(signals.astype(np.float64), sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
        # notch filter
        signals = notch_filter(signals, Fs=sfreq, freqs=50, verbose=False)
        # resample data
        signals = resample(signals.astype(np.float32), sfreq, 200, axis=1, filter='kaiser_best')
    elif model_name == "NeuroGPTModel":
        required_channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T2', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2']
        ch_names = [ch.upper() for ch in o_channels]

        channel_indices = []
        for ch in required_channels:
            if ch in ch_names:
                channel_indices.append(ch_names.index(ch))
            else:
                channel_indices.append(None)

        trial_data = []
        for ch_i, ch in zip(channel_indices, required_channels):
            if ch_i is not None:
                trial_data.append(signals[ch_i, :])  # Select the data for that channel
            else:
                trial_data.append(np.zeros(signals.shape[1]))  # Shape (n_timepoints)

        signals = np.array(trial_data)
        
        # Limit the raw data to a maximum of 30 minutes
        max_duration_s = 30 * 60  # 30 minutes in seconds
        if signals.shape[1] > max_duration_s * sfreq:
            signals = signals[:, :max_duration_s * sfreq]

        # set eeg reference to average
        signals = signals - np.mean(signals, axis=0, keepdims=True)
        # bandpass filter
        signals = filter_data(signals.astype(np.float64), sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
        # notch filter
        signals = notch_filter(signals, Fs=sfreq, freqs=50, verbose=False)
        # resample data
        signals = resample(signals.astype(np.float32), sfreq, 250, axis=1, filter='kaiser_best')
    elif model_name == "BENDRModel":
        reorder_channels = []
        new_ch_names = []
        ch_names = [ch.upper() for ch in o_channels]
        for key, value in channel_mapping.items():
            if key in ch_names:
                reorder_channels.append(ch_names.index(key))
                new_ch_names.append(key)
            else:
                found = False
                for v in value:
                    if v in ch_names:
                        reorder_channels.append(ch_names.index(v))
                        new_ch_names.append(v)
                        found = True
                        break
                if not found:
                    reorder_channels.append(len(ch_names))
                    new_ch_names.append("0")
                    signals = np.insert(signals, len(ch_names), 0, axis=1)
                    print(f"Channel {key} not found")
        target_channels = new_ch_names
        signals = signals[reorder_channels, :]

        # Limit the raw data to a maximum of 30 minutes
        max_duration_s = 30 * 60  # 30 minutes in seconds
        if signals.shape[1] > max_duration_s * sfreq:
            signals = signals[:, :max_duration_s * sfreq]

        # set eeg reference to average
        #signals = signals - np.mean(signals, axis=0, keepdims=True)
        # bandpass filter
        signals = filter_data(signals.astype(np.float64), sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
        # notch filter
        signals = notch_filter(signals, Fs=sfreq, freqs=50, verbose=False)
        # resample data
        signals = resample(signals.astype(np.float32), sfreq, 200, axis=1, filter='kaiser_best')
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Send the processed data to the writer process
    output_queue.put((idx, signals, label, chunk_len_s))
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

def get_channels(task_name):
    if task_name == "parkinsons_clinical":
        return ['P8', 'C2', 'PO8', 'PO7', 'P6', 'P4', 'CP1', 'FT7', 'Fz', 'Fp2', 'F2', 'Cz', 'C4', 'Fp1', 'P7', 'C5', 'TP7', 'P2', 'CP5', 'P1', 'F5', 'C3', 'FC6', 'FC1', 'C1', 'FC5', 'F1', 'FC3', 'O1', 'AF8', 'T7', 'CP2', 'O2', 'FCz', 'AF4', 'F6', 'F8', 'F4', 'CP4', 'CP6', 'P3', 'AFz', 'Oz', 'T8', 'C6', 'FC2', 'CP3', 'FC4', 'POz', 'FT8', 'TP8', 'AF3', 'AF7', 'P5', 'F3', 'F7']
    elif task_name == "abnormal_clinical":
        return ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']
    elif task_name == "epilepsy_clinical":
        return ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']
    elif task_name == "schizophrenia_clinical":
        return ['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'F6', 'F2', 'AF4', 'AF8']
    elif task_name == "mtbi_clinical":
        return ['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8', 'CPz']
    elif task_name == "ocd_clinical":
        return ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
    else:
        return ['AF3', 'AF4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CZ', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCZ', 'FZ', 'O1', 'O2', 'OZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO7', 'PO8', 'POZ', 'T7', 'T8', 'TP7', 'TP8']

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