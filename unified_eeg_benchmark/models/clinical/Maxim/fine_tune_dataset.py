import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from resampy import resample

class FinetuneDataset(Dataset):
    def __init__(self, h5_path: str, is_test_set: bool, recording_names=None):
        """
        Args:
            h5_path (string): Path to the HDF5 file.
            is_testing (bool): Whether this is the testing set.
            channels (list): List of channel names.
            recording_names (list, optional): Pre-selected list of recording names.
        """
        self.h5_path = h5_path
        self.is_test_set = is_test_set
        print("Inside FinetuneDataset")
        if recording_names is None:
            # Get list of all recording names from the HDF5 file
            with h5py.File(h5_path, 'r') as hf:
                self.recording_names = sorted(list(hf['/recordings'].keys()))
        else:
            self.recording_names = recording_names

    def __len__(self):
        return len(self.recording_names)
    
    def __getitem__(self, idx: int) -> dict:
        rec_name = self.recording_names[idx]
        
        with h5py.File(self.h5_path, 'r') as hf:
            recording_grp = hf[f'/recordings/{rec_name}']
            data = recording_grp['data'][:]
            sfreq = recording_grp['sfreq'][()]
            label = -1
            channels = [ch.decode('utf-8') for ch in recording_grp['channels'][:]]
        
            if not self.is_test_set and 'label' in recording_grp:
                label = recording_grp['label'][()]

        if sfreq > 300:
            data = resample(data, sfreq, 250, axis=-1)
            sfreq = 250
        
        signals = torch.from_numpy(data).float()
        label = torch.tensor(label).long()
        duration = signals.shape[-1] // sfreq
        
        return {
            "signals": signals,
            "output": label,
            "sr": sfreq,
            "dur": duration,
            "channels": channels,
            "dataset": "TUAB",
        }
    
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
        
        train_dataset = FinetuneDataset(self.h5_path, self.is_test_set, recording_names=train_recordings)
        val_dataset = FinetuneDataset(self.h5_path, self.is_test_set, recording_names=val_recordings)
        
        return train_dataset, val_dataset