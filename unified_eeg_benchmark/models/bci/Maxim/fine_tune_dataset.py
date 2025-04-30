from typing import List, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from ..LaBraM.utils_2 import map_label

class FinetuneDataset(Dataset):
    def __init__(
        self,
        X: List[np.ndarray],
        y : Optional[List[np.ndarray]],
        metas: List[Dict[str, Any]],
    ):
        self.data = []
        self.labels = []
        self.srs = []
        self.durs = []
        self.channels = []
        self.datasets = []

        if y is not None:
            for X_, y_, meta in zip(X, y, metas):
                assert len(X_) == len(y_), f"X and y must have the same length for dataset {meta['name']}"
                n_samples = len(X_)
                self.labels.extend([y_[i] for i in range(len(y_))])
        else:
            self.labels = None

        for X_, meta in zip(X, metas):
            n_samples = len(X_)
            self.data.extend([X_[i] for i in range(n_samples)])
            self.srs.extend([meta["sampling_frequency"] for _ in range(n_samples)])
            duration = X_.shape[-1] / meta["sampling_frequency"]
            self.durs.extend([duration for _ in range(n_samples)])
            self.channels.extend([meta["channel_names"] for _ in range(n_samples)])
            self.datasets.extend([meta["name"] for _ in range(n_samples)])
        
        self.task_name = metas[0]["task_name"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signals = self.data[idx]
        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor(map_label(label, self.task_name), dtype=torch.long)
        else:
            label = -1
        sr = self.srs[idx]
        dur = self.durs[idx]
        channels = self.channels[idx]
        dataset = self.datasets[idx]

        signals = torch.tensor(signals, dtype=torch.float32)
        
        return {
            "signals": signals,
            "output": label,
            "sr": sr,
            "dur": dur,
            "channels": channels,
            "dataset": dataset,
        }