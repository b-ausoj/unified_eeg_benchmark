from typing import List, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from ..LaBraM.utils_2 import map_label

class EmbeddingsDataset(Dataset):
    def __init__(
        self,
        embeddings: List[np.ndarray],
        labels: Optional[List[int]] = None,
    ):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.long)
            return embedding, label
        else:
            return embedding