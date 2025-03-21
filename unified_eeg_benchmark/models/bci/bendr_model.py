from ..abstract_model import AbstractModel
from typing import List, Dict, cast, Literal
import numpy as np
from .BENDR.make_dataset import make_dataset
import argparse
from pathlib import Path
from .BENDR import utils
import torch
import os
import random
import numpy as np
from mne.io import BaseRaw
from scipy import stats
import torch.nn as nn
import torch
from tqdm import tqdm
import math
from .BENDR.dn3_ext import ConvEncoderBENDR
from joblib import Memory
from ...utils.config import get_config_value


def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(7)   


class BENDRBCIModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        encoder = ConvEncoderBENDR(20, encoder_h=512, dropout=0., projection_head=False)
        encoder.load("/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/models/BENDR/checkpoints/encoder.pt")

        self.model = encoder
        self.scale_param    = torch.nn.Parameter(torch.tensor(1.))
        self.linear_probe   = torch.nn.Linear(4608, num_classes)
        self.drop           = torch.nn.Dropout(p=0.10)
        self.loss_fn        = torch.nn.CrossEntropyLoss()

    def mixup_data(self, x, y, alpha=None):
        
        lam = torch.rand(1).to(x) if alpha is None else alpha
        lam = torch.max(lam, 1 - lam)

        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]

        return mixed_x, mixed_y

    def forward(self, x):
        x = torch.cat([x, self.scale_param.repeat((x.shape[0], 1, x.shape[-1]))], dim=-2)
        
        h = self.model(x)
        h = h.flatten(1)
        h = self.drop(h)

        pred = self.linear_probe(h)
        return pred

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    running_loss, running_corrects, total_samples = 0.0, 0, 0

    for x, y in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        soft_labels = y.float().to(device)
        x, y = model.mixup_data(x.to(device), soft_labels)
        logits = model(x)
        loss = model.loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        y = torch.argmax(y, dim=1)
        running_corrects += torch.sum(preds == y).item()
        total_samples += x.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validation"):
            logits = model.forward(x.to(device))
            y = y.to(device).argmax(dim=1)
            loss = model.loss_fn(logits, y)
            
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_corrects += torch.sum(preds == y).item()
            total_samples += x.size(0)
            
            all_labels.append(y.cpu())
            all_logits.append(logits.cpu())

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    # Concatenate predictions and labels
    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    
    # Compute additional metrics using get_metrics
    metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
    results = utils.get_metrics(all_logits.numpy(), all_labels.numpy(), metrics, False)
    
    return epoch_loss, epoch_acc, results


def inference(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Testing"):
            logits = model.forward(x.to(device))
            pred = torch.argmax(logits, dim=-1)
            predictions.append(pred.cpu())
    predictions = torch.cat(predictions, dim=0)
    return predictions


class BENDRModel(AbstractModel):
    def __init__(
        self,
    ):
        super().__init__("BENDR Model")
        print("inside init")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BENDRBCIModel(num_classes=2).to(self.device)
        self.cache = Memory(location=get_config_value("cache"), verbose=0)

    def fit(self, X: List[np.ndarray|List[BaseRaw]], y: List[np.ndarray|List[str]], meta: List[Dict]) -> None:
        print("inside fit")
        task_name = meta[0]["task_name"]
        datasets = [self.cache.cache(make_dataset)(X_, y_, task_name, meta_["sampling_frequency"], meta_["channel_names"], train=True) for X_, y_, meta_ in zip(cast(List[np.ndarray], X), cast(List[np.ndarray], y), meta)]
        dataset_train_list = [dataset[0] for dataset in datasets]
        dataset_val_list = [dataset[1] for dataset in datasets]
        del X, y, meta

        dataset_train_list = [dataset for dataset in dataset_train_list if len(dataset) > 0]        
        ch_names_list_train = [dataset.ch_names for dataset in dataset_train_list]
        if dataset_val_list is not None:
            dataset_val_list = [dataset for dataset in dataset_val_list if len(dataset) > 0]
            ch_names_list_val = [dataset.ch_names for dataset in dataset_val_list]



        torch.cuda.empty_cache()

        batch_size = 64
        train_loader_list = [torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True) for train_dataset in dataset_train_list]
        valid_loader_list = [torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False) for valid_dataset in dataset_val_list]
        
        max_epochs = 20
        steps_per_epoch = math.ceil(sum([len(train_loader) for train_loader in train_loader_list]))
        max_lr = 4e-4
        
        
        # Set up optimizer and OneCycleLR scheduler
        optimizer = torch.optim.AdamW(
            list([self.model.scale_param])+
            list(self.model.model.parameters())+
            list(self.model.linear_probe.parameters()),
            lr=1e-5,
            weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs)

        #best_val_loss = float('inf')
        #best_model_state = None
        
        # Training loop
        for epoch in range(1, max_epochs + 1):
            print(f"Epoch {epoch}/{max_epochs}")
            for train_loader, ch_names in zip(train_loader_list, ch_names_list_train):
                train_loss, train_acc = train_epoch(self.model, train_loader, optimizer, scheduler, self.device)
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            
            for valid_loader, ch_names in zip(valid_loader_list, ch_names_list_val):
                val_loss, val_acc, val_metrics = validate_epoch(self.model, valid_loader, self.device)
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                print("  Val Metrics:", val_metrics)
            
            # Optionally save the best model based on validation loss
            #if val_loss < best_val_loss:
            #    best_val_loss = val_loss
            #    best_model_state = self.model.state_dict()
        
        # Load the best model (if saved)
        #if best_model_state is not None:
        #    self.model.load_state_dict(best_model_state)
    

    @torch.no_grad()
    def predict(self, X: List[np.ndarray|List[BaseRaw]], meta: List[Dict]) -> np.ndarray:
        print("inside predict")
        task_name = meta[0]["task_name"]
        dataset_test_list = [self.cache.cache(make_dataset)(X_, None, task_name, meta_["sampling_frequency"], meta_["channel_names"], train=False) for X_, meta_ in zip(cast(List[np.ndarray], X), meta)]
        dataset_test_list = [dataset for dataset in dataset_test_list if len(dataset) > 0]
        print("datasets length: ", len(dataset_test_list[0]))
        ch_names_list = [dataset.ch_names for dataset in dataset_test_list]
        # Inference on test set

        batch_size = 64
        test_loader_list = [torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False) for test_dataset in dataset_test_list]

        predictions = []
        for test_loader, ch_names in zip(test_loader_list, ch_names_list):
            predictions.append(inference(self.model, test_loader, self.device).cpu())
        
        predictions = torch.cat(predictions, dim=0).numpy()
        print(predictions.shape)

        reverse_label_mapping = {0: 'left_hand', 1: 'right_hand', 2: 'feet', 3: 'tongue'}
        mapped_pred = np.array([reverse_label_mapping[idx] for idx in predictions])
        
        print(mapped_pred.shape)
        return mapped_pred
        
