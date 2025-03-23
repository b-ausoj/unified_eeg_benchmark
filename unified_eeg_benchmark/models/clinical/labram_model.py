from ..abstract_model import AbstractModel
from typing import List, Dict, cast, Literal
import numpy as np
from .LaBraM.make_dataset import make_dataset, make_dataset_abnormal
import argparse
import logging
import socket
from pathlib import Path
from .LaBraM import utils
import torch
from timm.models import create_model
import torch.backends.cudnn as cudnn
from timm.utils import ModelEma
from timm.utils import NativeScaler
from collections import OrderedDict
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.utils.data.distributed import DistributedSampler
import os
import random
import numpy as np
import time
from datetime import datetime
import json
from .LaBraM.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from .LaBraM.engine_for_finetuning import train_one_epoch, evaluate
from einops import rearrange
from mne.io import BaseRaw
from scipy import stats
from .LaBraM import modeling_finetune # important to load the models
import torch.nn as nn
import torch
from tqdm import tqdm
import math
from joblib import Memory
from ...utils.config import get_config_value
import gc


def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(7)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class LaBraMBCIModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        checkpoint = torch.load("/itet-stor/jbuerki/home/unified_eeg_benchmark/unified_eeg_benchmark/models/clinical/LaBraM/checkpoints/labram-base.pth")
        new_checkpoint = {}
        for k,v in checkpoint['model'].items():
            if k.startswith('student.'):
                new_checkpoint[k[len('student.'):]] = v
        model = create_model("labram_base_patch200_200", 
                                # checkpoint_path= ,
                                qkv_bias=False,
                                rel_pos_bias=True,
                                num_classes=num_classes,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                attn_drop_rate=0.0,
                                drop_block_rate=None,
                                use_mean_pooling=True,
                                init_scale=0.001,
                                use_rel_pos_bias=True,
                                use_abs_pos_emb=True,
                                init_values=0.1,)
        model.load_state_dict(new_checkpoint, strict=False)
        for blk in model.blocks:
            for p in blk.parameters():
                p.requires_grad = True
        self.feature = model
        self.head = LinearWithConstraint(200, num_classes, max_norm=1)  # This layer follows the features
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, input_chans, window_size=1000, overlap=0.25):
        B, C, T = x.shape
        if T % 200 != 0: 
            x = x[:,:,0:T-T%200]
            T = T - T % 200        
        x = x / 100

        windows = self.sliding_window(x, window_size=window_size, overlap=overlap)
        
        min_val = x.min().item()
        max_val = x.max().item()
        #print(f"EEG values range: [{min_val}, {max_val}]")

        #pred = self.feature.forward_features(x, input_chans=input_chans, return_all_tokens=False)
        #pred = self.head(pred.flatten(1))
        
        predictions = []
        for window in windows:
            B, C, window_size = window.shape
            window = window.reshape((B, C, window_size // 200, 200))
            pred = self.feature.forward_features(window, input_chans=input_chans, return_all_tokens=False)
            pred = self.head(pred.flatten(1))
            predictions.append(pred.unsqueeze(0))  # [1, B, num_classes]
        
        # Average predictions from all windows
        predictions = torch.cat(predictions, dim=0)  # Shape: [num_windows, B, num_classes]
        avg_pred = predictions.mean(dim=0)  # Average across all windows
         
        return x, avg_pred
    
    def sliding_window(self, x, window_size=1000, overlap=0.25):
        """
        Split long EEG recordings into overlapping windows.
        
        Parameters:
        - x: EEG data of shape [batch_size, channels, time]
        - window_size: size of each window (e.g., 1000 samples i.e. 5 seconds)
        - overlap: overlap between windows (e.g., 0.25 for 25% overlap)
        
        Returns:
        - windows: A tensor of shape [num_windows, batch_size, channels, window_size]
        """
        B, C, T = x.shape
        stride = int(window_size * (1 - overlap))
        
        windows = []
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window = x[:, :, start:end]
            windows.append(window)
        
        return torch.stack(windows, dim=0)  # Shape: [num_windows, B, C, window_size]


def train_epoch(model, dataloader, optimizer, scheduler, device, input_chans):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    optimizer.zero_grad(set_to_none=True)
    for batch in tqdm(dataloader, desc="Training", leave=True):
        x, y = batch
        x = x.to(device)
        y = y.to(device).argmax(dim=1)
        
        _, logits = model(x, input_chans)
        loss = model.loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()  # update the learning rate
        optimizer.zero_grad(set_to_none=True)
        
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        running_corrects += torch.sum(preds == y).item()
        total_samples += x.size(0)

        del x, y, logits, loss  # Delete tensors no longer needed
        gc.collect()  # Invoke garbage collection
        torch.cuda.empty_cache()  # Clear cached memory on GPU
        
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, device, input_chans):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=True):
            x, y = batch
            x = x.to(device)
            y = y.to(device).argmax(dim=1)
            
            _, logits = model(x, input_chans)
            loss = model.loss_fn(logits, y)
            
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_corrects += torch.sum(preds == y).item()
            total_samples += x.size(0)
            
            all_labels.append(y.cpu())
            all_logits.append(logits.cpu())

            del x, y, logits  # Delete tensors no longer needed
            torch.cuda.empty_cache()  # Clear cached memory on GPU
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    # Concatenate predictions and labels
    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    
    # Compute additional metrics using get_metrics
    metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
    results = utils.get_metrics(all_logits.numpy(), all_labels.numpy(), metrics, False)
    
    return epoch_loss, epoch_acc, results

def inference(model, dataloader, device, input_chans):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=True):
            x, _ = batch
            x = x.to(device)
            _, logits = model(x, input_chans)
            preds = torch.argmax(logits, dim=1)
            predictions.append(preds.cpu())

            del x, logits  # Delete tensors no longer needed
            torch.cuda.empty_cache()  # Clear cached memory on GPU
    predictions = torch.cat(predictions, dim=0)
    return predictions


MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       return

   print("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       return

   print("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
   file_prefix = f"{host_name}_{timestamp}"

   try:
       print(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       print(f"Failed to capture memory snapshot {e}")
       return


class LaBraMModel(AbstractModel):
    def __init__(
        self,
    ):
        super().__init__("LaBraMModel")
        print("inside init")
        assert torch.cuda.is_available(), "CUDA is not available"
        
        start_record_memory_history()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LaBraMBCIModel(num_classes=2).to(self.device)
        self.cache = Memory(location=get_config_value("cache"), verbose=0)

    def fit(self, X: List[np.ndarray|List[BaseRaw]], y: List[np.ndarray|List[str]], meta: List[Dict]) -> None:
        print("inside fit")
        task_name = meta[0]["task_name"]
        if "Abnormal" in meta[0]["name"]:
            dataset_train_list = [self.cache.cache(make_dataset_abnormal)(X_, None, task_name, train=True) for X_, meta_ in zip(cast(List[BaseRaw], X), meta)]
        else:
            dataset_train_list = [self.cache.cache(make_dataset)(X_, y_, task_name, meta_["sampling_frequency"], meta_["channel_names"], train=True) for X_, y_, meta_ in zip(cast(List[np.ndarray], X), cast(List[np.ndarray], y), meta)]
        #dataset_train_list = [dataset[0] for dataset in datasets]
        dataset_val_list = None # dataset_train_list #[dataset[1] for dataset in datasets]
        del X, y, meta
        gc.collect()

        dataset_train_list = [dataset for dataset in dataset_train_list if len(dataset) > 0]        
        ch_names_list_train = [dataset.ch_names for dataset in dataset_train_list]
        if dataset_val_list is not None:
            dataset_val_list = [dataset for dataset in dataset_val_list if len(dataset) > 0]
            ch_names_list_val = [dataset.ch_names for dataset in dataset_val_list]

        torch.cuda.empty_cache()

        batch_size = 1
        train_loader_list = [torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True) for train_dataset in dataset_train_list]
        if dataset_val_list is not None:
            valid_loader_list = [torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False) for valid_dataset in dataset_val_list]
        else:
            valid_loader_list = None

        max_epochs = 2
        steps_per_epoch = math.ceil(sum([len(train_loader) for train_loader in train_loader_list]))
        max_lr = 4e-4
        
        
        # Set up optimizer and OneCycleLR scheduler
        optimizer = torch.optim.AdamW(list(self.model.head.parameters()) + list(self.model.feature.parameters()), lr=1e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        
        #best_val_loss = float('inf')
        #best_model_state = None


        # Training loop
        try:
            for epoch in range(1, max_epochs + 1):
                print(f"Epoch {epoch}/{max_epochs}")
                for train_loader, ch_names in zip(train_loader_list, ch_names_list_train):
                
                    input_chans = utils.get_input_chans(ch_names)
                    train_loss, train_acc = train_epoch(self.model, train_loader, optimizer, scheduler, self.device, input_chans)
                    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                
                    gc.collect()
                    torch.cuda.empty_cache()

                if valid_loader_list is not None:
                    for valid_loader, ch_names in zip(valid_loader_list, ch_names_list_val):
                        input_chans = utils.get_input_chans(ch_names)
                        val_loss, val_acc, val_metrics = validate_epoch(self.model, valid_loader, self.device, input_chans)
                        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                        print("  Val Metrics:", val_metrics)
            
            # Optionally save the best model based on validation loss
            #if val_loss < best_val_loss:
            #    best_val_loss = val_loss
            #    best_model_state = self.model.state_dict()
        
        # Load the best model (if saved)
        #if best_model_state is not None:
        #    self.model.load_state_dict(best_model_state)
        finally:
            stop_record_memory_history()
            export_memory_snapshot()

    @torch.no_grad()
    def predict(self, X: List[np.ndarray|List[BaseRaw]], meta: List[Dict]) -> np.ndarray:
        print("inside predict")
        task_name = meta[0]["task_name"]
        if "TUEG" in meta[0]["name"]:
            dataset_test_list = [self.cache.cache(make_dataset_abnormal)(X_, None, task_name, train=False) for X_, meta_ in zip(cast(List[np.ndarray], X), meta)]
        else:
            dataset_test_list = [self.cache.cache(make_dataset)(X_, None, task_name, meta_["sampling_frequency"], meta_["channel_names"], train=False) for X_, meta_ in zip(cast(List[np.ndarray], X), meta)]
        dataset_test_list = [dataset for dataset in dataset_test_list if len(dataset) > 0]
        print("datasets length: ", len(dataset_test_list[0]))
        ch_names_list = [dataset.ch_names for dataset in dataset_test_list]
        # Inference on test set

        batch_size = 1
        test_loader_list = [torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False) for test_dataset in dataset_test_list]

        predictions = []
        for test_loader, ch_names in zip(test_loader_list, ch_names_list):
            input_chans = utils.get_input_chans(ch_names)
            predictions.append(inference(self.model, test_loader, self.device, input_chans).cpu())
        
        predictions = torch.cat(predictions, dim=0).numpy()
        print(predictions.shape)

        if task_name == "parkinsons_clinical":
            reverse_label_mapping = {0: 'parkinsons', 1: 'no_parkinsons'}
        elif task_name == "schizophrenia_clinical":
            reverse_label_mapping = {0: 'schizophrenia', 1: 'no_schizophrenia'}
        elif task_name == "depression_clinical":
            reverse_label_mapping = {0: 'depression', 1: 'no_depression'}
        elif task_name == "mtbi_clinical":
            reverse_label_mapping = {0: True, 1: False}
        elif task_name == "ocd_clinical":
            reverse_label_mapping = {0: 'ocd', 1: 'no_ocd'}
        elif task_name == "abnormal_clinical":
            reverse_label_mapping = {0: 'abnormal', 1: 'normal'}
        elif task_name == "epilepsy_clinical":
            reverse_label_mapping = {0: 'epilepsy', 1: 'no_epilepsy'}
        mapped_pred = np.array([reverse_label_mapping[idx] for idx in predictions])
        
        print(mapped_pred.shape)
        return mapped_pred
        
