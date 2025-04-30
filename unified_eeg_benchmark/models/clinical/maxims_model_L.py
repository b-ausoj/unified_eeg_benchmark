from ..abstract_model import AbstractModel
from typing import List, Dict, cast, Literal
import numpy as np
from .Maxim.make_dataset import make_dataset
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse, calc_sample_weights
import logging
import torch
import numpy as np
from mne.io import BaseRaw
import torch.nn as nn
import torch
import json
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import gc
import os
from ...utils.config import get_config_value
import lightning as L
from .Maxim.fine_tuning_model import FineTuningModel
from .Maxim.mae_rope_encoder import EncoderViTRoPE
from .Maxim.engine_for_finetuning import (
    train_epoch,
    validate_epoch,
    inference,
    move_to_device,
)
from .Maxim.maxim_utils import (
    sample_collate_fn,
)


CHKPT_PATHS = [
    (
        "/itet-stor/jbuerki/home/unified_eeg_benchmark/"
        "unified_eeg_benchmark/models/clinical/Maxim/pretrain_ckpts/"
        "epoch=0-step=32807-val_loss=133.55.ckpt"
    ),
    (
        "/itet-stor/jbuerki/home/unified_eeg_benchmark/"
        "unified_eeg_benchmark/models/clinical/Maxim/pretrain_ckpts/"
        "epoch=5-step=181751-val_loss=130.43-lr.ckpt"
    ),
    (
        "/itet-stor/jbuerki/home/unified_eeg_benchmark/"
        "unified_eeg_benchmark/models/clinical/Maxim/pretrain_ckpts/"
        "epoch=7-step=221332-val_loss=129.99-lr.ckpt"
    ),
]

class MaximsModel(AbstractModel):
    def __init__(
        self,
        seed: int = 42,
    ):
        super().__init__("MaximsModel")
        print("inside init MaximsModel")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.use_cache = True

    def fit(self, X: List[np.ndarray|List[BaseRaw]], y: List[np.ndarray|List[str]], meta: List[Dict]) -> None:
        print("inside fit")
        logging.info("inside fit of MaximsModel")
        task_name = meta[0]["task_name"]

        out_dim = 2

        def load_encoder(chkpt_path: str) -> EncoderViTRoPE:
            channel_name_map_path = (
            "/itet-stor/jbuerki/home/unified_eeg_benchmark/"
            "unified_eeg_benchmark/models/clinical/Maxim/channels_to_id.json"
            )
            checkpoint = torch.load(chkpt_path, map_location=torch.device("cpu"))
            state_dict = checkpoint["state_dict"]
            state_dict = {
            k.replace("net.encoder.", ""): v
            for k, v in state_dict.items()
            if "net.encoder." in k
            }
            encoder = EncoderViTRoPE(channel_name_map_path)
            encoder.load_state_dict(state_dict)
            return encoder

        self.chkpt = 0
        encoder = load_encoder(CHKPT_PATHS[self.chkpt])
                
        self.model = FineTuningModel(
            encoder=encoder,
            frozen_encoder=False,
            out_dim=out_dim,
            task_name=task_name,
            task_type="Classification",
            learning_rate=0.01,
            mask_ratio=0,
        ).to(self.device)

        
        class_weights = torch.tensor(calc_class_weights(y)).to(self.device)
        print("class_weights", class_weights)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        train_dataset  = make_dataset(X, y, meta, task_name, self.name, is_test=False, use_cache=self.use_cache)

        train_val_split = 0.85
        if train_val_split is not None:
            train_dataset, val_dataset = train_dataset.split_train_val(val_split=1-train_val_split) # yes this is correct
        else:
            val_dataset = None

        sample_weights = calc_sample_weights(train_dataset).to(self.device)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # Important for oversampling
        )

        del X, y, meta
        gc.collect()
        torch.cuda.empty_cache()

        train_loader = DataLoader(
            train_dataset, batch_size=1, collate_fn=sample_collate_fn, sampler=sampler, num_workers=4, pin_memory=True
        )
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=1, collate_fn=sample_collate_fn, shuffle=False, num_workers=4, pin_memory=True
            )
        else:
            val_loader = None

        start_epoch = 1
        max_epochs = 10
        
        # Set up optimizer and OneCycleLR scheduler
        optimizer = torch.optim.AdamW(
            list(self.model.head.parameters()) + 
            list(self.model.encoder.parameters()), 
            lr=1e-6, 
            weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * max_epochs,
        )
        
        val_loss = float('inf')
        best_val_loss = float('inf')
        best_model_state = None

        # Create a checkpoint directory if it doesn't exist
        checkpoint_dir = get_config_value("chkpt")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set resume_checkpoint_path to a checkpoint file path to resume training, or leave as None to start fresh.
        resume_checkpoint_path = "/scratch/jbuerki/chkpt/epoch=39-step=90090-train_loss=0.17.ckpt"
        
        if resume_checkpoint_path is not None and os.path.isfile(resume_checkpoint_path):
            checkpoint = torch.load(resume_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            print(f"Resuming training from epoch {start_epoch}")
        
        # Training loop
        for epoch in range(start_epoch, max_epochs + 1):
            if epoch <= 5:
                self.model.freeze_encoder()
            else:
                self.model.unfreeze_encoder()
        
            print(f"Epoch {epoch}/{max_epochs}")
            train_loss, train_acc = train_epoch(self.model, train_loader, optimizer, scheduler, self.device)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {current_lr:.6f}")
        
            if val_loader is not None:
                val_loss, val_acc, val_metrics = validate_epoch(self.model, val_loader, self.device)
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                print("  Val Metrics:", val_metrics)
        
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
        
            # Save checkpoint for each epoch
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch={epoch}-step={epoch * len(train_loader)}-train_loss={train_loss:.2f}.ckpt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Load the best model (if saved)
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
      
    @torch.no_grad()
    def predict(self, X: List[np.ndarray|List[BaseRaw]], meta: List[Dict]) -> np.ndarray:
        print("inside predict")
        task_name = meta[0]["task_name"]
        test_dataset  = make_dataset(X, None, meta, task_name, self.name, is_test=True, use_cache=self.use_cache)    
        test_loader = DataLoader(
            test_dataset, batch_size=1, collate_fn=sample_collate_fn, shuffle=False, num_workers=4, pin_memory=True
        )
            
        predictions = inference(self.model, test_loader, self.device)

        mapped_pred = np.array([map_label_reverse(pred, task_name) for pred in predictions])
        
        print(mapped_pred)
        return mapped_pred
        
