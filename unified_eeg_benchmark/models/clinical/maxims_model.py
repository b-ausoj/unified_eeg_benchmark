from ..abstract_model import AbstractModel
from typing import List, Dict, cast, Literal
import numpy as np
from .Maxim.make_dataset import make_dataset
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse
import logging
import torch
import numpy as np
from mne.io import BaseRaw
import torch.nn as nn
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from ...utils.config import get_config_value
import gc
import os
import lightning as L
import torchaudio
from ...utils.config import get_config_value
from .Maxim.mae_rope_encoder import EncoderViTRoPE
from sklearn.neural_network import MLPClassifier
from ..bci.Maxim.simple_classifier import SimpleClassifier
from ..bci.Maxim.embeddings_dataset import EmbeddingsDataset
from ..bci.Maxim.engine_for_finetuning import (
    train_epoch,
    validate_epoch,
    inference,
    move_to_device,
)
# from ..bci.Maxim.engine_for_classifier import (
#     train_epoch,
#     validate_epoch,
#     inference,
#     move_to_device,
# )
from .Maxim.maxim_utils import (
    get_nr_y_patches,
    get_nr_x_patches,
    self_get_generic_channel_name,
    self_patch_size,
    self_encode_mean,
    self_win_shift_factor,
    self_win_shifts,
)
from .Maxim.transforms import (
    crop_spg,
    normalize_spg,
)

CHKPT_PATHS = [
    f"/itet-stor/{get_config_value("user")}/deepeye_storage/pretrain_ckpts/epoch=0-step=32807-val_loss=133.55.ckpt",
    f"/itet-stor/{get_config_value("user")}/deepeye_storage/pretrain_ckpts/epoch=5-step=181751-val_loss=130.43-lr.ckpt",
    f"/itet-stor/{get_config_value("user")}/deepeye_storage/pretrain_ckpts/epoch=7-step=221332-val_loss=129.99-lr.ckpt",
]       


def mean_aggregation(tokens):
    return torch.mean(torch.stack(tokens), dim=0)

def mean_aggregation2(data):
    return torch.mean(data, dim=0)

class FineTuningModel(L.LightningModule):
    def __init__(
        self,
        encoder,
        frozen_encoder,
        out_dim,
        task_name,
        task_type,
        learning_rate,
        mask_ratio,
    ):
        super(FineTuningModel, self).__init__()

        self.task_name = task_name
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio

        # Pretrained network
        self.encoder = encoder
        if frozen_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = True

        # Modular aggregation method on channel tokens
        self.win_shift_aggregation = mean_aggregation
        self.channel_aggregation = mean_aggregation2

        self.head = nn.Linear(encoder.encoder_embed_dim, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward_encoder(self, full_x):

        x_embeds = {}
        H_W = {}

        for win_size, x_win in full_x.items():
            if win_size != 4:
                continue
            spgs = x_win["batch"]
            channels = x_win["channels"]
            means = x_win["means"]
            stds = x_win["stds"]
            B, C, H, W = spgs.shape
            #print(f"[FT.forward] spgs.shape: {spgs.shape}") # [19, 1, 400, 1200]
            # TODO: split into less rows if necessary because of CUDA error
            #if nr_tokens > 80000:
            #    print(ard)
            #    pass

            # print(B, C, H, W)
            x_emb, _, _, _ = self.encoder(
                x=spgs,
                means=means,
                stds=stds,
                channels=channels,
                win_size=win_size,
                mask_ratio=self.mask_ratio,
            )
            # print("Got embedding: ", x_emb.shape)
            # TODO:
            x_embeds[win_size] = x_emb
            H_W[win_size] = (H, W)
            # print(f"[FT.forward, after self.encoder] x_emb.shape: {x_emb.shape}")

              # Release memory for intermediate tensors
            del spgs, channels, means, stds
            torch.cuda.empty_cache()
        
        # Pass through time-transformer (Get CLS Tokens)
        for win_size, x_emb in x_embeds.items():
            x_emb = x_emb[:, 0, :]
            # print("Got embedding2: ", x_emb.shape)
            x_embeds[win_size] = x_emb

        
        # Pass through channel-transformer
        tokens = []
        all_channels = []
        for win_size, x_emb in x_embeds.items():
            # print("Before: ", x_emb.shape)
            all_channels.append(x_emb)
            x_emb = torch.mean(x_emb, dim=0)
            # print("Got embedding3: ", x_emb.shape)
            tokens.append(x_emb)
            

        # print(f"[FT.forward] len(tokens): {len(tokens)}")
        # Average over all window shifts
        smart_token =  tokens[0] # self.win_shift_aggregation(tokens)
        # print(f"[FT.forward] smart_token.shape: {smart_token.shape}")
        # print("SMART TOKEN: ", smart_token.shape)

        return smart_token
    
    def forward(self, full_x):
        smart_token = self.forward_encoder(full_x)
        return self.head(smart_token)


def extract_embeddings(loader, fine_tuning_model: FineTuningModel, device):
    embeddings = []
    labels = []
    i = 0
    for inputs, label in tqdm(loader):
        # print(file)
        # print(label)
        # if (label == 0) and (i % 3 != 0):
        #    i += 1
        #    continue
        # print(label)
        inputs = move_to_device(inputs, device)  # Move inputs to the GPU
        with torch.no_grad():  # No need to compute gradients for inference
            output = fine_tuning_model.forward_encoder(inputs)
        embeddings.append(output.cpu().numpy())  # Move to CPU and convert to numpy
        if not isinstance(label, np.ndarray):
            if isinstance(label, torch.Tensor):
                if label.device != torch.device("cpu"):
                    label = label.cpu()
                label = label.numpy()
        labels.append(label)
        # Delete tensors to free GPU memory
        del inputs, output
        torch.cuda.empty_cache()
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    return embeddings, labels

def sample_collate_fn(batch):
    signals, target, sr, dur, channels, dataset = (
        batch[0]["signals"],
        batch[0]["output"],
        batch[0]["sr"],
        batch[0]["dur"],
        batch[0]["channels"],
        batch[0]["dataset"],
    )
    if dur > 1200:
        dur = 1200
        signals = signals[:, : 1200 * sr]

    # TODO: compute spectrograms for each win_size
    # gives a new dimension (S) in batch
    # need another extra transformer after the encoder
    # (B, 1, H, W) -> (S*B, 1, H, W)
    valid_win_shifts = [
        win_shift
        for win_shift in self_win_shifts
        if get_nr_y_patches(win_shift, sr) >= 1
        and get_nr_x_patches(win_shift, dur) >= 1
    ]

    channel_name_map_path = (
        "/itet-stor/jbuerki/home/unified_eeg_benchmark/unified_eeg_benchmark/models/clinical/Maxim/channels_to_id.json"
    )
    with open(channel_name_map_path, "r") as file:
        self_channel_name_map = json.load(file)

    # list holding assembled tensors for varying window shifts
    full_batch = {}

    for win_size in valid_win_shifts:

        fft = torchaudio.transforms.Spectrogram(
            n_fft=int(sr * win_size),
            win_length=int(sr * win_size),
            hop_length=int(sr * win_size * self_win_shift_factor),
            normalized=True,
        )

        spg_list = []
        chn_list = []
        mean_list = []
        std_list = []

        for signal, channel in zip(signals, channels):

            # Channel information
            channel_name = self_get_generic_channel_name(channel)
            channel = (
                self_channel_name_map[channel_name]
                if channel_name in self_channel_name_map
                else self_channel_name_map["None"]
            )

            # Spectrogram Computation & Cropping
            spg = fft(signal)
            spg = spg**2
            spg = crop_spg(spg, self_patch_size)

            H_new, W_new = spg.shape[0], spg.shape[1]
            h_new, w_new = H_new // self_patch_size, W_new // self_patch_size

            # Prepare channel information (per-patch)
            channel = torch.full((h_new, w_new), channel, dtype=torch.float16)

            spg, mean, std = normalize_spg(spg)
            mean = self_encode_mean(mean, win_size)
            std = self_encode_mean(std, win_size)

            spg_list.append(spg)
            chn_list.append(channel)
            mean_list.append(mean)
            std_list.append(std)

        win_batch = torch.stack(spg_list)
        win_channels = torch.stack(chn_list)
        win_means = torch.stack(mean_list)
        win_stds = torch.stack(std_list)

        win_batch.unsqueeze_(1)
        win_channels = win_channels.flatten(1)
        win_means = win_means.transpose(1, 2)
        win_stds = win_stds.transpose(1, 2)

        full_batch[win_size] = {
            "batch": win_batch,
            "channels": win_channels,
            "means": win_means,
            "stds": win_stds,
        }
        # print(f"[collate_fn] win_size={win_size}: {win_batch.shape}")

    # == Finished iterating over all possible window shifts

    return full_batch, target


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
        self.chunk_len_s = None
        self.train_on_embeddings = False
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
            frozen_encoder=True,
            out_dim=out_dim,
            task_name=task_name,
            task_type="Classification",
            learning_rate=0.01,
            mask_ratio=0,
        ).to(self.device)

        
        class_weights = torch.tensor(calc_class_weights(y)).to(self.device)
        print("class_weights", class_weights)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        train_dataset  = make_dataset(X, y, meta, task_name, self.name, self.chunk_len_s, is_train=True, use_cache=self.use_cache)

        train_val_split = None #0.85
        if train_val_split is not None:
            train_dataset, val_dataset = train_dataset.split_train_val(val_split=1-train_val_split) # yes this is correct
        else:
            val_dataset = None

        del X, y, meta
        gc.collect()
        torch.cuda.empty_cache()

        batch_size = 1

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=sample_collate_fn, shuffle=True, num_workers=0, pin_memory=True
        )
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, collate_fn=sample_collate_fn, shuffle=False, num_workers=0, pin_memory=True
            )
        else:
            val_loader = None

        max_epochs = 20
        steps_per_epoch = len(train_loader)
        max_lr = 4e-4
        
        
        # Set up optimizer and OneCycleLR scheduler
        optimizer = torch.optim.AdamW(
            list(self.model.head.parameters()) + 
            list(self.model.encoder.parameters()), 
            lr=1e-6, 
            weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        

        if self.train_on_embeddings:
            os.makedirs(os.path.dirname(os.path.join(get_config_value("data"), "embeddings")), exist_ok=True)
            train_embeddings_path = os.path.join(get_config_value("data"), "embeddings", f"{task_name}_{self.name}_train_embeddings_{self.chkpt}_{len(train_loader)}.npy")
            train_labels_path = os.path.join(get_config_value("data"), "embeddings", f"{task_name}_{self.name}_train_labels_{self.chkpt}_{len(train_loader)}.npy")
            if self.use_cache and os.path.exists(train_embeddings_path) and os.path.exists(train_labels_path):
                print("Loading cached embeddings - train")
                train_embeddings = np.load(train_embeddings_path)
                train_labels = np.load(train_labels_path)
            else:
                print("Extracting embeddings - train")
                train_embeddings, train_labels = extract_embeddings(train_loader, self.model, self.device)
                np.save(train_embeddings_path, train_embeddings)
                np.save(train_labels_path, train_labels)
            if val_loader is not None:
                val_embeddings_path = os.path.join(get_config_value("data"), "embeddings", f"{task_name}_{self.name}_val_embeddings_{self.chkpt}_{len(val_loader)}.npy")
                val_labels_path = os.path.join(get_config_value("data"), "embeddings", f"{task_name}_{self.name}_val_labels_{self.chkpt}_{len(val_loader)}.npy")
                if self.use_cache and os.path.exists(val_embeddings_path) and os.path.exists(val_labels_path):
                    print("Loading cached embeddings - val")
                    val_embeddings = np.load(val_embeddings_path)
                    val_labels = np.load(val_labels_path)
                else:
                    print("Extracting embeddings - val")
                    val_embeddings, val_labels = extract_embeddings(val_loader, self.model, self.device)
                    np.save(val_embeddings_path, val_embeddings)
                    np.save(val_labels_path, val_labels)

            batch_size = 32
            train_dataset = EmbeddingsDataset(train_embeddings, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
            if val_loader is not None:
                val_dataset = EmbeddingsDataset(val_embeddings, val_labels)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
            
            # Change the model to a simple classifier
            self.embedder = self.model
            self.model = SimpleClassifier(output_dim=2).to(self.device)
            self.model.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            
            optimizer = torch.optim.AdamW(
                list(self.model.net.parameters()), 
                lr=1e-6, 
                weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        
        best_val_loss = float('inf')
        best_model_state = None

        # Training loop
        for epoch in range(1, max_epochs + 1):
            print(f"Epoch {epoch}/{max_epochs}")
            train_loss, train_acc = train_epoch(self.model, train_loader, optimizer, scheduler, self.device)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {current_lr:.6f}")

            if val_loader is not None:
                val_loss, val_acc, val_metrics = validate_epoch(self.model, val_loader, self.device)
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                print("  Val Metrics:", val_metrics)
        
                # Optionally save the best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
        
        # Load the best model (if saved)
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
      
    @torch.no_grad()
    def predict(self, X: List[np.ndarray|List[BaseRaw]], meta: List[Dict]) -> np.ndarray:
        print("inside predict")
        task_name = meta[0]["task_name"]
        test_dataset  = make_dataset(X, None, meta, task_name, self.name, self.chunk_len_s, is_train=False, use_cache=self.use_cache)    
        test_loader = DataLoader(
            test_dataset, batch_size=1, collate_fn=sample_collate_fn, shuffle=False, num_workers=0, pin_memory=True
        )

        if self.train_on_embeddings:
            test_embeddings_path = os.path.join(get_config_value("data"), "embeddings", f"{task_name}_{self.name}_{meta[0]['name'].replace(' ', '_')}_test_embeddings_{self.chkpt}_{len(test_loader)}.npy")
            test_labels_path = os.path.join(get_config_value("data"), "embeddings", f"{task_name}_{self.name}_{meta[0]['name'].replace(' ', '_')}_test_labels_{self.chkpt}_{len(test_loader)}.npy")
          
            if self.use_cache and os.path.exists(test_embeddings_path) and os.path.exists(test_labels_path):
                print("Loading cached test embeddings - test")
                test_embeddings = np.load(test_embeddings_path)
                test_labels = np.load(test_labels_path)
            else:
                print("Extracting test embeddings - test")
                test_embeddings, test_labels = extract_embeddings(test_loader, self.model, self.device)
                os.makedirs(os.path.dirname(test_embeddings_path), exist_ok=True)
                os.makedirs(os.path.dirname(test_labels_path), exist_ok=True)
                np.save(test_embeddings_path, test_embeddings)
                np.save(test_labels_path, test_labels)
            test_dataset = EmbeddingsDataset(test_embeddings, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)
            
        predictions = inference(self.model, test_loader, self.device)

        mapped_pred = np.array([map_label_reverse(pred, task_name) for pred in predictions])
        
        print(mapped_pred)
        return mapped_pred
        
