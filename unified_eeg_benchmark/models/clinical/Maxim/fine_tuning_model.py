import torch
import torch.nn as nn
import pytorch_lightning as L
from .mae_rope_encoder import EncoderViTRoPE
from typing import Dict


class FineTuningModel(L.LightningModule):
    def __init__(
        self,
        encoder: EncoderViTRoPE,
        frozen_encoder: bool,
        out_dim: int,
        task_name: str,
        task_type: str,
        learning_rate: float,
        mask_ratio: float,
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
                param.requires_grad = False

        self.head = nn.Linear(encoder.encoder_embed_dim, out_dim)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True

    def forward_encoder(self, full_x: Dict[int, Dict[str, torch.Tensor]]) -> torch.Tensor:
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
        smart_token = tokens[0] # self.win_shift_aggregation(tokens) # tokens[0]
        # print(f"[FT.forward] smart_token.shape: {smart_token.shape}")
        # print("SMART TOKEN: ", smart_token.shape)
        return smart_token

    def forward(self, full_x: Dict[int, Dict[str, torch.Tensor]]) -> torch.Tensor:
        smart_token = self.forward_encoder(full_x)
        return self.head(smart_token)