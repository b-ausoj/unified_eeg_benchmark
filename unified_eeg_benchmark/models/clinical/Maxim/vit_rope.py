"""
This code was originally obtained from:
https://github.com/facebookresearch/deit
and
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial

import torch.nn.functional as F

from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from .models_v2 import (
    vit_models,
    Layer_scale_init_Block,
    Attention,
)

# from models_v2 import (
#     vit_models,
#     Layer_scale_init_Block,
#     Attention,
# )


def init_random_2d_freqs(
    dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True
):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat(
            [mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1
        )
        fy = torch.cat(
            [mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def compute_mixed_cis(
    freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int
):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_cis = []
        freqs_x = (
            (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
            .view(depth, N, num_heads, -1)
            .permute(0, 2, 1, 3)
        )
        freqs_y = (
            (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
            .view(depth, N, num_heads, -1)
            .permute(0, 2, 1, 3)
        )
        freqs_cis.append(torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y))

    return torch.cat(freqs_cis, dim=-1)


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # print("reshape_for_broadcast:", freqs_cis.shape, x.shape)
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]

    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class RoPEAttention(Attention):
    """Multi-head Attention block with rotary position embeddings."""

    def forward(self, x, freqs_cis):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(
            q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis
        )
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FlexibleRoPEAttention(Attention):
    """
    Multi-head Attention block with rotary position embeddings.

    Adjusted the RoPEAttention class to work with a variable number of prepended tokens,
    e.g. one cls token and multiple mean tokens.

    They are not taken into consideration when applying the rotary position embeddings.
    """

    def forward(self, x, freqs_cis, nr_meta_tokens=1):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q[:, :, nr_meta_tokens:], k[:, :, nr_meta_tokens:] = apply_rotary_emb(
            q[:, :, nr_meta_tokens:], k[:, :, nr_meta_tokens:], freqs_cis=freqs_cis
        )
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class RoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = RoPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(
            self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis)
        )
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


class Flexible_RoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications

    # Adjusted to work with FlexibleRoPEAttention.

    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = FlexibleRoPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis, nr_meta_tokens=1):
        x = x + self.drop_path(
            self.gamma_1
            * self.attn(
                self.norm1(x), freqs_cis=freqs_cis, nr_meta_tokens=nr_meta_tokens
            )
        )
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


# TODO: cleaner solution but it's ok for the moment
def select_freqs_cis(self, freqs_cis, H, W, win_size, device):
    h = H // self.patch_size
    w = W // self.patch_size

    win_shift = win_size * self.win_shift_factor

    # 1. Compute selection parameters
    y_nr_patches = h
    y_jump = int(self.max_win_size / win_size)

    x_nr_patches = w
    x_jump = int(win_shift / self.min_win_shift)

    # print("[select_freqs_cis] win_size:", win_size)
    # print(f"[select_freqs_cis] h: {h}, w: {w}")
    # print(
    #     f"[select_freqs_cis] y_nr_patches: {y_nr_patches}, x_nr_patches: {x_nr_patches}"
    # )
    # print(f"[select_freqs_cis] y_jump: {y_jump}, x_jump: {x_jump}")

    # print("[select_freqs_cis] freqs_cis.shape:", freqs_cis.shape)
    # print(
    #     f"[select_freqs_cis] max_y_patches: {self.max_y_patches}, max_x_patches: {self.max_x_patches}",
    # )
    assert (
        freqs_cis.shape[0] == self.max_y_patches * self.max_x_patches
    ), "freqs_cis shape does not match the expected shape."

    # 2. Select the freqs_cis rows
    # freqs_cis.shape = (N, d_head/2), where d_head = embed_dim // num_heads and /2 due to complex numbers
    freqs_cis_selected = []
    for i in range(0, y_nr_patches):
        row_start = i * self.max_x_patches * y_jump
        for j in range(0, x_nr_patches):
            freqs_cis_selected.append(freqs_cis[row_start + j * x_jump])
    freqs_cis_selected = torch.stack(freqs_cis_selected)

    # Send the newly created tensor to the same device as freqs_cis
    freqs_cis_selected = freqs_cis_selected.to(device)
    # print("[select_freqs_cis] freqs_cis.device:", freqs_cis.device)
    # print("[select_freqs_cis] freqs_cis_selected.device:", freqs_cis_selected.device)

    # print("[select_freqs_cis] freqs_cis_selected.shape:", freqs_cis_selected.shape)
    # assert freqs_cis_selected.shape[0] == h * w

    return freqs_cis_selected


class rope_vit_models(vit_models):
    def __init__(self, rope_theta=100.0, rope_mixed=False, use_ape=False, **kwargs):
        super().__init__(**kwargs)

        img_size = kwargs["img_size"] if "img_size" in kwargs else 224
        patch_size = kwargs["patch_size"] if "patch_size" in kwargs else 16
        num_heads = kwargs["num_heads"] if "num_heads" in kwargs else 12
        embed_dim = kwargs["embed_dim"] if "embed_dim" in kwargs else 768
        mlp_ratio = kwargs["mlp_ratio"] if "mlp_ratio" in kwargs else 4.0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        self.use_ape = use_ape
        if not self.use_ape:
            self.pos_embed = None

        self.rope_mixed = rope_mixed
        self.num_heads = num_heads
        self.patch_size = patch_size

        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)

            freqs = []
            for i, _ in enumerate(self.blocks):
                freqs.append(
                    init_random_2d_freqs(
                        dim=embed_dim // num_heads,
                        num_heads=num_heads,
                        theta=rope_theta,
                    )
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(self.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)

            t_x, t_y = init_t_xy(
                end_x=img_size // patch_size, end_y=img_size // patch_size
            )
            self.register_buffer("freqs_t_x", t_x)
            self.register_buffer("freqs_t_y", t_y)
        else:
            self.compute_cis = partial(
                compute_axial_cis, dim=embed_dim // num_heads, theta=rope_theta
            )

            freqs_cis = self.compute_cis(
                end_x=img_size // patch_size, end_y=img_size // patch_size
            )
            self.freqs_cis = freqs_cis

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "freqs"}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_ape:
            pos_embed = self.pos_embed
            if pos_embed.shape[-2] != x.shape[-2]:
                img_size = self.patch_embed.img_size
                patch_size = self.patch_embed.patch_size
                pos_embed = pos_embed.view(
                    1,
                    (img_size[1] // patch_size[1]),
                    (img_size[0] // patch_size[0]),
                    self.embed_dim,
                ).permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed,
                    size=(H // patch_size[1], W // patch_size[0]),
                    mode="bicubic",
                    align_corners=False,
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            x = x + pos_embed

        x = torch.cat((cls_tokens, x), dim=1)

        if self.rope_mixed:
            if self.freqs_t_x.shape[0] != x.shape[1] - 1:
                t_x, t_y = init_t_xy(
                    end_x=W // self.patch_size, end_y=H // self.patch_size
                )
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)

            for i, blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis[i])
        else:
            if self.freqs_cis.shape[0] != x.shape[1] - 1:
                freqs_cis = self.compute_cis(
                    end_x=W // self.patch_size, end_y=H // self.patch_size
                )
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)

            for i, blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis)

        x = self.norm(x)
        x = x[:, 0]

        return x


# RoPE-Axial
@register_model
def rope_axial_deit_small_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=100.0,
        rope_mixed=False,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rope_axial_deit_base_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=100.0,
        rope_mixed=False,
        **kwargs,
    )
    return model


@register_model
def rope_axial_deit_large_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=100.0,
        rope_mixed=False,
        **kwargs,
    )
    return model


# RoPE-Mixed
@register_model
def rope_mixed_deit_small_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rope_mixed_deit_base_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        **kwargs,
    )
    return model


@register_model
def rope_mixed_deit_large_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        **kwargs,
    )
    return model


# RoPE-Axial + APE
@register_model
def rope_axial_ape_deit_small_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=100.0,
        rope_mixed=False,
        use_ape=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rope_axial_ape_deit_base_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=100.0,
        rope_mixed=False,
        use_ape=True,
        **kwargs,
    )
    return model


@register_model
def rope_axial_ape_deit_large_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=100.0,
        rope_mixed=False,
        use_ape=True,
        **kwargs,
    )
    return model


# RoPE-Mixed + APE
@register_model
def rope_mixed_ape_deit_small_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        use_ape=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rope_mixed_ape_deit_base_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        use_ape=True,
        **kwargs,
    )
    return model


@register_model
def rope_mixed_ape_deit_large_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        use_ape=True,
        **kwargs,
    )
    return model
