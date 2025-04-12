"""
References:
    - VQGAN: https://github.com/CompVis/taming-transformers
    - MAE: https://github.com/facebookresearch/mae
"""

import numpy as np
import math
import functools
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Mlp
from timm.layers.helpers import to_2tuple
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from .dit import PatchEmbed


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, dim=1):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        if dim == 1:
            self.dims = [1, 2, 3]
        elif dim == 2:
            self.dims = [1, 2]
        else:
            raise NotImplementedError
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def mode(self):
        return self.mean


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        frame_height,
        frame_width,
        qkv_bias=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        rotary_freqs = RotaryEmbedding(
            dim=head_dim // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        ).get_axial_freqs(frame_height, frame_width)
        self.register_buffer("rotary_freqs", rotary_freqs, persistent=False)

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.frame_height * self.frame_width

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = rearrange(
            q,
            "b (H W) (h d) -> b h H W d",
            H=self.frame_height,
            W=self.frame_width,
            h=self.num_heads,
        )
        k = rearrange(
            k,
            "b (H W) (h d) -> b h H W d",
            H=self.frame_height,
            W=self.frame_width,
            h=self.num_heads,
        )
        v = rearrange(
            v,
            "b (H W) (h d) -> b h H W d",
            H=self.frame_height,
            W=self.frame_width,
            h=self.num_heads,
        )

        q = apply_rotary_emb(self.rotary_freqs, q)
        k = apply_rotary_emb(self.rotary_freqs, k)

        q = rearrange(q, "b h H W d -> b h (H W) d")
        k = rearrange(k, "b h H W d -> b h (H W) d")
        v = rearrange(v, "b h H W d -> b h (H W) d")

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h N d -> b N (h d)")

        x = self.proj(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        frame_height,
        frame_width,
        mlp_ratio=4.0,
        qkv_bias=False,
        attn_causal=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads,
            frame_height,
            frame_width,
            qkv_bias=qkv_bias,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_height=256,
        input_width=256,
        patch_size=16,
        enc_dim=768,
        enc_depth=6,
        enc_heads=12,
        dec_dim=768,
        dec_depth=6,
        dec_heads=12,
        mlp_ratio=4.0,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        use_variational=True,
        **kwargs,
    ):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.patch_size = patch_size
        self.seq_h = input_height // patch_size
        self.seq_w = input_width // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.patch_dim = 3 * patch_size**2

        self.latent_dim = latent_dim
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim

        # patch
        self.patch_embed = PatchEmbed(input_height, input_width, patch_size, 3, enc_dim)

        # encoder
        self.encoder = nn.ModuleList(
            [
                AttentionBlock(
                    enc_dim,
                    enc_heads,
                    self.seq_h,
                    self.seq_w,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(enc_depth)
            ]
        )
        self.enc_norm = norm_layer(enc_dim)

        # bottleneck
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.quant_conv = nn.Linear(enc_dim, mult * latent_dim)
        self.post_quant_conv = nn.Linear(latent_dim, dec_dim)

        # decoder
        self.decoder = nn.ModuleList(
            [
                AttentionBlock(
                    dec_dim,
                    dec_heads,
                    self.seq_h,
                    self.seq_w,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_dim)
        self.predictor = nn.Linear(dec_dim, self.patch_dim)  # decoder to patch

        # initialize this weight first
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        # patchify
        bsz, _, h, w = x.shape
        x = x.reshape(
            bsz,
            3,
            self.seq_h,
            self.patch_size,
            self.seq_w,
            self.patch_size,
        ).permute([0, 1, 3, 5, 2, 4])  # [b, c, h, p, w, p] --> [b, c, p, p, h, w]
        x = x.reshape(bsz, self.patch_dim, self.seq_h, self.seq_w)  # --> [b, cxpxp, h, w]
        x = x.permute([0, 2, 3, 1]).reshape(bsz, self.seq_len, self.patch_dim)  # --> [b, hxw, cxpxp]
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        # unpatchify
        x = x.reshape(bsz, self.seq_h, self.seq_w, self.patch_dim).permute([0, 3, 1, 2])  # [b, h, w, cxpxp] --> [b, cxpxp, h, w]
        x = x.reshape(
            bsz,
            3,
            self.patch_size,
            self.patch_size,
            self.seq_h,
            self.seq_w,
        ).permute([0, 1, 4, 2, 5, 3])  # [b, c, p, p, h, w] --> [b, c, h, p, w, p]
        x = x.reshape(
            bsz,
            3,
            self.input_height,
            self.input_width,
        )  # [b, c, hxp, wxp]
        return x

    def encode(self, x):
        # patchify
        x = self.patch_embed(x)

        # encoder
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)

        # bottleneck
        moments = self.quant_conv(x)
        if not self.use_variational:
            moments = torch.cat((moments, torch.zeros_like(moments)), 2)
        posterior = DiagonalGaussianDistribution(moments, deterministic=(not self.use_variational), dim=2)
        return posterior

    def decode(self, z):
        # bottleneck
        z = self.post_quant_conv(z)

        # decoder
        for blk in self.decoder:
            z = blk(z)
        z = self.dec_norm(z)

        # predictor
        z = self.predictor(z)

        # unpatchify
        dec = self.unpatchify(z)
        return dec

    def autoencode(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if self.use_variational and sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior, z

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def forward(self, inputs, labels, split="train"):
        rec, post, latent = self.autoencode(inputs)
        return rec, post, latent

    def get_last_layer(self):
        return self.predictor.weight


def ViT_L_20_Shallow_Encoder(**kwargs):
    if "latent_dim" in kwargs:
        latent_dim = kwargs.pop("latent_dim")
    else:
        latent_dim = 16
    return AutoencoderKL(
        latent_dim=latent_dim,
        patch_size=20,
        enc_dim=1024,
        enc_depth=6,
        enc_heads=16,
        dec_dim=1024,
        dec_depth=12,
        dec_heads=16,
        input_height=360,
        input_width=640,
        **kwargs,
    )


VAE_models = {
    "vit-l-20-shallow-encoder": ViT_L_20_Shallow_Encoder,
}
