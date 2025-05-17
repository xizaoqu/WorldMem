"""
Based on https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/attention.py
"""

from typing import Optional
from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
import numpy as np

class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        reference_length: int,
        rotary_emb: RotaryEmbedding,
        is_causal: bool = True,
        is_temporal_independent: bool = False,
        use_domain_adapter = False
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)

        self.use_domain_adapter = use_domain_adapter
        if self.use_domain_adapter:
            lora_rank = 8
            self.lora_A = nn.Linear(dim, lora_rank, bias=False)
            self.lora_B = nn.Linear(lora_rank, self.inner_dim * 3, bias=False)

        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.is_causal = is_causal
        self.is_temporal_independent = is_temporal_independent

        self.reference_length = reference_length

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        # if T>=9:
        #     try:
        #         # x = torch.cat([x[:,:-1],x[:,16-T:17-T],x[:,-1:]], dim=1)
        #         x = torch.cat([x[:,16-T:17-T],x], dim=1)
        #     except:
        #         import pdb;pdb.set_trace()
        #     print("="*50)
        #     print(x.shape)

        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        if self.use_domain_adapter:
            q_lora, k_lora, v_lora = self.lora_B(self.lora_A(x)).chunk(3, dim=-1)
            q = q+q_lora
            k = k+k_lora
            v = v+v_lora

        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        if self.is_temporal_independent:
            attn_bias = torch.ones((T, T), dtype=q.dtype, device=q.device)
            attn_bias = attn_bias.masked_fill(attn_bias == 1, float('-inf'))
            attn_bias[range(T), range(T)] = 0
        elif self.is_causal:
            attn_bias = torch.triu(torch.ones((T, T), dtype=q.dtype, device=q.device), diagonal=1)
            attn_bias = attn_bias.masked_fill(attn_bias == 1, float('-inf'))
            attn_bias[(T-self.reference_length):] = float('-inf')
            attn_bias[range(T), range(T)] = 0
        else:
            attn_bias = None

        try:
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_bias)
        except:
            import pdb;pdb.set_trace()

        x = rearrange(x, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)

        # if T>=10:
        #     try:
        #         # x = torch.cat([x[:,:-2],x[:,-1:]], dim=1)
        #         x = x[:,1:]
        #     except:
        #         import pdb;pdb.set_trace()
        #     print(x.shape)
        return x

class SpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        use_domain_adapter = False
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.use_domain_adapter = use_domain_adapter
        if self.use_domain_adapter:
            lora_rank = 8
            self.lora_A = nn.Linear(dim, lora_rank, bias=False)
            self.lora_B = nn.Linear(lora_rank, self.inner_dim * 3, bias=False)

        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        if self.use_domain_adapter:
            q_lora, k_lora, v_lora = self.lora_B(self.lora_A(x)).chunk(3, dim=-1)
            q = q+q_lora
            k = k+k_lora
            v = v+v_lora

        q = rearrange(q, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B T) h H W d", h=self.heads)

        freqs = self.rotary_emb.get_axial_freqs(H, W)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        # prepare for attn
        q = rearrange(q, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        k = rearrange(k, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        v = rearrange(v, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)

        x = rearrange(x, "(B T) h (H W) d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x

class MemTemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        is_causal: bool = True,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.is_causal = is_causal

        self.reference_length = 3

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)


        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        

        # q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        # k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # if T == 21000:
        #     # 手动计算缩放点积分数
        #     _, _, _, d_k = q.shape
        #     scores = torch.einsum("b h n d, b h m d -> b h n m", q, k) / (d_k ** 0.5)  # Shape: (B, T_q, T_k)

        #     # 计算注意力图 (Attention Map)
        #     attention_map = F.softmax(scores, dim=-1)  # Shape: (B, T_q, T_k)
        #     b_, h_, n_, m_ = attention_map.shape
        #     attention_map = attention_map.reshape(1, int(np.sqrt(b_/1)), int(np.sqrt(b_/1)), h_, n_, m_)
        #     attention_map = attention_map.mean(3)

        #     attn_bias = torch.zeros((T, T), dtype=q.dtype, device=q.device)
        #     T_origin = T - self.reference_length
        #     attn_bias[:T_origin, T_origin:] = 1
        #     attn_bias[range(T), range(T)] = 1

        #     attention_map = attention_map * attn_bias

            # # print 注意力图
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(21000, 21000, figsize=(9, 9))  # 调整figsize以适配图像大小

            # # 遍历3*3维度
            # for i in range(21000):
            #     for j in range(21000):
            #         # 取出第(i, j)个子图像
            #         img = attention_map[0, :, :, i, j].cpu().numpy()
            #         axes[i, j].imshow(img, cmap='viridis')  # 可以自定义cmap
            #         axes[i, j].axis('off')  # 隐藏坐标轴

            # # 调整子图间距
            # plt.tight_layout()
            # plt.savefig('attention_map.png')
            # import pdb; pdb.set_trace()
            # plt.close()

        attn_bias = torch.zeros((T, T), dtype=q.dtype, device=q.device)
        attn_bias = attn_bias.masked_fill(attn_bias == 0, float('-inf'))
        T_origin = T - self.reference_length
        attn_bias[:T_origin, T_origin:] = 0
        attn_bias[range(T), range(T)] = 0

        # if T==121000:
        #     import pdb;pdb.set_trace()

        try:
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_bias)
        except:
            import pdb;pdb.set_trace()

        x = rearrange(x, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x

class MemFullAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        reference_length: int,
        rotary_emb: RotaryEmbedding,
        is_causal: bool = True
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.is_causal = is_causal

        self.reference_length = reference_length

        self.store = None

    def forward(self, x: torch.Tensor, relative_embedding=False,
                extra_condition=None,
                state_embed_only_on_qk=False,
                reference_length=None):

        B, T, H, W, D = x.shape

        if state_embed_only_on_qk:
            q, k, _ = self.to_qkv(x+extra_condition).chunk(3, dim=-1)
            _, _, v = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        if relative_embedding:
            length = reference_length+1
            n_frames = T // length
            x = x.reshape(B, n_frames, length, H, W, D)

            x_list = []

            for i in range(n_frames):
                if i == n_frames-1:
                    q_i = rearrange(q[:, i*length:], "B T H W (h d) -> B h (T H W) d", h=self.heads)
                    k_i = rearrange(k[:, i*length+1:(i+1)*length], "B T H W (h d) -> B h (T H W) d", h=self.heads)
                    v_i = rearrange(v[:, i*length+1:(i+1)*length], "B T H W (h d) -> B h (T H W) d", h=self.heads)                
                else:
                    q_i = rearrange(q[:, i*length:i*length+1], "B T H W (h d) -> B h (T H W) d", h=self.heads)
                    k_i = rearrange(k[:, i*length+1:(i+1)*length], "B T H W (h d) -> B h (T H W) d", h=self.heads)
                    v_i = rearrange(v[:, i*length+1:(i+1)*length], "B T H W (h d) -> B h (T H W) d", h=self.heads)

                q_i, k_i, v_i = map(lambda t: t.contiguous(), (q_i, k_i, v_i))
                x_i = F.scaled_dot_product_attention(query=q_i, key=k_i, value=v_i)
                x_i = rearrange(x_i, "B h (T H W) d -> B T H W (h d)", B=B, H=H, W=W)
                x_i = x_i.to(q.dtype)
                x_list.append(x_i)
        
            x = torch.cat(x_list, dim=1)


        else:
            T_ = T - reference_length
            q = rearrange(q, "B T H W (h d) -> B h (T H W) d", h=self.heads)
            k = rearrange(k[:, T_:], "B T H W (h d) -> B h (T H W) d", h=self.heads)
            v = rearrange(v[:, T_:], "B T H W (h d) -> B h (T H W) d", h=self.heads)

            q, k, v = map(lambda t: t.contiguous(), (q, k, v))
            x = F.scaled_dot_product_attention(query=q, key=k, value=v)
            x = rearrange(x, "B h (T H W) d -> B T H W (h d)", B=B, H=H, W=W)
            x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)

        return x
