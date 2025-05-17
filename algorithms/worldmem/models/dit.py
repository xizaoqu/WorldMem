"""
References:
    - DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/unet3d.py
    - Latte: https://github.com/Vchitect/Latte/blob/main/models/latte.py
"""

from typing import Optional, Literal
import torch
from torch import nn
from .rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
from .attention import SpatialAxialAttention, TemporalAxialAttention, MemTemporalAxialAttention, MemFullAttention
from timm.models.vision_transformer import Mlp
from timm.layers.helpers import to_2tuple
import math
from collections import namedtuple
from typing import Optional, Callable
from .cameractrl_module import SimpleCameraPoseEncoder

def modulate(x, shift, scale):
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // scale.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift

def gate(x, g):
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_height=256,
        img_width=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = (img_height, img_width)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, random_sample=False):
        B, C, H, W = x.shape
        assert random_sample or (H == self.img_size[0] and W == self.img_size[1]), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "B C H W -> B (H W) C")
        else:
            x = rearrange(x, "B C H W -> B H W C")
        x = self.norm(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, freq_type='time_step'):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),  # hidden_size is diffusion model hidden size
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.freq_type = freq_type

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, freq_type='time_step'):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2

        if freq_type == 'time_step':
            freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        elif freq_type == 'spatial': # ~(-5 5)
            freqs = torch.linspace(1.0, half, half).to(device=t.device) * torch.pi
        elif freq_type == 'angle': # 0-360
            freqs = torch.linspace(1.0, half, half).to(device=t.device) * torch.pi / 180


        args = t[:, None].float() * freqs[None]
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, freq_type=self.freq_type)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        reference_length,
        mlp_ratio=4.0,
        is_causal=True,
        spatial_rotary_emb: Optional[RotaryEmbedding] = None,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
        reference_rotary_emb=None,
        use_plucker=False,
        relative_embedding=False,
        state_embed_only_on_qk=False,
        use_memory_attention=False,
        ref_mode='sequential'
    ):
        super().__init__()
        self.is_causal = is_causal
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.s_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_attn = SpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb
        )
        self.s_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.s_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.t_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_attn = TemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            is_causal=is_causal,
            rotary_emb=temporal_rotary_emb,
            reference_length=reference_length
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.t_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.use_memory_attention = use_memory_attention
        if self.use_memory_attention:
            self.r_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.ref_type = "full_ref"
            if self.ref_type == "temporal_ref":
                self.r_attn = MemTemporalAxialAttention(
                    hidden_size,
                    heads=num_heads,
                    dim_head=hidden_size // num_heads,
                    is_causal=is_causal,
                    rotary_emb=None
                )
            elif self.ref_type == "full_ref":
                self.r_attn = MemFullAttention(
                    hidden_size,
                    heads=num_heads,
                    dim_head=hidden_size // num_heads,
                    is_causal=is_causal,
                    rotary_emb=reference_rotary_emb,
                    reference_length=reference_length
                )
            self.r_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.r_mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
            self.r_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

            self.use_plucker = use_plucker
            if use_plucker:
                self.pose_cond_mlp = nn.Linear(hidden_size, hidden_size)
                self.temporal_pose_cond_mlp = nn.Linear(hidden_size, hidden_size)

        self.reference_length = reference_length
        self.relative_embedding = relative_embedding
        self.state_embed_only_on_qk = state_embed_only_on_qk

        self.ref_mode = ref_mode

        if self.ref_mode == 'parallel':
            self.parallel_map = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c, current_frame=None, timestep=None, is_last_block=False, 
        pose_cond=None, mode="training", c_action_cond=None, reference_length=None):
        B, T, H, W, D = x.shape

        # spatial block
        
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # temporal block
        if c_action_cond is not None:
            t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c_action_cond).chunk(6, dim=-1)
        else:
            t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        
        x_t = x + gate(self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x_t = x_t + gate(self.t_mlp(modulate(self.t_norm2(x_t), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        if self.ref_mode == 'sequential':
            x = x_t

        # memory block
        relative_embedding = self.relative_embedding # and mode == "training"

        if self.use_memory_attention:
            r_shift_msa, r_scale_msa, r_gate_msa, r_shift_mlp, r_scale_mlp, r_gate_mlp = self.r_adaLN_modulation(c).chunk(6, dim=-1)

            if pose_cond is not None:
                if self.use_plucker:
                    input_cond = self.pose_cond_mlp(pose_cond)
                    
                    if relative_embedding:
                        n_frames = x.shape[1] - reference_length
                        x1_relative_embedding = []
                        r_shift_msa_relative_embedding = []
                        r_scale_msa_relative_embedding = []
                        for i in range(n_frames):
                            x1_relative_embedding.append(torch.cat([x[:,i:i+1], x[:, -reference_length:]], dim=1).clone())
                            r_shift_msa_relative_embedding.append(torch.cat([r_shift_msa[:,i:i+1], r_shift_msa[:, -reference_length:]], dim=1).clone())
                            r_scale_msa_relative_embedding.append(torch.cat([r_scale_msa[:,i:i+1], r_scale_msa[:, -reference_length:]], dim=1).clone())
                        x1_zero_frame = torch.cat(x1_relative_embedding, dim=1)
                        r_shift_msa = torch.cat(r_shift_msa_relative_embedding, dim=1)
                        r_scale_msa = torch.cat(r_scale_msa_relative_embedding, dim=1)

                        # if current_frame == 18:
                        #     import pdb;pdb.set_trace()

                        if self.state_embed_only_on_qk:
                            attn_input = x1_zero_frame
                            extra_condition = input_cond                        
                        else:
                            attn_input = input_cond + x1_zero_frame
                            extra_condition = None
                    else:
                        attn_input = input_cond + x
                        extra_condition = None
                    # print("input_cond2:", input_cond.abs().mean())
                    # print("c:", c.abs().mean())
                    # input_cond = x1

                    x = x + gate(self.r_attn(modulate(self.r_norm1(attn_input), r_shift_msa, r_scale_msa), 
                                    relative_embedding=relative_embedding, 
                                    extra_condition=extra_condition, 
                                    state_embed_only_on_qk=self.state_embed_only_on_qk,
                                    reference_length=reference_length), r_gate_msa)
                else:
                    # pose_cond *= 0
                    x = x + gate(self.r_attn(modulate(self.r_norm1(x+pose_cond[:,:,None, None]), r_shift_msa, r_scale_msa), 
                                    current_frame=current_frame, timestep=timestep,
                                    is_last_block=is_last_block,
                                    reference_length=reference_length), r_gate_msa)
            else:
                x = x + gate(self.r_attn(modulate(self.r_norm1(x), r_shift_msa, r_scale_msa), current_frame=current_frame, timestep=timestep,
                                is_last_block=is_last_block), r_gate_msa)

            x = x + gate(self.r_mlp(modulate(self.r_norm2(x), r_shift_mlp, r_scale_mlp)), r_gate_mlp)

        if self.ref_mode == 'parallel':
            x = x_t + self.parallel_map(x)

        return x

        # print((x1-x2).abs().sum())
        # r_shift_msa, r_scale_msa, r_gate_msa, r_shift_mlp, r_scale_mlp, r_gate_mlp = self.r_adaLN_modulation(c).chunk(6, dim=-1)
        # x2 = x1 + gate(self.r_attn(modulate(self.r_norm1(x_), r_shift_msa, r_scale_msa)), r_gate_msa)
        # x2 = gate(self.r_mlp(modulate(self.r_norm2(x2), r_shift_mlp, r_scale_mlp)), r_gate_mlp)
        # x = x1 + x2

        # print(x.mean())
        # return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_h=18,
        input_w=32,
        patch_size=2,
        in_channels=16,
        hidden_size=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        action_cond_dim=25,
        pose_cond_dim=4,
        max_frames=32,
        reference_length=8,
        use_plucker=False,
        relative_embedding=False,
        state_embed_only_on_qk=False,
        use_memory_attention=False,
        add_timestamp_embedding=False,
        ref_mode='sequential'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames

        self.x_embedder = PatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.add_timestamp_embedding = add_timestamp_embedding
        if self.add_timestamp_embedding:
            self.timestamp_embedding = TimestepEmbedder(hidden_size)

        frame_h, frame_w = self.x_embedder.grid_size

        self.spatial_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads)
        # self.reference_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads // 2, freqs_for="pixel", max_freq=256)
        self.reference_rotary_emb = None

        self.external_cond = nn.Linear(action_cond_dim, hidden_size) if action_cond_dim > 0 else nn.Identity()

        # self.pose_cond = nn.Linear(pose_cond_dim, hidden_size) if pose_cond_dim > 0 else nn.Identity()
        
        self.use_plucker = use_plucker
        if not self.use_plucker:
            self.position_embedder = TimestepEmbedder(hidden_size, freq_type='spatial')
            self.angle_embedder = TimestepEmbedder(hidden_size, freq_type='angle')
        else:
            self.pose_embedder = SimpleCameraPoseEncoder(c_in=6, c_out=hidden_size)

        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    reference_length=reference_length,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                    reference_rotary_emb=self.reference_rotary_emb,
                    use_plucker=self.use_plucker,
                    relative_embedding=relative_embedding,
                    state_embed_only_on_qk=state_embed_only_on_qk,
                    use_memory_attention=use_memory_attention,
                    ref_mode=ref_mode
                )
                for _ in range(depth)
            ]
        )
        self.use_memory_attention = use_memory_attention
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.use_memory_attention:
            if not self.use_plucker:
                nn.init.normal_(self.position_embedder.mlp[0].weight, std=0.02)
                nn.init.normal_(self.position_embedder.mlp[2].weight, std=0.02)

                nn.init.normal_(self.angle_embedder.mlp[0].weight, std=0.02)
                nn.init.normal_(self.angle_embedder.mlp[2].weight, std=0.02)
            
            if self.add_timestamp_embedding:
                nn.init.normal_(self.timestamp_embedding.mlp[0].weight, std=0.02)
                nn.init.normal_(self.timestamp_embedding.mlp[2].weight, std=0.02)


        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)

            if self.use_plucker and self.use_memory_attention:
                nn.init.constant_(block.pose_cond_mlp.weight, 0)
                nn.init.constant_(block.pose_cond_mlp.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, H, W, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, action_cond=None, pose_cond=None, current_frame=None, mode=None, 
                    reference_length=None, frame_idx=None):
        """
        Forward pass of DiT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B, T,) tensor of diffusion timesteps
        """

        B, T, C, H, W = x.shape

        # add spatial embeddings
        x = rearrange(x, "b t c h w -> (b t) c h w")

        x = self.x_embedder(x)  # (B*T, C, H, W) -> (B*T, H/2, W/2, D) , C = 16, D = d_model
        # restore shape
        x = rearrange(x, "(b t) h w d -> b t h w d", t=T)
        # embed noise steps
        t = rearrange(t, "b t -> (b t)")

        c_t = self.t_embedder(t)  # (N, D)
        c = c_t.clone()
        c = rearrange(c, "(b t) d -> b t d", t=T)

        if torch.is_tensor(action_cond):
            try:
                c_action_cond = c + self.external_cond(action_cond)
            except:
                import pdb;pdb.set_trace()
        else:
            c_action_cond = None
        
        if torch.is_tensor(pose_cond):
            if not self.use_plucker:
                pose_cond = pose_cond.to(action_cond.dtype)
                b_, t_, d_ = pose_cond.shape
                pos_emb = self.position_embedder(rearrange(pose_cond[...,:3], "b t d -> (b t d)"))
                angle_emb = self.angle_embedder(rearrange(pose_cond[...,3:], "b t d -> (b t d)"))
                pos_emb = rearrange(pos_emb, "(b t d) c -> b t d c", b=b_, t=t_, d=3).sum(-2)
                angle_emb = rearrange(angle_emb, "(b t d) c -> b t d c", b=b_, t=t_, d=2).sum(-2)
                pc = pos_emb + angle_emb
            else:
                pose_cond = pose_cond[:, :, ::40, ::40]
                # pc = self.pose_embedder(pose_cond)[0]
                # pc = pc.permute(0,2,3,4,1)
                pc = self.pose_embedder(pose_cond)
                pc = pc.permute(1,0,2,3,4)

                if torch.is_tensor(frame_idx) and self.add_timestamp_embedding:
                    bb = frame_idx.shape[1]
                    frame_idx = rearrange(frame_idx, "t b -> (b t)")
                    frame_idx = self.timestamp_embedding(frame_idx)
                    frame_idx = rearrange(frame_idx, "(b t) d -> b t d", b=bb)
                    pc = pc + frame_idx[:, :, None, None]

                # pc = pc + rearrange(c_t.clone(), "(b t) d -> b t d", t=T)[:,:,None,None] # add time condition for different timestep scaling 
        else:
            pc = None
        
        for i, block in enumerate(self.blocks):
            x = block(x, c, current_frame=current_frame, timestep=t, is_last_block= (i+1 == len(self.blocks)), 
                pose_cond=pc, mode=mode, c_action_cond=c_action_cond, reference_length=reference_length)  # (N, T, H, W, D)
        x = self.final_layer(x, c) # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)
        return x


def DiT_S_2(action_cond_dim, pose_cond_dim, reference_length, 
use_plucker, relative_embedding, 
state_embed_only_on_qk, use_memory_attention, add_timestamp_embedding,
ref_mode):
    return DiT(
        patch_size=2,
        hidden_size=1024,
        depth=16,
        num_heads=16,
        action_cond_dim=action_cond_dim,
        pose_cond_dim=pose_cond_dim,
        reference_length=reference_length,
        use_plucker=use_plucker,
        relative_embedding=relative_embedding,
        state_embed_only_on_qk=state_embed_only_on_qk,
        use_memory_attention=use_memory_attention,
        add_timestamp_embedding=add_timestamp_embedding,
        ref_mode=ref_mode
    )


DiT_models = {"DiT-S/2": DiT_S_2}
