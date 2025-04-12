"""
Adapted from https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/utils.py
Action format derived from VPT https://github.com/openai/Video-Pre-Training
Adapted from https://github.com/etched-ai/open-oasis/blob/master/utils.py
"""

import math
import torch
from torch import nn
from torchvision.io import read_image, read_video
from torchvision.transforms.functional import resize
from einops import rearrange
from typing import Mapping, Sequence
from einops import rearrange, parse_shape


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    f, b = t.shape
    out = a[t]
    return out.reshape(f, b, *((1,) * (len(x_shape) - 2)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)



def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]


def one_hot_actions(actions: Sequence[Mapping[str, int]]) -> torch.Tensor:
    actions_one_hot = torch.zeros(len(actions), len(ACTION_KEYS))
    for i, current_actions in enumerate(actions):
        for j, action_key in enumerate(ACTION_KEYS):
            if action_key.startswith("camera"):
                if action_key == "cameraX":
                    value = current_actions["camera"][0]
                elif action_key == "cameraY":
                    value = current_actions["camera"][1]
                else:
                    raise ValueError(f"Unknown camera action key: {action_key}")
                max_val = 20
                bin_size = 0.5
                num_buckets = int(max_val / bin_size)
                value = (value - num_buckets) / num_buckets
                assert -1 - 1e-3 <= value <= 1 + 1e-3, f"Camera action value must be in [-1, 1], got {value}"
            else:
                value = current_actions[action_key]
                assert 0 <= value <= 1, f"Action value must be in [0, 1] got {value}"
            actions_one_hot[i, j] = value

    return actions_one_hot


IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
VIDEO_EXTENSIONS = {"mp4"}


def load_prompt(path, video_offset=None, n_prompt_frames=1):
    if path.lower().split(".")[-1] in IMAGE_EXTENSIONS:
        print("prompt is image; ignoring video_offset and n_prompt_frames")
        prompt = read_image(path)
        # add frame dimension
        prompt = rearrange(prompt, "c h w -> 1 c h w")
    elif path.lower().split(".")[-1] in VIDEO_EXTENSIONS:
        prompt = read_video(path, pts_unit="sec")[0]
        if video_offset is not None:
            prompt = prompt[video_offset:]
        prompt = prompt[:n_prompt_frames]
    else:
        raise ValueError(f"unrecognized prompt file extension; expected one in {IMAGE_EXTENSIONS} or {VIDEO_EXTENSIONS}")
    assert prompt.shape[0] == n_prompt_frames, f"input prompt {path} had less than n_prompt_frames={n_prompt_frames} frames"
    prompt = resize(prompt, (360, 640))
    # add batch dimension
    prompt = rearrange(prompt, "t c h w -> 1 t c h w")
    prompt = prompt.float() / 255.0
    return prompt


def load_actions(path, action_offset=None):
    if path.endswith(".actions.pt"):
        actions = one_hot_actions(torch.load(path))
    elif path.endswith(".one_hot_actions.pt"):
        actions = torch.load(path, weights_only=True)
    else:
        raise ValueError("unrecognized action file extension; expected '*.actions.pt' or '*.one_hot_actions.pt'")
    if action_offset is not None:
        actions = actions[action_offset:]
    actions = torch.cat([torch.zeros_like(actions[:1]), actions], dim=0)
    # add batch dimension
    actions = rearrange(actions, "t d -> 1 t d")
    return actions
