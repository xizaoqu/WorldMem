import os
import io
import tarfile
import numpy as np
import torch
from typing import Sequence, Mapping
from omegaconf import DictConfig
from pytorchvideo.data.encoded_video import EncodedVideo
import random

from .base_video_dataset import BaseVideoDataset




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
    "cameraY",
    "cameraX",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]

def convert_action_space(actions):
    vec_25 = torch.zeros(len(actions), len(ACTION_KEYS))
    vec_25[actions[:,0]==1, 11] = 1
    vec_25[actions[:,0]==2, 12] = 1
    vec_25[actions[:,4]==11, 16] = -1
    vec_25[actions[:,4]==13, 16] = 1
    vec_25[actions[:,3]==11, 15] = -1
    vec_25[actions[:,3]==13, 15] = 1
    vec_25[actions[:,5]==6, 24] = 1
    vec_25[actions[:,5]==1, 24] = 1
    vec_25[actions[:,1]==1, 13] = 1
    vec_25[actions[:,1]==2, 14] = 1
    vec_25[actions[:,7]==1, 2] = 1
    return vec_25

# Dataset class
class MinecraftVideoDataset(BaseVideoDataset):
    """
    Minecraft video dataset for training and validation.

    Args:
        cfg (DictConfig): Configuration object.
        split (str): Dataset split ("training" or "validation").
    """
    def __init__(self, cfg: DictConfig, split: str = "training"):
        if split == "test":
            split = "validation"
        self.wo_updown = getattr(cfg, "wo_updown", False)
        super().__init__(cfg, split)
        self.n_frames = cfg.n_frames_valid if split == "validation" and hasattr(cfg, "n_frames_valid") else cfg.n_frames
        self.memory_condition_length = cfg.memory_condition_length
        self.customized_validation = cfg.customized_validation
        if split == "training":
            self.angle_range = cfg.angle_range
            self.pos_range = cfg.pos_range
        self.add_timestamp_embedding = cfg.add_timestamp_embedding
        self.training_dropout = 0.1
        self.memory_condition_length = getattr(cfg, "memory_condition_length", False)
        self.sample_more_event = getattr(cfg, "sample_more_event", False)
        self.causal_frame = getattr(cfg, "causal_frame", False)

    def get_data_paths(self, split: str):
        """
        Retrieve all video file paths for the given split.

        Args:
            split (str): Dataset split ("training" or "validation").

        Returns:
            List[Path]: List of video file paths.
        """
        data_dir = self.save_dir / split
        paths = sorted(list(data_dir.glob("**/*.mp4")), key=lambda x: x.name)

        if self.wo_updown:
            # Filter out paths containing "w_updown"
            paths = [p for p in paths if "w_updown" not in str(p)]
        
        if split == "validation" and self.wo_updown:
            paths = [p for p in paths if "w_updown" not in str(p)]
        elif split == "validation":
            paths = [p for p in paths if "w_updown" in str(p)]

        if not paths:
            sub_dirs = os.listdir(data_dir)
            for sub_dir in sub_dirs:
                sub_path = data_dir / sub_dir
                paths += sorted(list(sub_path.glob("**/*.mp4")), key=lambda x: x.name)
        return paths

    def download_dataset(self):
        pass
    
    def __getitem__(self, idx: int):
        """
        Retrieve a single data sample by index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]: Video, actions, poses, and timestamps.
        """
        max_retries = 1000
        for _ in range(max_retries):
            try:
                return self.load_data(idx)
            except Exception as e:
                print(f"Retrying due to error: {e}")
                idx = (idx + 1) % len(self)

    def load_data(self, idx):
        # === 1. Remap index and skip first few frames ===
        idx = self.idx_remap[idx]
        file_idx, frame_idx = self.split_idx(idx)
        frame_idx += 100 # initial few frames are low quality

        # === 2. Load paths and data arrays ===
        video_path = self.data_paths[file_idx]
        action_path = video_path.with_suffix(".npz")
        data = np.load(action_path)
        actions_pool = convert_action_space(data["actions"])
        poses_pool = data["poses"]

        # Fix corrupted height (maybe) in the first frame
        poses_pool[0, 1] = poses_pool[1, 1]
        assert poses_pool[:, 1].ptp() < 2, f"Height variation too large: {poses_pool[:, 1].ptp()} - {video_path}"

        # Pad poses if shorter than actions
        if len(poses_pool) < len(actions_pool):
            poses_pool = np.pad(poses_pool, ((1, 0), (0, 0)))

        # === 3. Load video clip ===
        video_raw = EncodedVideo.from_path(video_path, decode_audio=False)
        fps = 10
        clip = video_raw.get_clip(
            start_sec=frame_idx / fps,
            end_sec=(frame_idx + self.n_frames) / fps
        )["video"]
        video = clip.permute(1, 2, 3, 0).numpy()

        actions = np.copy(actions_pool[frame_idx : frame_idx + self.n_frames])
        poses = np.copy(poses_pool[frame_idx : frame_idx + self.n_frames])

        # === 4. Normalize poses relative to current segment ===
        def normalize_pose(pose, ref_pose):
            pose[:, :3] -= ref_pose[:1, :3]
            pose[:, -1] = -pose[:, -1]
            pose[:, 3:] %= 360
            return pose

        poses_pool = normalize_pose(poses_pool, poses)
        poses = normalize_pose(poses, poses)

        assert len(video) >= self.n_frames, f"{video_path}"

        # === 5. Sample memory frames for training ===
        if self.split == "training" and self.memory_condition_length > 0:
            use_memory = random.uniform(0, 1) > self.training_dropout

            if use_memory:
                # Compute pose distance between current and candidate frames
                dis = np.abs(poses[:, None] - poses_pool[None, :])
                dis[..., 3:][dis[..., 3:] > 180] = 360 - dis[..., 3:][dis[..., 3:] > 180]

                spatial_match = (dis[..., :3] <= self.pos_range).sum(-1) >= 3
                angular_match = (dis[..., 3:] <= self.angle_range).sum(-1) >= 2
                not_exact_match = ((dis[..., :3] > 0).sum(-1) >= 1) | ((dis[..., 3:] > 0).sum(-1) >= 1)

                valid_index = (spatial_match & angular_match & not_exact_match).sum(0)
                valid_index[:100] = 0  # skip unstable early frames

                # Exclude future if causality and timestamp are enabled
                if self.add_timestamp_embedding and self.causal_frame and (actions_pool[:frame_idx, 24] == 1).sum() > 0:
                    valid_index[frame_idx:] = 0

                # Select indices satisfying condition
                mask = valid_index >= 1
                mask[0] = False
                candidate_indices = np.argwhere(mask)

                # Backup candidates with weaker conditions
                mask2 = valid_index >= 0
                mask2[0] = False

                count = min(self.memory_condition_length, candidate_indices.shape[0])
                selected = candidate_indices[np.random.choice(candidate_indices.shape[0], count, replace=True)][:, 0]

                if count < self.memory_condition_length:
                    extra = np.argwhere(mask2)
                    extra = extra[np.random.choice(extra.shape[0], self.memory_condition_length - count, replace=True)][:, 0]
                    selected = np.concatenate([selected, extra])

                # Prioritize event-trigger frames if applicable
                if self.sample_more_event and random.uniform(0, 1) > 0.3:
                    event_idx = torch.nonzero(actions_pool[:frame_idx, 24] == 1)[:, 0]
                    if len(event_idx) > self.memory_condition_length // 2:
                        event_idx = event_idx[-self.memory_condition_length // 2:]
                    if len(event_idx) > 0:
                        selected[-len(event_idx):] = event_idx + 4

            else:
                selected = np.full(self.memory_condition_length, random.randint(0, frame_idx))

            # === 6. Retrieve video frames for selected memory indices ===
            video_pool = []
            for si in selected:
                frame = video_raw.get_clip(start_sec=si / fps, end_sec=(si + 1) / fps)["video"][:, 0].permute(1, 2, 0)
                video_pool.append(frame)

            video = np.concatenate([video, np.stack(video_pool)], axis=0)
            actions = np.concatenate([actions, actions_pool[selected]], axis=0)
            poses = np.concatenate([poses, poses_pool[selected]], axis=0)
            timestamp = np.concatenate([np.arange(frame_idx, frame_idx + self.n_frames), selected])
        else:
            timestamp = np.arange(self.n_frames)

        # === 7. Convert video to torch format ===
        video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()

        # === 9. Return all items ===
        return (
            video[:: self.frame_skip],
            actions[:: self.frame_skip],
            poses[:: self.frame_skip],
            timestamp
        )
