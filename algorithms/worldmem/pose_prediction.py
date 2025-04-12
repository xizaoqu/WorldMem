from omegaconf import DictConfig
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from algorithms.common.metrics import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetVideoDistance,
)
from .df_base import DiffusionForcingBase
from utils.logging_utils import log_video, get_validation_metrics_for_videos
from .models.vae import VAE_models
from .models.dit import DiT_models
from einops import rearrange
from torch import autocast
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from .models.pose_prediction import PosePredictionNet
import torchvision.transforms.functional as TF
import random
from torchvision.transforms import InterpolationMode
from PIL import Image
import math
from packaging import version as pver
import torch.distributed as dist
import matplotlib.pyplot as plt

import torch
import math
import wandb

import torch.nn as nn
from algorithms.common.base_pytorch_algo import BasePytorchAlgo

class PosePrediction(BasePytorchAlgo):

    def __init__(self, cfg: DictConfig):

        super().__init__(cfg)

    def _build_model(self):
        self.pose_prediction_model = PosePredictionNet()
        vae = VAE_models["vit-l-20-shallow-encoder"]()
        self.vae = vae.eval()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        xs, conditions, pose_conditions= batch
        pose_conditions[:,:,3:] = pose_conditions[:,:,3:] // 15
        xs = self.encode(xs)

        b,f,c,h,w = xs.shape
        xs = xs[:,:-1].reshape(-1, c, h, w)
        conditions = conditions[:,1:].reshape(-1, 25)
        offset_gt = pose_conditions[:,1:] - pose_conditions[:,:-1]
        pose_conditions = pose_conditions[:,:-1].reshape(-1, 5)
        offset_gt = offset_gt.reshape(-1, 5)
        offset_gt[:, 3][offset_gt[:, 3]==23] = -1
        offset_gt[:, 3][offset_gt[:, 3]==-23] = 1
        offset_gt[:, 4][offset_gt[:, 4]==23] = -1
        offset_gt[:, 4][offset_gt[:, 4]==-23] = 1

        offset_pred = self.pose_prediction_model(xs, conditions, pose_conditions)
        criterion = nn.MSELoss()
        loss = criterion(offset_pred, offset_gt)
        if batch_idx % 200 == 0:
            self.log("training/loss", loss.cpu())
        output_dict = {
            "loss": loss}
        return output_dict

    def encode(self, x):
        # vae encoding
        B = x.shape[1]
        T = x.shape[0]
        H, W = x.shape[-2:]
        scaling_factor = 0.07843137255

        x = rearrange(x, "t b c h w -> (t b) c h w")
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                x = self.vae.encode(x * 2 - 1).mean * scaling_factor
        x = rearrange(x, "(t b) (h w) c -> t b c h w", t=T, h=H // self.vae.patch_size, w=W // self.vae.patch_size)
        # x = x[:, :n_prompt_frames]
        return x

    def decode(self, x):
        total_frames = x.shape[0]
        scaling_factor = 0.07843137255
        x = rearrange(x, "t b c h w -> (t b) (h w) c")
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                x = (self.vae.decode(x / scaling_factor) + 1) / 2

        x = rearrange(x, "(t b) c h w-> t b c h w", t=total_frames)
        return x

    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, pose_conditions= batch
        pose_conditions[:,:,3:] = pose_conditions[:,:,3:] // 15
        xs = self.encode(xs)

        b,f,c,h,w = xs.shape
        xs = xs[:,:-1].reshape(-1, c, h, w)
        conditions = conditions[:,1:].reshape(-1, 25)
        offset_gt = pose_conditions[:,1:] - pose_conditions[:,:-1]
        pose_conditions = pose_conditions[:,:-1].reshape(-1, 5)
        offset_gt = offset_gt.reshape(-1, 5)
        offset_gt[:, 3][offset_gt[:, 3]==23] = -1
        offset_gt[:, 3][offset_gt[:, 3]==-23] = 1
        offset_gt[:, 4][offset_gt[:, 4]==23] = -1
        offset_gt[:, 4][offset_gt[:, 4]==-23] = 1

        offset_pred = self.pose_prediction_model(xs, conditions, pose_conditions)

        criterion = nn.MSELoss()
        loss = criterion(offset_pred, offset_gt)

        if batch_idx % 200 == 0:
            self.log("validation/loss", loss.cpu())
        output_dict = {
            "loss": loss}
        return

    @torch.no_grad()
    def interactive(self, batch, context_frames, device):
        with torch.cuda.amp.autocast():
            condition_similar_length = self.condition_similar_length
            # xs_raw, conditions, pose_conditions, c2w_mat, masks, frame_idx = self._preprocess_batch(batch)

            first_frame, new_conditions, new_pose_conditions, new_c2w_mat, new_frame_idx = batch

            if self.frames is None:
                first_frame_encode = self.encode(first_frame[None, None].to(device))
                self.frames = first_frame_encode.to(device)
                self.actions = new_conditions[None, None].to(device)
                self.poses = new_pose_conditions[None, None].to(device)
                self.memory_c2w = new_c2w_mat[None, None].to(device)
                self.frame_idx = torch.tensor([[new_frame_idx]]).to(device)
                return first_frame
            else:
                self.actions = torch.cat([self.actions, new_conditions[None, None].to(device)])
                self.poses = torch.cat([self.poses, new_pose_conditions[None, None].to(device)])
                self.memory_c2w = torch.cat([self.memory_c2w, new_c2w_mat[None, None].to(device)])
                self.frame_idx = torch.cat([self.frame_idx, torch.tensor([[new_frame_idx]]).to(device)])

            conditions = self.actions.clone()
            pose_conditions = self.poses.clone()
            c2w_mat = self.memory_c2w .clone()
            frame_idx = self.frame_idx.clone()


            curr_frame = 0
            horizon = 1
            batch_size = 1
            n_frames = curr_frame + horizon
            # context
            n_context_frames = context_frames // self.frame_stack
            xs_pred = self.frames[:n_context_frames].clone()
            curr_frame += n_context_frames

            pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")

            # generation on frame
            scheduling_matrix = self._generate_scheduling_matrix(horizon)
            chunk = torch.randn((horizon, batch_size, *xs_pred.shape[2:])).to(xs_pred.device)
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)

            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_tokens frames
            start_frame = max(0, curr_frame + horizon - self.n_tokens)

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + horizon,
                }
            )

            if condition_similar_length:

                if curr_frame < condition_similar_length:
                    random_idx = [i for i in range(curr_frame)] + [0] * (condition_similar_length-curr_frame)
                    random_idx = np.repeat(np.array(random_idx)[:,None], xs_pred.shape[1], -1)
                else:
                    num_samples = 10000
                    radius = 30
                    samples = torch.rand((num_samples, 1), device=pose_conditions.device)
                    angles = 2 * np.pi * torch.rand((num_samples,), device=pose_conditions.device)
                    # points = radius * torch.sqrt(samples) * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
                    
                    points = generate_points_in_sphere(num_samples, radius).to(pose_conditions.device)
                    points = points[:, None].repeat(1, pose_conditions.shape[1], 1)
                    points += pose_conditions[curr_frame, :, :3][None]
                    fov_half_h = torch.tensor(105/2, device=pose_conditions.device)
                    fov_half_v = torch.tensor(75/2, device=pose_conditions.device)
                    # in_fov1 = is_inside_fov(points, pose_conditions[curr_frame, :, [0, 2]], pose_conditions[curr_frame, :, -1], fov_half)

                    in_fov1 = is_inside_fov_3d_hv(points, pose_conditions[curr_frame, :, :3], 
                        pose_conditions[curr_frame, :, -2], pose_conditions[curr_frame, :, -1],
                        fov_half_h, fov_half_v)

                    in_fov_list = []
                    for pc in pose_conditions[:curr_frame]:
                        in_fov_list.append(is_inside_fov_3d_hv(points, pc[:, :3], pc[:, -2], pc[:, -1],
                                                        fov_half_h, fov_half_v))
                    
                    in_fov_list = torch.stack(in_fov_list)
                    # v3
                    random_idx = []

                    for csl in range(self.condition_similar_length // 2):
                        overlap_ratio = ((in_fov1[None].bool() & in_fov_list).sum(1))/in_fov1.sum()
                        # mask = distance > (in_fov1.bool().sum(0) / 4)
                        #_, r_idx = torch.topk(overlap_ratio / tensor_max_with_number((frame_idx[curr_frame] - frame_idx[:curr_frame]), 10), k=1, dim=0)
                        
                        # if csl > self.condition_similar_length:
                        # 	_, r_idx = torch.topk(overlap_ratio, k=1, dim=0)
                        # else:
                        # 	_, r_idx = torch.topk(overlap_ratio / tensor_max_with_number((frame_idx[curr_frame] - frame_idx[:curr_frame]), 10), k=1, dim=0)

                        _, r_idx = torch.topk(overlap_ratio, k=1, dim=0)
                        # _, r_idx = torch.topk(overlap_ratio / tensor_max_with_number((frame_idx[curr_frame] - frame_idx[:curr_frame]), 10), k=1, dim=0)

                        # if curr_frame >=93:
                        #     import pdb;pdb.set_trace()

                        # start_time = time.time()
                        cos_sim = F.cosine_similarity(xs_pred.to(r_idx.device)[r_idx[:, range(in_fov1.shape[1])], 
                            range(in_fov1.shape[1])], xs_pred.to(r_idx.device)[:curr_frame], dim=2)
                        cos_sim = cos_sim.mean((-2,-1))

                        mask_sim = cos_sim>0.9
                        in_fov_list = in_fov_list & ~mask_sim[:,None].to(in_fov_list.device)

                        random_idx.append(r_idx)

                    for bi in range(conditions.shape[1]):
                        if len(torch.nonzero(conditions[:,bi,24] == 1))==0:
                            pass
                        else:
                            last_idx = torch.nonzero(conditions[:,bi,24] == 1)[-1]
                            in_fov_list[:last_idx,:,bi] = False

                    for csl in range(self.condition_similar_length // 2):
                        overlap_ratio = ((in_fov1[None].bool() & in_fov_list).sum(1))/in_fov1.sum()
                        # mask = distance > (in_fov1.bool().sum(0) / 4)
                        #_, r_idx = torch.topk(overlap_ratio / tensor_max_with_number((frame_idx[curr_frame] - frame_idx[:curr_frame]), 10), k=1, dim=0)
                        
                        # if csl > self.condition_similar_length:
                        # 	_, r_idx = torch.topk(overlap_ratio, k=1, dim=0)
                        # else:
                        # 	_, r_idx = torch.topk(overlap_ratio / tensor_max_with_number((frame_idx[curr_frame] - frame_idx[:curr_frame]), 10), k=1, dim=0)

                        _, r_idx = torch.topk(overlap_ratio, k=1, dim=0)
                        # _, r_idx = torch.topk(overlap_ratio / tensor_max_with_number((frame_idx[curr_frame] - frame_idx[:curr_frame]), 10), k=1, dim=0)

                        # if curr_frame >=93:
                        #     import pdb;pdb.set_trace()

                        # start_time = time.time()
                        cos_sim = F.cosine_similarity(xs_pred.to(r_idx.device)[r_idx[:, range(in_fov1.shape[1])], 
                            range(in_fov1.shape[1])], xs_pred.to(r_idx.device)[:curr_frame], dim=2)
                        cos_sim = cos_sim.mean((-2,-1))

                        mask_sim = cos_sim>0.9
                        in_fov_list = in_fov_list & ~mask_sim[:,None].to(in_fov_list.device)

                        random_idx.append(r_idx)
                    
                    random_idx = torch.cat(random_idx).cpu()
                    condition_similar_length = len(random_idx)
                
                xs_pred = torch.cat([xs_pred, xs_pred[random_idx[:,range(xs_pred.shape[1])], range(xs_pred.shape[1])].clone()], 0)

            if condition_similar_length:
                # import pdb;pdb.set_trace()
                padding = torch.zeros((condition_similar_length,) + conditions.shape[1:], device=conditions.device, dtype=conditions.dtype)
                input_condition = torch.cat([conditions[start_frame : curr_frame + horizon], padding], dim=0)
                if self.pose_cond_dim:
                    # if not self.use_plucker:
                    input_pose_condition = torch.cat([pose_conditions[start_frame : curr_frame + horizon], pose_conditions[random_idx[:,range(xs_pred.shape[1])], range(xs_pred.shape[1])]], dim=0).clone()

                if self.use_plucker:
                    if self.all_zero_frame:
                        frame_idx_list = []
                        input_pose_condition = []
                        for i in range(start_frame, curr_frame + horizon):
                            input_pose_condition.append(convert_to_plucker(torch.cat([c2w_mat[i:i+1],c2w_mat[random_idx[:,range(xs_pred.shape[1])], range(xs_pred.shape[1])]]).clone(), 0, focal_length=self.focal_length, is_old_setting=self.old_setting).to(xs_pred.dtype))
                            frame_idx_list.append(torch.cat([frame_idx[i:i+1]-frame_idx[i:i+1], frame_idx[random_idx[:,range(xs_pred.shape[1])], range(xs_pred.shape[1])]-frame_idx[i:i+1]]))
                        input_pose_condition = torch.cat(input_pose_condition)
                        frame_idx_list = torch.cat(frame_idx_list)

                        # print(frame_idx_list[:,0])
                    else:
                        # print(curr_frame-start_frame)
                        # input_pose_condition = torch.cat([c2w_mat[start_frame : curr_frame + horizon], c2w_mat[random_idx[:,range(xs_pred.shape[1])], range(xs_pred.shape[1])]], dim=0).clone()
                        # import pdb;pdb.set_trace()
                        if self.last_frame_refer:
                            input_pose_condition = torch.cat([c2w_mat[start_frame : curr_frame + horizon], c2w_mat[-1:]], dim=0).clone()
                        else:
                            input_pose_condition = torch.cat([c2w_mat[start_frame : curr_frame + horizon], c2w_mat[random_idx[:,range(xs_pred.shape[1])], range(xs_pred.shape[1])]], dim=0).clone()
                        
                        if self.zero_curr:
                            # print("="*50)
                            input_pose_condition = convert_to_plucker(input_pose_condition, curr_frame-start_frame, focal_length=self.focal_length, is_old_setting=self.old_setting)
                            # input_pose_condition[:curr_frame-start_frame] = input_pose_condition[curr_frame-start_frame:curr_frame-start_frame+1]
                        # input_pose_condition = convert_to_plucker(input_pose_condition, -self.condition_similar_length-1, focal_length=self.focal_length)
                        else:
                            input_pose_condition = convert_to_plucker(input_pose_condition, -condition_similar_length, focal_length=self.focal_length, is_old_setting=self.old_setting)
                        frame_idx_list = None
                else:
                    input_pose_condition = torch.cat([pose_conditions[start_frame : curr_frame + horizon], pose_conditions[random_idx[:,range(xs_pred.shape[1])], range(xs_pred.shape[1])]], dim=0).clone()
                    frame_idx_list = None
            else:
                input_condition = conditions[start_frame : curr_frame + horizon]
                input_pose_condition = None
                frame_idx_list = None
                
            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m]))[
                    :, None
                ].repeat(batch_size, axis=1)
                to_noise_levels = np.concatenate(
                    (
                        np.zeros((curr_frame,), dtype=np.int64),
                        scheduling_matrix[m + 1],
                    )
                )[
                    :, None
                ].repeat(batch_size, axis=1)

                if condition_similar_length:
                    from_noise_levels = np.concatenate([from_noise_levels, np.zeros((condition_similar_length,from_noise_levels.shape[-1]), dtype=np.int32)], axis=0)
                    to_noise_levels = np.concatenate([to_noise_levels, np.zeros((condition_similar_length,from_noise_levels.shape[-1]), dtype=np.int32)], axis=0)

                from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
                to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)


                if input_pose_condition is not None:
                    input_pose_condition = input_pose_condition.to(xs_pred.dtype)
                
                xs_pred[start_frame:] = self.diffusion_model.sample_step(
                    xs_pred[start_frame:],
                    input_condition,
                    input_pose_condition,
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                    current_frame=curr_frame,
                    mode="validation",
                    reference_length=condition_similar_length,
                    frame_idx=frame_idx_list
                )

                # if curr_frame > 14:
                #     import pdb;pdb.set_trace()

                # if xs_pred_back is not None:
                #     xs_pred = torch.cat([xs_pred[:6], xs_pred_back[6:12], xs_pred[6:]], dim=0)
        
            # import pdb;pdb.set_trace()
            if condition_similar_length: # and curr_frame+1!=n_frames:
                xs_pred = xs_pred[:-condition_similar_length]

            curr_frame += horizon
            pbar.update(horizon)

            self.frames = torch.cat([self.frames, xs_pred[n_context_frames:]])

            xs_pred = self.decode(xs_pred[n_context_frames:])

            return xs_pred[-1,0].cpu()

