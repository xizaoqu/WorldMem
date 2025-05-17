import os
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
from packaging import version as pver
from einops import rearrange
from tqdm import tqdm
from omegaconf import DictConfig
from lightning.pytorch.utilities.types import STEP_OUTPUT
from algorithms.common.metrics import (
    LearnedPerceptualImagePatchSimilarity,
)
from utils.logging_utils import log_video, get_validation_metrics_for_videos
from .df_base import DiffusionForcingBase
from .models.vae import VAE_models
from .models.diffusion import Diffusion
from .models.pose_prediction import PosePredictionNet
import glob

# Utility Functions
def euler_to_rotation_matrix(pitch, yaw):
    """
    Convert pitch and yaw angles (in radians) to a 3x3 rotation matrix.
    Supports batch input.

    Args:
        pitch (torch.Tensor): Pitch angles in radians.
        yaw (torch.Tensor): Yaw angles in radians.

    Returns:
        torch.Tensor: Rotation matrix of shape (batch_size, 3, 3).
    """
    cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    R_pitch = torch.stack([
        torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch),
        torch.zeros_like(pitch), cos_pitch, -sin_pitch,
        torch.zeros_like(pitch), sin_pitch, cos_pitch
    ], dim=-1).reshape(-1, 3, 3)

    R_yaw = torch.stack([
        cos_yaw, torch.zeros_like(yaw), sin_yaw,
        torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
        -sin_yaw, torch.zeros_like(yaw), cos_yaw
    ], dim=-1).reshape(-1, 3, 3)

    return torch.matmul(R_yaw, R_pitch)


def euler_to_camera_to_world_matrix(pose):
    """
    Convert (x, y, z, pitch, yaw) to a 4x4 camera-to-world transformation matrix using torch.
    Supports both (5,) and (f, b, 5) shaped inputs.

    Args:
        pose (torch.Tensor): Pose tensor of shape (5,) or (f, b, 5).

    Returns:
        torch.Tensor: Camera-to-world transformation matrix of shape (4, 4).
    """

    origin_dim = pose.ndim
    if origin_dim == 1:
        pose = pose.unsqueeze(0).unsqueeze(0)  # Convert (5,) -> (1, 1, 5)
    elif origin_dim == 2:
        pose = pose.unsqueeze(0)

    x, y, z, pitch, yaw = pose[..., 0], pose[..., 1], pose[..., 2], pose[..., 3], pose[..., 4]
    pitch, yaw = torch.deg2rad(pitch), torch.deg2rad(yaw)

    # Compute rotation matrix (batch mode)
    R = euler_to_rotation_matrix(pitch, yaw)  # Shape (f*b, 3, 3)

    # Create the 4x4 transformation matrix
    eye = torch.eye(4, dtype=torch.float32, device=pose.device)
    camera_to_world = eye.repeat(R.shape[0], 1, 1)  # Shape (f*b, 4, 4)

    # Assign rotation
    camera_to_world[:, :3, :3] = R

    # Assign translation
    camera_to_world[:, :3, 3] = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)

    # Reshape back to (f, b, 4, 4) if needed
    if origin_dim == 3:
        return camera_to_world.view(pose.shape[0], pose.shape[1], 4, 4)
    elif origin_dim == 2:
        return camera_to_world.view(pose.shape[0], 4, 4)
    else:
        return camera_to_world.squeeze(0).squeeze(0)  # Convert (1,1,4,4) -> (4,4)

def is_inside_fov_3d_hv(points, center, center_pitch, center_yaw, fov_half_h, fov_half_v):
    """
    Check whether points are within a given 3D field of view (FOV) 
    with separately defined horizontal and vertical ranges.

    The center view direction is specified by pitch and yaw (in degrees).

    :param points: (N, B, 3) Sample point coordinates
    :param center: (3,) Center coordinates of the FOV
    :param center_pitch: Pitch angle of the center view (in degrees)
    :param center_yaw: Yaw angle of the center view (in degrees)
    :param fov_half_h: Horizontal half-FOV angle (in degrees)
    :param fov_half_v: Vertical half-FOV angle (in degrees)
    :return: Boolean tensor (N, B), indicating whether each point is inside the FOV
    """
    # Compute vectors relative to the center
    vectors = points - center  # shape (N, B, 3)
    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]
    
    # Compute horizontal angle (yaw): measured with respect to the z-axis as the forward direction,
    # and the x-axis as left-right, resulting in a range of -180 to 180 degrees.
    azimuth = torch.atan2(x, z) * (180 / math.pi)
    
    # Compute vertical angle (pitch): measured with respect to the horizontal plane,
    # resulting in a range of -90 to 90 degrees.
    elevation = torch.atan2(y, torch.sqrt(x**2 + z**2)) * (180 / math.pi)
    
    # Compute the angular difference from the center view (handling circular angle wrap-around)
    diff_azimuth = (azimuth - center_yaw).abs() % 360
    diff_elevation = (elevation - center_pitch).abs() % 360
    
    # Adjust values greater than 180 degrees to the shorter angular difference
    diff_azimuth = torch.where(diff_azimuth > 180, 360 - diff_azimuth, diff_azimuth)
    diff_elevation = torch.where(diff_elevation > 180, 360 - diff_elevation, diff_elevation)
    
    # Check if both horizontal and vertical angles are within their respective FOV limits
    return (diff_azimuth < fov_half_h) & (diff_elevation < fov_half_v)
    
def generate_points_in_sphere(n_points, radius):
    # Sample three independent uniform distributions
    samples_r = torch.rand(n_points)       # For radius distribution
    samples_phi = torch.rand(n_points)     # For azimuthal angle phi
    samples_u = torch.rand(n_points)       # For polar angle theta

    # Apply cube root to ensure uniform volumetric distribution
    r = radius * torch.pow(samples_r, 1/3)
    # Azimuthal angle phi uniformly distributed in [0, 2π]
    phi = 2 * math.pi * samples_phi
    # Convert u to theta to ensure cos(theta) is uniformly distributed
    theta = torch.acos(1 - 2 * samples_u)

    # Convert spherical coordinates to Cartesian coordinates
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points

def tensor_max_with_number(tensor, number):
    number_tensor = torch.tensor(number, dtype=tensor.dtype, device=tensor.device)
    result = torch.max(tensor, number_tensor)
    return result

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')
    
def camera_to_world_to_world_to_camera(camera_to_world: torch.Tensor) -> torch.Tensor:
    """
    Convert Camera-to-World matrices to World-to-Camera matrices for a tensor with shape (f, b, 4, 4).

    Args:
        camera_to_world (torch.Tensor): A tensor of shape (f, b, 4, 4), where:
            f = number of frames,
            b = batch size.

    Returns:
        torch.Tensor: A tensor of shape (f, b, 4, 4) representing the World-to-Camera matrices.
    """
    # Ensure input is a 4D tensor
    assert camera_to_world.ndim == 4 and camera_to_world.shape[2:] == (4, 4), \
        "Input must be of shape (f, b, 4, 4)"
    
    # Extract the rotation (R) and translation (T) parts
    R = camera_to_world[:, :, :3, :3]  # Shape: (f, b, 3, 3)
    T = camera_to_world[:, :, :3, 3]   # Shape: (f, b, 3)
    
    # Initialize an identity matrix for the output
    world_to_camera = torch.eye(4, device=camera_to_world.device).unsqueeze(0).unsqueeze(0)
    world_to_camera = world_to_camera.repeat(camera_to_world.size(0), camera_to_world.size(1), 1, 1)  # Shape: (f, b, 4, 4)
    
    # Compute the rotation (transpose of R)
    world_to_camera[:, :, :3, :3] = R.transpose(2, 3)
    
    # Compute the translation (-R^T * T)
    world_to_camera[:, :, :3, 3] = -torch.matmul(R.transpose(2, 3), T.unsqueeze(-1)).squeeze(-1)
    
    return world_to_camera.to(camera_to_world.dtype)

def convert_to_plucker(poses, curr_frame, focal_length, image_width, image_height):

    intrinsic = np.asarray([focal_length * image_width,
                                focal_length * image_height,
                                0.5 * image_width,
                                0.5 * image_height], dtype=np.float32)

    c2ws = get_relative_pose(poses, zero_first_frame_scale=curr_frame)
    c2ws = rearrange(c2ws, "t b m n -> b t m n")

    K = torch.as_tensor(intrinsic, device=poses.device, dtype=poses.dtype).repeat(c2ws.shape[0],c2ws.shape[1],1)  # [B, F, 4]
    plucker_embedding = ray_condition(K, c2ws, image_height, image_width, device=c2ws.device)
    plucker_embedding = rearrange(plucker_embedding, "b t h w d -> t b h w d").contiguous()

    return plucker_embedding


def get_relative_pose(abs_c2ws, zero_first_frame_scale):
    abs_w2cs = camera_to_world_to_world_to_camera(abs_c2ws)
    target_cam_c2w = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).to(abs_c2ws.device).to(abs_c2ws.dtype)
    abs2rel = target_cam_c2w @ abs_w2cs[zero_first_frame_scale]
    ret_poses = [abs2rel @ abs_c2w for abs_c2w in abs_c2ws]
    ret_poses = torch.stack(ret_poses)
    return ret_poses

def ray_condition(K, c2w, H, W, device):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i, device=device, dtype=c2w.dtype)  # [B, HxW]
    xs = -(i - cx) / fx * zs
    ys = -(j - cy) / fy * zs 

    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6

    return plucker

def random_transform(tensor):
    """
    Apply the same random translation, rotation, and scaling to all frames in the batch.

    Args:
        tensor (torch.Tensor): Input tensor of shape (F, B, 3, H, W).

    Returns:
        torch.Tensor: Transformed tensor of shape (F, B, 3, H, W).
    """
    if tensor.ndim != 5:
        raise ValueError("Input tensor must have shape (F, B, 3, H, W)")

    F, B, C, H, W = tensor.shape

    # Generate random transformation parameters
    max_translate = 0.2  # Translate up to 20% of width/height
    max_rotate = 30      # Rotate up to 30 degrees
    max_scale = 0.2      # Scale change by up to +/- 20%

    translate_x = random.uniform(-max_translate, max_translate) * W
    translate_y = random.uniform(-max_translate, max_translate) * H
    rotate_angle = random.uniform(-max_rotate, max_rotate)
    scale_factor = 1 + random.uniform(-max_scale, max_scale)

    # Apply the same transformation to all frames and batches

    tensor = tensor.reshape(F*B, C, H, W)
    transformed_tensor = TF.affine(
        tensor,
        angle=rotate_angle,
        translate=(translate_x, translate_y),
        scale=scale_factor,
        shear=(0, 0),
        interpolation=InterpolationMode.BILINEAR,
        fill=0
    )

    transformed_tensor = transformed_tensor.reshape(F, B, C, H, W)
    return transformed_tensor

def save_tensor_as_png(tensor, file_path):
    """
    Save a 3*H*W tensor as a PNG image.

    Args:
        tensor (torch.Tensor): Input tensor of shape (3, H, W).
        file_path (str): Path to save the PNG file.
    """
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Input tensor must have shape (3, H, W)")

    # Convert tensor to PIL Image
    image = TF.to_pil_image(tensor)

    # Save image
    image.save(file_path)

class WorldMemMinecraft(DiffusionForcingBase):
    """
    Video generation for MineCraft with memory.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the WorldMemMinecraft class with the given configuration.

        Args:
            cfg (DictConfig): Configuration object.
        """
        self.n_tokens = cfg.n_frames // cfg.frame_stack # number of max tokens for the model
        self.n_frames = cfg.n_frames
        if hasattr(cfg, "n_tokens"):
            self.n_tokens = cfg.n_tokens // cfg.frame_stack
        self.memory_condition_length = cfg.memory_condition_length
        self.pose_cond_dim = getattr(cfg, "pose_cond_dim", 5)

        self.use_plucker = cfg.use_plucker
        self.relative_embedding = cfg.relative_embedding
        self.state_embed_only_on_qk = getattr(cfg, "state_embed_only_on_qk", True)
        self.use_memory_attention = getattr(cfg, "use_memory_attention", True)
        self.add_timestamp_embedding = cfg.add_timestamp_embedding
        self.ref_mode = getattr(cfg, "ref_mode", 'sequential')
        self.log_curve = getattr(cfg, "log_curve", False)
        self.focal_length =  getattr(cfg, "focal_length", 0.35)
        self.log_video = cfg.log_video
        self.self_consistency_eval = getattr(cfg, "self_consistency_eval", False)
        self.next_frame_length = getattr(cfg, "next_frame_length", 1)
        self.require_pose_prediction = getattr(cfg, "require_pose_prediction", False)

        super().__init__(cfg)
            
    def _build_model(self):

        self.diffusion_model = Diffusion(
            reference_length=self.memory_condition_length,
            x_shape=self.x_stacked_shape,
            action_cond_dim=self.action_cond_dim,
            pose_cond_dim=self.pose_cond_dim,
            is_causal=self.causal,
            cfg=self.cfg.diffusion,
            is_dit=True,
            use_plucker=self.use_plucker,
            relative_embedding=self.relative_embedding,
            state_embed_only_on_qk=self.state_embed_only_on_qk,
            use_memory_attention=self.use_memory_attention,
            add_timestamp_embedding=self.add_timestamp_embedding,
            ref_mode=self.ref_mode
        )

        self.validation_lpips_model = LearnedPerceptualImagePatchSimilarity()
        vae = VAE_models["vit-l-20-shallow-encoder"]()
        self.vae = vae.eval()

        if self.require_pose_prediction:
            self.pose_prediction_model = PosePredictionNet()

    def _generate_noise_levels(self, xs: torch.Tensor, masks = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        num_frames, batch_size, *_ = xs.shape
        match self.cfg.noise_level:
            case "random_all":  # entirely random noise levels
                noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)
            case "same":
                noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)
                noise_levels[1:] = noise_levels[0]

        if masks is not None:
            # for frames that are not available, treat as full noise
            discard = torch.all(~rearrange(masks.bool(), "(t fs) b -> t b fs", fs=self.frame_stack), -1)
            noise_levels = torch.where(discard, torch.full_like(noise_levels, self.timesteps - 1), noise_levels)

        return noise_levels

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Perform a single training step.

        This function processes the input batch,
        encodes the input frames, generates noise levels, and computes the loss using the diffusion model.

        Args:
            batch: Input batch of data containing frames, conditions, poses, etc.
            batch_idx: Index of the current batch.

        Returns:
            dict: A dictionary containing the training loss.
        """
        xs, conditions, pose_conditions, c2w_mat, frame_idx = self._preprocess_batch(batch)

        if self.use_plucker:
            if self.relative_embedding:
                input_pose_condition = []
                frame_idx_list = []
                for i in range(self.n_frames):
                    input_pose_condition.append(
                        convert_to_plucker(
                            torch.cat([c2w_mat[i:i + 1], c2w_mat[-self.memory_condition_length:]]).clone(),
                            0,
                            focal_length=self.focal_length,
                            image_height=xs.shape[-2],image_width=xs.shape[-1]
                        ).to(xs.dtype)
                    )
                    frame_idx_list.append(
                        torch.cat([
                            frame_idx[i:i + 1] - frame_idx[i:i + 1],
                            frame_idx[-self.memory_condition_length:] - frame_idx[i:i + 1]
                        ]).clone()
                    )
                input_pose_condition = torch.cat(input_pose_condition)
                frame_idx_list = torch.cat(frame_idx_list)
            else:
                input_pose_condition = convert_to_plucker(
                    c2w_mat, 0, focal_length=self.focal_length
                ).to(xs.dtype)
                frame_idx_list = frame_idx
        else:
            input_pose_condition = pose_conditions.to(xs.dtype)
            frame_idx_list = None

        xs = self.encode(xs)

        noise_levels = self._generate_noise_levels(xs)

        if self.memory_condition_length:
            noise_levels[-self.memory_condition_length:] = self.diffusion_model.stabilization_level
            conditions[-self.memory_condition_length:] *= 0

        _, loss = self.diffusion_model(
            xs,
            conditions,
            input_pose_condition,
            noise_levels=noise_levels,
            reference_length=self.memory_condition_length,
            frame_idx=frame_idx_list
        )

        if self.memory_condition_length:
            loss = loss[:-self.memory_condition_length]

        loss = self.reweight_loss(loss, None)

        if batch_idx % 20 == 0:
            self.log("training/loss", loss.cpu())

        return {"loss": loss}
    
    def on_validation_epoch_end(self, namespace="validation") -> None:
        if not self.validation_step_outputs:
            return
        
        xs_pred = []
        xs = []
        for pred, gt in self.validation_step_outputs:
            xs_pred.append(pred)
            xs.append(gt)

        xs_pred = torch.cat(xs_pred, 1)
        if gt is not None:
            xs = torch.cat(xs, 1)
        else:
            xs = None

        if self.logger and self.log_video:
            log_video(
                xs_pred,
                xs,
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=self.context_frames,
                logger=self.logger.experiment,
            )

        if xs is not None:
            metric_dict = get_validation_metrics_for_videos(
                xs_pred, xs, 
                lpips_model=self.validation_lpips_model)
            
            self.log_dict(
                {"mse": metric_dict['mse'],
                "psnr": metric_dict['psnr'],
                "lpips": metric_dict['lpips']},
                sync_dist=True
            )

            if self.log_curve:
                psnr_values = metric_dict['frame_wise_psnr'].cpu().tolist()
                frames = list(range(len(psnr_values)))
                line_plot = wandb.plot.line_series(
                    xs = frames,
                    ys = [psnr_values],
                    keys = ["PSNR"],
                    title = "Frame-wise PSNR",
                    xname = "Frame index"
                )

                self.logger.experiment.log({"frame_wise_psnr_plot": line_plot})
      
        elif self.self_consistency_eval:
            metric_dict = get_validation_metrics_for_videos(
                xs_pred[:1],
                xs_pred[-1:],
                lpips_model=self.validation_lpips_model,
            )            
            self.log_dict(
                {"lpips": metric_dict['lpips'],
                "mse": metric_dict['mse'],
                "psnr": metric_dict['psnr']},
                sync_dist=True
            )

        self.validation_step_outputs.clear()

    def _preprocess_batch(self, batch):

        xs, conditions, pose_conditions, frame_index = batch

        if self.action_cond_dim:
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b t d -> t b d").contiguous()
        else:
            raise NotImplementedError("Only support external cond.")

        pose_conditions = rearrange(pose_conditions, "b t d -> t b d").contiguous()
        c2w_mat = euler_to_camera_to_world_matrix(pose_conditions)
        xs = rearrange(xs, "b t c ... -> t b c ...").contiguous()
        frame_index = rearrange(frame_index, "b t -> t b").contiguous()

        return xs, conditions, pose_conditions, c2w_mat, frame_index
    
    def encode(self, x):
        # vae encoding
        T = x.shape[0]
        H, W = x.shape[-2:]
        scaling_factor = 0.07843137255

        x = rearrange(x, "t b c h w -> (t b) c h w")
        with torch.no_grad():
            x = self.vae.encode(x * 2 - 1).mean * scaling_factor
        x = rearrange(x, "(t b) (h w) c -> t b c h w", t=T, h=H // self.vae.patch_size, w=W // self.vae.patch_size)
        return x

    def decode(self, x):
        total_frames = x.shape[0]
        scaling_factor = 0.07843137255
        x = rearrange(x, "t b c h w -> (t b) (h w) c")
        with torch.no_grad():
            x = (self.vae.decode(x / scaling_factor) + 1) / 2
        x = rearrange(x, "(t b) c h w-> t b c h w", t=total_frames)
        return x

    def _generate_condition_indices(self, curr_frame, memory_condition_length, xs_pred, pose_conditions, frame_idx, horizon):
        """
        Generate indices for condition similarity based on the current frame and pose conditions.
        """
        if curr_frame < memory_condition_length:
            random_idx = [i for i in range(curr_frame)] + [0] * (memory_condition_length - curr_frame)
            random_idx = np.repeat(np.array(random_idx)[:, None], xs_pred.shape[1], -1)
        else:
            # Generate points in a sphere and filter based on field of view
            num_samples = 10000
            radius = 30
            points = generate_points_in_sphere(num_samples, radius).to(pose_conditions.device)
            points = points[:, None].repeat(1, pose_conditions.shape[1], 1)
            points += pose_conditions[curr_frame, :, :3][None]
            fov_half_h = torch.tensor(105 / 2, device=pose_conditions.device)
            fov_half_v = torch.tensor(75 / 2, device=pose_conditions.device)

            # in_fov1 = is_inside_fov_3d_hv(
            #     points, pose_conditions[curr_frame, :, :3],
            #     pose_conditions[curr_frame, :, -2], pose_conditions[curr_frame, :, -1],
            #     fov_half_h, fov_half_v
            # )

            in_fov1 = torch.stack([
                is_inside_fov_3d_hv(points, pc[:, :3], pc[:, -2], pc[:, -1], fov_half_h, fov_half_v)
                for pc in pose_conditions[curr_frame:curr_frame+horizon]
            ])

            in_fov1 = torch.sum(in_fov1, 0) > 0

            # Compute overlap ratios and select indices
            in_fov_list = torch.stack([
                is_inside_fov_3d_hv(points, pc[:, :3], pc[:, -2], pc[:, -1], fov_half_h, fov_half_v)
                for pc in pose_conditions[:curr_frame]
            ])

            random_idx = []
            for _ in range(memory_condition_length):
                overlap_ratio = ((in_fov1.bool() & in_fov_list).sum(1)) / in_fov1.sum()
                
                confidence = overlap_ratio + (curr_frame - frame_idx[:curr_frame]) / curr_frame * (-0.2)

                if len(random_idx) > 0:
                    confidence[torch.cat(random_idx)] = -1e10
                _, r_idx = torch.topk(confidence, k=1, dim=0)
                random_idx.append(r_idx[0])

                # choice 1: directly remove overlapping region
                occupied_mask = in_fov_list[r_idx[0, range(in_fov1.shape[-1])], :, range(in_fov1.shape[-1])].permute(1,0)
                in_fov1 = in_fov1 & ~occupied_mask

                # choice 2: apply similarity filter 
                # cos_sim = F.cosine_similarity(xs_pred.to(r_idx.device)[r_idx[:, range(in_fov1.shape[1])], 
                #     range(in_fov1.shape[1])], xs_pred.to(r_idx.device)[:curr_frame], dim=2)
                # cos_sim = cos_sim.mean((-2,-1))

                # mask_sim = cos_sim>0.9
                # in_fov_list = in_fov_list & ~mask_sim[:,None].to(in_fov_list.device)

            random_idx = torch.stack(random_idx).cpu()

        return random_idx

    def _prepare_conditions(self, 
                            start_frame, curr_frame, horizon, conditions, 
                            pose_conditions, c2w_mat, frame_idx, random_idx,
                            image_width, image_height):
        """
        Prepare input conditions and pose conditions for sampling.
        """

        padding = torch.zeros((len(random_idx),) + conditions.shape[1:], device=conditions.device, dtype=conditions.dtype)
        input_condition = torch.cat([conditions[start_frame:curr_frame + horizon], padding], dim=0)

        batch_size = conditions.shape[1]

        if self.use_plucker:
            if self.relative_embedding:
                frame_idx_list = []
                input_pose_condition = []
                for i in range(start_frame, curr_frame + horizon):
                    input_pose_condition.append(convert_to_plucker(torch.cat([c2w_mat[i:i+1],c2w_mat[random_idx[:,range(batch_size)], range(batch_size)]]).clone(), 0, focal_length=self.focal_length,
                                                image_width=image_width, image_height=image_height).to(conditions.dtype))
                    frame_idx_list.append(torch.cat([frame_idx[i:i+1]-frame_idx[i:i+1], frame_idx[random_idx[:,range(batch_size)], range(batch_size)]-frame_idx[i:i+1]]))
                input_pose_condition = torch.cat(input_pose_condition)
                frame_idx_list = torch.cat(frame_idx_list)

            else:
                input_pose_condition = torch.cat([c2w_mat[start_frame : curr_frame + horizon], c2w_mat[random_idx[:,range(batch_size)], range(batch_size)]], dim=0).clone()
                input_pose_condition = convert_to_plucker(input_pose_condition, 0, focal_length=self.focal_length)
                frame_idx_list = None
        else:
            input_pose_condition = torch.cat([pose_conditions[start_frame : curr_frame + horizon], pose_conditions[random_idx[:,range(batch_size)], range(batch_size)]], dim=0).clone()
            frame_idx_list = None

        return input_condition, input_pose_condition, frame_idx_list

    def _prepare_noise_levels(self, scheduling_matrix, m, curr_frame, batch_size, memory_condition_length):
        """
        Prepare noise levels for the current sampling step.
        """
        from_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m]))[:, None].repeat(batch_size, axis=1)
        to_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m + 1]))[:, None].repeat(batch_size, axis=1)
        if memory_condition_length:
            from_noise_levels = np.concatenate([from_noise_levels, np.zeros((memory_condition_length, from_noise_levels.shape[-1]), dtype=np.int32)], axis=0)
            to_noise_levels = np.concatenate([to_noise_levels, np.zeros((memory_condition_length, from_noise_levels.shape[-1]), dtype=np.int32)], axis=0)
        from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
        to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)
        return from_noise_levels, to_noise_levels

    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        """
        Perform a single validation step.

        This function processes the input batch, encodes frames, generates predictions using a sliding window approach,
        and handles condition similarity logic for sampling. The results are decoded and stored for evaluation.

        Args:
            batch: Input batch of data containing frames, conditions, poses, etc.
            batch_idx: Index of the current batch.
            namespace: Namespace for logging (default: "validation").

        Returns:
            None: Appends the predicted and ground truth frames to `self.validation_step_outputs`.
        """
        # Preprocess the input batch
        memory_condition_length = self.memory_condition_length
        xs_raw, conditions, pose_conditions, c2w_mat, frame_idx = self._preprocess_batch(batch)


        # Encode frames in chunks if necessary
        total_frame = xs_raw.shape[0]
        if total_frame > 10:
            xs = torch.cat([
                self.encode(xs_raw[int(total_frame * i / 10):int(total_frame * (i + 1) / 10)]).cpu()
                for i in range(10)
            ])
        else:
            xs = self.encode(xs_raw).cpu()

        n_frames, batch_size, *_ = xs.shape
        curr_frame = 0

        # Initialize context frames
        n_context_frames = self.context_frames // self.frame_stack
        xs_pred = xs[:n_context_frames].clone()
        curr_frame += n_context_frames

        # Progress bar for sampling
        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")

        while curr_frame < n_frames:
            # Determine the horizon for the current chunk
            horizon = min(n_frames - curr_frame, self.chunk_size) if self.chunk_size > 0 else n_frames - curr_frame
            assert horizon <= self.n_tokens, "Horizon exceeds the number of tokens."

            # Generate scheduling matrix and initialize noise
            scheduling_matrix = self._generate_scheduling_matrix(horizon)
            chunk = torch.randn((horizon, batch_size, *xs_pred.shape[2:]))
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise).to(xs_pred.device)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            # Sliding window: only input the last `n_tokens` frames
            start_frame = max(0, curr_frame + horizon - self.n_tokens)
            pbar.set_postfix({"start": start_frame, "end": curr_frame + horizon})

            # Handle condition similarity logic
            if memory_condition_length:
                random_idx = self._generate_condition_indices(
                    curr_frame, memory_condition_length, xs_pred, pose_conditions, frame_idx, horizon
                )

                xs_pred = torch.cat([xs_pred, xs_pred[random_idx[:, range(xs_pred.shape[1])], range(xs_pred.shape[1])].clone()], 0)

            # Prepare input conditions and pose conditions
            input_condition, input_pose_condition, frame_idx_list = self._prepare_conditions(
                start_frame, curr_frame, horizon, conditions, pose_conditions, c2w_mat, frame_idx, random_idx,
                image_width=xs_raw.shape[-1], image_height=xs_raw.shape[-2]
            )

            # Perform sampling for each step in the scheduling matrix
            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels, to_noise_levels = self._prepare_noise_levels(
                    scheduling_matrix, m, curr_frame, batch_size, memory_condition_length
                )

                xs_pred[start_frame:] = self.diffusion_model.sample_step(
                    xs_pred[start_frame:].to(input_condition.device),
                    input_condition,
                    input_pose_condition,
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                    current_frame=curr_frame,
                    mode="validation",
                    reference_length=memory_condition_length,
                    frame_idx=frame_idx_list
                ).cpu()

            # Remove condition similarity frames if applicable
            if memory_condition_length:
                xs_pred = xs_pred[:-memory_condition_length]

            curr_frame += horizon
            pbar.update(horizon)

        # Decode predictions and ground truth
        xs_pred = self.decode(xs_pred[n_context_frames:].to(conditions.device))
        xs_decode = self.decode(xs[n_context_frames:].to(conditions.device))

        # Store results for evaluation
        self.validation_step_outputs.append((xs_pred, xs_decode))
        return

    @torch.no_grad()
    def interactive(self, first_frame, new_actions, first_pose, device,
                    memory_latent_frames, memory_actions, memory_poses, memory_c2w, memory_frame_idx):
    
        memory_condition_length = self.memory_condition_length

        if memory_latent_frames is None:
            first_frame = torch.from_numpy(first_frame)
            new_actions = torch.from_numpy(new_actions)
            first_pose = torch.from_numpy(first_pose)
            first_frame_encode = self.encode(first_frame[None, None].to(device))
            memory_latent_frames = first_frame_encode.cpu()
            memory_actions = new_actions[None, None].to(device)
            memory_poses = first_pose[None, None].to(device)
            new_c2w_mat = euler_to_camera_to_world_matrix(first_pose)
            memory_c2w = new_c2w_mat[None, None].to(device)
            memory_frame_idx = torch.tensor([[0]]).to(device)
            return first_frame.cpu().numpy(), memory_latent_frames.cpu().numpy(), memory_actions.cpu().numpy(), memory_poses.cpu().numpy(), memory_c2w.cpu().numpy(), memory_frame_idx.cpu().numpy()
        else:
            memory_latent_frames = torch.from_numpy(memory_latent_frames)
            memory_actions = torch.from_numpy(memory_actions).to(device)
            memory_poses = torch.from_numpy(memory_poses).to(device)
            memory_c2w = torch.from_numpy(memory_c2w).to(device)
            memory_frame_idx = torch.from_numpy(memory_frame_idx).to(device)
            new_actions = new_actions.to(device)

        curr_frame = 0
        batch_size = 1
        horizon = self.next_frame_length
        n_frames = curr_frame + horizon
        # context
        n_context_frames = len(memory_latent_frames)
        xs_pred = memory_latent_frames[:n_context_frames].clone()
        curr_frame += n_context_frames

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")

        new_pose_condition_list = []
        last_frame = xs_pred[-1].clone()
        last_pose_condition = memory_poses[-1].clone()
        curr_actions = new_actions.clone()
        for hi in range(len(new_actions)):
            last_pose_condition[:,3:] = last_pose_condition[:,3:] // 15
            new_pose_condition_offset = self.pose_prediction_model(last_frame.to(device), curr_actions[None, hi], last_pose_condition)
            new_pose_condition_offset[:,3:] = torch.round(new_pose_condition_offset[:,3:])
            new_pose_condition = last_pose_condition + new_pose_condition_offset
            new_pose_condition[:,3:] = new_pose_condition[:,3:] * 15
            new_pose_condition[:,3:] %= 360
            last_pose_condition = new_pose_condition.clone()
            new_pose_condition_list.append(new_pose_condition[None])
        new_pose_condition_list = torch.cat(new_pose_condition_list, 0)
        
        ai = 0
        while ai < len(new_actions):
            next_horizon = min(horizon, len(new_actions) - ai)
            last_frame = xs_pred[-1].clone()
            curr_actions = new_actions[ai:ai+next_horizon].clone()

            new_pose_condition = new_pose_condition_list[ai:ai+next_horizon].clone()

            new_c2w_mat = euler_to_camera_to_world_matrix(new_pose_condition)
            memory_poses = torch.cat([memory_poses, new_pose_condition])
            memory_actions = torch.cat([memory_actions, curr_actions[:, None]])
            memory_c2w = torch.cat([memory_c2w, new_c2w_mat])
            new_indices = memory_frame_idx[-1,0] + torch.arange(next_horizon, device=memory_frame_idx.device) + 1

            memory_frame_idx = torch.cat([memory_frame_idx, new_indices[:, None]])

            conditions = memory_actions.clone()
            pose_conditions = memory_poses.clone()
            c2w_mat = memory_c2w .clone()
            frame_idx = memory_frame_idx.clone()

            # generation on frame
            scheduling_matrix = self._generate_scheduling_matrix(next_horizon)
            chunk = torch.randn((next_horizon, batch_size, *xs_pred.shape[2:])).to(xs_pred.device)
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)

            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_tokens frames
            start_frame = max(0, curr_frame - self.n_tokens)

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + next_horizon,
                }
            )

            # Handle condition similarity logic
            if memory_condition_length:
                random_idx = self._generate_condition_indices(
                    curr_frame, memory_condition_length, xs_pred, pose_conditions, frame_idx, next_horizon
                )
                
                # random_idx = np.unique(random_idx)[:, None]
                # memory_condition_length = len(random_idx)
                xs_pred = torch.cat([xs_pred, xs_pred[random_idx[:, range(xs_pred.shape[1])], range(xs_pred.shape[1])].clone()], 0)

            # Prepare input conditions and pose conditions
            input_condition, input_pose_condition, frame_idx_list = self._prepare_conditions(
                start_frame, curr_frame, next_horizon, conditions, pose_conditions, c2w_mat, frame_idx, random_idx,
                image_width=first_frame.shape[-1], image_height=first_frame.shape[-2]
            )

            # Perform sampling for each step in the scheduling matrix
            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels, to_noise_levels = self._prepare_noise_levels(
                    scheduling_matrix, m, curr_frame, batch_size, memory_condition_length
                )

                xs_pred[start_frame:] = self.diffusion_model.sample_step(
                    xs_pred[start_frame:].to(input_condition.device),
                    input_condition,
                    input_pose_condition,
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                    current_frame=curr_frame,
                    mode="validation",
                    reference_length=memory_condition_length,
                    frame_idx=frame_idx_list
                ).cpu()


            if memory_condition_length:
                xs_pred = xs_pred[:-memory_condition_length]

            curr_frame += next_horizon
            pbar.update(next_horizon)
            ai += next_horizon

        memory_latent_frames = torch.cat([memory_latent_frames, xs_pred[n_context_frames:]])
        xs_pred = self.decode(xs_pred[n_context_frames:].to(device)).cpu()

        return xs_pred.cpu().numpy(), memory_latent_frames.cpu().numpy(), memory_actions.cpu().numpy(), \
            memory_poses.cpu().numpy(), memory_c2w.cpu().numpy(), memory_frame_idx.cpu().numpy()
