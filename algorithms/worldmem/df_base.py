"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Optional
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any
from einops import rearrange

from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .models.diffusion import Diffusion


class DiffusionForcingBase(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.frame_stack = cfg.frame_stack
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.guidance_scale = cfg.guidance_scale
        self.context_frames = cfg.context_frames
        self.chunk_size = cfg.chunk_size
        self.action_cond_dim = cfg.action_cond_dim
        self.causal = cfg.causal

        self.uncertainty_scale = cfg.uncertainty_scale
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise

        self.cfg.diffusion.cum_snr_decay = self.cfg.diffusion.cum_snr_decay ** (self.frame_stack * cfg.frame_skip)

        self.validation_step_outputs = []
        super().__init__(cfg)

    def _build_model(self):
        self.diffusion_model = Diffusion(
            x_shape=self.x_stacked_shape,
            action_cond_dim=self.action_cond_dim,
            is_causal=self.causal,
            cfg=self.cfg.diffusion,
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

    def configure_optimizers(self):
        params = tuple(self.diffusion_model.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )
        return optimizer_dynamics

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.cfg.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)

        rand_length = torch.randint(3,xs.shape[0]-2, (1,))[0].item()
        xs = torch.cat([xs[:rand_length], xs[rand_length-3:rand_length-1]])
        conditions = torch.cat([conditions[:rand_length], conditions[rand_length-3:rand_length-1]])
        masks = torch.cat([masks[:rand_length], masks[rand_length-3:rand_length-1]])
        noise_levels=self._generate_noise_levels(xs)
        noise_levels[:rand_length] = 15 # stable_noise_levels
        noise_levels[rand_length+1:] = 15 # stable_noise_levels

        xs_pred, loss = self.diffusion_model(xs, conditions, noise_levels=noise_levels)
        loss = self.reweight_loss(loss, masks)

        # log the loss
        if batch_idx % 20 == 0:
            self.log("training/loss", loss)

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)
        n_frames, batch_size, *_ = xs.shape
        xs_pred = []
        curr_frame = 0

        # context
        n_context_frames = self.context_frames // self.frame_stack
        xs_pred = xs[:n_context_frames].clone()
        curr_frame += n_context_frames

        if self.condtion_similar_length:
            n_frames -= self.condtion_similar_length

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            scheduling_matrix = self._generate_scheduling_matrix(horizon)

            chunk = torch.randn((horizon, batch_size, *self.x_stacked_shape), device=self.device)
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

            if self.condtion_similar_length:
                xs_pred = torch.cat([xs_pred, xs[curr_frame-self.condtion_similar_length:curr_frame].clone()], 0)

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

                if self.condtion_similar_length:
                    from_noise_levels = np.concatenate([from_noise_levels, np.array([[0,0,0,0]*self.condtion_similar_length])], axis=0)
                    to_noise_levels = np.concatenate([to_noise_levels, np.array([[0,0,0,0]*self.condtion_similar_length])], axis=0)

                from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
                to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)

                # update xs_pred by DDIM or DDPM sampling
                # input frames within the sliding window

                try:
                    input_condition = conditions[start_frame : curr_frame + horizon].clone()
                except:
                    import pdb;pdb.set_trace()
                if self.condtion_similar_length:
                    input_condition = torch.cat([conditions[start_frame : curr_frame + horizon], conditions[-self.condtion_similar_length:]], dim=0)
                xs_pred[start_frame:] = self.diffusion_model.sample_step(
                    xs_pred[start_frame:],
                    input_condition,
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                )

            if self.condtion_similar_length:
                xs_pred = xs_pred[:-self.condtion_similar_length]

            curr_frame += horizon
            pbar.update(horizon)

        if self.condtion_similar_length:
            xs = xs[:-self.condtion_similar_length]
        # FIXME: loss
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweight_loss(loss, masks)
        self.validation_step_outputs.append((xs_pred.detach().cpu(), xs.detach().cpu()))

        return loss

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    def _generate_scheduling_matrix(self, horizon: int):
        match self.cfg.scheduling_matrix:
            case "pyramid":
                return self._generate_pyramid_scheduling_matrix(horizon, self.uncertainty_scale)
            case "full_sequence":
                return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
            case "autoregressive":
                return self._generate_pyramid_scheduling_matrix(horizon, self.sampling_timesteps)
            case "trapezoid":
                return self._generate_trapezoid_scheduling_matrix(horizon, self.uncertainty_scale)

    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def _generate_trapezoid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
                scheduling_matrix[m, -t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def reweight_loss(self, loss, weight=None):
        # Note there is another part of loss reweighting (fused_snr) inside the Diffusion class!
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(
                weight,
                "(t fs) b ... -> t b fs ..." + " 1" * expand_dim,
                fs=self.frame_stack,
            )
            loss = loss * weight

        return loss.mean()

    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        masks = torch.ones(n_frames, batch_size).to(xs.device)
        n_frames = n_frames // self.frame_stack

        if self.action_cond_dim:
            conditions = batch[1]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()

            # f, _, _ = conditions.shape
            # predefined_1 = torch.tensor([0,0,0,1]).to(conditions.device)
            # predefined_2 = torch.tensor([0,0,1,0]).to(conditions.device)
            # conditions[:f//2] = predefined_1
            # conditions[f//2:] = predefined_2
        else:
            conditions = [None for _ in range(n_frames)]

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        return xs, conditions, masks

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return xs * std + mean

    def _unstack_and_unnormalize(self, xs):
        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        return self._unnormalize_x(xs)
