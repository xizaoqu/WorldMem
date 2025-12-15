from typing import Optional
import wandb
import numpy as np
import torch
import os

import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import matplotlib.animation as animation
from pathlib import Path
import imageio
        
plt.set_loglevel("warning")

from torchmetrics.functional import mean_squared_error, peak_signal_noise_ratio
from torchmetrics.functional import (
    structural_similarity_index_measure,
    universal_image_quality_index,
)
from algorithms.common.metrics import (
    FrechetVideoDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance,
)


# FIXME: clean up & check this util
def log_video(
    observation_hat,
    observation_gt=None,
    step=0,
    namespace="train",
    prefix="video",
    context_frames=0,
    color=(255, 0, 0),
    logger=None,
    fps=15,
    format="mp4",
    save_local=True,
    local_save_dir=None,
):
    """
    take in video tensors in range [-1, 1] and log into wandb

    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param step: an int indicating the step number
    :param namespace: a string specify a name space this video logging falls under, e.g. train, val
    :param prefix: a string specify a prefix for the video name
    :param context_frames: an int indicating how many frames in observation_hat are ground truth given as context
    :param color: a tuple of 3 numbers specifying the color of the border for ground truth frames
    :param logger: optional logger to use. use global wandb if not specified
    :param fps: frames per second for the video (default: 15)
    :param format: video format, either "mp4" or "gif" (default: "mp4")
    :param save_local: whether to save videos to local disk (default: True)
    :param local_save_dir: directory to save local videos. If None, uses hydra output dir
    """
    import cv2
    import hydra
    from pathlib import Path
    
    # Get local rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if not logger:
        logger = wandb
    
    # Prepare video tensors
    observation_hat_np = observation_hat.detach().cpu().numpy()
    if observation_gt is not None:
        observation_gt_np = observation_gt.detach().cpu().numpy()
    else:
        observation_gt_np = None
    
    # Normalize to 0-255
    observation_hat_np = np.transpose(np.clip(observation_hat_np, a_min=0.0, a_max=1.0) * 255, (1, 0, 2, 3, 4)).astype(np.uint8)
    if observation_gt_np is not None:
        observation_gt_np = np.transpose(np.clip(observation_gt_np, a_min=0.0, a_max=1.0) * 255, (1, 0, 2, 3, 4)).astype(np.uint8)
    
    n_samples = len(observation_hat_np)
    
    # Setup local save directory
    if save_local:
        if local_save_dir is None:
            try:
                hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
                output_dir = Path(hydra_cfg.runtime.output_dir)
            except:
                output_dir = Path.cwd() / "outputs"
            local_save_dir = output_dir / "videos" / namespace
        else:
            local_save_dir = Path(local_save_dir)
        
        local_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pred videos locally
        pred_dir = local_save_dir / "pred"
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Save gt videos locally if available
        if observation_gt_np is not None:
            gt_dir = local_save_dir / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save videos
    for i in range(n_samples):
        video_pred = observation_hat_np[i]  # (T, C, H, W)
        
        if save_local:
            # Save prediction video
            if step is not None:
                video_filename_pred = f"{prefix}_{i}_rank{local_rank}_step{step}.{format}"
            else:
                video_filename_pred = f"{prefix}_{i}_rank{local_rank}.{format}"
            
            video_path_pred = pred_dir / video_filename_pred
            _save_video_to_file(video_pred, str(video_path_pred), fps)
            
            # Save ground truth video if available
            if observation_gt_np is not None:
                video_gt = observation_gt_np[i]
                if step is not None:
                    video_filename_gt = f"{prefix}_{i}_rank{local_rank}_step{step}.{format}"
                else:
                    video_filename_gt = f"{prefix}_{i}_rank{local_rank}.{format}"
                
                video_path_gt = gt_dir / video_filename_gt
                _save_video_to_file(video_gt, str(video_path_gt), fps)
        
        # Log to wandb (only rank 0 to avoid duplicate logging)
        if local_rank == 0 and logger:
            # Concatenate pred and gt side by side for visualization
            if observation_gt_np is not None:
                video_combined = torch.cat([
                    torch.from_numpy(observation_hat_np),
                    torch.from_numpy(observation_gt_np)
                ], -2).numpy()  # Concatenate along width
                logger.log(
                    {
                        f"{namespace}/{prefix}_{i}": wandb.Video(video_combined[i], fps=fps, format=format),
                        f"trainer/global_step": step,
                    }
                )
            else:
                logger.log(
                    {
                        f"{namespace}/{prefix}_{i}": wandb.Video(video_pred, fps=fps, format=format),
                        f"trainer/global_step": step,
                    }
                )


def _save_video_to_file(video_tensor, output_path, fps=15):
    """
    Save a video tensor to file using imageio (better compatibility than cv2).
    
    :param video_tensor: numpy array of shape (T, C, H, W) with values in [0, 255]
    :param output_path: path to save the video
    :param fps: frames per second
    """

    T, C, H, W = video_tensor.shape
    
    # Convert from (T, C, H, W) to (T, H, W, C)
    video_tensor = np.transpose(video_tensor, (0, 2, 3, 1))
    
    # Ensure uint8
    video_tensor = video_tensor.astype(np.uint8)
    
    # Save using imageio with H.264 codec (best compatibility)
    writer = imageio.get_writer(
        output_path, 
        fps=fps,
        codec='libx264',  # H.264 codec - widely supported
        quality=8,  # Good quality (scale 0-10, 10 is best)
        pixelformat='yuv420p',  # Standard pixel format for compatibility
        macro_block_size=1  # Better quality
    )
    
    for frame in video_tensor:
        writer.append_data(frame)
    
    writer.close()
        



def get_validation_metrics_for_videos(
    observation_hat,
    observation_gt,
    lpips_model: Optional[LearnedPerceptualImagePatchSimilarity] = None,
    fid_model: Optional[FrechetInceptionDistance] = None,
    fvd_model: Optional[FrechetVideoDistance] = None,
    lpips_batch_size: int = 100,
):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param lpips_model: a LearnedPerceptualImagePatchSimilarity object from algorithm.common.metrics
    :param fid_model: a FrechetInceptionDistance object  from algorithm.common.metrics
    :param fvd_model: a FrechetVideoDistance object  from algorithm.common.metrics
    :param lpips_batch_size: batch size for LPIPS calculation to avoid OOM (default: 100)
    :return: a tuple of metrics
    """
    frame, batch, channel, height, width = observation_hat.shape
    output_dict = {}
    observation_gt = observation_gt.type_as(observation_hat)  # some metrics don't fully support fp16

    if frame < 9:
        fvd_model = None  # FVD requires at least 9 frames

    observation_hat = observation_hat.float()
    observation_gt = observation_gt.float()

    # Clip to [0, 1] range before computing metrics (matching video saving behavior)
    observation_hat_clipped = torch.clamp(observation_hat, 0.0, 1.0)
    observation_gt_clipped = torch.clamp(observation_gt, 0.0, 1.0)

    # Compute video-wise PSNR: frame-wise average per video, then average across videos
    video_psnr_list = []
    for b in range(batch):
        frame_psnr_for_video = []
        for f in range(frame):
            frame_psnr = peak_signal_noise_ratio(observation_hat_clipped[f, b], observation_gt_clipped[f, b], data_range=1.0)
            frame_psnr_for_video.append(frame_psnr)
        video_psnr = torch.stack(frame_psnr_for_video).mean()
        video_psnr_list.append(video_psnr)
    output_dict["psnr"] = torch.stack(video_psnr_list).mean()
    
    observation_hat_clipped = observation_hat_clipped.view(-1, channel, height, width)
    observation_gt_clipped = observation_gt_clipped.view(-1, channel, height, width)

    # Compute MSE on clipped data
    output_dict["mse"] = mean_squared_error(observation_hat_clipped, observation_gt_clipped)
    # output_dict["ssim"] = structural_similarity_index_measure(observation_hat_clipped, observation_gt_clipped, data_range=1.0)
    # output_dict["uiqi"] = universal_image_quality_index(observation_hat_clipped, observation_gt_clipped)

    # LPIPS computation
    if lpips_model is not None:
        # Process LPIPS in batches to avoid OOM
        num_frames = observation_hat_clipped.shape[0]
        
        for i in range(0, num_frames, lpips_batch_size):
            batch_end = min(i + lpips_batch_size, num_frames)
            observation_hat_batch = observation_hat_clipped[i:batch_end]
            observation_gt_batch = observation_gt_clipped[i:batch_end]
            
            lpips_model.update(observation_hat_batch, observation_gt_batch)
            
            # Free GPU memory after each batch
            del observation_hat_batch, observation_gt_batch
            torch.cuda.empty_cache()
        
        lpips = lpips_model.compute().item()
        # Reset the states of non-functional metrics
        output_dict["lpips"] = lpips
        lpips_model.reset()

    # FID computation
    if fid_model is not None:
        observation_hat_uint8 = (observation_hat_clipped * 255).type(torch.uint8)
        observation_gt_uint8 = (observation_gt_clipped * 255).type(torch.uint8)
        fid_model.update(observation_gt_uint8, real=True)
        fid_model.update(observation_hat_uint8, real=False)
        fid = fid_model.compute()
        output_dict["fid"] = fid
        # Reset the states of non-functional metrics
        fid_model.reset()

    return output_dict


def is_grid_env(env_id):
    return "maze2d" in env_id or "diagonal2d" in env_id


def get_maze_grid(env_id):
    # import gym
    # maze_string = gym.make(env_id).str_maze_spec
    if "large" in env_id:
        maze_string = "############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############"
    if "medium" in env_id:
        maze_string = "########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########"
    if "umaze" in env_id:
        maze_string = "#####\\#GOO#\\###O#\\#OOO#\\#####"
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]


def get_random_start_goal(env_id, batch_size):
    maze_grid = get_maze_grid(env_id)
    s2i = {"O": 0, "#": 1, "G": 2}
    maze_grid = [[s2i[s] for s in r] for r in maze_grid]
    maze_grid = np.array(maze_grid)
    x, y = np.nonzero(maze_grid == 0)
    indices = np.random.randint(len(x), size=batch_size)
    start = np.stack([x[indices], y[indices]], -1) + 1
    x, y = np.nonzero(maze_grid == 2)
    goal = np.concatenate([x, y], -1)
    goal = np.tile(goal[None, :], (batch_size, 1)) + 1
    return start, goal


def plot_maze_layout(ax, maze_grid):
    ax.clear()

    if maze_grid is not None:
        for i, row in enumerate(maze_grid):
            for j, cell in enumerate(row):
                if cell == "#":
                    square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                    ax.add_patch(square)

    ax.set_aspect("equal")
    ax.grid(True, color="white", linewidth=4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)
    ax.set_facecolor("lightgray")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xticks(np.arange(0.5, len(maze_grid) + 0.5))
    ax.set_yticks(np.arange(0.5, len(maze_grid[0]) + 0.5))
    ax.set_xlim(0.5, len(maze_grid) + 0.5)
    ax.set_ylim(0.5, len(maze_grid[0]) + 0.5)
    ax.grid(True, color="white", which="minor", linewidth=4)


def plot_start_goal(ax, start_goal: None):
    def draw_star(center, radius, num_points=5, color="black"):
        angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (2 * num_points)
        inner_radius = radius / 2.0

        points = []
        for angle in angles:
            points.extend(
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                    center[1] + inner_radius * np.sin(angle + np.pi / num_points),
                ]
            )

        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        ax.add_patch(star)

    start_x, start_y = start_goal[0]
    start_outer_circle = plt.Circle((start_x, start_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(start_outer_circle)
    start_inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    ax.add_patch(start_inner_circle)

    goal_x, goal_y = start_goal[1]
    goal_outer_circle = plt.Circle((goal_x, goal_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(goal_outer_circle)
    draw_star((goal_x, goal_y), radius=0.08)


def make_trajectory_images(env_id, trajectory, batch_size, start, goal, plot_end_points=True):
    images = []
    for batch_idx in range(batch_size):
        fig, ax = plt.subplots()
        if is_grid_env(env_id):
            maze_grid = get_maze_grid(env_id)
        else:
            maze_grid = None
        plot_maze_layout(ax, maze_grid)
        ax.scatter(trajectory[:, batch_idx, 0], trajectory[:, batch_idx, 1], c=np.arange(len(trajectory)), cmap="Reds"),
        if plot_end_points:
            start_goal = (start[batch_idx], goal[batch_idx])
            plot_start_goal(ax, start_goal)
        # plt.title(f"sample_{batch_idx}")
        fig.tight_layout()
        fig.canvas.draw()
        img_shape = fig.canvas.get_width_height()[::-1] + (4,)
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy().reshape(img_shape)
        images.append(img)

        plt.close()
    return images


def make_convergence_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    open_loop_horizon,
    namespace,
    interval=100,
    plot_end_points=True,
    batch_idx=0,
):
    # - plan_history: contains for each time step all the MPC predicted plans for each pyramid noise level.
    #                 Structured as a list of length (episode_len // open_loop_horizon), where each
    #                 element corresponds to a control_time_step and stores a list of length pyramid_height,
    #                 where each element is a plan at a different pyramid noise level and stored as a tensor of
    #                 shape (episode_len // open_loop_horizon - control_time_step,
    #                        batch_size, x_stacked_shape)

    # select index and prune history
    start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(plan_history, trajectory, goal, open_loop_horizon)

    # animate the convergence of the first plan
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame):
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[0][frame]
        plan_history_m = plan_history_m.numpy()
        ax.scatter(
            plan_history_m[:, 0],
            plan_history_m[:, 1],
            c=np.arange(len(plan_history_m))[::-1],
            cmap="Reds",
        )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    frames = tqdm(range(len(plan_history[0])), desc="Making convergence animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_convergence.mp4"
    ani.save(filename, writer="ffmpeg", fps=5)
    return filename


def prune_history(plan_history, trajectory, goal, open_loop_horizon):
    dist = np.linalg.norm(
        trajectory[:, :2] - np.array(goal)[None],
        axis=-1,
    )
    reached = dist < 0.2
    if reached.any():
        cap_idx = np.argmax(reached)
        trajectory = trajectory[: cap_idx + open_loop_horizon + 1]
        plan_history = plan_history[: cap_idx // open_loop_horizon + 2]

    pruned_plan_history = []
    for plans in plan_history:
        pruned_plan_history.append([])
        for m in range(len(plans)):
            plan = plans[m]
            pruned_plan_history[-1].append(plan)
        plan = pruned_plan_history[-1][-1]
        dist = np.linalg.norm(plan.numpy()[:, :2] - np.array(goal)[None], axis=-1)
        reached = dist < 0.2
        if reached.any():
            cap_idx = np.argmax(reached) + 1
            pruned_plan_history[-1] = [p[:cap_idx] for p in pruned_plan_history[-1]]
    return trajectory, pruned_plan_history


def make_mpc_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    open_loop_horizon,
    namespace,
    interval=100,
    plot_end_points=True,
    batch_idx=0,
):
    # - plan_history: contains for each time step all the MPC predicted plans for each pyramid noise level.
    #                 Structured as a list of length (episode_len // open_loop_horizon), where each
    #                 element corresponds to a control_time_step and stores a list of length pyramid_height,
    #                 where each element is a plan at a different pyramid noise level and stored as a tensor of
    #                 shape (episode_len // open_loop_horizon - control_time_step,
    #                        batch_size, x_stacked_shape)

    # select index and prune history
    start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(plan_history, trajectory, goal, open_loop_horizon)

    # animate the convergence of the plans
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    trajectory_colors = np.linspace(0, 1, len(trajectory))

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame):
        control_time_step = 0
        while frame >= 0:
            frame -= len(plan_history[control_time_step])
            control_time_step += 1
        control_time_step -= 1
        m = frame + len(plan_history[control_time_step])
        num_steps_taken = 1 + open_loop_horizon * control_time_step
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[control_time_step][m]
        plan_history_m = plan_history_m.numpy()
        ax.scatter(
            trajectory[:num_steps_taken, 0],
            trajectory[:num_steps_taken, 1],
            c=trajectory_colors[:num_steps_taken],
            cmap="Blues",
        )
        ax.scatter(
            plan_history_m[:, 0],
            plan_history_m[:, 1],
            c=np.arange(len(plan_history_m))[::-1],
            cmap="Reds",
        )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    num_frames = sum([len(p) for p in plan_history])
    frames = tqdm(range(num_frames), desc="Making MPC animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_mpc.mp4"
    ani.save(filename, writer="ffmpeg", fps=5)

    return filename
