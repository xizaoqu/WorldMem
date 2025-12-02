#!/usr/bin/env python3
"""
Calculate FID (Fréchet Inception Distance) between predicted and ground truth videos.

Usage:
    python calculate_fid.py --videos_dir /path/to/videos
    python calculate_fid.py --videos_dir /path/to/videos --batch_size 32
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
from torchmetrics.image.fid import FrechetInceptionDistance


def load_video_frames(video_path, max_frames=None):
    """
    Load frames from a video file.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to load (None = all frames)
        
    Returns:
        torch.Tensor: Video frames with shape (T, C, H, W) in range [0, 255]
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames loaded from {video_path}")
    
    # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
    frames = np.stack(frames, axis=0)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
    
    return frames


def load_videos_from_directory(video_dir, max_frames_per_video=None, max_videos=None):
    """
    Load all videos from a directory.
    
    Args:
        video_dir: Directory containing .mp4 files
        max_frames_per_video: Maximum frames to load per video
        max_videos: Maximum number of videos to load
        
    Returns:
        torch.Tensor: All frames concatenated with shape (N, C, H, W)
    """
    video_dir = Path(video_dir)
    video_paths = sorted(list(video_dir.glob("**/*.mp4")))
    
    if max_videos:
        video_paths = video_paths[:max_videos]
    
    all_frames = []
    
    print(f"Loading videos from {video_dir}")
    print(f"Found {len(video_paths)} videos")
    
    for video_path in tqdm(video_paths, desc="Loading videos"):
        try:
            frames = load_video_frames(video_path, max_frames=max_frames_per_video)
            all_frames.append(frames)
        except Exception as e:
            print(f"\nWarning: Failed to load {video_path.name}: {e}")
            continue
    
    if len(all_frames) == 0:
        raise ValueError(f"No videos loaded from {video_dir}")
    
    # Concatenate all frames: (N_videos, T, C, H, W) -> (N_total_frames, C, H, W)
    all_frames = torch.cat(all_frames, dim=0)
    
    print(f"Loaded {all_frames.shape[0]} frames total")
    print(f"Frame shape: {all_frames.shape[1:]}")
    
    return all_frames


def calculate_fid(pred_dir, gt_dir, batch_size=32, device='cuda', 
                  max_frames_per_video=None, max_videos=None):
    """
    Calculate FID between predicted and ground truth videos.
    
    Args:
        pred_dir: Directory containing predicted videos
        gt_dir: Directory containing ground truth videos
        batch_size: Batch size for FID calculation
        device: Device to use ('cuda' or 'cpu')
        max_frames_per_video: Maximum frames to load per video
        max_videos: Maximum number of videos to load from each directory
        
    Returns:
        float: FID score
    """
    print("="*60)
    print("FID Calculation")
    print("="*60)
    print(f"Pred directory: {pred_dir}")
    print(f"GT directory: {gt_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print("="*60 + "\n")
    
    # Check if directories exist
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    
    if not pred_dir.exists():
        raise ValueError(f"Pred directory does not exist: {pred_dir}")
    if not gt_dir.exists():
        raise ValueError(f"GT directory does not exist: {gt_dir}")
    
    # Load videos
    print("\n[1/3] Loading predicted videos...")
    pred_frames = load_videos_from_directory(
        pred_dir, 
        max_frames_per_video=max_frames_per_video,
        max_videos=max_videos
    )
    
    print("\n[2/3] Loading ground truth videos...")
    gt_frames = load_videos_from_directory(
        gt_dir,
        max_frames_per_video=max_frames_per_video,
        max_videos=max_videos
    )
    
    # Initialize FID model
    print("\n[3/3] Calculating FID...")
    fid_model = FrechetInceptionDistance(normalize=True).to(device)
    
    # Process pred frames in batches
    print("Processing predicted frames...")
    num_pred_frames = pred_frames.shape[0]
    for i in tqdm(range(0, num_pred_frames, batch_size)):
        batch = pred_frames[i:i+batch_size]
        batch = batch.to(device)
        fid_model.update(batch, real=False)
    
    # Process gt frames in batches
    print("Processing ground truth frames...")
    num_gt_frames = gt_frames.shape[0]
    for i in tqdm(range(0, num_gt_frames, batch_size)):
        batch = gt_frames[i:i+batch_size]
        batch = batch.to(device)
        fid_model.update(batch, real=True)
    
    # Compute FID
    fid_score = fid_model.compute().item()
    
    return fid_score


def main():
    parser = argparse.ArgumentParser(
        description="Calculate FID between predicted and ground truth videos"
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="/mnt/worldmem_valid/outputs/2025-12-01/08-09-46/videos/test_vis",
        help="Base directory containing 'pred' and 'gt' subdirectories"
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="Override pred directory (default: {videos_dir}/pred)"
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Override gt directory (default: {videos_dir}/gt)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for FID calculation (default: 32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)"
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=None,
        help="Maximum frames to load per video (default: None, load all)"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=50,
        help="Maximum number of videos to load (default: None, load all)"
    )
    
    args = parser.parse_args()
    
    # Determine pred and gt directories
    videos_dir = Path(args.videos_dir)
    
    if args.pred_dir:
        pred_dir = Path(args.pred_dir)
    else:
        pred_dir = videos_dir / "pred"
    
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
    else:
        gt_dir = videos_dir / "gt"
    
    # Calculate FID
    try:
        fid_score = calculate_fid(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            batch_size=args.batch_size,
            device=args.device,
            max_frames_per_video=args.max_frames_per_video,
            max_videos=args.max_videos
        )
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"FID Score: {fid_score:.4f}")
        print("="*60)
        
        # Save results to file
        output_file = videos_dir / "fid_results.txt"
        with open(output_file, 'w') as f:
            f.write(f"FID Score: {fid_score:.4f}\n")
            f.write(f"Pred directory: {pred_dir}\n")
            f.write(f"GT directory: {gt_dir}\n")
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

