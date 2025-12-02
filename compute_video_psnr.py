#!/usr/bin/env python3
"""
Compute PSNR for each video pair in gt and pred directories.
"""
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import imageio


def load_video(video_path):
    """
    Load a video file and return frames as a numpy array.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        np.ndarray: Video frames of shape (T, C, H, W) normalized to [0, 1]
    """
    reader = imageio.get_reader(str(video_path))
    frames = []
    for frame in reader:
        # frame is already RGB
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    reader.close()
    
    if len(frames) == 0:
        raise ValueError(f"No frames loaded from {video_path}")
    
    # Convert to numpy array: (T, H, W, C) -> (T, C, H, W)
    frames = np.stack(frames, axis=0)
    frames = np.transpose(frames, (0, 3, 1, 2))
    
    return frames


def calculate_psnr(pred, gt, data_range=1.0):
    """
    Calculate PSNR between two images/videos.
    
    Args:
        pred: Predicted frames
        gt: Ground truth frames
        data_range: Data range (default: 1.0)
        
    Returns:
        float: PSNR value in dB
    """
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((data_range ** 2) / mse)


def compute_psnr(pred_frames, gt_frames, data_range=1.0):
    """
    Compute PSNR between predicted and ground truth frames.
    
    Args:
        pred_frames: Predicted frames array (T, C, H, W)
        gt_frames: Ground truth frames array (T, C, H, W)
        data_range: Data range of the frames (default: 1.0 for [0, 1] range)
        
    Returns:
        dict: Dictionary containing overall PSNR and frame-wise PSNR
    """
    # Ensure same shape
    assert pred_frames.shape == gt_frames.shape, \
        f"Shape mismatch: pred {pred_frames.shape} vs gt {gt_frames.shape}"
    
    T, C, H, W = pred_frames.shape
    
    # Compute frame-wise PSNR
    frame_wise_psnr = []
    for t in range(T):
        psnr = calculate_psnr(pred_frames[t], gt_frames[t], data_range=data_range)
        frame_wise_psnr.append(psnr)
    
    # Compute overall PSNR
    overall_psnr = calculate_psnr(pred_frames, gt_frames, data_range=data_range)
    
    return {
        'overall_psnr': overall_psnr,
        'frame_wise_psnr': frame_wise_psnr,
        'mean_frame_psnr': np.mean(frame_wise_psnr),
        'std_frame_psnr': np.std(frame_wise_psnr),
        'num_frames': T
    }


def main():
    # Directories
    gt_dir = Path("/mnt/WorldMem/outputs/2025-12-01/23-39-44/videos/test_vis/gt")
    pred_dir = Path("/mnt/WorldMem/outputs/2025-12-01/23-39-44/videos/test_vis/pred")
    
    # Get all video files in gt directory
    gt_videos = sorted(gt_dir.glob("*.mp4"))
    
    print(f"Found {len(gt_videos)} videos to process")
    print("=" * 80)
    
    results = []
    
    for gt_path in tqdm(gt_videos, desc="Computing PSNR"):
        video_name = gt_path.name
        pred_path = pred_dir / video_name
        
        if not pred_path.exists():
            print(f"Warning: Prediction video not found for {video_name}")
            continue
        
        try:
            # Load videos
            gt_frames = load_video(gt_path)
            pred_frames = load_video(pred_path)
            
            # Compute PSNR
            psnr_results = compute_psnr(pred_frames, gt_frames, data_range=1.0)
            
            # Store results
            result = {
                'video_name': video_name,
                'overall_psnr': psnr_results['overall_psnr'],
                'mean_frame_psnr': psnr_results['mean_frame_psnr'],
                'std_frame_psnr': psnr_results['std_frame_psnr'],
                'num_frames': psnr_results['num_frames']
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {video_name}: {str(e)}")
            continue
    
    # Create DataFrame for better visualization
    df = pd.DataFrame(results)
    
    # Print results
    print("\n" + "=" * 80)
    print("PSNR Results for Each Video:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("=" * 80)
    print(f"Average PSNR across all videos: {df['overall_psnr'].mean():.4f} ± {df['overall_psnr'].std():.4f}")
    print(f"Min PSNR: {df['overall_psnr'].min():.4f} ({df.loc[df['overall_psnr'].idxmin(), 'video_name']})")
    print(f"Max PSNR: {df['overall_psnr'].max():.4f} ({df.loc[df['overall_psnr'].idxmax(), 'video_name']})")
    print(f"Median PSNR: {df['overall_psnr'].median():.4f}")
    
    # Group by video ID (video_0, video_1, etc.) and rank
    df['video_id'] = df['video_name'].str.extract(r'(video_\d+)')[0]
    df['rank'] = df['video_name'].str.extract(r'rank(\d+)')[0].astype(int)
    
    print("\n" + "=" * 80)
    print("Average PSNR by Video ID:")
    print("=" * 80)
    video_group = df.groupby('video_id')['overall_psnr'].agg(['mean', 'std', 'count'])
    print(video_group.to_string())
    
    print("\n" + "=" * 80)
    print("Average PSNR by Rank:")
    print("=" * 80)
    rank_group = df.groupby('rank')['overall_psnr'].agg(['mean', 'std', 'count'])
    print(rank_group.to_string())
    
    # Save results to CSV
    output_csv = gt_dir.parent / "psnr_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Save summary to text file
    output_txt = gt_dir.parent / "psnr_summary.txt"
    with open(output_txt, 'w') as f:
        f.write("PSNR Results Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Average PSNR: {df['overall_psnr'].mean():.4f} ± {df['overall_psnr'].std():.4f}\n")
        f.write(f"Min PSNR: {df['overall_psnr'].min():.4f}\n")
        f.write(f"Max PSNR: {df['overall_psnr'].max():.4f}\n")
        f.write(f"Median PSNR: {df['overall_psnr'].median():.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Full Results:\n")
        f.write("=" * 80 + "\n")
        f.write(df.to_string(index=False))
    print(f"Summary saved to: {output_txt}")


if __name__ == "__main__":
    main()

