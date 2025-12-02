"""
MineDojo Episode Collection Script

This script generates trajectory data from MineDojo environment using a simple agent
that randomly explores the environment. It supports parallel data collection and
saves video and action/pose data.
"""

import argparse
import math
import multiprocessing as mp
import os
import os.path as osp
import random
from typing import Dict, Optional, Tuple

import cv2
import minedojo
import numpy as np
from tqdm import tqdm

# Action mappings for the agent
# Format: [forward/back, ?, ?, pitch, yaw, ?, ?, ?]
ACTIONS: Dict[str, np.ndarray] = {
    'forward': np.array([1, 0, 0, 12, 12, 0, 0, 0]),
    'back': np.array([2, 0, 0, 12, 12, 0, 0, 0]),
    'left': np.array([0, 0, 0, 12, 11, 0, 0, 0]),
    'right': np.array([0, 0, 0, 12, 13, 0, 0, 0]),
    'up': np.array([0, 0, 0, 11, 12, 0, 0, 0]),
    'down': np.array([0, 0, 0, 13, 12, 0, 0, 0]),
    'noop': np.array([0, 0, 0, 12, 12, 0, 0, 0])
}


def sample_action(prob_forward: float) -> str:
    """
    Sample an action based on forward probability.
    
    Args:
        prob_forward: Probability of moving forward or backward
        
    Returns:
        Action name string
    """
    prob_turn = (1 - prob_forward) / 2
    action = np.random.choice(
        ['forward', 'back', 'left', 'right', 'up', 'down'],
        p=[prob_forward / 2 - 0.1, prob_forward / 2 - 0.1, prob_turn, prob_turn, 0.1, 0.1]
    )
    return action


class SimpleAgent:
    """
    Simple agent that explores the environment with random actions.
    
    Attributes:
        action_repeat: Number of times to repeat the same action
        prob_forward: Probability of moving forward/backward
        max_consec_fwd: Maximum consecutive forward actions (currently unused)
    """
    
    def __init__(self, prob_forward: float, action_repeat: int, max_consec_fwd: int):
        """
        Initialize the SimpleAgent.
        
        Args:
            prob_forward: Probability of moving forward or backward
            action_repeat: How many steps to repeat each sampled action
            max_consec_fwd: Maximum consecutive forward movements
        """
        self.action_repeat = action_repeat
        self.prob_forward = prob_forward
        self.max_consec_fwd = max_consec_fwd
        self.reset()

    def reset(self) -> None:
        """Reset the agent's internal state."""
        self.n_fwd = 0
        self.counter = 0
        self.action = None

    def sample(self, pos: np.ndarray) -> np.ndarray:
        """
        Sample an action given the current position.
        
        Args:
            pos: Current position array containing [x, y, z, pitch, yaw]
            
        Returns:
            Action array
        """
        prob_forward = self.prob_forward

        # Sample new action if needed (or if previous was up/down)
        if (self.action is None or 
            self.counter % self.action_repeat == 0 or 
            self.action in ['up', 'down']):
            self.action = sample_action(prob_forward)

        self.counter += 1
        return ACTIONS[self.action]


def collect_episode(env, agent: SimpleAgent, traj_length: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Collect a single episode of trajectory data.
    
    Args:
        env: MineDojo environment
        agent: Agent to collect data with
        traj_length: Length of trajectory to collect
        
    Returns:
        Tuple of (rgb observations, actions, poses) or None if collection fails
    """
    agent.reset()

    # Retry environment reset until successful
    success = False
    max_retries = 10
    retries = 0
    while not success and retries < max_retries:
        try:
            obs = env.reset()
            success = True
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                print(f"Failed to reset environment after {max_retries} retries")
                return None

    observations = [obs['rgb']]
    actions = [env.action_space.no_op()]
    pose = [np.concatenate([
        obs['location_stats']['pos'],
        obs['location_stats']['pitch'],
        obs['location_stats']['yaw']
    ])]

    for ei in range(traj_length):
        curr_actions = agent.sample(pose[ei])
        obs, reward, done, info = env.step(curr_actions)
        
        actions.append(curr_actions)
        observations.append(obs['rgb'])
        pose.append(np.concatenate([
            obs['location_stats']['pos'],
            obs['location_stats']['pitch'],
            obs['location_stats']['yaw']
        ]))
    
    rgb = np.stack(observations, axis=0)
    actions = np.array(actions, dtype=np.int32)
    pose = np.array(pose)

    return rgb, actions, pose


def worker(worker_id: int, args: argparse.Namespace) -> None:
    """
    Worker process for parallel data collection.
    
    Args:
        worker_id: Unique ID for this worker process
        args: Command-line arguments
    """
    # Create worker-specific output directory
    worker_output_dir = osp.join(args.output_dir, f'{worker_id}')
    os.makedirs(worker_output_dir, exist_ok=True)
    
    # Set worker-specific random seeds for reproducibility
    # Use a large offset between workers to ensure independent random streams
    worker_seed = args.base_seed + worker_id * 10000
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    agent = SimpleAgent(args.prob_forward, args.action_repeat, args.max_consec_fwd)

    # Calculate number of episodes for this worker
    num_episodes = args.num_episodes // args.n_parallel
    if worker_id < (args.num_episodes % args.n_parallel):
        num_episodes += 1
    
    pbar = tqdm(total=num_episodes, position=worker_id, desc=f"Worker {worker_id}")
    episode_count = 0
    
    while episode_count < num_episodes:
        # Create environment with unique seeds for each worker and episode
        # Ensure world_seed and seed are different for each worker and episode
        episode_seed_base = worker_seed + episode_count * 100
        world_seed = episode_seed_base
        env_seed = episode_seed_base + 1
        
        env = minedojo.make(
            task_id="open-ended",
            image_size=(360, 640),
            world_seed=world_seed,
            seed=env_seed,
            generate_world_type='specified_biome',
            specified_biome=args.env_type,
            initial_weather='rain'
        )

        # Collect episode data
        out = collect_episode(env, agent, args.traj_length)
        if out is None:
            env.close()
            continue

        rgb, actions, poses = out
        
        # Save video
        video_fname = osp.join(worker_output_dir, f'{episode_count:06d}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_fname, fourcc, 10.0, (rgb.shape[3], rgb.shape[2]))
        
        for t in range(rgb.shape[0]):
            frame = rgb[t].transpose(1, 2, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()

        # Save actions and poses
        action_fname = osp.join(worker_output_dir, f'{episode_count:06d}.npz')
        np.savez_compressed(action_fname, actions=actions, poses=poses)
        
        episode_count += 1
        env.close()
        pbar.update(1)
    
    pbar.close()


def main(args: argparse.Namespace) -> None:
    """
    Main function to orchestrate parallel data collection.
    
    Args:
        args: Command-line arguments
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # Create and start worker processes
    procs = [mp.Process(target=worker, args=(i, args)) for i in range(args.n_parallel)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate MineDojo trajectory data with parallel collection'
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default='test',
        help='Output directory for generated data'
    )
    parser.add_argument(
        '--env_type',
        type=str,
        default='test',
        help='Biome type for environment generation'
    )
    parser.add_argument(
        '-z', '--n_parallel',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '-a', '--action_repeat',
        type=int,
        default=5,
        help='Number of times to repeat each action (default: 5)'
    )
    parser.add_argument(
        '-p', '--prob_forward',
        type=float,
        default=0.7,
        help='Probability of forward/backward actions (default: 0.7)'
    )
    parser.add_argument(
        '-m', '--max_consec_fwd',
        type=int,
        default=50,
        help='Maximum consecutive forward movements (default: 50)'
    )
    parser.add_argument(
        '-t', '--traj_length',
        type=int,
        default=1500,
        help='Length of each trajectory (default: 1500)'
    )
    parser.add_argument(
        '-n', '--num_episodes',
        type=int,
        default=100000,
        help='Total number of episodes to generate (default: 100000)'
    )
    parser.add_argument(
        '-r', '--resolution',
        type=int,
        default=128,
        help='Resolution (currently unused, default: 128)'
    )
    parser.add_argument(
        '-rh', '--resolution_h',
        type=int,
        default=360,
        help='Height resolution (currently unused, default: 360)'
    )
    parser.add_argument(
        '-rw', '--resolution_w',
        type=int,
        default=640,
        help='Width resolution (currently unused, default: 640)'
    )
    parser.add_argument(
        '--base_seed',
        type=int,
        default=42,
        help='Base RNG seed; worker i uses base_seed+i (default: 42)'
    )
    
    args = parser.parse_args()
    main(args)
