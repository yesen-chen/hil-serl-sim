#!/usr/bin/env python3
"""
Replay script: Load saved buffer data and replay robot actions in simulation environment
"""

import sys
sys.path.insert(0, '../../../')

import os
import glob
import pickle as pkl
import numpy as np
import time
from absl import app, flags
from natsort import natsorted
import cv2

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "pick_cube_sim", "Name of experiment corresponding to folder.")
flags.DEFINE_string("pkl_file", None, "Path to pkl file (demo or buffer data).")
flags.DEFINE_integer("start_episode", 0, "Start episode index to replay.")
flags.DEFINE_integer("num_episodes", 1, "Number of episodes to replay.")
flags.DEFINE_boolean("save_video", False, "Save replay video.")
flags.DEFINE_string("video_path", "./replay_videos", "Path to save replay videos.")
flags.DEFINE_float("replay_speed", 1.0, "Replay speed multiplier (1.0 = normal speed).")
flags.DEFINE_boolean("show_rewards", True, "Display rewards during replay.")
flags.DEFINE_boolean("display_window", True, "Display real-time MuJoCo window.")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def load_pkl_data(pkl_file):
    """Load transitions from a pkl file (demo or buffer data)
    
    Args:
        pkl_file: Path to pkl file
    
    Returns:
        List of transitions
    """
    if not os.path.exists(pkl_file):
        raise ValueError(f"Pkl file does not exist: {pkl_file}")
    
    print_green(f"Loading pkl file: {pkl_file}")
    with open(pkl_file, "rb") as f:
        transitions = pkl.load(f)
    
    print_green(f"Total transitions loaded: {len(transitions)}")
    return transitions


def split_into_episodes(transitions):
    """Split transitions into episodes
    
    Args:
        transitions: List of transitions
    
    Returns:
        List of episodes, each episode is a list of transitions
    """
    episodes = []
    current_episode = []
    
    for transition in transitions:
        current_episode.append(transition)
        
        # Episode ends when done is True
        if transition.get('dones', False):
            episodes.append(current_episode)
            current_episode = []
    
    # If there's an incomplete episode at the end
    if len(current_episode) > 0:
        episodes.append(current_episode)
    
    return episodes


def replay_episode(env, episode, replay_speed=1.0, show_rewards=True, save_video=False, video_path=None, episode_idx=0, display_window=True):
    """Replay an episode
    
    Args:
        env: Environment instance
        episode: List of transitions
        replay_speed: Playback speed multiplier
        show_rewards: Whether to display rewards
        save_video: Whether to save video
        video_path: Path to save videos
        episode_idx: Episode index
        display_window: Whether to display real-time window
    """
    print_yellow(f"\n{'='*60}")
    print_yellow(f"Replaying Episode {episode_idx} (Length: {len(episode)} steps)")
    print_yellow(f"{'='*60}\n")
    
    # Reset environment
    obs, _ = env.reset()
    
    # Video saving setup
    video_writer = None
    if save_video:
        os.makedirs(video_path, exist_ok=True)
        video_file = os.path.join(video_path, f"episode_{episode_idx:04d}.mp4")
        
        # Get first frame to determine video dimensions
        frame = env.render()
        if frame is not None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 15  # Adjust based on actual environment frame rate
            video_writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
            print_green(f"Saving video to: {video_file}")
    
    total_reward = 0.0
    step_delay = 1.0 / 15.0 / replay_speed  # Assuming 15Hz, adjust based on replay_speed
    
    for step_idx, transition in enumerate(episode):
        # Get saved action
        action = transition['actions']
        reward = transition['rewards']
        done = transition['dones']
        
        # Execute action
        obs, _, _, _, info = env.step(action)
        
        # Render
        frame = env.render()
        
        # Save video frame (frame is not None only in rgb_array mode)
        if save_video and video_writer is not None and frame is not None:
            # OpenCV uses BGR format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        # Display information
        total_reward += reward
        if show_rewards:
            print(f"Step {step_idx + 1}/{len(episode)}: "
                  f"Action: {action}, "
                  f"Reward: {reward:.3f}, "
                  f"Total Reward: {total_reward:.3f}, "
                  f"Done: {done}")
        
        # Control playback speed
        time.sleep(step_delay)
        
        # If episode ends
        if done:
            print_green(f"\nEpisode {episode_idx} completed!")
            print_green(f"Total steps: {step_idx + 1}")
            print_green(f"Total reward: {total_reward:.3f}")
            if 'succeed' in info:
                print_green(f"Success: {info['succeed']}")
            break
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print_green(f"Video saved successfully!")
    
    return total_reward


def main(_):
    # Check parameters
    if FLAGS.pkl_file is None:
        raise ValueError("Please specify --pkl_file")
    else:
        pkl_file = os.path.abspath(FLAGS.pkl_file)
    
    # Load configuration
    assert FLAGS.exp_name in CONFIG_MAPPING, f"Experiment {FLAGS.exp_name} not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # Create environment (for replay, classifier not needed)
    print_green("Initializing environment...")
    
    # Choose render mode based on requirements
    # Use human mode for real-time display; use rgb_array mode for video saving
    if FLAGS.display_window and not FLAGS.save_video:
        render_mode = "human"
        print_green("Using render_mode='human' for real-time display")
    else:
        render_mode = "rgb_array"
        print_green("Using render_mode='rgb_array' for video recording or headless mode")
    
    env = config.get_environment(
        fake_env=True,  # No keyboard interaction wrapper needed
        save_video=False,  # We handle video saving ourselves
        classifier=False,  # Classifier not needed for replay
        render_mode=render_mode
    )
    
    print_green("Environment initialized successfully!")
    
    # Load pkl data
    print_green("\nLoading pkl data...")
    transitions = load_pkl_data(pkl_file)
    
    # Split into episodes
    print_green("\nSplitting transitions into episodes...")
    episodes = split_into_episodes(transitions)
    print_green(f"Total episodes: {len(episodes)}")
    
    # Calculate episode range to replay
    start_idx = FLAGS.start_episode
    end_idx = min(start_idx + FLAGS.num_episodes, len(episodes))
    
    if start_idx >= len(episodes):
        raise ValueError(f"start_episode ({start_idx}) >= total episodes ({len(episodes)})")
    
    print_yellow(f"\nReplaying episodes {start_idx} to {end_idx - 1}")
    
    # Replay episodes
    total_rewards = []
    for episode_idx in range(start_idx, end_idx):
        total_reward = replay_episode(
            env=env,
            episode=episodes[episode_idx],
            replay_speed=FLAGS.replay_speed,
            show_rewards=FLAGS.show_rewards,
            save_video=FLAGS.save_video,
            video_path=FLAGS.video_path,
            episode_idx=episode_idx,
            display_window=FLAGS.display_window
        )
        total_rewards.append(total_reward)
        
        # Wait before playing next episode
        if episode_idx < end_idx - 1:
            print("\nPress Enter to continue to next episode...")
            input()
    
    # Statistics
    print_yellow(f"\n{'='*60}")
    print_yellow("Replay Summary")
    print_yellow(f"{'='*60}")
    print_green(f"Episodes replayed: {len(total_rewards)}")
    print_green(f"Average reward: {np.mean(total_rewards):.3f}")
    print_green(f"Min reward: {np.min(total_rewards):.3f}")
    print_green(f"Max reward: {np.max(total_rewards):.3f}")
    
    env.close()
    print_green("\nReplay completed!")


if __name__ == "__main__":
    app.run(main)

