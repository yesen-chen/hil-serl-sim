#!/bin/bash

# Replay buffer数据脚本示例
# 使用方法: bash replay_buffer.sh

# 设置checkpoint路径（根据你的实际路径修改）
CHECKPOINT_PATH="./checkpoint/pick_cube_sim"

# 基本使用：replay第一个episode（带实时窗口显示）
python ../../replay_buffer.py \
    --exp_name=pick_cube_sim \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --start_episode=0 \
    --num_episodes=1 \
    --replay_speed=1.0 \
    --show_rewards=True \
    --display_window=True

# 其他使用示例：

# 1. Replay多个episodes并保存视频
# python ../../replay_buffer.py \
#     --exp_name=pick_cube_sim \
#     --checkpoint_path=${CHECKPOINT_PATH} \
#     --start_episode=0 \
#     --num_episodes=5 \
#     --save_video=True \
#     --video_path=./replay_videos \
#     --replay_speed=1.0

# 2. 快速播放特定episode
# python ../../replay_buffer.py \
#     --exp_name=pick_cube_sim \
#     --checkpoint_path=${CHECKPOINT_PATH} \
#     --start_episode=10 \
#     --num_episodes=1 \
#     --replay_speed=2.0 \
#     --show_rewards=True

# 3. Replay特定的buffer文件
# python ../../replay_buffer.py \
#     --exp_name=pick_cube_sim \
#     --checkpoint_path=${CHECKPOINT_PATH} \
#     --buffer_file=transitions_5000.pkl \
#     --start_episode=0 \
#     --num_episodes=3

