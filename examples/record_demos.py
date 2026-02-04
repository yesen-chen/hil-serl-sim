# import debugpy
# debugpy.listen(10010)
# print('wait debugger')
# debugpy.wait_for_client()
# print("Debugger Attached")

import sys
sys.path.insert(0, '../../../')
import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    
    obs, info = env.reset()
    print("=" * 50)
    print("Reset done！")
    print("Instructions:")
    print("  - Use keyboard to control the robot (W/A/S/D/J/K move, L toggle gripper)")
    print("  - Press ; key to toggle manual intervention mode")
    print("  - Press Enter to start/stop data collection")
    print("=" * 50)
    print("Waiting for key press to start recording...")
    
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed, desc="Success trajectories")
    trajectory = []
    returns = 0
    is_recording = False  # Whether currently recording
    
    while success_count < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        
        # 检查采集状态变化
        current_recording = info.get("recording", False)
        if current_recording != is_recording:
            is_recording = current_recording
            if is_recording:
                print("\n>>> 开始采集数据...")
                trajectory = []  # 开始新的轨迹
                returns = 0
            else:
                print("\n>>> 暂停采集数据")
        
        # 只有在采集模式下才记录数据
        if is_recording:
            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)
            pbar.set_description(f"采集中 | Return: {returns:.2f} | 步数: {len(trajectory)}")
        else:
            pbar.set_description(f"等待中 | 成功: {success_count}/{success_needed}")

        obs = next_obs
        
        # 处理 episode 结束
        if done and is_recording:
            if info["succeed"]:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)
                print(f"\n✓ 轨迹成功！已采集 {success_count}/{success_needed} 条轨迹")
            else:
                print(f"\n✗ 轨迹失败，未保存")
            
            trajectory = []
            returns = 0
            is_recording = False  # 重置采集状态，等待下次回车
            obs, info = env.reset()
            print("环境已重置，按回车键开始下一次采集...")
        elif done and not is_recording:
            # 不在采集模式下 episode 结束，直接重置
            obs, info = env.reset()
            
    pbar.close()
    
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"\n保存 {success_needed} 条演示数据到 {file_name}")

if __name__ == "__main__":
    app.run(main)
