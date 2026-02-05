import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from examples.experiments.config import DefaultTrainingConfig
from so101_sim.envs.so101_pick_cube_gym_env import So101PickCubeGymEnv


class EnvConfig:
    REALSENSE_CAMERAS = {
        "wrist_1": {"serial_number": "sim_cam_1", "dim": (1280, 720), "exposure": 40000},
        "wrist_2": {"serial_number": "sim_cam_2", "dim": (1280, 720), "exposure": 40000},
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[150:450, 350:1100],
        "wrist_2": lambda img: img[100:500, 400:900],
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["joint_pos", "joint_delta"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, render_mode="human"):
        env = So101PickCubeGymEnv(render_mode=render_mode, image_obs=True, hz=15, config=EnvConfig())
        if not fake_env:
            env = KeyBoardIntervention2(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        classifier=False
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier(obs)) > 0.85 and obs["state"][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env


import glfw
import gymnasium as gym


class KeyBoardIntervention2(gym.ActionWrapper):
    """
    键盘控制 SO101 机械臂的 6 个关节:
    - 1/Q: Joint 0 (Rotation) 正/负方向
    - 2/W: Joint 1 (Pitch) 正/负方向
    - 3/E: Joint 2 (Elbow) 正/负方向
    - 4/R: Joint 3 (Wrist_Pitch) 正/负方向
    - 5/T: Joint 4 (Wrist_Roll) 正/负方向
    - 6/Y: Joint 5 (Jaw/Gripper) 正/负方向
    - ;: 切换人为干预模式
    - Enter: 切换录制状态
    """
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.left, self.right = False, False
        self.action_indices = action_indices

        self.intervened = False
        self.recording = False
        self.action_length = 1.0
        self.current_action = np.zeros(6, dtype=np.float64)
        # 正方向键: 1-6, 负方向键: q, w, e, r, t, y
        self.key_states = {
            "pos_0": False, "neg_0": False,  # Joint 0: 1/Q
            "pos_1": False, "neg_1": False,  # Joint 1: 2/W
            "pos_2": False, "neg_2": False,  # Joint 2: 3/E
            "pos_3": False, "neg_3": False,  # Joint 3: 4/R
            "pos_4": False, "neg_4": False,  # Joint 4: 5/T
            "pos_5": False, "neg_5": False,  # Joint 5: 6/Y
        }

        glfw.set_key_callback(self.env._viewer.viewer.window, self.glfw_on_key)

    def glfw_on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            # 正方向: 1-6
            if key == glfw.KEY_1:
                self.key_states["pos_0"] = True
            elif key == glfw.KEY_2:
                self.key_states["pos_1"] = True
            elif key == glfw.KEY_3:
                self.key_states["pos_2"] = True
            elif key == glfw.KEY_4:
                self.key_states["pos_3"] = True
            elif key == glfw.KEY_5:
                self.key_states["pos_4"] = True
            elif key == glfw.KEY_6:
                self.key_states["pos_5"] = True
            # 负方向: Q, W, E, R, T, Y
            elif key == glfw.KEY_Q:
                self.key_states["neg_0"] = True
            elif key == glfw.KEY_W:
                self.key_states["neg_1"] = True
            elif key == glfw.KEY_E:
                self.key_states["neg_2"] = True
            elif key == glfw.KEY_R:
                self.key_states["neg_3"] = True
            elif key == glfw.KEY_T:
                self.key_states["neg_4"] = True
            elif key == glfw.KEY_Y:
                self.key_states["neg_5"] = True
            # 功能键
            elif key == glfw.KEY_SEMICOLON:
                self.intervened = not self.intervened
                self.env.intervened = self.intervened
                print(f"Intervention toggled: {self.intervened}")
            elif key == glfw.KEY_ENTER:
                self.recording = not self.recording
                status = "Start recording" if self.recording else "Stop recording"
                print(f"Recording toggled: {self.recording} ({status})")
                if self.recording:
                    self.env.env_step = 0
                    print("env_step reset to 0")

        elif action == glfw.RELEASE:
            # 正方向: 1-6
            if key == glfw.KEY_1:
                self.key_states["pos_0"] = False
            elif key == glfw.KEY_2:
                self.key_states["pos_1"] = False
            elif key == glfw.KEY_3:
                self.key_states["pos_2"] = False
            elif key == glfw.KEY_4:
                self.key_states["pos_3"] = False
            elif key == glfw.KEY_5:
                self.key_states["pos_4"] = False
            elif key == glfw.KEY_6:
                self.key_states["pos_5"] = False
            # 负方向: Q, W, E, R, T, Y
            elif key == glfw.KEY_Q:
                self.key_states["neg_0"] = False
            elif key == glfw.KEY_W:
                self.key_states["neg_1"] = False
            elif key == glfw.KEY_E:
                self.key_states["neg_2"] = False
            elif key == glfw.KEY_R:
                self.key_states["neg_3"] = False
            elif key == glfw.KEY_T:
                self.key_states["neg_4"] = False
            elif key == glfw.KEY_Y:
                self.key_states["neg_5"] = False

        # 计算各关节动作: 正方向 - 负方向
        self.current_action = np.array([
            int(self.key_states["pos_0"]) - int(self.key_states["neg_0"]),
            int(self.key_states["pos_1"]) - int(self.key_states["neg_1"]),
            int(self.key_states["pos_2"]) - int(self.key_states["neg_2"]),
            int(self.key_states["pos_3"]) - int(self.key_states["neg_3"]),
            int(self.key_states["pos_4"]) - int(self.key_states["neg_4"]),
            int(self.key_states["pos_5"]) - int(self.key_states["neg_5"]),
        ], dtype=np.float64)
        self.current_action *= self.action_length

    def action(self, action: np.ndarray) -> np.ndarray:
        expert_a = self.current_action.copy()

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a
        if self.intervened:
            return expert_a, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        info["recording"] = self.recording
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.recording = False
        print("Environment reset. Press Enter to start recording...")
        return obs, info

