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
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.left, self.right = False, False
        self.action_indices = action_indices

        self.intervened = False
        self.recording = False
        self.action_length = 1.0
        self.current_action = np.zeros(6, dtype=np.float64)
        self.key_states = {
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "5": 0,
            "6": 0,
        }

        glfw.set_key_callback(self.env._viewer.viewer.window, self.glfw_on_key)

    def glfw_on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            direction = -1 if (mods & glfw.MOD_CONTROL) else 1
            if key == glfw.KEY_1:
                self.key_states["1"] = direction
            elif key == glfw.KEY_2:
                self.key_states["2"] = direction
            elif key == glfw.KEY_3:
                self.key_states["3"] = direction
            elif key == glfw.KEY_4:
                self.key_states["4"] = direction
            elif key == glfw.KEY_5:
                self.key_states["5"] = direction
            elif key == glfw.KEY_6:
                self.key_states["6"] = direction
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
            if key == glfw.KEY_1:
                self.key_states["1"] = 0
            elif key == glfw.KEY_2:
                self.key_states["2"] = 0
            elif key == glfw.KEY_3:
                self.key_states["3"] = 0
            elif key == glfw.KEY_4:
                self.key_states["4"] = 0
            elif key == glfw.KEY_5:
                self.key_states["5"] = 0
            elif key == glfw.KEY_6:
                self.key_states["6"] = 0

        self.current_action = [
            self.key_states["1"],
            self.key_states["2"],
            self.key_states["3"],
            self.key_states["4"],
            self.key_states["5"],
            self.key_states["6"],
        ]
        self.current_action = np.array(self.current_action, dtype=np.float64)
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

