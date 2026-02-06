from pathlib import Path
from typing import Any, Dict, Literal, Tuple
import sys

# Allow direct execution without installing the package.
_PACKAGE_ROOT = Path(__file__).parent.parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import gymnasium as gym
import mujoco
import numpy as np
import time
import cv2

from so101_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv


_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "so101_pick_cube.xml"

_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
_HOME_QPOS = np.asarray([0.0, -1.5708, 1.5708, 0.0, 1.5708, 1.0])
#_HOME_QPOS = np.asarray([0.0, -3.1416, 0.0, 0.0, 0.0, 0.0])
_DEFAULT_ACTION_SCALE = np.asarray([0.05, 0.05, 0.05, 0.08, 0.08, 0.2])

_SAMPLING_BOUNDS = np.asarray([[-0.15, -0.4], [0.15, -0.3]])


class So101PickCubeGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray | None = None,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        config=None,
        hz: int = 10,
    ):
        self.hz = hz
        self._action_scale = (
            np.asarray(action_scale, dtype=np.float32)
            if action_scale is not None
            else _DEFAULT_ACTION_SCALE.copy()
        )

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self._camera_names = ("front", "wrist")
        self._camera_ids = [self._model.camera(name).id for name in self._camera_names]
        self.image_obs = image_obs
        self.env_step = 0
        self.intervened = False
        self._prev_qpos = None

        self._joint_ids = [self._model.joint(name).id for name in _JOINT_NAMES]
        self._qpos_indices = np.asarray(
            [self._model.jnt_qposadr[jid] for jid in self._joint_ids]
        )
        self._ctrl_ids = np.asarray(
            [self._model.actuator(name).id for name in _JOINT_NAMES]
        )
        self._site_id = self._model.site("tcp").id
        self._qpos_min = np.asarray([self._model.jnt_range[jid][0] for jid in self._joint_ids])
        self._qpos_max = np.asarray([self._model.jnt_range[jid][1] for jid in self._joint_ids])

        self._block_qpos_adr = int(np.array(self._model.joint("block").qposadr).item())
        self._block_z = self._model.geom("block").size[2]

        image_keys = None
        if config is not None and hasattr(config, "REALSENSE_CAMERAS"):
            image_keys = list(config.REALSENSE_CAMERAS)
        if not image_keys or len(image_keys) < 2:
            image_keys = ["wrist_1", "wrist_2"]
        if len(image_keys) > 2:
            image_keys = image_keys[:2]
        self._image_keys = image_keys

        # joint_delta 的最大范围约为 action_scale（单步最大变化量）
        max_delta = self._action_scale.astype(np.float32)
        state_space = gym.spaces.Dict(
            {
                "joint_pos": gym.spaces.Box(
                    low=self._qpos_min.astype(np.float32),
                    high=self._qpos_max.astype(np.float32),
                    dtype=np.float32,
                ),
                "joint_delta": gym.spaces.Box(
                    low=-max_delta,
                    high=max_delta,
                    dtype=np.float32,
                ),
            }
        )
        print("state_space = ", state_space)
        if self.image_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": state_space,
                    "images": gym.spaces.Dict(
                        {
                            key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                            for key in self._image_keys
                        }
                    ),
                }
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": state_space,
                    "block_pos": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                }
            )

        self.action_space = gym.spaces.Box(
            low=-np.ones((6,), dtype=np.float32),
            high=np.ones((6,), dtype=np.float32),
            dtype=np.float32,
        )

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self._viewer = MujocoRenderer(self.model, self.data)
        self._viewer.render(self.render_mode)

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[self._qpos_indices] = _HOME_QPOS
        self._data.ctrl[self._ctrl_ids] = _HOME_QPOS

        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.qpos[self._block_qpos_adr : self._block_qpos_adr + 3] = (
            block_xy[0],
            block_xy[1],
            self._block_z,
        )
        self._data.qpos[self._block_qpos_adr + 3 : self._block_qpos_adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0]
        )

        mujoco.mj_forward(self._model, self._data)
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.06
        self.env_step = 0
        self._prev_qpos = None
        obs = self._compute_observation()
        return obs, {"succeed": False}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 6:
            raise ValueError(f"Expected 6D action, got {action.shape}")

        current_qpos = self._data.qpos[self._qpos_indices].copy()
        delta = action * self._action_scale
        target_qpos = np.clip(current_qpos + delta, self._qpos_min, self._qpos_max)
        self._data.ctrl[self._ctrl_ids] = target_qpos

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        #rew = self._compute_reward()
        self.env_step += 1

        success = self._compute_success()
        if success:
            print(f'success!')
            rew = 1
        else:
            rew = 0
            pass
        terminated = success or self.env_step >= 1000
        truncated = False

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)
            dt = time.time() - start_time
            if self.intervened:
                time.sleep(max(0, (1.0 / self.hz) - dt))
            if self.image_obs:
                img_front = obs["images"][self._image_keys[0]]
                img_wrist = obs["images"][self._image_keys[1]]
                img_front_bgr = cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR)
                img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)
                scale = 2
                img_front_large = cv2.resize(
                    img_front_bgr,
                    (img_front_bgr.shape[1] * scale, img_front_bgr.shape[0] * scale),
                )
                img_wrist_large = cv2.resize(
                    img_wrist_bgr,
                    (img_wrist_bgr.shape[1] * scale, img_wrist_bgr.shape[0] * scale),
                )
                combined = np.hstack((img_front_large, img_wrist_large))
                cv2.imshow("SO101 Camera Views (Front | Wrist)", combined)
                cv2.waitKey(1)

        return obs, rew, terminated, truncated, {"succeed": success}

    def render(self):
        rendered_frames = []
        for cam_id in self._camera_ids:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    def _compute_observation(self) -> dict:
        obs: Dict[str, Any] = {"state": {}}

        joint_pos = self._data.qpos[self._qpos_indices].copy().astype(np.float32)
        if self._prev_qpos is None:
            joint_delta = np.zeros_like(joint_pos)
        else:
            joint_delta = (joint_pos - self._prev_qpos).astype(np.float32)
        self._prev_qpos = joint_pos.copy()

        obs["state"] = {
            "joint_pos": joint_pos,
            "joint_delta": joint_delta,
        }

        if self.image_obs:
            images = {}
            front_img, wrist_img = self.render()
            images[self._image_keys[0]] = front_img
            images[self._image_keys[1]] = wrist_img
            obs["images"] = images
        else:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["block_pos"] = block_pos

        return obs

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.site_xpos[self._site_id].copy()
        dist = np.linalg.norm(block_pos - tcp_pos)
        r_close = np.exp(-10 * dist)
        r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init + 1e-6)
        return 0.3 * r_close + 0.7 * r_lift

    def _compute_success(self) -> bool:
        block_pos = self._data.sensor("block_pos").data
        return bool(block_pos[2] >= self._z_success)


if __name__ == "__main__":
    env = So101PickCubeGymEnv(render_mode="human", image_obs=True)
    env.reset()
    for _ in range(2000):
        env.step(np.random.uniform(-1, 1, 6))
    env.close()

