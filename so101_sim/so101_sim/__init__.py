from so101_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gym.envs.registration import register

register(
    id="SO101Sim-v0",
    entry_point="so101_sim.envs:So101GymEnv",
    max_episode_steps=200,
)
register(
    id="SO101SimVision-v0",
    entry_point="so101_sim.envs:So101GymEnv",
    max_episode_steps=200,
    kwargs={"image_obs": True},
)

register(
    id="SO101PickCube-v0",
    entry_point="so101_sim.envs:So101PickCubeGymEnv",
    max_episode_steps=200,
)
register(
    id="SO101PickCubeVision-v0",
    entry_point="so101_sim.envs:So101PickCubeGymEnv",
    max_episode_steps=200,
    kwargs={"image_obs": True},
)

