# Intro
This package provides a simple SO101 arm simulator written in MuJoCo.
It includes a state-based and a vision-based gym environment.

# Installation
- From `hil-serl-sim` folder, cd into `so101_sim`.
- In your conda environment, run `pip install -e .` to install this package.

# Explore the Environments
- Run `python so101_sim/envs/so101_gym_env.py` to visualize the task.

# Notes
- If you see `egl` errors on CPU machines:
```bash
export MUJOCO_GL=egl
conda install -c conda-forge libstdcxx-ng
```

