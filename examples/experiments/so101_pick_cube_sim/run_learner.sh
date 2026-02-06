export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=so101_pick_cube_sim \
    --checkpoint_path=first_run \
    --demo_path=/home/zhang/robot/hil-serl-sim/examples/experiments/so101_pick_cube_sim/demo_data/so101_pick_cube_sim_20_demos_2026-02-06_15-14-16.pkl \
    --learner \