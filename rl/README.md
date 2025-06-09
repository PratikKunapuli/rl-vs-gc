### Reinforcement Learning (RL)
This folder contains implementations of the reinforcement learning (RL) scripts used to train and evaluate in the various environments supported. The scripts are slightly modified from teh vanilla IsaacLab training and evaluation scripts.

## Training
Training is done with the `train_rslrl.py` script. Almost all of the configuration is done via Hydra for setting the environment parameters such as reward shaping, curriculum, initialization, and trajectory, as well as agent parameters such as learning rate, number of updates, etc. 

### Example:
```bash
python train_rslrl.py --task Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0 --num_envs 4096 agent.num_steps_per_env=64 --experiment_name PaperModels --run_name AM_0DOF_RL_Opt_Lissajous_NoFF 'env.lissajous_amplitudes_rand_ranges=[2.0, 2.0, 2.0, 2.0]' 'env.lissajous_frequencies_rand_ranges=[3.0, 3.0, 3.0, 2.0]' 'env.lissajous_phases_rand_ranges=[3.14159, 3.14159, 3.14159, 3.14159]' 'env.lissajous_offsets_rand_ranges=[0.5, 0.5, 0.5, 3.14159]'  env.pos_radius_curriculum=39000000 env.init_cfg="rand" 'env.init_pos_ranges=[0.5, 0.5, 0.5]' 'env.init_lin_vel_ranges=[0.1, 0.1, 0.1]' 'env.init_yaw_ranges=[3.14159]' 'env.init_ang_vel_ranges=[0.1, 0.1, 0.1]' agent.max_iterations=1200 --device cuda:0 'env.yaw_error_reward_scale=-4.0' env.trajectory_horizon=0
```

The training pipeline will create a logging folder under `rl/logs/rsl_rl/{Experiment Name}/{Run Name}/` and model weights will periodically be saved there. 

## Evaluation
Evaluation can be performed by running the `eval_rslrl.py` script and pointing to a pre-trained experiment and run name, as well as dynamically configurable env parameters via Hydra. 

Required Flags:
- `--task` is the IsaacLab env name
- `--num_envs` is the number of evaluation environments to use (1000)
- `--experiment_name` is the name of the experiment 
- `--load_run` will load the last checkpoint in a folder if the folder name is specified, otherwise the specific model weights 

Optional Flags:
- `--video` will record a video
- `--follow_robot N` will change the camera to track robot N
- `--save_prefix PREFIX` will prepend a prefix to the saved rollout data and video 

### Example:
```bash
python eval_rslrl.py --video --task Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0 --num_envs 1000  env.yaw_error_reward_scale=-4.0 env.pos_radius=0.1 --follow_robot 0 env.viz_mode="robot" --save_prefix "4d_lissajous_rand_init_" 'env.lissajous_amplitudes_rand_ranges=[2.0, 2.0, 2.0, 2.0]' 'env.lissajous_frequencies_rand_ranges=[3.0, 3.0, 3.0, 2.0]' 'env.lissajous_phases_rand_ranges=[3.14159, 3.14159, 3.14159, 3.14159]' 'env.lissajous_offsets_rand_ranges=[0.5, 0.5, 0.5, 3.14159]' 'env.init_cfg=rand' 'env.init_pos_ranges=[0.5, 0.5, 0.5]' 'env.init_lin_vel_ranges=[0.1, 0.1, 0.1]' 'env.init_yaw_ranges=[3.14159]' 'env.init_ang_vel_ranges=[0.1, 0.1, 0.1]' --device cuda:1 --experiment_name PaperModels --load_run AM_0DOF_RL_Opt_Lissajous_FF env.trajectory_horizon=10
```