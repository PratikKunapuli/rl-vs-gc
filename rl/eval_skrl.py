# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys 
from omni.isaac.lab.app import AppLauncher

from utils import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--baseline", type=bool, default=False, help="Use baseline policy.")
parser.add_argument("--case_study", type=bool, default=False, help="Use case study policy.")
parser.add_argument("--save_prefix", type=str, default="", help="Prefix for saving files.")
parser.add_argument("--follow_robot", type=int, default=-1, help="Follow robot index.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")

cli_args.add_skrl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
args_cli.enable_cameras = True
args_cli.headless = True # make false to see the simulation

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import yaml
import envs
from controllers.decoupled_controller import DecoupledController
import time
import numpy as np

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.3.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
# from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
from utils.skrl_wrapper import IsaacLabWrapper
from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg, DirectMARLEnvCfg

# config shortcuts
algorithm = args_cli.algorithm.lower()

@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    experiment_cfg["agent"]["experiment"]["directory"] = args_cli.experiment_name
    if args_cli.load_run:
        experiment_cfg["agent"]["experiment"]["experiment_name"] = args_cli.load_run


    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"], experiment_cfg["agent"]["experiment"]["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.join(log_root_path, "checkpoints", args_cli.checkpoint)
        resume_path = os.path.abspath(resume_path)
    else:
        resume_path = os.path.join(log_root_path, "checkpoints", "best_agent.pt")
        resume_path = os.path.abspath(resume_path)
    log_dir = os.path.dirname(os.path.dirname(resume_path))


    if not args_cli.baseline:
        policy_path = log_dir
    else:
        env_cfg.gc_mode = True
        if "Crazyflie" in args_cli.task:
            env_cfg.task_body = "body"
            env_cfg.goal_body = "body"
            env_cfg.reward_task_body = "endeffector"
            env_cfg.reward_goal_body = "endeffector"

            policy_path = "./baseline_cf_0dof/"
        else:
            env_cfg.task_body = "COM"
            env_cfg.goal_body = "COM"
            env_cfg.reward_task_body = "root"
            env_cfg.reward_goal_body = "root"

            policy_path = "./baseline_0dof_ee_reward_tune/"

    env_cfg.eval_mode = True
    env_cfg.viewer.resolution = (1920, 1080)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if os.path.exists(os.path.join(log_dir, "params/env.yaml")):
        with open(os.path.join(log_dir, "params/env.yaml")) as f:
            hydra_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
        # f = os.path.join(log_dir, "params/env.yaml")
        # loader = yaml.YAML(typ="safe")
        # hydra_cfg = loader.load(f)

        
        if "use_yaw_representation" in hydra_cfg:
            env_cfg.use_yaw_representation = hydra_cfg["use_yaw_representation"]
        if "use_full_ori_matrix" in hydra_cfg:
            env_cfg.use_full_ori_matrix = hydra_cfg["use_full_ori_matrix"]
        
        if not ("Ball" in args_cli.task):
            if "scale_reward_with_time" in hydra_cfg:
                env_cfg.scale_reward_with_time = hydra_cfg["scale_reward_with_time"]
            if "yaw_error_reward_scale" in hydra_cfg:
                env_cfg.yaw_error_reward_scale = hydra_cfg["yaw_error_reward_scale"]
            if "yaw_distance_reward_scale" in hydra_cfg:
                env_cfg.yaw_distance_reward_scale = hydra_cfg["yaw_distance_reward_scale"]
            if "yaw_smooth_transition_scale" in hydra_cfg:
                env_cfg.yaw_smooth_transition_scale = hydra_cfg["yaw_smooth_transition_scale"]
            if "yaw_radius" in hydra_cfg:
                env_cfg.yaw_radius = hydra_cfg["yaw_radius"]
            if "pos_distance_reward_scale" in hydra_cfg:
                env_cfg.pos_distance_reward_scale = hydra_cfg["pos_distance_reward_scale"]
            if "pos_error_reward_scale" in hydra_cfg:
                env_cfg.pos_error_reward_scale = hydra_cfg["pos_error_reward_scale"]
            if "lin_vel_reward_scale" in hydra_cfg:
                env_cfg.lin_vel_reward_scale = hydra_cfg["lin_vel_reward_scale"]
            if "ang_vel_reward_scale" in hydra_cfg:
                env_cfg.ang_vel_reward_scale = hydra_cfg["ang_vel_reward_scale"]
            if "combined_alpha" in hydra_cfg:
                env_cfg.combined_alpha = hydra_cfg["combined_alpha"]
            if "combined_tolerance" in hydra_cfg:
                env_cfg.combined_tolerance = hydra_cfg["combined_tolerance"]
            if "combined_reward_scale" in hydra_cfg:
                env_cfg.combined_reward_scale = hydra_cfg["combined_reward_scale"]

    if env_cfg.use_yaw_representation:
        # env_cfg.num_observations += 4
        env_cfg.num_observations += 1
    
    if env_cfg.use_full_ori_matrix:
        # env_cfg.num_observations += 6
        env_cfg.num_observations += 9

    if "Traj" in args_cli.task:
        env_cfg.goal_cfg = "rand"

    env_cfg.seed = args_cli.seed

    robot_index_prefix = ""
    if args_cli.case_study:
        # Manual override of env cfg
        env_cfg.goal_cfg = "fixed"
        # env_cfg.goal_pos = [0.0, 0.0, 0.5]
        env_cfg.goal_pos = [0.0, 0.0, 3.0]
        env_cfg.goal_ori = [0.7071068, 0.0, 0.0, 0.7071068]
        env_cfg.init_cfg = "default"

        # Camera settings
        if "Crazyflie" in args_cli.task:
            env_cfg.viewer.eye = (0.25, 0.25, 3.25)
            # env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
        else:
            env_cfg.viewer.eye = (0.75, 0.75, 3.75)
            # env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 3.0)
        env_cfg.viewer.resolution = (1080, 1920)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
            
    else:
        if args_cli.follow_robot >= 0:
            if "Crazyflie" in args_cli.task:
                env_cfg.viewer.eye = (-0.5, 0.5, 0.5)
                env_cfg.viewer.resolution = (1920, 1080)
                env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
                env_cfg.viewer.origin_type = "asset_root"
                env_cfg.viewer.env_index = args_cli.follow_robot
                env_cfg.viewer.asset_name = "robot"
            else:
                env_cfg.viewer.eye = (0.75, 0.75, 0.75)
                env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
                env_cfg.viewer.resulution = (1080, 1920)
                env_cfg.viewer.origin_type = "asset_root"
                env_cfg.viewer.env_index = args_cli.follow_robot
                env_cfg.viewer.asset_name = "robot"
            robot_index_prefix = f"_robot_{args_cli.follow_robot}"

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    save_prefix = args_cli.save_prefix
    if args_cli.case_study:
        save_prefix = "case_study_"
    
    if "Ball" in args_cli.task:
        save_prefix += "ball_catch_"

    if "Traj" in args_cli.task:
        save_prefix += "eval_traj_track_" + str(int(1/env_cfg.traj_update_dt)) + "Hz_"

    
    # save_prefix = "ball_catch_side_view_"
    video_name = save_prefix + "_eval_video" + robot_index_prefix
    if args_cli.baseline:
        video_folder_path = f"{policy_path}"
    else:
        video_folder_path = os.path.join(policy_path, "videos", "eval")

    video_kwargs = {
        "video_folder": video_folder_path,
        "step_trigger": lambda step: step == 0,
        # "episode_trigger": lambda episode: (episode % args.save_interval) == 0,
        "video_length": args_cli.video_length,
        "name_prefix": video_name
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    device = env.unwrapped.device

    if args_cli.baseline:
        vehicle_mass = env.vehicle_mass
        arm_mass = env.arm_mass
        inertia =  env.quad_inertia
        arm_offset = env.arm_offset
        pos_offset = env.position_offset
        ori_offset = env.orientation_offset
        # Hand-tuned gains
        # agent = DecoupledController(args_cli.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device)
        
        
        if "Crazyflie" not in args_cli.task:
            # Optuna-tuned gains for EE-Reward
            agent = DecoupledController(args_cli.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
                                        kp_pos_gain_xy=43.507, kp_pos_gain_z=24.167, kd_pos_gain_xy=9.129, kd_pos_gain_z=6.081,
                                        kp_att_gain_xy=998.777, kp_att_gain_z=18.230, kd_att_gain_xy=47.821, kd_att_gain_z=8.818)
        else:
            # Crazyflie DC
            agent = DecoupledController(args_cli.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
                                        kp_pos_gain_xy=6.5, kp_pos_gain_z=15.0, kd_pos_gain_xy=4.0, kd_pos_gain_z=9.0,
                                        kp_att_gain_xy=544, kp_att_gain_z=544, kd_att_gain_xy=46.64, kd_att_gain_z=46.64, 
                                        skip_precompute=True, vehicle="Crazyflie", control_mode="CTATT", print_debug=False)
            
        # Optuna-tuned gains for EE-LQR Cost (equal pos and yaw weight)
        # agent = DecoupledController(args_cli.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
        #                             kp_pos_gain_xy=24.675, kp_pos_gain_z=31.101, kd_pos_gain_xy=7.894, kd_pos_gain_z=8.207,
        #                             kp_att_gain_xy=950.228, kp_att_gain_z=10.539, kd_att_gain_xy=39.918, kd_att_gain_z=5.719)
        
        # Optuna-tuned gains for COM-Reward
        # agent = DecoupledController(args_cli.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
        #                             kp_pos_gain_xy=38.704, kp_pos_gain_z=39.755, kd_pos_gain_xy=10.413, kd_pos_gain_z=13.509,
        #                             kp_att_gain_xy=829.511, kp_att_gain_z=1.095, kd_att_gain_xy=38.383, kd_att_gain_z=4.322)
        
        # Optuna-tuned gains for COM-LQR Cost (equal pos and yaw weight)
        # agent = DecoupledController(args_cli.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
        #                             kp_pos_gain_xy=49.960, kp_pos_gain_z=23.726, kd_pos_gain_xy=13.218, kd_pos_gain_z=6.878,
        #                             kp_att_gain_xy=775.271, kp_att_gain_z=3.609, kd_att_gain_xy=41.144, kd_att_gain_z=1.903)
        
        # Optuna-tuned gains for COM-LQR Cost (environment has further away goals)
        # agent = DecoupledController(args_cli.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
        #                             kp_pos_gain_xy=24.172, kp_pos_gain_z=28.362, kd_pos_gain_xy=6.149, kd_pos_gain_z=8.881,
        #                             kp_att_gain_xy=955.034, kp_att_gain_z=14.370, kd_att_gain_xy=36.101, kd_att_gain_z=8.828)
    
    else:
        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
            env = multi_agent_to_single_agent(env)

        # wrap around environment for skrl
        env = IsaacLabWrapper(env) # custom wrapper that retains dict observation space

        # configure and instantiate the skrl runner
        # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
        runner = Runner(env, experiment_cfg)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        # set agent to evaluation mode
        runner.agent.set_running_mode("eval")

    # reset environment
    if args_cli.baseline:
        obs_dict, info = env.reset()
    else:
        obs, _ = env.reset()
        obs_dict = env.getObservationsDict()

    full_state_size = obs_dict["full_state"].shape[1]
    full_states = torch.zeros((args_cli.num_envs, 500, full_state_size), dtype=torch.float32).to(device)
    rewards = torch.zeros((args_cli.num_envs, 500), dtype=torch.float32).to(device)
    steps = 0
    done = False
    done_count = 0
    times = []
    # simulate environment
    with torch.no_grad():
        while simulation_app.is_running():
            while steps < 500 and not done:
                # run everything in inference mode
                with torch.inference_mode():
                    full_states[:, steps, :] = obs_dict["full_state"]

                    start = time.time()
                    # agent stepping
                    if args_cli.baseline:
                        # action = agent.get_action(obs_dict["full_state"])
                        actions = agent.get_action(obs_dict["gc"])
                        # print("Obs: ", obs_dict["gc"][args_cli.follow_robot])
                    else:
                        obs_tensor = obs_dict["policy"]
                        actions = runner.agent.act(obs_tensor, timestep=0, timesteps=0)[0]
                    times.append(time.time() - start)

                    if args_cli.baseline:
                        obs_dict, reward, terminated, truncated, info = env.step(actions)
                    else:
                        # env stepping
                        obs, reward, terminated, truncated, info = env.step(actions)
                        obs_dict = env.getObservationsDict()
                        reward = reward.squeeze()
                    done_count += terminated.sum().item() + truncated.sum().item()
                    rewards[:, steps] = reward.detach()
                    steps += 1
                    print("Step: ", steps)

            print("Full states shape: ", full_states.shape)
            torch.save(full_states, os.path.join(policy_path, save_prefix + "eval_full_states.pt"))
            torch.save(rewards, os.path.join(policy_path, save_prefix + "eval_rewards.pt"))

            print("Final Info: \n\n", info, "\n")

            print("\nAverage inference time: ", np.mean(times))

            # close the simulator
            env.close()
            simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()