import argparse
import sys 
from omni.isaac.lab.app import AppLauncher

# local imports
from utils import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CleanRL. ")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--goal_task", type=str, default="rand", help="Goal task for the environment.")
parser.add_argument("--frame", type=str, default="root", help="Frame of the task.")
parser.add_argument("--baseline", type=bool, default=False, help="Use baseline policy.")
parser.add_argument("--baseline_gains", type=str, default=None, help="Baseline gains to use.")
parser.add_argument("--use_integral_terms", type=bool, default=False, help="Use integral terms in the controller.")
parser.add_argument("--case_study", type=bool, default=False, help="Use case study policy.")
parser.add_argument("--save_prefix", type=str, default="", help="Prefix for saving files.")
parser.add_argument("--follow_robot", type=int, default=-1, help="Follow robot index.")


# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
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


import os
import random
import time
from dataclasses import dataclass
import ast
import re
# import ruamel.yaml as yaml
import yaml

import gymnasium as gym
import envs
from controllers.decoupled_controller import DecoupledController
from controllers.gc_params import gc_params_dict


from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from omni.isaac.lab.utils.io import load_yaml

import numpy as np
import torch

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # torch.manual_seed(args_cli.seed)
    # env_cfg = parse_env_cfg(
    #     args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    # )

    
    # agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print("Resume path: ", resume_path)
    log_dir = os.path.dirname(resume_path)
    print("Log dir: ", log_dir)
    
    if not args_cli.baseline:
        policy_path = log_dir
    else:
        # policy_path = "./baseline_0dof/"
        # policy_path = "./baseline_0dof_com_lqr_tune/"
        # policy_path = "./baseline_0dof_com_reward_tune/"
        # policy_path = "./baseline_0dof_ee_reward_tune/"
        # policy_path = "./baseline_0dof_ee_lqr_tune/"

        
        # env_cfg.sim_rate_hz = 100
        # env_cfg.policy_rate_hz = 50
        # env_cfg.sim.dt = 1/env_cfg.sim_rate_hz
        # env_cfg.decimation = env_cfg.sim_rate_hz // env_cfg.policy_rate_hz
        # env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.gc_mode = True
        if "Crazyflie" in args_cli.task:
            env_cfg.task_body = "body"
            env_cfg.goal_body = "body"
            # env_cfg.reward_task_body = "endeffector"
            # env_cfg.reward_goal_body = "endeffector"
            env_cfg.reward_task_body = "body"
            env_cfg.reward_goal_body = "body"

            # policy_path = "./baseline_cf_0dof/"
        else:
            env_cfg.task_body = "COM"
            env_cfg.goal_body = "COM"
            env_cfg.reward_task_body = "root"
            env_cfg.reward_goal_body = "root"

            # policy_path = "./baseline_0dof_ee_reward_tune/"

        task_name = args_cli.task            
        if args_cli.use_integral_terms:
            task_name = args_cli.task + "-Integral"
        elif args_cli.baseline_gains is not None:
            task_name = args_cli.task + "-" + args_cli.baseline_gains
        
        if task_name in gc_params_dict.keys():
            policy_path = gc_params_dict[task_name]["log_dir"]
        else:
            print(f"[ERROR] Task name {task_name} not found in gc_params_dict.")
            print(f"Available tasks: {gc_params_dict.keys()}")
            return
            

        # env_cfg.yaw_distance_reward_scale = 5.0
    # else:
    #     print("\n\nSaved args: ", saved_args_cli)
    #     print("Keys: ", saved_args_cli.keys())
    #     env_cfg = update_env_cfg(env_cfg, saved_args_cli)

    env_cfg.eval_mode = True
    env_cfg.viewer.resolution = (1920, 1080)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    

    # If ".hydra/config.yaml" is present, load some of the reward scalars from there
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

    # else:
    #     yaml_base = "./logs/rsl_rl/AM_0DOF_Hover/2024-09-14_14-38-12_rsl_rl_test_default_1024_env_pos_distance_15_yaw_error_-2.0_no_smooth_transition_full_ori"
    #     with open(os.path.join(yaml_base, "params/env.yaml"), "r") as f:
    #         hydra_cfg = yaml.load(f, Loader=yaml.FullLoader)
    #         if "use_yaw_representation" in hydra_cfg:
    #             env_cfg.use_yaw_representation = hydra_cfg["use_yaw_representation"]
    #         if "yaw_error_reward_scale" in hydra_cfg:
    #             env_cfg.yaw_error_reward_scale = hydra_cfg["yaw_error_reward_scale"]
    #         if "yaw_distance_reward_scale" in hydra_cfg:
    #             env_cfg.yaw_distance_reward_scale = hydra_cfg["yaw_distance_reward_scale"]
    #         if "yaw_smooth_transition_scale" in hydra_cfg:
    #             env_cfg.yaw_smooth_transition_scale = hydra_cfg["yaw_smooth_transition_scale"]
    #         if "yaw_radius" in hydra_cfg:
    #             env_cfg.yaw_radius = hydra_cfg["yaw_radius"]
            
    #         if "use_full_ori_matrix" in hydra_cfg:
    #             env_cfg.use_full_ori_matrix = hydra_cfg["use_full_ori_matrix"]
            
    #         if "scale_reward_with_time" in hydra_cfg:
    #             env_cfg.scale_reward_with_time = hydra_cfg["scale_reward_with_time"]

    # env_cfg.yaw_radius = 0.5
    
    if env_cfg.use_yaw_representation:
        # env_cfg.num_observations += 4
        env_cfg.num_observations += 1
    
    if env_cfg.use_full_ori_matrix:
        # env_cfg.num_observations += 6
        env_cfg.num_observations += 9

    if "Traj" in args_cli.task:
        env_cfg.goal_cfg = "rand"
        # env_cfg.trajectory_params["x_amp"] = 1.0
        # env_cfg.trajectory_params["x_freq"] = 0.5
        # env_cfg.trajectory_params["y_amp"] = 2.0
        # env_cfg.trajectory_params["y_freq"] = 1.0
        # env_cfg.trajectory_params["z_amp"] = 0.0
        # env_cfg.trajectory_params["z_offset"] = 0.5
        # env_cfg.trajectory_params["yaw_amp"] = 1.0
        # env_cfg.trajectory_params["yaw_freq"] = 1.0
        # env_cfg.traj_update_dt = 1.0
        # env_cfg.traj_update_dt = 2.0

    env_cfg.seed = args_cli.seed

    # import code; code.interact(local=locals())
    print("\n\nUpdated env cfg: ", env_cfg)

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
                if "Viz" in args_cli.save_prefix:
                    env_cfg.viewer.eye = (0, 0, 5.5)
                    env_cfg.viewer.lookat = (0, 0, 0)
                    env_cfg.viewer.resulution = (720, 720)
                    # env_cfg.viewer.origin_type = "asset_root"
                    env_cfg.viewer.origin_type = "env"
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


    
    # env_cfg.viewer.eye = (3.0, 1.5, 2.0)
    # env_cfg.viewer.resolution = (1920, 1080)
    # env_cfg.viewer.lookat = (0.0, 1.5, 0.5)
    # env_cfg.viewer.origin_type = "env"
    # env_cfg.viewer.env_index = 0

    # Manual override of env cfg
    # env_cfg.goal_pos_range = 2.0
    # env_cfg.goal_yaw_range = 0.0 #0.0 1.5708  3.14159



    envs = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    save_prefix = args_cli.save_prefix
    if args_cli.case_study:
        save_prefix = "case_study_"
    
    if "Ball" in args_cli.task:
        save_prefix += "ball_catch_"

    if "Traj" in args_cli.task:
        save_prefix += "eval_traj_track_" + str(int(1/env_cfg.traj_update_dt)) + "Hz_"

    
    # save_prefix = "ball_catch_side_view_"
    if "Traj" in args_cli.task:
        viz_mode = env_cfg.viz_mode
    else:
        viz_mode = ""
        
    video_name = save_prefix + "_eval_video" + robot_index_prefix + "_viz_" + viz_mode
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
    envs = gym.wrappers.RecordVideo(envs, **video_kwargs)
    device = envs.unwrapped.device


    if args_cli.baseline:
        env = envs.unwrapped
        vehicle_mass = envs.unwrapped.vehicle_mass
        arm_mass = envs.unwrapped.arm_mass
        inertia =  envs.unwrapped.quad_inertia
        arm_offset = envs.unwrapped.arm_offset
        pos_offset = envs.unwrapped.position_offset
        ori_offset = envs.unwrapped.orientation_offset

        if "Traj" in args_cli.task:
            feed_forward = True
        else:
            feed_forward = False

        # Hand-tuned gains
        # agent = DecoupledController(envs.num_envs, 0, envs.vehicle_mass, envs.arm_mass, envs.quad_inertia, envs.arm_offset, envs.orientation_offset, com_pos_w=None, device=device)
        
        
        if "Crazyflie" not in args_cli.task:
            # Optuna-tuned gains for EE-Reward
            # use_feed_forward = "Traj" in args_cli.task and "Integral" not in args_cli.task
            control_params_dict = gc_params_dict[task_name]["controller_params"]
            agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
                                        **control_params_dict)
        else:
            # Crazyflie DC
            control_params_dict = gc_params_dict[task_name]["controller_params"]
            # agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
            #                             kp_pos_gain_xy=6.5, kp_pos_gain_z=15.0, kd_pos_gain_xy=4.0, kd_pos_gain_z=9.0,
            #                             kp_att_gain_xy=544, kp_att_gain_z=544, kd_att_gain_xy=46.64, kd_att_gain_z=46.64, 
            #                             skip_precompute=True, vehicle="Crazyflie", control_mode="CTBM", print_debug=False, feed_forward=feed_forward)
            
            agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
                                        vehicle="Crazyflie", **control_params_dict)
            
        # Optuna-tuned gains for EE-LQR Cost (equal pos and yaw weight)
        # agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
        #                             kp_pos_gain_xy=24.675, kp_pos_gain_z=31.101, kd_pos_gain_xy=7.894, kd_pos_gain_z=8.207,
        #                             kp_att_gain_xy=950.228, kp_att_gain_z=10.539, kd_att_gain_xy=39.918, kd_att_gain_z=5.719)
        
        # Optuna-tuned gains for COM-Reward
        # agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
        #                             kp_pos_gain_xy=38.704, kp_pos_gain_z=39.755, kd_pos_gain_xy=10.413, kd_pos_gain_z=13.509,
        #                             kp_att_gain_xy=829.511, kp_att_gain_z=1.095, kd_att_gain_xy=38.383, kd_att_gain_z=4.322)
        
        # Optuna-tuned gains for COM-LQR Cost (equal pos and yaw weight)
        # agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
        #                             kp_pos_gain_xy=49.960, kp_pos_gain_z=23.726, kd_pos_gain_xy=13.218, kd_pos_gain_z=6.878,
        #                             kp_att_gain_xy=775.271, kp_att_gain_z=3.609, kd_att_gain_xy=41.144, kd_att_gain_z=1.903)
        
        # Optuna-tuned gains for COM-LQR Cost (environment has further away goals)
        # agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
        #                             kp_pos_gain_xy=24.172, kp_pos_gain_z=28.362, kd_pos_gain_xy=6.149, kd_pos_gain_z=8.881,
        #                             kp_att_gain_xy=955.034, kp_att_gain_z=14.370, kd_att_gain_xy=36.101, kd_att_gain_z=8.828)
    
    else:
        envs = RslRlVecEnvWrapper(envs) # This calls Reset!!
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        ppo_runner = OnPolicyRunner(envs, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        agent = ppo_runner.get_inference_policy(device=envs.unwrapped.device)

        # actor_params =  sum(p.numel() for p in ppo_runner.alg.actor_critic.actor.parameters() if p.requires_grad)
        # critic_params = sum(p.numel() for p in ppo_runner.alg.actor_critic.critic.parameters() if p.requires_grad)
        # print(f"Actor params: {actor_params}, Critic params: {critic_params}")
        # print("Total params: ", actor_params + critic_params)
        # input()

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
        )
        
    
    if args_cli.baseline:
        obs_dict, info = envs.reset()
        obs = obs_dict["policy"]
        # print(envs.vehicle_mass)

        # Refresh GC models for mass, inertia and ttw
        # agent.reset_dr_terms(None, env.vehicle_mass, env.vehicle_inertia, env._thrust_to_weight)
    else:
        # obs, dict_obs = envs.reset()
        obs, dict_obs = envs.get_observations()
        obs_dict = dict_obs['observations']

    print("Starting obs: ", obs_dict["full_state"])

    ee_start = obs_dict["full_state"][:, 13:16]
    goal_start = obs_dict["full_state"][:, 26:26 + 3]
    # print("starting norm: ", torch.norm(ee_start - goal_start, dim=1))
    # input("Check and press Enter to continue...")
    # import code; code.interact(local=locals())
    max_steps = int(env_cfg.episode_length_s * env_cfg.policy_rate_hz)


    full_state_size = obs_dict["full_state"].shape[1]
    full_states = torch.zeros((args_cli.num_envs, max_steps, full_state_size), dtype=torch.float32).to(device)
    rewards = torch.zeros((args_cli.num_envs, max_steps), dtype=torch.float32).to(device)
    actions_log = torch.zeros((args_cli.num_envs, max_steps, 4), dtype=torch.float32).to(device)


    steps = 0
    done = False
    done_count = 0
    times = []
    # input("Press Enter to continue...")
    with torch.no_grad():
        while simulation_app.is_running():
            while steps < max_steps and not done:
                obs_tensor = obs_dict["policy"]
                full_states[:, steps, :] = obs_dict["full_state"]
                # print("Full State: ", obs_dict["full_state"][0, 33:])

                start = time.time()
                if args_cli.baseline:
                    actions = agent.get_action(obs_dict["gc"])
                else:
                    actions = agent(obs_tensor)
                actions_log[:, steps] = actions
                times.append(time.time() - start)

                if args_cli.baseline:
                    obs_dict, reward, terminated, truncated, info = envs.step(actions)
                    done_count += terminated.sum().item() + truncated.sum().item()
                else:
                    obs, reward, dones, extras = envs.step(actions)
                    # print("Reward: ", reward)
                    done_count += dones.sum().item()
                    obs_dict = extras["observations"]
                    info = extras
                rewards[:, steps] = reward.detach()

                steps += 1
                print("Step: ", steps)

            print("Full states shape: ", full_states.shape)
            torch.save(full_states, os.path.join(policy_path, save_prefix + "eval_full_states.pt"))
            torch.save(rewards, os.path.join(policy_path, save_prefix + "eval_rewards.pt"))

            print("Final Info: \n\n", info, "\n")

            print("\nAverage inference time: ", np.mean(times))

            quad_pos = full_states[args_cli.follow_robot, :-1, 0:3].cpu().numpy()
            quad_quat = full_states[args_cli.follow_robot, :-1, 3:7].cpu().numpy()
            quad_vel = full_states[args_cli.follow_robot, :-1, 7:10].cpu().numpy()
            quad_ang_vel = full_states[args_cli.follow_robot, :-1, 10:13].cpu().numpy()
            ee_pos = full_states[args_cli.follow_robot, :-1, 13:16].cpu().numpy()
            goal_pos = full_states[args_cli.follow_robot, :-1, 26:26 + 3].cpu().numpy()
            



            if args_cli.follow_robot >= 0:
                import matplotlib.pyplot as plt
                import omni.isaac.lab.utils.math as isaac_math_utils
                quad_euler = isaac_math_utils.euler_xyz_from_quat(full_states[args_cli.follow_robot, :-1, 3:7])
                quad_roll = quad_euler[0].cpu().numpy()
                quad_pitch = quad_euler[1].cpu().numpy()
                quad_yaw = quad_euler[2].cpu().numpy()
                T = rewards.shape[1] - 1
                x =  np.arange(T) * (1/env_cfg.policy_rate_hz)

                save_path = video_folder_path +"/" + video_name + ".png"
                
                fig = plt.figure(figsize=(10, 10))
                plt.subplot(4, 3, 1)
                plt.plot(x, quad_pos[:, 0], label="Quad X")
                plt.plot(x, goal_pos[:, 0], label="Goal X")
                plt.legend(loc="best")
                plt.subplot(4, 3, 2)
                plt.plot(x, quad_pos[:, 1], label="Quad Y")
                plt.plot(x, goal_pos[:, 1], label="Goal Y")
                plt.legend(loc="best")
                plt.subplot(4, 3, 3)
                plt.plot(x, quad_pos[:, 2], label="Quad Z")
                plt.plot(x, goal_pos[:, 2], label="Goal Z")
                plt.legend(loc="best")
                plt.subplot(4, 3, 4)
                plt.plot(x, quad_vel[:, 0], label="Quad Vel X")
                plt.legend(loc="best")
                plt.subplot(4, 3, 5)
                plt.plot(x, quad_vel[:, 1], label="Quad Vel Y")
                plt.legend(loc="best")
                plt.subplot(4, 3, 6)
                plt.plot(x, quad_vel[:, 2], label="Quad Vel Z")
                plt.legend(loc="best")
                plt.subplot(4, 3, 7)
                plt.plot(x, quad_roll, label="Quad Roll")
                plt.legend(loc="best")
                plt.subplot(4, 3, 8)
                plt.plot(x, quad_pitch, label="Quad Pitch")
                plt.legend(loc="best")
                plt.subplot(4, 3, 9)
                plt.plot(x, quad_yaw, label="Quad Yaw")
                plt.legend(loc="best")
                plt.subplot(4, 3, 10)
                plt.plot(x, quad_ang_vel[:, 0], label="Quad Ang Vel X")
                plt.legend(loc="best")
                plt.subplot(4, 3, 11)
                plt.plot(x, quad_ang_vel[:, 1], label="Quad Ang Vel Y")
                plt.legend(loc="best")
                plt.subplot(4, 3, 12)
                plt.plot(x, quad_ang_vel[:, 2], label="Quad Ang Vel Z")
                plt.legend(loc="best")
                plt.tight_layout()
                plt.savefig(save_path)
                print(f"Saved plot to {save_path}")
                plt.close(fig)

                # Plot actions
                fig = plt.figure(figsize=(10, 10))
                actions_log = actions_log.cpu().numpy()
                plt.subplot(2, 2, 1)
                plt.plot(x, actions_log[args_cli.follow_robot, :-1, 0], label="Action 1")
                plt.legend(loc="best")
                plt.subplot(2, 2, 2)
                plt.plot(x, actions_log[args_cli.follow_robot, :-1, 1], label="Action 2")
                plt.legend(loc="best")
                plt.subplot(2, 2, 3)
                plt.plot(x, actions_log[args_cli.follow_robot, :-1, 2], label="Action 3")
                plt.legend(loc="best")
                plt.subplot(2, 2, 4)
                plt.plot(x, actions_log[args_cli.follow_robot, :-1, 3], label="Action 4")
                plt.legend(loc="best")
                plt.tight_layout()
                save_path = video_folder_path + "/" + video_name + "_actions.png"
                plt.savefig(save_path)
                print(f"Saved actions plot to {save_path}")
                plt.close(fig)

            envs.close()
            simulation_app.close()

    
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()