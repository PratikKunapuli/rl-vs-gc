import argparse
from omni.isaac.lab.app import AppLauncher

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
parser.add_argument("--load_path", type=str, default=None, required=True, help="Policy to evaluate.")
parser.add_argument("--goal_task", type=str, default="rand", help="Goal task for the environment.")
parser.add_argument("--frame", type=str, default="root", help="Frame of the task.")
parser.add_argument("--baseline", type=bool, default=False, help="Use baseline policy.")
parser.add_argument("--case_study", type=bool, default=False, help="Use case study policy.")

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.enable_cameras = True
args_cli.headless = True # make false to see the simulation

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import random
import time
from dataclasses import dataclass
import ast
import re
import yaml

import gymnasium as gym
import envs
from policies import Agent, DecoupledController

from omni.isaac.lab_tasks.utils import parse_env_cfg

import numpy as np
import torch

def process_args(args_string):
    clean_str = args_string.strip()[10:-1]
    clean_str = clean_str.replace("=", ":")
    clean_str = clean_str.replace("None", "None").replace("True", "True").replace("False", "False")
    clean_str = re.sub(r'(\w+):', r'"\1":', clean_str)

    try:
        result_dict = ast.literal_eval(f"{{{clean_str}}}")
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Error converting string to dictionary: {e}")
    
    return result_dict

def update_env_cfg(env_cfg, args_cli):
    env_cfg.goal_cfg = args_cli['goal_task']
    env_cfg.task_body = args_cli['frame']
    env_cfg.sim_rate_hz = args_cli['sim_rate_hz']
    env_cfg.policy_rate_hz = args_cli['policy_rate_hz']
    env_cfg.sim.dt = 1/env_cfg.sim_rate_hz
    env_cfg.decimation = env_cfg.sim_rate_hz // env_cfg.policy_rate_hz
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.pos_radius = args_cli['pos_radius']

    # env_cfg.use_yaw_representation = True
    return env_cfg

class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["policy"]


def main():
    torch.manual_seed(args_cli.seed)
    if not args_cli.baseline:
        # find the policy to load
        policy_path = args_cli.load_path
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy {policy_path} not found.")

        if "model" in policy_path:
            model_path = policy_path
            policy_path = os.path.dirname(policy_path)
        else:
            model_path = os.path.join(policy_path, "best_cleanrl_model.pt")


        with open(os.path.join(policy_path, "args_cli.txt"), "r") as f:
            raw_string = f.read()
            if "Namespace" in raw_string:
                saved_args_cli = process_args(raw_string)
            else:
                saved_args_cli = ast.literal_eval(raw_string)
    else:
        policy_path = "./baseline_0dof/"


    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=True
    )


    if args_cli.baseline:
        env_cfg.sim_rate_hz = 100
        env_cfg.policy_rate_hz = 50
        env_cfg.sim.dt = 1/env_cfg.sim_rate_hz
        env_cfg.decimation = env_cfg.sim_rate_hz // env_cfg.policy_rate_hz
        env_cfg.sim.render_interval = env_cfg.decimation

        env_cfg.yaw_distance_reward_scale = 5.0
    else:
        print("\n\nSaved args: ", saved_args_cli)
        print("Keys: ", saved_args_cli.keys())
        env_cfg = update_env_cfg(env_cfg, saved_args_cli)

    env_cfg.eval_mode = True
    env_cfg.viewer.resolution = (1920, 1080)
    

    # If ".hydra/config.yaml" is present, load some of the reward scalars from there
    if os.path.exists(os.path.join(policy_path, ".hydra/config.yaml")):
        with open(os.path.join(policy_path, ".hydra/config.yaml"), "r") as f:
            hydra_cfg = yaml.safe_load(f)
            if "use_yaw_representation" in hydra_cfg["env"]:
                env_cfg.use_yaw_representation = hydra_cfg["env"]["use_yaw_representation"]
            if "yaw_error_reward_scale" in hydra_cfg["env"]:
                env_cfg.yaw_error_reward_scale = hydra_cfg["env"]["yaw_error_reward_scale"]
            if "yaw_distance_reward_scale" in hydra_cfg["env"]:
                env_cfg.yaw_distance_reward_scale = hydra_cfg["env"]["yaw_distance_reward_scale"]
            if "yaw_smooth_transition_scale" in hydra_cfg["env"]:
                env_cfg.yaw_smooth_transition_scale = hydra_cfg["env"]["yaw_smooth_transition_scale"]
            if "yaw_radius" in hydra_cfg["env"]:
                env_cfg.yaw_radius = hydra_cfg["env"]["yaw_radius"]

    env_cfg.yaw_radius = 0.5
    
    if env_cfg.use_yaw_representation:
        env_cfg.num_observations += 1

    if "Traj" in args_cli.task:
        env_cfg.goal_cfg = "fixed"
        env_cfg.trajectory_params["x_amp"] = 1.0
        env_cfg.trajectory_params["x_freq"] = 0.5
        env_cfg.trajectory_params["y_amp"] = 2.0
        env_cfg.trajectory_params["y_freq"] = 1.0
        env_cfg.trajectory_params["z_amp"] = 0.0
        env_cfg.trajectory_params["z_offset"] = 0.5
        env_cfg.trajectory_params["yaw_amp"] = 1.0
        env_cfg.trajectory_params["yaw_freq"] = 1.0
        env_cfg.traj_update_dt = 1.0

    print("\n\nUpdated env cfg: ", env_cfg)

    if args_cli.case_study:
        # Manual override of env cfg
        env_cfg.goal_cfg = "fixed"
        env_cfg.goal_pos = [0.0, 0.0, 0.5]
        env_cfg.goal_ori = [0.7071068, 0.0, 0.0, 0.7071068]
        env_cfg.init_cfg = "default"

        # Camera settings
        env_cfg.viewer.eye = (0.75, 0.75, 1.25)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0

    
    # env_cfg.viewer.eye = (3.0, 1.5, 2.0)
    # env_cfg.viewer.resolution = (1920, 1080)
    # env_cfg.viewer.lookat = (0.0, 1.5, 0.5)
    # env_cfg.viewer.origin_type = "env"
    # env_cfg.viewer.env_index = 0

    envs = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    save_prefix = ""
    if args_cli.case_study:
        save_prefix = "case_study_"
    
    if "Ball" in args_cli.task:
        save_prefix += "ball_catch_"
        save_prefix += "seed_" + str(args_cli.seed) + "_"

    if "Traj" in args_cli.task:
        save_prefix += "eval_traj_track_" + str(int(1/env_cfg.traj_update_dt)) + "Hz_"
        save_prefix += "seed_" + str(args_cli.seed) + "_"

    
    # save_prefix = "ball_catch_side_view_"

   

    video_kwargs = {
        "video_folder": f"{policy_path}",
        "step_trigger": lambda step: step == 0,
        # "episode_trigger": lambda episode: (episode % args.save_interval) == 0,
        "video_length": args_cli.video_length,
        "name_prefix": save_prefix + "eval_video"
    }
    envs = gym.wrappers.RecordVideo(envs, **video_kwargs)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    envs.single_action_space_shape = (np.array(envs.single_action_space.shape[1]).prod(),)
    envs.single_observation_space_shape = (np.array(envs.single_observation_space.shape[1]).prod(),)


    # import code; code.interact(local=locals())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args_cli.baseline:
        vehicle_mass = envs.vehicle_mass
        arm_mass = envs.arm_mass
        inertia =  envs.quad_inertia
        arm_offset = envs.arm_offset
        pos_offset = envs.position_offset
        ori_offset = envs.orientation_offset
        agent = DecoupledController(envs.num_envs, 0, envs.vehicle_mass, envs.arm_mass, envs.quad_inertia, envs.arm_offset, envs.orientation_offset, com_pos_w=None, device=envs.device)
    else:
        agent = Agent(envs).to(device)
        agent.load_state_dict(torch.load(model_path))



    obs_dict, info = envs.reset()
    full_state_size = obs_dict["full_state"].shape[1]
    full_states = torch.zeros((envs.num_envs, 500, full_state_size), dtype=torch.float32).to(device)
    rewards = torch.zeros((envs.num_envs, 500), dtype=torch.float32).to(device)


    steps = 0
    done = False
    done_count = 0
    # input("Press Enter to continue...")
    with torch.no_grad():
        while simulation_app.is_running():
            while done_count < 1:
                obs_tensor = obs_dict["policy"]
                full_states[:, steps, :] = obs_dict["full_state"]

                if args_cli.baseline:
                    action = agent.get_action(obs_dict["full_state"])
                else:
                    action = agent.predict(obs_tensor, deterministic=True)

                obs_dict, reward, terminated, truncated, info = envs.step(action)
                rewards[:, steps] = reward.detach()
                done_count += terminated.sum().item() + truncated.sum().item()

                steps += 1
                print("Step: ", steps)

            torch.save(full_states, os.path.join(policy_path, save_prefix + "eval_full_states.pt"))
            torch.save(rewards, os.path.join(policy_path, save_prefix + "eval_rewards.pt"))

            print("Final Info: \n\n", info, "\n")
            envs.close()
            simulation_app.close()

    


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()