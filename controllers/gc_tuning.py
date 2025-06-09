# Launch Sim window
import argparse
import sys
# from isaacsim import SimulationApp
from omni.isaac.lab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run demo with Isaac Sim")
parser.add_argument("--video", action="store_true", help="Record video")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=100, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-AerialManipulator-0DOF-Debug-Hover-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
AppLauncher.add_app_launcher_args(parser)


# args_cli = parser.parse_args()
args_cli, hydra_args = parser.parse_known_args()


# simulation_app = SimulationApp(vars(args_cli))

# args_cli.headless=False
args_cli.headless=True
# args_cli.enable_cameras=True
args_cli.enable_cameras=False

sys.argv = [sys.argv[0]] + hydra_args
# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config

from omni.isaac.lab_tasks.utils import parse_env_cfg
import gymnasium as gym
import torch
# from envs.hover import hover_env
import envs
import envs.hover
# from AerialManipulation.envs.hover import hover_env

from controllers.decoupled_controller import DecoupledController

import optuna

use_integral_terms = False
use_feed_forward_terms = False


def eval_trial(trial):
    obs_dict, info = env.reset()
    # Get mass from env
    vehicle_mass = env.vehicle_mass # this is pulled from the body "vehicle" in the USD file
    # vehicle_mass = torch.tensor([0.706028], device=env.device)
    arm_mass = env.arm_mass 
    # arm_mass = env.arm_mass - vehicle_mass
    inertia =  env.quad_inertia
    arm_offset = env.arm_offset
    pos_offset = env.position_offset
    ori_offset = env.orientation_offset
    steps = 0
    terminated_count = 0

    pos_kp_gain_xy = trial.suggest_float("pos_kp_gain_xy", 0.0, 50.0)
    pos_kp_gain_z = trial.suggest_float("pos_kp_gain_z", 0.0, 50.0)
    pos_kd_gain_xy = trial.suggest_float("pos_kd_gain_xy", 0.0, 20.0)
    pos_kd_gain_z = trial.suggest_float("pos_kd_gain_z", 0.0, 20.0)
    ori_kp_gain_xy = trial.suggest_float("ori_kp_gain_xy", 0.0, 1000.0)
    ori_kp_gain_z = trial.suggest_float("ori_kp_gain_z", 0.0, 20.0)
    ori_kd_gain_xy = trial.suggest_float("ori_kd_gain_xy", 0.0, 200.0)
    ori_kd_gain_z = trial.suggest_float("ori_kd_gain_z", 0.0, 10.0)

    if use_integral_terms:
        pos_ki_gain_xy = trial.suggest_float("pos_ki_gain_xy", 0.0, 20.0)
        pos_ki_gain_z = trial.suggest_float("pos_ki_gain_z", 0.0, 20.0)
        ori_ki_gain_xy = trial.suggest_float("ori_ki_gain_xy", 0.0, 200.0)
        ori_ki_gain_z = trial.suggest_float("ori_ki_gain_z", 0.0, 10.0)

    rewards = torch.zeros(args_cli.num_envs, device=env.device)

    if "Crazyflie" in args_cli.task:
        vehicle="Crazyflie"
        skip_precompute=True
    else:
        vehicle="AM"
        skip_precompute=False

    if "Traj" in args_cli.task:
        if not use_integral_terms:
            gc = DecoupledController(args_cli.num_envs, 0, vehicle_mass, arm_mass, inertia, arm_offset, ori_offset, print_debug=False, com_pos_w=None, device=env.device,
                                    kp_pos_gain_xy=pos_kp_gain_xy, kp_pos_gain_z=pos_kp_gain_z, kd_pos_gain_xy=pos_kd_gain_xy, kd_pos_gain_z=pos_kd_gain_z,
                                    kp_att_gain_xy=ori_kp_gain_xy, kp_att_gain_z=ori_kp_gain_z, kd_att_gain_xy=ori_kd_gain_xy, kd_att_gain_z=ori_kd_gain_z,
                                    tuning_mode=False, feed_forward=use_feed_forward_terms, vehicle=vehicle, skip_precompute=skip_precompute)
        else:
            gc = DecoupledController(args_cli.num_envs, 0, vehicle_mass, arm_mass, inertia, arm_offset, ori_offset, print_debug=False, com_pos_w=None, device=env.device,
                                    kp_pos_gain_xy=pos_kp_gain_xy, kp_pos_gain_z=pos_kp_gain_z, kd_pos_gain_xy=pos_kd_gain_xy, kd_pos_gain_z=pos_kd_gain_z,
                                    kp_att_gain_xy=ori_kp_gain_xy, kp_att_gain_z=ori_kp_gain_z, kd_att_gain_xy=ori_kd_gain_xy, kd_att_gain_z=ori_kd_gain_z,
                                    ki_pos_gain_xy=pos_ki_gain_xy, ki_pos_gain_z=pos_ki_gain_z, ki_att_gain_xy=ori_ki_gain_xy, ki_att_gain_z=ori_ki_gain_z,
                                    tuning_mode=False, feed_forward=False, use_integral=True, vehicle=vehicle, skip_precompute=skip_precompute)
    else:
        gc = DecoupledController(args_cli.num_envs, 0, vehicle_mass, arm_mass, inertia, arm_offset, ori_offset, print_debug=False, com_pos_w=None, device=env.device,
                                kp_pos_gain_xy=pos_kp_gain_xy, kp_pos_gain_z=pos_kp_gain_z, kd_pos_gain_xy=pos_kd_gain_xy, kd_pos_gain_z=pos_kd_gain_z,
                                kp_att_gain_xy=ori_kp_gain_xy, kp_att_gain_z=ori_kp_gain_z, kd_att_gain_xy=ori_kd_gain_xy, kd_att_gain_z=ori_kd_gain_z,
                                tuning_mode=False, feed_forward=use_feed_forward_terms, vehicle=vehicle, skip_precompute=skip_precompute)

    while steps < 500:
        obs_tensor = obs_dict["policy"]
        # full_state = obs_dict["full_state"]
        # action_gc = gc.get_action(full_state)
        gc_obs = obs_dict["gc"]
        action_gc = gc.get_action(gc_obs)

        action = action_gc.to(obs_tensor.device)

        # # If we want metrics from the state directly intead of the reward we can use these.
        # goal_pos_w = full_state[:, 26:26 + 3]
        # goal_ori_w = full_state[:, 26 + 3:26 + 7]
        # ee_pos = full_state[:, 13:16]
        # ee_ori_quat = full_state[:, 16:20]
        # ee_vel = full_state[:, 20:23]
        # ee_omega = full_state[:, 23:26]
        # quad_pos = full_state[:, :3]
        # quad_ori_quat = full_state[:, 3:7]
        # quad_vel = full_state[:, 7:10]
        # quad_omega = full_state[:, 10:13]

        obs_dict, reward, terminated, truncated, info = env.step(action)

        if use_integral_terms:
            reset_mask = torch.logical_or(terminated, truncated)
            gc.reset_integral_terms(reset_mask)

        rewards += reward
        terminated_count += terminated.sum().item()
        steps += 1
    
    avg_reward = rewards.mean().item()

    return avg_reward

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    # master_env_cfg = parse_env_cfg(args_cli.task, num_envs= args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    # env_cfg = master_env_cfg.copy()

    env_cfg.goal_cfg = "rand" 


    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim_rate_hz = 100
    env_cfg.policy_rate_hz = 50
    env_cfg.sim.dt = 1/env_cfg.sim_rate_hz
    env_cfg.decimation = env_cfg.sim_rate_hz // env_cfg.policy_rate_hz
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.gc_mode = True
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if "Crazyflie" in args_cli.task:
        env_cfg.task_body = "body"
        env_cfg.goal_body = "body"
    else:
        env_cfg.task_body = "COM"
        env_cfg.goal_body = "COM"

    # Reward shaping
    env_cfg.pos_radius = 0.1
    # env_cfg.pos_distance_reward_scale = 0.0
    # env_cfg.pos_error_reward_scale = -2.0
    # env_cfg.yaw_error = -2.0
    # env_cfg.yaw_smooth_transition_scale = 0.0

    print("Args Task: ", args_cli.task)


    global env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = env.unwrapped

    # study_name = "DR ALL ang_vel -0.05 Control Delay 40ms Fixed Position Radius 0.1: " + args_cli.task
    study_name = "CTBR sim2real tau=0.017 Position Radius 0.1: " + args_cli.task

    if use_integral_terms:
        study_name += " Integral Terms"
    
    if use_feed_forward_terms:
        study_name += " Feed Forward Terms"

    study_name += " Hover" if 0.0 in env_cfg.lissajous_amplitudes_rand_ranges else " Trajectory Tracking"
    
    study = optuna.create_study(direction="maximize",
                                study_name=study_name, storage="sqlite:///database_gc_tuning.sqlite3", load_if_exists=True,
    )
    study.optimize(eval_trial, n_trials=250)
    
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    print("Summary: ")
    print(study.trials_dataframe())
    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)



    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

    simulation_app.close()