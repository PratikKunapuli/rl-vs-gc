import torch
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import pandas as pd
import os
import yaml

from matplotlib import rc
rc('font', size=10)
rc('legend', fontsize=10)
rc('ytick', labelsize=8)
rc('xtick', labelsize=8)
sns.set_context("paper")
sns.set_theme()

import omni.isaac.lab.utils.math as isaac_math_utils

import math_utilities as math_utils

def plot_data(rl_eval_path, dc_eval_path=None, save_prefix=""):
    """
    Make plots for the evaluation.

    full_states: torch.Tensor (num_envs, 500, full_state_size) [33]
    policy_path: str, path to the policy for saving the plots
    """

    """Plot the errors from the RL policy."""
    # Load the RL policy errors
    rl_state_path = os.path.join(rl_eval_path, save_prefix +  "eval_full_states.pt")
    rl_rewards_path = os.path.join(rl_eval_path, save_prefix +  "eval_rewards.pt")
    rl_full_states = torch.load(rl_state_path)
    rl_rewards = torch.load(rl_rewards_path)
    rl_save_path = rl_eval_path

    if dc_eval_path is not None:
        dc_state_path = os.path.join(dc_eval_path, save_prefix + "eval_full_states.pt")
        dc_rewards_path = os.path.join(dc_eval_path, save_prefix + "eval_rewards.pt")
        dc_full_states = torch.load(dc_state_path)
        dc_rewards = torch.load(dc_rewards_path)
        dc_save_path = dc_eval_path
    else:
        dc_full_states = torch.rand(rl_full_states.shape).to(rl_full_states.device)
        dc_rewards = torch.rand(rl_rewards.shape).to(rl_rewards.device)
        dc_save_path = rl_save_path
    
    goal_pos_w = rl_full_states[:, :-1, -7:-4]
    goal_ori_w = rl_full_states[:, :-1, -4:]
    ee_pos_w = rl_full_states[:, :-1, 13:16]
    ee_ori_w = rl_full_states[:, :-1, 16:20]
    
    dc_goal_pos_w = dc_full_states[:, :-1, -7:-4]
    dc_goal_ori_w = dc_full_states[:, :-1, -4:]
    dc_ee_pos_w = dc_full_states[:, :-1, 13:16]
    dc_ee_ori_w = dc_full_states[:, :-1, 16:20]

    

    pos_error = torch.norm(goal_pos_w - ee_pos_w, dim=-1).cpu()
    print("Individual converged error for RL: ", pos_error[:, -1])

    pos_error_df = pd.DataFrame(pos_error.numpy().T, columns=[f"env_{i}" for i in range(pos_error.shape[0])])
    pos_error_df = pos_error_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Position Error')
    pos_error_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    pos_error_df['Timesteps'] = pos_error_df['Timesteps'] * 0.02



    mean_pos_error = pos_error.mean(dim=0).numpy()
    std_pos_error = pos_error.std(dim=0).numpy()
    time_axis = torch.arange(mean_pos_error.shape[0]) * 0.02


    print("Mean converged error for RL: ", mean_pos_error[-1])
    print("Std converged error for RL: ", std_pos_error[-1])

    

    dc_pos_error = torch.norm(dc_goal_pos_w - dc_ee_pos_w, dim=-1).cpu()
    dc_mean_pos_error = dc_pos_error.mean(dim=0).numpy()
    dc_std_pos_error = dc_pos_error.std(dim=0).numpy()

    dc_pos_error_df = pd.DataFrame(dc_pos_error.numpy().T, columns=[f"env_{i}" for i in range(dc_pos_error.shape[0])])
    dc_pos_error_df = dc_pos_error_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Position Error')
    dc_pos_error_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    dc_pos_error_df['Timesteps'] = dc_pos_error_df['Timesteps'] * 0.02

    print("Mean converged error for Decoupled: ", dc_mean_pos_error[-1])
    print("Std converged error for Decoupled: ", dc_std_pos_error[-1])

    sns.set_theme()
    sns.set_context("paper")


    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x=time_axis, y=mean_pos_error, label="RL Controller", errorbar='sd')
    plt.fill_between(time_axis, mean_pos_error - std_pos_error, mean_pos_error + std_pos_error, alpha=0.2)
    plt.plot([0, time_axis[-1]], [0.05, 0.05], 'r--', label="Threshold")
    sns.lineplot(x=time_axis, y=dc_mean_pos_error, label="Decoupled Controller")
    plt.fill_between(time_axis, dc_mean_pos_error - dc_std_pos_error, dc_mean_pos_error + dc_std_pos_error, alpha=0.2)
    plt.title("End Effector Position Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.savefig(os.path.join(rl_save_path, save_prefix + "mean_pos_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_save_path, save_prefix + "mean_pos_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x='Timesteps', y='Position Error', hue='Environment', data=pos_error_df, legend=True)
    plt.title("End Effector Position Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.savefig(os.path.join(rl_save_path, save_prefix + "pos_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')


    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x='Timesteps', y='Position Error', hue='Environment', data=dc_pos_error_df, legend=True)
    plt.title("End Effector Position Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.savefig(os.path.join(dc_save_path, save_prefix + "pos_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')
    

    # We want to use the isaac_math_utils to convert the quaternion to a pure yaw quaternion
    goal_ori_yaw_quat = isaac_math_utils.yaw_quat(goal_ori_w)
    ee_ori_yaw_quat = isaac_math_utils.yaw_quat(ee_ori_w)

    dc_goal_ori_yaw_quat = isaac_math_utils.yaw_quat(dc_goal_ori_w)
    dc_ee_ori_yaw_quat = isaac_math_utils.yaw_quat(dc_ee_ori_w)

    # yaw_error = (isaac_math_utils.wrap_to_pi(math_utils.yaw_from_quat(goal_ori_yaw_quat)) - isaac_math_utils.wrap_to_pi(math_utils.yaw_from_quat(ee_ori_yaw_quat))).cpu()
    # yaw_error = isaac_math_utils.quat_error_magnitude(goal_ori_yaw_quat, ee_ori_yaw_quat).cpu()
    # dc_yaw_error = isaac_math_utils.quat_error_magnitude(dc_goal_ori_yaw_quat, dc_ee_ori_yaw_quat).cpu()

    yaw_error = math_utils.yaw_error_from_quats(goal_ori_w, ee_ori_w, 0).cpu()
    dc_yaw_error = math_utils.yaw_error_from_quats(dc_goal_ori_w, dc_ee_ori_w, 0).cpu()

    yaw_error_df = pd.DataFrame(yaw_error.numpy().T, columns=[f"env_{i}" for i in range(yaw_error.shape[0])])
    yaw_error_df = yaw_error_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Yaw Error')
    yaw_error_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    yaw_error_df['Timesteps'] = yaw_error_df['Timesteps'] * 0.02

    dc_yaw_error_df = pd.DataFrame(dc_yaw_error.numpy().T, columns=[f"env_{i}" for i in range(dc_yaw_error.shape[0])])
    dc_yaw_error_df = dc_yaw_error_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Yaw Error')
    dc_yaw_error_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    dc_yaw_error_df['Timesteps'] = dc_yaw_error_df['Timesteps'] * 0.02


    mean_yaw_error = yaw_error.mean(dim=0).numpy()
    std_yaw_error = yaw_error.std(dim=0).numpy()

    dc_mean_yaw_error = dc_yaw_error.mean(dim=0).numpy()
    dc_std_yaw_error = dc_yaw_error.std(dim=0).numpy()

    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x=time_axis, y=mean_yaw_error, label="RL Controller", errorbar='sd')
    plt.fill_between(time_axis, mean_yaw_error - std_yaw_error, mean_yaw_error + std_yaw_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=dc_mean_yaw_error, label="Decoupled Controller")
    plt.fill_between(time_axis, dc_mean_yaw_error - dc_std_yaw_error, dc_mean_yaw_error + dc_std_yaw_error, alpha=0.2)
    plt.title("End Effector Yaw Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.legend()
    plt.savefig(os.path.join(rl_save_path, save_prefix + "mean_yaw_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_save_path, save_prefix + "mean_yaw_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x='Timesteps', y='Yaw Error', hue='Environment', data=yaw_error_df, legend=True)
    plt.title("End Effector Yaw Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.savefig(os.path.join(rl_save_path, save_prefix + "yaw_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x='Timesteps', y='Yaw Error', hue='Environment', data=dc_yaw_error_df, legend=True)
    plt.title("End Effector Yaw Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.savefig(os.path.join(dc_save_path, save_prefix + "yaw_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')

    rewards = rl_rewards[:, :-1].cpu()
    dc_rewards = dc_rewards[:, :-1].cpu()

    # Look at hydra config for the reward scales
    if os.path.exists(os.path.join(rl_eval_path, ".hydra/config.yaml")):
        with open(os.path.join(rl_eval_path, ".hydra/config.yaml"), "r") as f:
            hydra_cfg = yaml.safe_load(f)
            max_reward = 0.0
            for key in hydra_cfg["env"]:
                if "distance_reward_scale" in key:
                    max_reward += hydra_cfg["env"][key]
    elif os.path.exists(os.path.join(rl_eval_path, "params/env.yaml")):
        with open(os.path.join(rl_eval_path, "params/env.yaml"), "r") as f:
            hydra_cfg = yaml.load(f, yaml.FullLoader)
            max_reward = 0.0
            for key in hydra_cfg:
                if "distance_reward_scale" in key:
                    max_reward += hydra_cfg[key]
    else:
        max_reward = 15.0

    # # max_reward = 15.0
    # max_reward = 20.0
    min_reward = 0.0
    # normalize the rewards
    rewards = (rewards - min_reward) / (max_reward - min_reward)
    dc_rewards = (dc_rewards - min_reward) / (max_reward - min_reward)

    rewards_df = pd.DataFrame(rewards.numpy().T, columns=[f"env_{i}" for i in range(rewards.shape[0])])
    rewards_df = rewards_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Reward')
    rewards_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    rewards_df['Timesteps'] = rewards_df['Timesteps'] * 0.02

    dc_rewards_df = pd.DataFrame(dc_rewards.numpy().T, columns=[f"env_{i}" for i in range(dc_rewards.shape[0])])
    dc_rewards_df = dc_rewards_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Reward')
    dc_rewards_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    dc_rewards_df['Timesteps'] = dc_rewards_df['Timesteps'] * 0.02

    mean_rewards = rewards.mean(dim=0).numpy()
    std_rewards = rewards.std(dim=0).numpy()
    time_axis = torch.arange(mean_rewards.shape[0]) * 0.02

    dc_mean_rewards = dc_rewards.mean(dim=0).numpy()
    dc_std_rewards = dc_rewards.std(dim=0).numpy()

    sns.set_theme()
    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x=time_axis, y=mean_rewards, label="RL Controller", errorbar='sd')
    plt.fill_between(time_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    sns.lineplot(x=time_axis, y=dc_mean_rewards, label="Decoupled Controller")
    plt.fill_between(time_axis, dc_mean_rewards - dc_std_rewards, dc_mean_rewards + dc_std_rewards, alpha=0.2)
    plt.title("Rewards")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Reward")
    plt.legend()
    plt.savefig(os.path.join(rl_save_path, save_prefix + "mean_rewards.pdf"), dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_save_path, save_prefix + "mean_rewards.pdf"), dpi=300, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x='Timesteps', y='Reward', hue='Environment', data=rewards_df, legend=True)
    plt.title("Rewards")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Reward")
    plt.savefig(os.path.join(rl_save_path, save_prefix + "rewards.pdf"), dpi=300, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x='Timesteps', y='Reward', hue='Environment', data=dc_rewards_df, legend=True)
    plt.title("Rewards")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Reward")
    plt.savefig(os.path.join(dc_save_path, save_prefix + "rewards.pdf"), dpi=300, format='pdf', bbox_inches='tight')


    # Make one plot that is 1x3 of the mean pos error, mean yaw error, and mean reward, and keep the legend outside the plot at the bottom below all the plots
    fig, axs = plt.subplots(1, 3, figsize=(9.75, 2.5), dpi=300)
    sns.lineplot(x=time_axis, y=mean_pos_error, label="RL Controller", errorbar='sd', ax=axs[0], legend=False)
    axs[0].fill_between(time_axis, mean_pos_error - std_pos_error, mean_pos_error + std_pos_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=dc_mean_pos_error, label="Decoupled Controller", ax=axs[0], legend=False)
    axs[0].fill_between(time_axis, dc_mean_pos_error - dc_std_pos_error, dc_mean_pos_error + dc_std_pos_error, alpha=0.2)
    axs[0].set_title("End Effector Position Error")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Error (m)")
    if "case_study" in save_prefix:
        axs[0].set_ylim(-0.02, 0.3)
    else:
        axs[0].set_ylim(-0.1, 2.0)
    axs[0].set_xlim(0, 5.0)
    
    
    sns.lineplot(x=time_axis, y=mean_yaw_error, label="RL Controller", errorbar='sd', ax=axs[1], legend=False)
    axs[1].fill_between(time_axis, mean_yaw_error - std_yaw_error, mean_yaw_error + std_yaw_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=dc_mean_yaw_error, label="Decoupled Controller", ax=axs[1], legend=False)
    axs[1].fill_between(time_axis, dc_mean_yaw_error - dc_std_yaw_error, dc_mean_yaw_error + dc_std_yaw_error, alpha=0.2)
    axs[1].set_title("End Effector Yaw Error")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Error (rad)")
    axs[1].set_ylim(-0.1, 3.141)
    axs[1].set_xlim(0, 5.0)
    
    
    sns.lineplot(x=time_axis, y=mean_rewards, label="RL Controller", errorbar='sd', ax=axs[2], legend=False)
    axs[2].fill_between(time_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    sns.lineplot(x=time_axis, y=dc_mean_rewards, label="Decoupled Controller", ax=axs[2], legend=False)
    axs[2].fill_between(time_axis, dc_mean_rewards - dc_std_rewards, dc_mean_rewards + dc_std_rewards, alpha=0.2)
    axs[2].set_title("Rewards")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Normalized Reward")
    if "case_study" in save_prefix:
        axs[2].set_ylim(0.7, 1.01)
    else:
        axs[2].set_ylim(-0.1, 1.1)
    axs[2].set_xlim(0, 5.0)
    
    plt.tight_layout()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))

    plt.savefig(os.path.join(rl_save_path, save_prefix + "mean_combined_plots.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_save_path, save_prefix + "mean_combined_plots.pdf"), dpi=1000, format='pdf', bbox_inches='tight')

    if "case_study" in save_prefix:
        # combined plot of just position and yaw error
        fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.5), dpi=1000)
        sns.lineplot(x=time_axis, y=mean_pos_error, label="RL Controller", errorbar='sd', ax=axs[0], legend=False)
        axs[0].fill_between(time_axis, mean_pos_error - std_pos_error, mean_pos_error + std_pos_error, alpha=0.2)
        sns.lineplot(x=time_axis, y=dc_mean_pos_error, label="Decoupled Controller", ax=axs[0], legend=False)
        axs[0].fill_between(time_axis, dc_mean_pos_error - dc_std_pos_error, dc_mean_pos_error + dc_std_pos_error, alpha=0.2)
        # axs[0].set_title("End Effector Position Error")
        axs[0].set_xlabel("Time (s)", size=8)
        axs[0].set_ylabel("Pos Error (m)", size=8)
        axs[0].set_ylim(-0.02, 0.2)

        sns.lineplot(x=time_axis, y=mean_yaw_error, label="RL Controller", errorbar='sd', ax=axs[1], legend=False)
        axs[1].fill_between(time_axis, mean_yaw_error - std_yaw_error, mean_yaw_error + std_yaw_error, alpha=0.2)
        sns.lineplot(x=time_axis, y=dc_mean_yaw_error, label="Decoupled Controller", ax=axs[1], legend=False)
        axs[1].fill_between(time_axis, dc_mean_yaw_error - dc_std_yaw_error, dc_mean_yaw_error + dc_std_yaw_error, alpha=0.2)
        # axs[1].set_title("End Effector Yaw Error")
        axs[1].set_xlabel("Time (s)", size=8)
        axs[1].set_ylabel("Yaw Error (rad)", size=8)
        axs[1].set_ylim(-0.1, 2)
        fig.suptitle("Initial Yaw Offset")
        # plt.tight_layout()

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))

        plt.savefig(os.path.join(rl_save_path, save_prefix + "mean_combined_plots_no_reward.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(dc_save_path, save_prefix + "mean_combined_plots_no_reward.pdf"), dpi=1000, format='pdf', bbox_inches='tight')


def get_data_from_tensor(data):
    goal_pos_w = data[:, :-1, -7:-4]
    goal_ori_w = data[:, :-1, -4:]
    ee_pos_w = data[:, :-1, 13:16]
    ee_ori_w = data[:, :-1, 16:20]

    return goal_pos_w, goal_ori_w, ee_pos_w, ee_ori_w

def plot_traj_tracking(rl_eval_path, dc_eval_path, prefix="", seed="0"):

    sns.set_theme()
    sns.set_context("paper")
    print(sns.plotting_context())
    

    rl_data_path = os.path.join(rl_eval_path, prefix + "_seed_" + seed + "_eval_full_states.pt")
    dc_data_path = os.path.join(dc_eval_path, prefix + "_seed_" + seed + "_eval_full_states.pt")
    rl_data = torch.load(rl_data_path)
    dc_data = torch.load(dc_data_path)

    rl_goal_pos_w, rl_goal_ori_w, rl_ee_pos_w, rl_ee_ori_w = get_data_from_tensor(rl_data)
    dc_goal_pos_w, dc_goal_ori_w, dc_ee_pos_w, dc_ee_ori_w = get_data_from_tensor(dc_data)

    rl_pos_error = torch.norm(rl_goal_pos_w - rl_ee_pos_w, dim=-1).cpu()
    dc_pos_error = torch.norm(dc_goal_pos_w - dc_ee_pos_w, dim=-1).cpu()

    rl_yaw_error = math_utils.yaw_error_from_quats(rl_goal_ori_w, rl_ee_ori_w, 0).cpu()
    dc_yaw_error = math_utils.yaw_error_from_quats(dc_goal_ori_w, dc_ee_ori_w, 0).cpu()

    rl_pos_error_df = pd.DataFrame(rl_pos_error.numpy().T, columns=[f"env_{i}" for i in range(rl_pos_error.shape[0])])
    rl_pos_error_df = rl_pos_error_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Position Error')
    rl_pos_error_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    rl_pos_error_df['Timesteps'] = rl_pos_error_df['Timesteps'] * 0.02

    dc_pos_error_df = pd.DataFrame(dc_pos_error.numpy().T, columns=[f"env_{i}" for i in range(dc_pos_error.shape[0])])
    dc_pos_error_df = dc_pos_error_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Position Error')
    dc_pos_error_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    dc_pos_error_df['Timesteps'] = dc_pos_error_df['Timesteps'] * 0.02

    rl_yaw_error_df = pd.DataFrame(rl_yaw_error.numpy().T, columns=[f"env_{i}" for i in range(rl_yaw_error.shape[0])])
    rl_yaw_error_df = rl_yaw_error_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Yaw Error')
    rl_yaw_error_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    rl_yaw_error_df['Timesteps'] = rl_yaw_error_df['Timesteps'] * 0.02

    dc_yaw_error_df = pd.DataFrame(dc_yaw_error.numpy().T, columns=[f"env_{i}" for i in range(dc_yaw_error.shape[0])])
    dc_yaw_error_df = dc_yaw_error_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Yaw Error')
    dc_yaw_error_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    dc_yaw_error_df['Timesteps'] = dc_yaw_error_df['Timesteps'] * 0.02


    mean_rl_pos_error = rl_pos_error.mean(dim=0).numpy()
    std_rl_pos_error = rl_pos_error.std(dim=0).numpy()
    time_axis = torch.arange(mean_rl_pos_error.shape[0]) * 0.02

    
    mean_dc_pos_error = dc_pos_error.mean(dim=0).numpy()
    std_dc_pos_error = dc_pos_error.std(dim=0).numpy()

    mean_rl_yaw_error = rl_yaw_error.mean(dim=0).numpy()
    std_rl_yaw_error = rl_yaw_error.std(dim=0).numpy()

    mean_dc_yaw_error = dc_yaw_error.mean(dim=0).numpy()
    std_dc_yaw_error = dc_yaw_error.std(dim=0).numpy()

    RL_RMSE_pos = np.sqrt(np.mean(mean_rl_pos_error**2))
    DC_RMSE_pos = np.sqrt(np.mean(mean_dc_pos_error**2))
    RL_RMSE_yaw = np.sqrt(np.mean(mean_rl_yaw_error**2))
    DC_RMSE_yaw = np.sqrt(np.mean(mean_dc_yaw_error**2))

    print("RL RMSE Pos: ", RL_RMSE_pos)
    print("DC RMSE Pos: ", DC_RMSE_pos)
    print("RL RMSE Yaw: ", RL_RMSE_yaw)
    print("DC RMSE Yaw: ", DC_RMSE_yaw)

    
    plt.figure(figsize=(3.25, 2.5), dpi=1000)
    sns.lineplot(x=time_axis, y=mean_rl_pos_error, label="RL Controller", errorbar='sd')
    plt.fill_between(time_axis, mean_rl_pos_error - std_rl_pos_error, mean_rl_pos_error + std_rl_pos_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=mean_dc_pos_error, label="Decoupled Controller")
    plt.fill_between(time_axis, mean_dc_pos_error - std_dc_pos_error, mean_dc_pos_error + std_dc_pos_error, alpha=0.2)
    plt.title("End Effector Position Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.savefig(os.path.join(rl_eval_path, prefix + "_mean_pos_error.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_eval_path, prefix + "_mean_pos_error.pdf"), dpi=1000, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(3.25, 2.5), dpi=1000)
    sns.lineplot(x=time_axis, y=mean_rl_yaw_error, label="RL Controller", errorbar='sd')
    plt.fill_between(time_axis, mean_rl_yaw_error - std_rl_yaw_error, mean_rl_yaw_error + std_rl_yaw_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=mean_dc_yaw_error, label="Decoupled Controller")
    plt.fill_between(time_axis, mean_dc_yaw_error - std_dc_yaw_error, mean_dc_yaw_error + std_dc_yaw_error, alpha=0.2)
    plt.title("End Effector Yaw Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.legend()
    plt.savefig(os.path.join(rl_eval_path, prefix + "_mean_yaw_error.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_eval_path, prefix + "_mean_yaw_error.pdf"), dpi=1000, format='pdf', bbox_inches='tight')

    # combined plot of just position and yaw error
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.5), dpi=1000)
    sns.lineplot(x=time_axis, y=mean_rl_pos_error, label="RL Controller", errorbar='sd', ax=axs[0], legend=False)
    axs[0].fill_between(time_axis, mean_rl_pos_error - std_rl_pos_error, mean_rl_pos_error + std_rl_pos_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=mean_dc_pos_error, label="Decoupled Controller", ax=axs[0], legend=False)
    axs[0].fill_between(time_axis, mean_dc_pos_error - std_dc_pos_error, mean_dc_pos_error + std_dc_pos_error, alpha=0.2)
    # axs[0].set_title("End Effector Position Error")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Pos Error (m)")
    # axs[0].set_ylim(-0.02, 0.2)

    sns.lineplot(x=time_axis, y=mean_rl_yaw_error, label="RL Controller", errorbar='sd', ax=axs[1], legend=False)
    axs[1].fill_between(time_axis, mean_rl_yaw_error - std_rl_yaw_error, mean_rl_yaw_error + std_rl_yaw_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=mean_dc_yaw_error, label="Decoupled Controller", ax=axs[1], legend=False)
    axs[1].fill_between(time_axis, mean_dc_yaw_error - std_dc_yaw_error, mean_dc_yaw_error + std_dc_yaw_error, alpha=0.2)
    # axs[1].set_title("End Effector Yaw Error")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Yaw Error (rad)")
    # axs[1].set_ylim(-0.1, 2)
    fig.suptitle("Lissajous Tracking at 50Hz")
    plt.tight_layout()

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))

    plt.savefig(os.path.join(rl_eval_path, prefix + "_mean_combined_plots.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_eval_path, prefix + "_mean_combined_plots.pdf"), dpi=1000, format='pdf', bbox_inches='tight')    


def plot_ball_catching(rl_eval_path, dc_eval_path):
    sns.set_theme()
    sns.set_context("paper")
    print(sns.plotting_context())
    

    rl_rewards_path = os.path.join(rl_eval_path, "ball_catch_eval_rewards.pt")
    dc_rewards_path = os.path.join(dc_eval_path, "ball_catch_eval_rewards.pt")
    rl_rewards = torch.load(rl_rewards_path)
    dc_rewards = torch.load(dc_rewards_path)

    balls_caught_per_env = rl_rewards.sum(dim=1)
    baseline_mean_balls_caught = dc_rewards.sum(dim=1).mean()

    print("Balls caught per env: ", balls_caught_per_env)
    print("Mean balls caught: ", balls_caught_per_env.mean())
    print("Baseline mean balls caught: ", baseline_mean_balls_caught)


def plot_paper_figs(rl_eval_path_list, dc_eval_path):
    sns.set_theme({'legend.frameon': False})
    sns.set_context("paper")
    
    case_study_data_path = "case_study_eval_full_states.pt"
    generalized_data_path = "eval_full_states.pt"
    generalized_rewards_path = "eval_rewards.pt"
    ball_catch_rewards_path = "ball_catch_eval_rewards.pt"
    traj_rate_list = ["0", "1", "5" ,"10", "25", "50"]

    N = 5
    M = 100
    T = 500

    # Load RL Full Data
    generalized_data = torch.zeros(N, M, T, 33)
    generalized_rewards = torch.zeros(N, M, T)
    case_study_data = torch.zeros(N, M, T, 33)
    ball_catching_rewards = torch.zeros(N, M, T)

    for i in range(len(rl_eval_path_list)):
        rl_eval_path = rl_eval_path_list[i]
        single_trial_generalized = torch.load(os.path.join(rl_eval_path, generalized_data_path))
        single_trial_case_study = torch.load(os.path.join(rl_eval_path, case_study_data_path))
        single_trial_rewards = torch.load(os.path.join(rl_eval_path, generalized_rewards_path))
        single_trial_ball_catch_rewards = torch.load(os.path.join(rl_eval_path, ball_catch_rewards_path))

        # print("rewards shape: ", single_trial_rewards.shape)
        # print("ball catching rewards shape: ", single_trial_ball_catch_rewards.shape)

        generalized_data[i] = single_trial_generalized
        case_study_data[i] = single_trial_case_study
        generalized_rewards[i] = single_trial_rewards
        ball_catching_rewards[i] = single_trial_ball_catch_rewards

    dc_full_data = torch.load(os.path.join(dc_eval_path, generalized_data_path))
    dc_rewards = torch.load(os.path.join(dc_eval_path, generalized_rewards_path))
    dc_case_study_data = torch.load(os.path.join(dc_eval_path, case_study_data_path))
    dc_ball_catch_rewards = torch.load(os.path.join(dc_eval_path, ball_catch_rewards_path))


    generalized_rl_pos_norm = torch.norm(generalized_data[:, :, :-1, -7:-4] - generalized_data[:, :, :-1, 13:16], dim=-1)
    generalized_rl_yaw_error = math_utils.yaw_error_from_quats(generalized_data[:, :, :-1, -4:], generalized_data[:, :, :-1, 16:20], 0)
    case_study_rl_pos_norm = torch.norm(case_study_data[:, :, :-1, -7:-4] - case_study_data[:, :, :-1, 13:16], dim=-1)
    case_study_rl_yaw_error = math_utils.yaw_error_from_quats(case_study_data[:, :, :-1, -4:], case_study_data[:, :, :-1, 16:20], 0)
    generalized_dc_pos_norm = torch.norm(dc_full_data[:, :-1, -7:-4] - dc_full_data[:, :-1, 13:16], dim=-1)
    generalized_dc_yaw_error = math_utils.yaw_error_from_quats(dc_full_data[:, :-1, -4:], dc_full_data[:, :-1, 16:20], 0)
    case_study_dc_pos_norm = torch.norm(dc_case_study_data[:, :-1, -7:-4] - dc_case_study_data[:, :-1, 13:16], dim=-1)
    case_study_dc_yaw_error = math_utils.yaw_error_from_quats(dc_case_study_data[:, :-1, -4:], dc_case_study_data[:, :-1, 16:20], 0)


    max_reward = 15.0
    min_reward = torch.min(torch.min(generalized_rewards[:,:,:-1]), torch.min(dc_rewards[:,:-1])).cpu().item()
    generalized_rewards = (generalized_rewards[:,:,:-1] - min_reward) / (max_reward - min_reward)
    dc_rewards = (dc_rewards[:,:-1] - min_reward) / (max_reward - min_reward)

    generalized_mean_pos_error = generalized_rl_pos_norm.mean(dim=1).mean(dim=0).cpu().numpy()
    generalized_std_pos_error = generalized_rl_pos_norm.mean(dim=1).std(dim=0).cpu().numpy()
    generalized_mean_yaw_error = generalized_rl_yaw_error.mean(dim=1).mean(dim=0).cpu().numpy()
    generalized_std_yaw_error = generalized_rl_yaw_error.mean(dim=1).std(dim=0).cpu().numpy()
    generalized_mean_rewards = generalized_rewards.mean(dim=1).mean(dim=0).cpu().numpy()
    generalized_std_rewards = generalized_rewards.mean(dim=1).std(dim=0).cpu().numpy()
    rl_mean_balls_caught = ball_catching_rewards.sum(dim=-1).mean(dim=1).cpu().numpy()
    rl_std_balls_caught = ball_catching_rewards.sum(dim=-1).mean(dim=1).cpu().numpy()
    case_study_mean_pos_error = case_study_rl_pos_norm.mean(dim=1).mean(dim=0).cpu().numpy()
    case_study_std_pos_error = case_study_rl_pos_norm.mean(dim=1).std(dim=0).cpu().numpy()
    case_study_mean_yaw_error = case_study_rl_yaw_error.mean(dim=1).mean(dim=0).cpu().numpy()
    case_study_std_yaw_error = case_study_rl_yaw_error.mean(dim=1).std(dim=0).cpu().numpy()
    case_study_mean_rewards = ball_catching_rewards.mean(dim=1).mean(dim=0).cpu().numpy()
    case_study_std_rewards = ball_catching_rewards.mean(dim=1).std(dim=0).cpu().numpy()
    generalized_dc_mean_pos_error = generalized_dc_pos_norm.mean(dim=0).cpu().numpy()
    generalized_dc_std_pos_error = generalized_dc_pos_norm.std(dim=0).cpu().numpy()
    generalized_dc_mean_yaw_error = generalized_dc_yaw_error.mean(dim=0).cpu().numpy()
    generalized_dc_std_yaw_error = generalized_dc_yaw_error.std(dim=0).cpu().numpy()
    generalized_dc_mean_rewards = dc_rewards.mean(dim=0).cpu().numpy()
    generalized_dc_std_rewards = dc_rewards.std(dim=0).cpu().numpy()
    dc_mean_balls_caught = dc_ball_catch_rewards.sum(dim=-1).mean().cpu().numpy()
    dc_std_balls_caught = dc_ball_catch_rewards.sum(dim=-1).std().cpu().numpy()
    case_study_dc_mean_pos_error = case_study_dc_pos_norm.mean(dim=0).cpu().numpy()
    case_study_dc_std_pos_error = case_study_dc_pos_norm.std(dim=0).cpu().numpy()
    case_study_dc_mean_yaw_error = case_study_dc_yaw_error.mean(dim=0).cpu().numpy()
    case_study_dc_std_yaw_error = case_study_dc_yaw_error.std(dim=0).cpu().numpy()
    case_study_dc_mean_rewards = dc_case_study_data.mean(dim=0).cpu().numpy()
    case_study_dc_std_rewards = dc_case_study_data.std(dim=0).cpu().numpy()


    time_axis = torch.arange(generalized_mean_pos_error.shape[0]) * 0.02

    print("RL: ")
    print("Final converged error general mean: ", generalized_mean_pos_error[-1])
    print("Final converged error general std: ", generalized_std_pos_error[-1])
    print("Final converged error case study mean: ", case_study_mean_pos_error[-1])
    print("Final converged error case study yaw: ", case_study_mean_yaw_error[-1])
    # print("Final converged error case study std: ", case_study_std_pos_error[-1])
    print("Final converged error yaw mean: ", generalized_mean_yaw_error[-1])
    print("Final converged error yaw std: ", generalized_std_yaw_error[-1])

    print("\n\nDC: ")
    print("Final converged error general mean: ", generalized_dc_mean_pos_error[-1])
    print("Final converged error general std: ", generalized_dc_std_pos_error[-1])
    print("Final converged error case study mean: ", case_study_dc_mean_pos_error[-1])
    print("Final converged error case study yaw: ", case_study_dc_mean_yaw_error[-1])
    print("Final converged error yaw mean: ", generalized_dc_mean_yaw_error[-1])
    print("Final converged error yaw std: ", generalized_dc_std_yaw_error[-1])



    # Make one plot that is 1x3 of the mean pos error, mean yaw error, and mean reward, and keep the legend outside the plot at the bottom below all the plots
    fig, axs = plt.subplots(1, 3, figsize=(9.75, 2.5), dpi=1000)
    sns.lineplot(x=time_axis, y=generalized_mean_pos_error, label="RL Controller", errorbar='sd', ax=axs[0], legend=False)
    axs[0].fill_between(time_axis, generalized_mean_pos_error - generalized_std_pos_error, generalized_mean_pos_error + generalized_std_pos_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=generalized_dc_mean_pos_error, label="Decoupled Controller", ax=axs[0], legend=False)
    # axs[0].fill_between(time_axis, generalized_dc_mean_pos_error - generalized_dc_std_pos_error, generalized_dc_mean_pos_error + generalized_dc_std_pos_error, alpha=0.2)
    # axs[0].set_title("End Effector Position Error")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position Error (m)")
    axs[0].set_ylim(-0.1, 2.0)
    # axs[0].set_xlim(0, 5.0)

    # sub region of the original image
    axins = inset_axes(axs[0],1,1, loc=2,bbox_to_anchor=(0.21, 0.8),bbox_transform=axs[0].figure.transFigure)
    x1, x2, y1, y2 = 6, 10, -0.01, 0.04
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_yticks([0, 0.02, 0.04])
    axins.tick_params(axis='y', labelsize=8, pad=1)
    axins.set_xticks([])
    sns.lineplot(x=time_axis, y=generalized_mean_pos_error, label="RL Controller", errorbar='sd', ax=axins, legend=False)
    axins.fill_between(time_axis, generalized_mean_pos_error - generalized_std_pos_error, generalized_mean_pos_error + generalized_std_pos_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=generalized_dc_mean_pos_error, label="Decoupled Controller", ax=axins, legend=False)
    axins.set(xlabel=None, ylabel=None)
    mark_inset(axs[0], axins, loc1=3, loc2=4, fc="none", ec="0.5")

    sns.lineplot(x=time_axis, y=generalized_mean_yaw_error, label="RL Controller", errorbar='sd', ax=axs[1], legend=False)
    axs[1].fill_between(time_axis, generalized_mean_yaw_error - generalized_std_yaw_error, generalized_mean_yaw_error + generalized_std_yaw_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=generalized_dc_mean_yaw_error, label="Decoupled Controller", ax=axs[1], legend=False)
    # axs[1].fill_between(time_axis, generalized_dc_mean_yaw_error - generalized_dc_std_yaw_error, generalized_dc_mean_yaw_error + generalized_dc_std_yaw_error, alpha=0.2)
    # axs[1].set_title("End Effector Yaw Error")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Yaw Error (rad)")
    axs[1].set_ylim(-0.1, 3.141)
    # axs[1].set_xlim(0, 5.0)

    axins = inset_axes(axs[1],1,1, loc=2,bbox_to_anchor=(0.537, 0.8),bbox_transform=axs[1].figure.transFigure)
    x1, x2, y1, y2 = 6, 10, -0.001, 0.016
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_yticks([0, 0.008, 0.016])
    axins.tick_params(axis='y', labelsize=8, pad=1)
    axins.set_xticks([])
    sns.lineplot(x=time_axis, y=generalized_mean_yaw_error, label="RL Controller", errorbar='sd', ax=axins, legend=False)
    axins.fill_between(time_axis, generalized_mean_yaw_error - generalized_std_yaw_error, generalized_mean_yaw_error + generalized_std_yaw_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=generalized_dc_mean_pos_error, label="Decoupled Controller", ax=axins, legend=False)
    axins.set(xlabel=None, ylabel=None)
    mark_inset(axs[1], axins, loc1=3, loc2=4, fc="none", ec="0.5")

    sns.lineplot(x=time_axis, y=generalized_mean_rewards, label="RL Controller", errorbar='sd', ax=axs[2], legend=False)
    axs[2].fill_between(time_axis, generalized_mean_rewards - generalized_std_rewards, generalized_mean_rewards + generalized_std_rewards, alpha=0.2)
    sns.lineplot(x=time_axis, y=generalized_dc_mean_rewards, label="Decoupled Controller", ax=axs[2], legend=False)
    # axs[2].fill_between(time_axis, generalized_dc_mean_rewards - generalized_dc_std_rewards, generalized_dc_mean_rewards + generalized_dc_std_rewards, alpha=0.2)
    # axs[2].set_title("Rewards")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Normalized Reward")
    axs[2].set_ylim(-0.1, 1.1)
    # axs[2].set_xlim(0, 5.0)

    axins = inset_axes(axs[2],1,1, loc=2,bbox_to_anchor=(0.865, 0.8),bbox_transform=axs[2].figure.transFigure)
    x1, x2, y1, y2 = 6, 10, 0.995, 1.001
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_yticks([0.996, 0.998, 1.0])
    axins.tick_params(axis='y', labelsize=8, pad=1)
    axins.set_xticks([])
    sns.lineplot(x=time_axis, y=generalized_mean_rewards, label="RL Controller", errorbar='sd', ax=axins, legend=False)
    axins.fill_between(time_axis, generalized_mean_rewards - generalized_std_rewards, generalized_mean_rewards + generalized_std_rewards, alpha=0.2)
    sns.lineplot(x=time_axis, y=generalized_dc_mean_rewards, label="Decoupled Controller", ax=axins, legend=False)
    axins.set(xlabel=None, ylabel=None)
    mark_inset(axs[2], axins, loc1=1, loc2=2, fc="none", ec="0.5")

    plt.tight_layout()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))
    
    # plt.savefig(os.path.join(rl_eval_path_list[0], "paper_figures", "mean_combined_plots.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_eval_path, "paper_figures", "mean_combined_plots.pdf"), dpi=1000, format='pdf', bbox_inches='tight')

    # Make plot for single column, just position and yaw error case study
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.5), dpi=1000)
    sns.lineplot(x=time_axis, y=case_study_mean_pos_error, label="RL Controller", errorbar='sd', ax=axs[0], legend=False)
    axs[0].fill_between(time_axis, case_study_mean_pos_error - case_study_std_pos_error, case_study_mean_pos_error + case_study_std_pos_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=case_study_dc_mean_pos_error, label="Decoupled Controller", ax=axs[0], legend=False)
    # axs[0].fill_between(time_axis, case_study_dc_mean_pos_error - case_study_dc_std_pos_error, case_study_dc_mean_pos_error + case_study_dc_std_pos_error, alpha=0.2)
    # axs[0].set_title("End Effector Position Error")
    axs[0].set_xlabel("Time (s)", size=8,)
    axs[0].set_ylabel("Position Error (m)", size=8,)
    axs[0].tick_params(axis='x', labelsize=8, pad=1)
    axs[0].tick_params(axis='y', labelsize=8, pad=1)
    axs[0].set_ylim(-0.001, 0.25)
    axs[0].set_xlim(0, 5.0)

    sns.lineplot(x=time_axis, y=case_study_mean_yaw_error, label="RL Controller", errorbar='sd', ax=axs[1], legend=False)
    axs[1].fill_between(time_axis, case_study_mean_yaw_error - case_study_std_yaw_error, case_study_mean_yaw_error + case_study_std_yaw_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=case_study_dc_mean_yaw_error, label="Decoupled Controller", ax=axs[1], legend=False)
    # axs[1].fill_between(time_axis, case_study_dc_mean_yaw_error - case_study_dc_std_yaw_error, case_study_dc_mean_yaw_error + case_study_dc_std_yaw_error, alpha=0.2)
    # axs[1].set_title("End Effector Yaw Error")
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    axs[1].tick_params(axis='x', labelsize=8, pad=1)
    axs[1].tick_params(axis='y', labelsize=8, pad=1)
    axs[1].set_xlabel("Time (s)", size=8,)
    axs[1].set_ylabel("Yaw Error (rad)", size=8,)
    axs[1].set_ylim(-0.01, 1.75)
    axs[1].set_xlim(0, 5.0)
    
    plt.tight_layout()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))

    # plt.savefig(os.path.join(rl_eval_path_list[0], "paper_figures", "mean_combined_plots_no_reward.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_eval_path, "paper_figures", "case_study_mean_combined_plot.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    

    #Make a bar plot for the balls caught
    print("RL Balls Caught: ", rl_mean_balls_caught.mean())
    print("RL Balls Caught: ", rl_mean_balls_caught.mean()/ 5.0 * 100)
    print("DC Balls Caught: ", dc_mean_balls_caught)
    print("DC Balls Caught: ", dc_mean_balls_caught /5.0 * 100)
    print("RL Balls Caught STD: ", rl_std_balls_caught.std())
    print("DC Balls Caught STD: ", dc_std_balls_caught)
    df = pd.DataFrame()
    df['Balls Caught'] = np.concatenate([rl_mean_balls_caught, np.asarray([dc_mean_balls_caught])])
    df['Controller'] = np.concatenate([np.asarray(["RL Controller", "RL Controller", "RL Controller", "RL Controller", "RL Controller", "Decoupled Controller"])])

    fig, axs = plt.subplots(1, 1, figsize=(3.25, 2.5), dpi=1000)
    sns.barplot(x="Controller", y="Balls Caught", data=df, ax=axs, errorbar='sd', hue="Controller")
    axs.set_ylabel("Balls Caught")
    axs.set_ylim(0, 5)
    axs.set(xlabel=None)
    plt.tight_layout()
    plt.savefig(os.path.join(dc_eval_path, "paper_figures", "balls_caught.pdf"), dpi=1000, format='pdf', bbox_inches='tight')

    # Traj tracking plot
    rl_traj_data = torch.zeros(len(traj_rate_list), N, M, T, 33)
    rl_traj_rewards = torch.zeros(len(traj_rate_list), N, M, T)
    for i, rate in enumerate(traj_rate_list):
        traj_data_path = "eval_traj_track_" + rate + "Hz_eval_full_states.pt"
        traj_reward_path = "eval_traj_track_" + rate + "Hz_eval_rewards.pt"
        for n in range(len(rl_eval_path_list)):
            rl_eval_path = rl_eval_path_list[n]
            single_trial_traj = torch.load(os.path.join(rl_eval_path, traj_data_path))
            single_trial_traj_rewards = torch.load(os.path.join(rl_eval_path, traj_reward_path))

            rl_traj_data[i, n] = single_trial_traj
            rl_traj_rewards[i, n] = single_trial_traj_rewards
    
    dc_traj_data = torch.zeros(len(traj_rate_list), M, T, 33)
    dc_traj_rewards = torch.zeros(len(traj_rate_list), M, T)
    for i, rate in enumerate(traj_rate_list):
        traj_data_path = "eval_traj_track_" + rate + "Hz_eval_full_states.pt"
        traj_reward_path = "eval_traj_track_" + rate + "Hz_eval_rewards.pt"
        dc_traj_data[i] = torch.load(os.path.join(dc_eval_path, traj_data_path))
        dc_traj_rewards[i] = torch.load(os.path.join(dc_eval_path, traj_reward_path))

    rl_rmse_pos = (rl_traj_data[:, :, :, :-1, -7:-4] - rl_traj_data[:, :, :, :-1, 13:16]).norm(dim=-1).square().mean(dim=-1).sqrt()
    dc_rmse_pos = (dc_traj_data[:, :, :-1, -7:-4] - dc_traj_data[:, :, :-1, 13:16]).norm(dim=-1).square().mean(dim=-1).sqrt()
    rl_rmse_yaw = math_utils.yaw_error_from_quats(rl_traj_data[:, :, :, :-1, -4:], rl_traj_data[:, :, :, :-1, 16:20], 0).square().mean(dim=-1).sqrt()
    dc_rmse_yaw = math_utils.yaw_error_from_quats(dc_traj_data[:, :, :-1, -4:], dc_traj_data[:, :, :-1, 16:20], 0).square().mean(dim=-1).sqrt()


    rl_rmse_pos_mean = rl_rmse_pos.mean(dim=2).cpu().numpy()
    dc_rmse_pos_mean = dc_rmse_pos.mean(dim=-1).cpu().numpy()

    rl_rmse_yaw_mean = rl_rmse_yaw.mean(dim=2).cpu().numpy()
    dc_rmse_yaw_mean = dc_rmse_yaw.mean(dim=-1).cpu().numpy()

    traj_rate_list_label = traj_rate_list
    traj_rate_list_label[0] = "0.5"
    for i in range(len(traj_rate_list_label)):
        traj_rate_list_label[i] += " Hz"
    
    rl_rate_label_list = []
    for i in traj_rate_list_label:
        rl_rate_label_list.extend([i] * N)
    rl_controller_label = ['RL Controller'] * len(rl_rate_label_list)
    
    df = pd.DataFrame()
    df['RMSE Pos'] = np.concatenate([rl_rmse_pos_mean.flatten(), dc_rmse_pos_mean.flatten()])
    df['RMSE Yaw'] = np.concatenate([rl_rmse_yaw_mean.flatten(), dc_rmse_yaw_mean.flatten()])
    df['traj_rate'] = np.concatenate([np.asarray(rl_rate_label_list), np.asarray(traj_rate_list_label)])
    df['Controller'] = np.concatenate([np.asarray(rl_controller_label), np.asarray(['Decoupled Controller'] * len(dc_rmse_pos_mean.flatten()))])

    # Fig size for double column IEEE figure: 6.5 x 2.5
    fig, axs = plt.subplots(1, 1, figsize=(3.5, 2.5), dpi=1000)
    sns.barplot(x="traj_rate", y="RMSE Pos", hue="Controller", data=df, ax=axs, errorbar="sd")
    axs.set_ylabel("RMS Position Error (m)")
    # axs.set_ylim(0, 0.2)
    axs.set_xlabel("Trajectory Rate (Hz)")
    handles, labels = axs.get_legend_handles_labels()
    axs.legend(handles=handles, labels=labels)
    sns.move_legend(axs, "upper center", ncol=2, bbox_to_anchor=(0.5, 1.25))
    # handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles=handles, labels=labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(dc_eval_path, "paper_figures", "traj_tracking_pos.pdf"), dpi=1000, format='pdf', bbox_inches='tight')

    fig, axs = plt.subplots(1, 1, figsize=(3.5, 2.5), dpi=1000)
    sns.barplot(x="traj_rate", y="RMSE Yaw", hue="Controller", data=df, ax=axs, errorbar="sd")
    axs.set_ylabel("RMS Yaw Error (rad)")
    # axs.set_ylim(0, 0.2)
    axs.set_xlabel("Trajectory Rate (Hz)")
    handles, labels = axs.get_legend_handles_labels()
    axs.legend(handles=handles, labels=labels)
    sns.move_legend(axs, "upper center", ncol=2, bbox_to_anchor=(0.5, 1.25))
    # handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))


    plt.tight_layout()
    plt.savefig(os.path.join(dc_eval_path, "paper_figures", "traj_tracking_yaw.pdf"), dpi=1000, format='pdf', bbox_inches='tight')

    

if __name__ == "__main__":
    # rl_eval_path = "../rl/runs/AM_0DOF_hover_pos_and_yaw_yaw_error_scale_-0.1_custom_yaw_error_1/"
    # rl_eval_path = "../rl/runs/AM_0DOF_hover_pos_and_yaw_yaw_error_scale_-0.1_custom_yaw_func_anneal_lr_1/"
    # rl_eval_path = "../rl/runs/AM_0DOF_hover_pos_and_yaw_yaw_distance_scale_5_pos_distance_15_smooth_transition_1"

    # rl_eval_path = "../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_14-38-12_rsl_rl_test_default_1024_env_pos_distance_15_yaw_error_-2.0_no_smooth_transition_full_ori/"
    # rl_eval_path = "../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_17-44-44_paper_model_0/"

    
    # plot_data(rl_eval_path, dc_eval_path)
    # plot_data(rl_eval_path, dc_eval_path, "case_study_")
    

    # plot_ball_catching(rl_eval_path, dc_eval_path)



    rl_seeds_list = []
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_17-44-44_paper_model_0/")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_17-45-04_paper_model_1/")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_18-24-39_paper_model_2/")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_18-24-40_paper_model_3/")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_18-24-50_paper_model_4/")
    dc_eval_path = "../rl/baseline_0dof/"
    plot_paper_figs(rl_seeds_list, dc_eval_path)

    # rl_eval_path = "../rl/runs/AM_0DOF_hover_pos_and_yaw_yaw_error_scale_-2.0_smooth_transition_rework_10_yaw_error_urdf_match_1"
    # dc_eval_path = "../rl/baseline_0dof/"
    # plot_traj_tracking(rl_eval_path, dc_eval_path, "eval_traj_track_1Hz", seed="0")