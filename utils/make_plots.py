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

        generalized_data[i,:,:,:] = single_trial_generalized
        case_study_data[i,:,:,:] = single_trial_case_study
        generalized_rewards[i,:,:] = single_trial_rewards
        ball_catching_rewards[i,:,:] = single_trial_ball_catch_rewards

    # DC Full Data (M, T, 33)
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

    # print("Generalized RL Pos Norm: ", generalized_rl_pos_norm.mean(dim=1).mean(dim=0)[0])
    # print("generalized_dc_pos_norm mean: ", generalized_dc_pos_norm.mean(dim=0)[0])

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
    plt.savefig(os.path.join(dc_eval_path, "paper_figures", "mean_combined_plots.png"), dpi=1000, format='png', bbox_inches='tight')

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
    axs[0].set_ylim(-0.001, 0.15)
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
    axs[1].set_ylim(-0.01, 1.6)
    axs[1].set_xlim(0, 5.0)
    
    plt.tight_layout()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))

    # plt.savefig(os.path.join(rl_eval_path_list[0], "paper_figures", "mean_combined_plots_no_reward.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_eval_path, "paper_figures", "case_study_mean_combined_plot.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_eval_path, "paper_figures", "case_study_mean_combined_plot.png"), dpi=1000, format='png', bbox_inches='tight')

    #Make a bar plot for the balls caught
    print("RL Balls Caught: ", rl_mean_balls_caught.mean())
    print("RL Balls Caught (%): ", rl_mean_balls_caught.mean()/ 5.0 * 100)
    print("DC Balls Caught: ", dc_mean_balls_caught)
    print("DC Balls Caught (%): ", dc_mean_balls_caught /5.0 * 100)
    print("RL Balls Caught STD: ", rl_mean_balls_caught.std())
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
    plt.savefig(os.path.join(dc_eval_path, "paper_figures", "balls_caught.png"), dpi=1000, format='png', bbox_inches='tight')


    return

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

def plot_gc_tuning_variants():
    dc_hand_tune_path = "../rl/baseline_0dof/"
    dc_com_tune_path = "../rl/baseline_0dof_com_lqr_tune/"
    dc_com_reward_tune_path = "../rl/baseline_0dof_com_reward_tune/"
    dc_ee_tune_path = "../rl/baseline_0dof_ee_lqr_tune/"
    dc_ee_reward_tune_path = "../rl/baseline_0dof_ee_reward_tune/"

    dc_hand_tune_data = torch.load(os.path.join(dc_hand_tune_path, "eval_full_states.pt"))
    dc_com_tune_data = torch.load(os.path.join(dc_com_tune_path, "eval_full_states.pt"))
    dc_com_reward_tune_data = torch.load(os.path.join(dc_com_reward_tune_path, "eval_full_states.pt"))
    dc_ee_tune_data = torch.load(os.path.join(dc_ee_tune_path, "eval_full_states.pt"))
    dc_ee_reward_tune_data = torch.load(os.path.join(dc_ee_reward_tune_path, "eval_full_states.pt"))

    dc_hand_tune_rewards = torch.load(os.path.join(dc_hand_tune_path, "eval_rewards.pt"))
    dc_com_tune_rewards = torch.load(os.path.join(dc_com_tune_path, "eval_rewards.pt"))
    dc_com_reward_tune_rewards = torch.load(os.path.join(dc_com_reward_tune_path, "eval_rewards.pt"))
    dc_ee_tune_rewards = torch.load(os.path.join(dc_ee_tune_path, "eval_rewards.pt"))
    dc_ee_reward_tune_rewards = torch.load(os.path.join(dc_ee_reward_tune_path, "eval_rewards.pt"))

    dc_hand_tune_case_study_data = torch.load(os.path.join(dc_hand_tune_path, "case_study_eval_full_states.pt"))
    dc_com_tune_case_study_data = torch.load(os.path.join(dc_com_tune_path, "case_study_eval_full_states.pt"))
    dc_com_reward_tune_case_study_data = torch.load(os.path.join(dc_com_reward_tune_path, "case_study_eval_full_states.pt"))
    dc_ee_tune_case_study_data = torch.load(os.path.join(dc_ee_tune_path, "case_study_eval_full_states.pt"))
    dc_ee_reward_tune_case_study_data = torch.load(os.path.join(dc_ee_reward_tune_path, "case_study_eval_full_states.pt"))

    dc_hand_tune_pos_norm = torch.norm(dc_hand_tune_data[:, :-1, -7:-4] - dc_hand_tune_data[:, :-1, 13:16], dim=-1)
    dc_hand_tune_yaw_error = math_utils.yaw_error_from_quats(dc_hand_tune_data[:, :-1, -4:], dc_hand_tune_data[:, :-1, 16:20], 0)
    dc_com_tune_pos_norm = torch.norm(dc_com_tune_data[:, :-1, -7:-4] - dc_com_tune_data[:, :-1, 13:16], dim=-1)
    dc_com_tune_yaw_error = math_utils.yaw_error_from_quats(dc_com_tune_data[:, :-1, -4:], dc_com_tune_data[:, :-1, 16:20], 0)
    dc_com_reward_tune_pos_norm = torch.norm(dc_com_reward_tune_data[:, :-1, -7:-4] - dc_com_reward_tune_data[:, :-1, 13:16], dim=-1)
    dc_com_reward_tune_yaw_error = math_utils.yaw_error_from_quats(dc_com_reward_tune_data[:, :-1, -4:], dc_com_reward_tune_data[:, :-1, 16:20], 0)
    dc_ee_tune_pos_norm = torch.norm(dc_ee_tune_data[:, :-1, -7:-4] - dc_ee_tune_data[:, :-1, 13:16], dim=-1)
    dc_ee_tune_yaw_error = math_utils.yaw_error_from_quats(dc_ee_tune_data[:, :-1, -4:], dc_ee_tune_data[:, :-1, 16:20], 0)
    dc_ee_reward_tune_pos_norm = torch.norm(dc_ee_reward_tune_data[:, :-1, -7:-4] - dc_ee_reward_tune_data[:, :-1, 13:16], dim=-1)
    dc_ee_reward_tune_yaw_error = math_utils.yaw_error_from_quats(dc_ee_reward_tune_data[:, :-1, -4:], dc_ee_reward_tune_data[:, :-1, 16:20], 0)

    dc_hand_tune_case_study_pos_norm = torch.norm(dc_hand_tune_case_study_data[:, :-1, -7:-4] - dc_hand_tune_case_study_data[:, :-1, 13:16], dim=-1)
    dc_hand_tune_case_study_yaw_error = math_utils.yaw_error_from_quats(dc_hand_tune_case_study_data[:, :-1, -4:], dc_hand_tune_case_study_data[:, :-1, 16:20], 0)
    dc_com_tune_case_study_pos_norm = torch.norm(dc_com_tune_case_study_data[:, :-1, -7:-4] - dc_com_tune_case_study_data[:, :-1, 13:16], dim=-1)
    dc_com_tune_case_study_yaw_error = math_utils.yaw_error_from_quats(dc_com_tune_case_study_data[:, :-1, -4:], dc_com_tune_case_study_data[:, :-1, 16:20], 0)
    dc_com_reward_tune_case_study_pos_norm = torch.norm(dc_com_reward_tune_case_study_data[:, :-1, -7:-4] - dc_com_reward_tune_case_study_data[:, :-1, 13:16], dim=-1)
    dc_com_reward_tune_case_study_yaw_error = math_utils.yaw_error_from_quats(dc_com_reward_tune_case_study_data[:, :-1, -4:], dc_com_reward_tune_case_study_data[:, :-1, 16:20], 0)
    dc_ee_tune_case_study_pos_norm = torch.norm(dc_ee_tune_case_study_data[:, :-1, -7:-4] - dc_ee_tune_case_study_data[:, :-1, 13:16], dim=-1)
    dc_ee_tune_case_study_yaw_error = math_utils.yaw_error_from_quats(dc_ee_tune_case_study_data[:, :-1, -4:], dc_ee_tune_case_study_data[:, :-1, 16:20], 0)
    dc_ee_reward_tune_case_study_pos_norm = torch.norm(dc_ee_reward_tune_case_study_data[:, :-1, -7:-4] - dc_ee_reward_tune_case_study_data[:, :-1, 13:16], dim=-1)
    dc_ee_reward_tune_case_study_yaw_error = math_utils.yaw_error_from_quats(dc_ee_reward_tune_case_study_data[:, :-1, -4:], dc_ee_reward_tune_case_study_data[:, :-1, 16:20], 0)

    max_reward = 15.0
    min_reward = torch.min(dc_hand_tune_rewards[:,:-1]).cpu().item()
    dc_hand_tune_rewards = (dc_hand_tune_rewards[:,:-1] - min_reward) / (max_reward - min_reward)
    dc_com_tune_rewards = (dc_com_tune_rewards[:,:-1] - min_reward) / (max_reward - min_reward)
    dc_com_reward_tune_rewards = (dc_com_reward_tune_rewards[:,:-1] - min_reward) / (max_reward - min_reward)
    dc_ee_tune_rewards = (dc_ee_tune_rewards[:,:-1] - min_reward) / (max_reward - min_reward)
    dc_ee_reward_tune_rewards = (dc_ee_reward_tune_rewards[:,:-1] - min_reward) / (max_reward - min_reward)


    dc_hand_tune_mean_pos_error = dc_hand_tune_pos_norm.mean(dim=0).cpu().numpy()
    dc_com_tune_mean_pos_error = dc_com_tune_pos_norm.mean(dim=0).cpu().numpy()
    dc_com_reward_tune_mean_pos_error = dc_com_reward_tune_pos_norm.mean(dim=0).cpu().numpy()
    dc_ee_tune_mean_pos_error = dc_ee_tune_pos_norm.mean(dim=0).cpu().numpy()
    dc_ee_reward_tune_mean_pos_error = dc_ee_reward_tune_pos_norm.mean(dim=0).cpu().numpy()

    dc_hand_tune_mean_yaw_error = dc_hand_tune_yaw_error.mean(dim=0).cpu().numpy()
    dc_com_tune_mean_yaw_error = dc_com_tune_yaw_error.mean(dim=0).cpu().numpy()
    dc_com_reward_tune_mean_yaw_error = dc_com_reward_tune_yaw_error.mean(dim=0).cpu().numpy()
    dc_ee_tune_mean_yaw_error = dc_ee_tune_yaw_error.mean(dim=0).cpu().numpy()
    dc_ee_reward_tune_mean_yaw_error = dc_ee_reward_tune_yaw_error.mean(dim=0).cpu().numpy()

    dc_hand_tune_mean_rewards = dc_hand_tune_rewards.mean(dim=0).cpu().numpy()
    dc_com_tune_mean_rewards = dc_com_tune_rewards.mean(dim=0).cpu().numpy()
    dc_com_reward_tune_mean_rewards = dc_com_reward_tune_rewards.mean(dim=0).cpu().numpy()
    dc_ee_tune_mean_rewards = dc_ee_tune_rewards.mean(dim=0).cpu().numpy()
    dc_ee_reward_tune_mean_rewards = dc_ee_reward_tune_rewards.mean(dim=0).cpu().numpy()

    dc_hand_tune_case_study_pos_error = dc_hand_tune_case_study_pos_norm.mean(dim=0).cpu().numpy()
    dc_hand_tune_case_study_yaw_error = dc_hand_tune_case_study_yaw_error.mean(dim=0).cpu().numpy()
    dc_com_tune_case_study_pos_error = dc_com_tune_case_study_pos_norm.mean(dim=0).cpu().numpy()
    dc_com_tune_case_study_yaw_error = dc_com_tune_case_study_yaw_error.mean(dim=0).cpu().numpy()
    dc_com_reward_tune_case_study_pos_error = dc_com_reward_tune_case_study_pos_norm.mean(dim=0).cpu().numpy()
    dc_com_reward_tune_case_study_yaw_error = dc_com_reward_tune_case_study_yaw_error.mean(dim=0).cpu().numpy()
    dc_ee_tune_case_study_pos_error = dc_ee_tune_case_study_pos_norm.mean(dim=0).cpu().numpy()
    dc_ee_tune_case_study_yaw_error = dc_ee_tune_case_study_yaw_error.mean(dim=0).cpu().numpy()
    dc_ee_reward_tune_case_study_pos_error = dc_ee_reward_tune_case_study_pos_norm.mean(dim=0).cpu().numpy()
    dc_ee_reward_tune_case_study_yaw_error = dc_ee_reward_tune_case_study_yaw_error.mean(dim=0).cpu().numpy()

    time_axis = torch.arange(dc_hand_tune_mean_pos_error.shape[0]) * 0.02

    plt.figure()
    plt.plot(time_axis, dc_hand_tune_mean_pos_error, label="Hand Tuned Controller")
    plt.plot(time_axis, dc_com_tune_mean_pos_error, label="CoM LQR Tuned Controller")
    plt.plot(time_axis, dc_com_reward_tune_mean_pos_error, label="CoM Reward Tuned Controller")
    plt.plot(time_axis, dc_ee_tune_mean_pos_error, label="EE LQR Tuned Controller")
    plt.plot(time_axis, dc_ee_reward_tune_mean_pos_error, label="EE Reward Tuned Controller")
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.legend()
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_pos_error.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_pos_error.png"), dpi=1000, format='png', bbox_inches='tight')

    plt.figure()
    plt.plot(time_axis, dc_hand_tune_mean_yaw_error, label="Hand Tuned Controller")
    plt.plot(time_axis, dc_com_tune_mean_yaw_error, label="CoM LQR Tuned Controller")
    plt.plot(time_axis, dc_com_reward_tune_mean_yaw_error, label="CoM Reward Tuned Controller")
    plt.plot(time_axis, dc_ee_tune_mean_yaw_error, label="EE LQR Tuned Controller")
    plt.plot(time_axis, dc_ee_reward_tune_mean_yaw_error, label="EE Reward Tuned Controller")
    plt.xlabel("Time (s)")
    plt.ylabel("Yaw Error (rad)")
    plt.legend()
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_yaw_error.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_yaw_error.png"), dpi=1000, format='png', bbox_inches='tight')

    plt.figure()
    plt.plot(time_axis, dc_hand_tune_mean_rewards, label="Hand Tuned Controller")
    plt.plot(time_axis, dc_com_tune_mean_rewards, label="CoM LQR Tuned Controller")
    plt.plot(time_axis, dc_com_reward_tune_mean_rewards, label="CoM Reward Tuned Controller")
    plt.plot(time_axis, dc_ee_tune_mean_rewards, label="EE LQR Tuned Controller")
    plt.plot(time_axis, dc_ee_reward_tune_mean_rewards, label="EE Reward Tuned Controller")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Reward")
    plt.legend()
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_rewards.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_rewards.png"), dpi=1000, format='png', bbox_inches='tight')

    # combined plot
    fig, axs = plt.subplots(1, 3, figsize=(9.75, 2.5), dpi=1000)
    sns.lineplot(x=time_axis, y=dc_hand_tune_mean_pos_error, label="Hand Tuned Controller", ax=axs[0], legend=False)
    sns.lineplot(x=time_axis, y=dc_com_tune_mean_pos_error, label="CoM LQR Tuned Controller", ax=axs[0], legend=False)
    sns.lineplot(x=time_axis, y=dc_com_reward_tune_mean_pos_error, label="CoM Reward Tuned Controller", ax=axs[0], legend=False)
    sns.lineplot(x=time_axis, y=dc_ee_tune_mean_pos_error, label="EE LQR Tuned Controller", ax=axs[0], legend=False)
    sns.lineplot(x=time_axis, y=dc_ee_reward_tune_mean_pos_error, label="EE Reward Tuned Controller", ax=axs[0], legend=False)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position Error (m)")
    axs[0].set_ylim(-0.1, 2.0)

    sns.lineplot(x=time_axis, y=dc_hand_tune_mean_yaw_error, label="Hand Tuned Controller", ax=axs[1], legend=False)
    sns.lineplot(x=time_axis, y=dc_com_tune_mean_yaw_error, label="CoM LQR Tuned Controller", ax=axs[1], legend=False)
    sns.lineplot(x=time_axis, y=dc_com_reward_tune_mean_yaw_error, label="CoM Reward Tuned Controller", ax=axs[1], legend=False)
    sns.lineplot(x=time_axis, y=dc_ee_tune_mean_yaw_error, label="EE Tuned Controller", ax=axs[1], legend=False)
    sns.lineplot(x=time_axis, y=dc_ee_reward_tune_mean_yaw_error, label="EE Reward Tuned Controller", ax=axs[1], legend=False)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Yaw Error (rad)")
    axs[1].set_ylim(-0.1, 3.141)

    sns.lineplot(x=time_axis, y=dc_hand_tune_mean_rewards, label="Hand Tuned Controller", ax=axs[2], legend=False)
    sns.lineplot(x=time_axis, y=dc_com_tune_mean_rewards, label="CoM LQR Tuned Controller", ax=axs[2], legend=False)
    sns.lineplot(x=time_axis, y=dc_com_reward_tune_mean_rewards, label="CoM Reward Tuned Controller", ax=axs[2], legend=False)
    sns.lineplot(x=time_axis, y=dc_ee_tune_mean_rewards, label="EE Tuned Controller", ax=axs[2], legend=False)
    sns.lineplot(x=time_axis, y=dc_ee_reward_tune_mean_rewards, label="EE Reward Tuned Controller", ax=axs[2], legend=False)
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Normalized Reward")
    axs[2].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.25))
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_combined_plots.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_combined_plots.png"), dpi=1000, format='png', bbox_inches='tight')


    # case study pos plot
    plt.figure()
    plt.plot(time_axis, dc_hand_tune_case_study_pos_error, label="Hand Tuned Controller")
    plt.plot(time_axis, dc_com_tune_case_study_pos_error, label="CoM LQR Tuned Controller")
    plt.plot(time_axis, dc_com_reward_tune_case_study_pos_error, label="CoM Reward Tuned Controller")
    plt.plot(time_axis, dc_ee_tune_case_study_pos_error, label="EE LQR Tuned Controller")
    plt.plot(time_axis, dc_ee_reward_tune_case_study_pos_error, label="EE Reward Tuned Controller")
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.legend()
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_case_study_pos_error.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_case_study_pos_error.png"), dpi=1000, format='png', bbox_inches='tight')

    # case study yaw plot
    plt.figure()
    plt.plot(time_axis, dc_hand_tune_case_study_yaw_error, label="Hand Tuned Controller")
    plt.plot(time_axis, dc_com_tune_case_study_yaw_error, label="CoM LQR Tuned Controller")
    plt.plot(time_axis, dc_com_reward_tune_case_study_yaw_error, label="CoM Reward Tuned Controller")
    plt.plot(time_axis, dc_ee_tune_case_study_yaw_error, label="EE Tuned Controller")
    plt.plot(time_axis, dc_ee_reward_tune_case_study_yaw_error, label="EE Reward Tuned Controller")
    plt.xlabel("Time (s)")
    plt.ylabel("Yaw Error (rad)")
    plt.legend()
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_case_study_yaw_error.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_case_study_yaw_error.png"), dpi=1000, format='png', bbox_inches='tight')

    # Case study combined plot
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.5), dpi=1000)
    sns.lineplot(x=time_axis, y=dc_hand_tune_case_study_pos_error, label="Hand Tuned Controller", ax=axs[0], legend=False)
    sns.lineplot(x=time_axis, y=dc_com_tune_case_study_pos_error, label="CoM LQR Tuned Controller", ax=axs[0], legend=False)
    sns.lineplot(x=time_axis, y=dc_com_reward_tune_case_study_pos_error, label="CoM Reward Tuned Controller", ax=axs[0], legend=False)
    sns.lineplot(x=time_axis, y=dc_ee_tune_case_study_pos_error, label="EE LQR Tuned Controller", ax=axs[0], legend=False)
    sns.lineplot(x=time_axis, y=dc_ee_reward_tune_case_study_pos_error, label="EE Reward Tuned Controller", ax=axs[0], legend=False)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position Error (m)")
    axs[0].set_ylim(-0.01, 0.15)

    sns.lineplot(x=time_axis, y=dc_hand_tune_case_study_yaw_error, label="Hand Tuned Controller", ax=axs[1], legend=False)
    sns.lineplot(x=time_axis, y=dc_com_tune_case_study_yaw_error, label="CoM LQR Tuned Controller", ax=axs[1], legend=False)
    sns.lineplot(x=time_axis, y=dc_com_reward_tune_case_study_yaw_error, label="CoM Reward Tuned Controller", ax=axs[1], legend=False)
    sns.lineplot(x=time_axis, y=dc_ee_tune_case_study_yaw_error, label="EE Tuned Controller", ax=axs[1], legend=False)
    sns.lineplot(x=time_axis, y=dc_ee_reward_tune_case_study_yaw_error, label="EE Reward Tuned Controller", ax=axs[1], legend=False)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Yaw Error (rad)")
    axs[1].set_ylim(-0.1, 1.6)

    plt.tight_layout()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.25))
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_case_study_combined_plots.pdf"), dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_hand_tune_path, "gc_tuning", "gc_tuning_case_study_combined_plots.png"), dpi=1000, format='png', bbox_inches='tight')


def load_rl_data(rl_seeds_list):
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

    for i in range(len(rl_seeds_list)):
        rl_eval_path = rl_seeds_list[i]
        single_trial_generalized = torch.load(os.path.join(rl_eval_path, generalized_data_path))
        single_trial_case_study = torch.load(os.path.join(rl_eval_path, case_study_data_path))
        single_trial_rewards = torch.load(os.path.join(rl_eval_path, generalized_rewards_path))
        # single_trial_ball_catch_rewards = torch.load(os.path.join(rl_eval_path, ball_catch_rewards_path))

        # print("rewards shape: ", single_trial_rewards.shape)
        # print("ball catching rewards shape: ", single_trial_ball_catch_rewards.shape)

        generalized_data[i,:,:,:] = single_trial_generalized
        case_study_data[i,:,:,:] = single_trial_case_study
        generalized_rewards[i,:,:] = single_trial_rewards
        # ball_catching_rewards[i,:,:] = single_trial_ball_catch_rewards

    # DC Full Data (M, T, 33)
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

    # print("Generalized RL Pos Norm: ", generalized_rl_pos_norm.mean(dim=1).mean(dim=0)[0])
    # print("generalized_dc_pos_norm mean: ", generalized_dc_pos_norm.mean(dim=0)[0])

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

    rl_mean_pos_error = generalized_mean_pos_error
    rl_mean_yaw_error = generalized_mean_yaw_error
    rl_mean_rewards = generalized_mean_rewards

    return rl_mean_pos_error, rl_mean_yaw_error, rl_mean_rewards

def load_gc_data():
    dc_hand_tune_path = "../rl/baseline_0dof/"
    dc_com_tune_path = "../rl/baseline_0dof_com_lqr_tune/"
    dc_com_reward_tune_path = "../rl/baseline_0dof_com_reward_tune/"
    dc_ee_tune_path = "../rl/baseline_0dof_ee_lqr_tune/"
    dc_ee_reward_tune_path = "../rl/baseline_0dof_ee_reward_tune/"

    dc_hand_tune_data = torch.load(os.path.join(dc_hand_tune_path, "eval_full_states.pt"))
    dc_com_tune_data = torch.load(os.path.join(dc_com_tune_path, "eval_full_states.pt"))
    dc_com_reward_tune_data = torch.load(os.path.join(dc_com_reward_tune_path, "eval_full_states.pt"))
    dc_ee_tune_data = torch.load(os.path.join(dc_ee_tune_path, "eval_full_states.pt"))
    dc_ee_reward_tune_data = torch.load(os.path.join(dc_ee_reward_tune_path, "eval_full_states.pt"))

    dc_hand_tune_rewards = torch.load(os.path.join(dc_hand_tune_path, "eval_rewards.pt"))
    dc_com_tune_rewards = torch.load(os.path.join(dc_com_tune_path, "eval_rewards.pt"))
    dc_com_reward_tune_rewards = torch.load(os.path.join(dc_com_reward_tune_path, "eval_rewards.pt"))
    dc_ee_tune_rewards = torch.load(os.path.join(dc_ee_tune_path, "eval_rewards.pt"))
    dc_ee_reward_tune_rewards = torch.load(os.path.join(dc_ee_reward_tune_path, "eval_rewards.pt"))

    dc_hand_tune_case_study_data = torch.load(os.path.join(dc_hand_tune_path, "case_study_eval_full_states.pt"))
    dc_com_tune_case_study_data = torch.load(os.path.join(dc_com_tune_path, "case_study_eval_full_states.pt"))
    dc_com_reward_tune_case_study_data = torch.load(os.path.join(dc_com_reward_tune_path, "case_study_eval_full_states.pt"))
    dc_ee_tune_case_study_data = torch.load(os.path.join(dc_ee_tune_path, "case_study_eval_full_states.pt"))
    dc_ee_reward_tune_case_study_data = torch.load(os.path.join(dc_ee_reward_tune_path, "case_study_eval_full_states.pt"))

    dc_hand_tune_pos_norm = torch.norm(dc_hand_tune_data[:, :-1, -7:-4] - dc_hand_tune_data[:, :-1, 13:16], dim=-1)
    dc_hand_tune_yaw_error = math_utils.yaw_error_from_quats(dc_hand_tune_data[:, :-1, -4:], dc_hand_tune_data[:, :-1, 16:20], 0)
    dc_com_tune_pos_norm = torch.norm(dc_com_tune_data[:, :-1, -7:-4] - dc_com_tune_data[:, :-1, 13:16], dim=-1)
    dc_com_tune_yaw_error = math_utils.yaw_error_from_quats(dc_com_tune_data[:, :-1, -4:], dc_com_tune_data[:, :-1, 16:20], 0)
    dc_com_reward_tune_pos_norm = torch.norm(dc_com_reward_tune_data[:, :-1, -7:-4] - dc_com_reward_tune_data[:, :-1, 13:16], dim=-1)
    dc_com_reward_tune_yaw_error = math_utils.yaw_error_from_quats(dc_com_reward_tune_data[:, :-1, -4:], dc_com_reward_tune_data[:, :-1, 16:20], 0)
    dc_ee_tune_pos_norm = torch.norm(dc_ee_tune_data[:, :-1, -7:-4] - dc_ee_tune_data[:, :-1, 13:16], dim=-1)
    dc_ee_tune_yaw_error = math_utils.yaw_error_from_quats(dc_ee_tune_data[:, :-1, -4:], dc_ee_tune_data[:, :-1, 16:20], 0)
    dc_ee_reward_tune_pos_norm = torch.norm(dc_ee_reward_tune_data[:, :-1, -7:-4] - dc_ee_reward_tune_data[:, :-1, 13:16], dim=-1)
    dc_ee_reward_tune_yaw_error = math_utils.yaw_error_from_quats(dc_ee_reward_tune_data[:, :-1, -4:], dc_ee_reward_tune_data[:, :-1, 16:20], 0)

    dc_hand_tune_case_study_pos_norm = torch.norm(dc_hand_tune_case_study_data[:, :-1, -7:-4] - dc_hand_tune_case_study_data[:, :-1, 13:16], dim=-1)
    dc_hand_tune_case_study_yaw_error = math_utils.yaw_error_from_quats(dc_hand_tune_case_study_data[:, :-1, -4:], dc_hand_tune_case_study_data[:, :-1, 16:20], 0)
    dc_com_tune_case_study_pos_norm = torch.norm(dc_com_tune_case_study_data[:, :-1, -7:-4] - dc_com_tune_case_study_data[:, :-1, 13:16], dim=-1)
    dc_com_tune_case_study_yaw_error = math_utils.yaw_error_from_quats(dc_com_tune_case_study_data[:, :-1, -4:], dc_com_tune_case_study_data[:, :-1, 16:20], 0)
    dc_com_reward_tune_case_study_pos_norm = torch.norm(dc_com_reward_tune_case_study_data[:, :-1, -7:-4] - dc_com_reward_tune_case_study_data[:, :-1, 13:16], dim=-1)
    dc_com_reward_tune_case_study_yaw_error = math_utils.yaw_error_from_quats(dc_com_reward_tune_case_study_data[:, :-1, -4:], dc_com_reward_tune_case_study_data[:, :-1, 16:20], 0)
    dc_ee_tune_case_study_pos_norm = torch.norm(dc_ee_tune_case_study_data[:, :-1, -7:-4] - dc_ee_tune_case_study_data[:, :-1, 13:16], dim=-1)
    dc_ee_tune_case_study_yaw_error = math_utils.yaw_error_from_quats(dc_ee_tune_case_study_data[:, :-1, -4:], dc_ee_tune_case_study_data[:, :-1, 16:20], 0)
    dc_ee_reward_tune_case_study_pos_norm = torch.norm(dc_ee_reward_tune_case_study_data[:, :-1, -7:-4] - dc_ee_reward_tune_case_study_data[:, :-1, 13:16], dim=-1)
    dc_ee_reward_tune_case_study_yaw_error = math_utils.yaw_error_from_quats(dc_ee_reward_tune_case_study_data[:, :-1, -4:], dc_ee_reward_tune_case_study_data[:, :-1, 16:20], 0)

    max_reward = 15.0
    min_reward = torch.min(dc_hand_tune_rewards[:,:-1]).cpu().item()
    dc_hand_tune_rewards = (dc_hand_tune_rewards[:,:-1] - min_reward) / (max_reward - min_reward)
    dc_com_tune_rewards = (dc_com_tune_rewards[:,:-1] - min_reward) / (max_reward - min_reward)
    dc_com_reward_tune_rewards = (dc_com_reward_tune_rewards[:,:-1] - min_reward) / (max_reward - min_reward)
    dc_ee_tune_rewards = (dc_ee_tune_rewards[:,:-1] - min_reward) / (max_reward - min_reward)
    dc_ee_reward_tune_rewards = (dc_ee_reward_tune_rewards[:,:-1] - min_reward) / (max_reward - min_reward)


    dc_hand_tune_mean_pos_error = dc_hand_tune_pos_norm.mean(dim=0).cpu().numpy()
    dc_com_tune_mean_pos_error = dc_com_tune_pos_norm.mean(dim=0).cpu().numpy()
    dc_com_reward_tune_mean_pos_error = dc_com_reward_tune_pos_norm.mean(dim=0).cpu().numpy()
    dc_ee_tune_mean_pos_error = dc_ee_tune_pos_norm.mean(dim=0).cpu().numpy()
    dc_ee_reward_tune_mean_pos_error = dc_ee_reward_tune_pos_norm.mean(dim=0).cpu().numpy()

    dc_hand_tune_mean_yaw_error = dc_hand_tune_yaw_error.mean(dim=0).cpu().numpy()
    dc_com_tune_mean_yaw_error = dc_com_tune_yaw_error.mean(dim=0).cpu().numpy()
    dc_com_reward_tune_mean_yaw_error = dc_com_reward_tune_yaw_error.mean(dim=0).cpu().numpy()
    dc_ee_tune_mean_yaw_error = dc_ee_tune_yaw_error.mean(dim=0).cpu().numpy()
    dc_ee_reward_tune_mean_yaw_error = dc_ee_reward_tune_yaw_error.mean(dim=0).cpu().numpy()

    dc_hand_tune_mean_rewards = dc_hand_tune_rewards.mean(dim=0).cpu().numpy()
    dc_com_tune_mean_rewards = dc_com_tune_rewards.mean(dim=0).cpu().numpy()
    dc_com_reward_tune_mean_rewards = dc_com_reward_tune_rewards.mean(dim=0).cpu().numpy()
    dc_ee_tune_mean_rewards = dc_ee_tune_rewards.mean(dim=0).cpu().numpy()
    dc_ee_reward_tune_mean_rewards = dc_ee_reward_tune_rewards.mean(dim=0).cpu().numpy()

    dc_hand_tune_case_study_pos_error = dc_hand_tune_case_study_pos_norm.mean(dim=0).cpu().numpy()
    dc_hand_tune_case_study_yaw_error = dc_hand_tune_case_study_yaw_error.mean(dim=0).cpu().numpy()
    dc_com_tune_case_study_pos_error = dc_com_tune_case_study_pos_norm.mean(dim=0).cpu().numpy()
    dc_com_tune_case_study_yaw_error = dc_com_tune_case_study_yaw_error.mean(dim=0).cpu().numpy()
    dc_com_reward_tune_case_study_pos_error = dc_com_reward_tune_case_study_pos_norm.mean(dim=0).cpu().numpy()
    dc_com_reward_tune_case_study_yaw_error = dc_com_reward_tune_case_study_yaw_error.mean(dim=0).cpu().numpy()
    dc_ee_tune_case_study_pos_error = dc_ee_tune_case_study_pos_norm.mean(dim=0).cpu().numpy()
    dc_ee_tune_case_study_yaw_error = dc_ee_tune_case_study_yaw_error.mean(dim=0).cpu().numpy()
    dc_ee_reward_tune_case_study_pos_error = dc_ee_reward_tune_case_study_pos_norm.mean(dim=0).cpu().numpy()
    dc_ee_reward_tune_case_study_yaw_error = dc_ee_reward_tune_case_study_yaw_error.mean(dim=0).cpu().numpy()

    pos_error_list = [dc_hand_tune_mean_pos_error, dc_com_tune_mean_pos_error, dc_com_reward_tune_mean_pos_error, dc_ee_tune_mean_pos_error, dc_ee_reward_tune_mean_pos_error]
    yaw_error_list = [dc_hand_tune_mean_yaw_error, dc_com_tune_mean_yaw_error, dc_com_reward_tune_mean_yaw_error, dc_ee_tune_mean_yaw_error, dc_ee_reward_tune_mean_yaw_error]
    rewards_list = [dc_hand_tune_mean_rewards, dc_com_tune_mean_rewards, dc_com_reward_tune_mean_rewards, dc_ee_tune_mean_rewards, dc_ee_reward_tune_mean_rewards]

    return pos_error_list, yaw_error_list, rewards_list

def plot_settling_time_and_error(threshold=0.95):

    rl_mean_pos_error, rl_mean_yaw_error, rl_mean_rewards = load_rl_data(rl_seeds_list)
    gc_mean_pos_list, gc_mean_yaw_list, gc_mean_rewards_list  = load_gc_data()

    # find the steady state error, and then compute the time to reach threshold*steady_state_error
    rl_steady_state_error = rl_mean_pos_error[-1]
    
    # find index when the error is less than threshold*steady_state_error
    error_threshold = np.max(rl_mean_pos_error)  - threshold * (np.max(rl_mean_pos_error) - rl_steady_state_error)
    rl_settling_time_idx = (np.flatnonzero(rl_mean_pos_error > threshold)[-1] + 1) * 0.02
    # print("RL steady state error: ", rl_steady_state_error)
    # print("RL error threshold: ", error_threshold)
    # print("RL settling time: ", rl_settling_time_idx)

    gc_steady_state_error_list = []
    gc_settling_time_idx_list = []
    gc_labels = ["Hand Tuned", "CoM LQR Tuned", "CoM Reward Tuned", "EE LQR Tuned", "EE Reward Tuned"]
    for i, gc_mean_data in enumerate(gc_mean_pos_list):
        gc_steady_state_error = gc_mean_data[-1]
        error_threshold = gc_mean_data[0] - threshold * (gc_mean_data[0] - gc_steady_state_error)
        gc_settling_time_idx = np.argmax(gc_mean_data < error_threshold) * 0.02
        gc_steady_state_error_list.append(gc_steady_state_error)
        gc_settling_time_idx_list.append(gc_settling_time_idx)

    # Make a scatter plot where each point is a different controller
    plt.figure()
    plt.scatter(rl_steady_state_error, rl_settling_time_idx, label="RL", marker="*")
    for i in range(len(gc_steady_state_error_list)):
        plt.scatter(gc_steady_state_error_list[i], gc_settling_time_idx_list[i], label="DC " + gc_labels[i], marker="*")
    plt.xlabel("Steady State Error (m)")
    plt.ylabel("Settling Time (s)")
    plt.legend(loc="best")
    # fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.25))
    plt.savefig("settling_time_vs_steady_state_error.pdf", dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig("settling_time_vs_steady_state_error.png", dpi=1000, format='png', bbox_inches='tight')

def plot_rl_pareto():
    yaw_conditions = ["none", "half", "full"]
    pos_conditions = ["none", "half", "full"]
    rl_base_path = "../rl/logs/rsl_rl/AM_0DOF_Hover/2024-10-01_17-52-33_new_defaults_anneal_50e6_seed_0/"
    dc_base_path = "../rl/baseline_0dof_ee_reward_tune/"
    rl_crash_base_path = "../rl/logs/rsl_rl/AM_0_DOF/2024-10-07_13-27-55_LQR_reward_-2_pos_-2_yaw_-10_crash/"
    rl_COM_base_path = "../rl/logs/rsl_rl/AM_0DOF_COM/2024-10-07_18-59-53_LQR_-2_pos_-2_yaw_-10_crash_COM_all_bodies_reward_change/"

    rl_pos_settling_times = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_yaw_settling_times = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_pos_rmse = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_yaw_rmse = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_pos_errors = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_yaw_errors = np.zeros((len(pos_conditions), len(yaw_conditions)))

    dc_pos_settling_times = np.zeros((len(pos_conditions), len(yaw_conditions)))
    dc_yaw_settling_times = np.zeros((len(pos_conditions), len(yaw_conditions)))
    dc_pos_rmse = np.zeros((len(pos_conditions), len(yaw_conditions)))
    dc_yaw_rmse = np.zeros((len(pos_conditions), len(yaw_conditions)))
    dc_pos_errors = np.zeros((len(pos_conditions), len(yaw_conditions)))
    dc_yaw_errors = np.zeros((len(pos_conditions), len(yaw_conditions)))

    rl_crash_pos_settling_times = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_crash_yaw_settling_times = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_crash_pos_rmse = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_crash_yaw_rmse = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_crash_pos_errors = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_crash_yaw_errors = np.zeros((len(pos_conditions), len(yaw_conditions)))

    rl_COM_pos_settling_times = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_COM_yaw_settling_times = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_COM_pos_rmse = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_COM_yaw_rmse = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_COM_pos_errors = np.zeros((len(pos_conditions), len(yaw_conditions)))
    rl_COM_yaw_errors = np.zeros((len(pos_conditions), len(yaw_conditions)))

    pos_errors = np.zeros((len(pos_conditions), len(yaw_conditions), 249)) # 499
    yaw_errors = np.zeros((len(pos_conditions), len(yaw_conditions), 249)) # 499
    rewards = np.zeros((len(pos_conditions), len(yaw_conditions), 249)) # 499

    pos_errors_dc = np.zeros((len(pos_conditions), len(yaw_conditions), 249)) # 499
    yaw_errors_dc = np.zeros((len(pos_conditions), len(yaw_conditions), 249)) # 499
    rewards_dc = np.zeros((len(pos_conditions), len(yaw_conditions), 249)) # 499

    pos_errors_rl_crash = np.zeros((len(pos_conditions), len(yaw_conditions), 249)) # 499
    yaw_errors_rl_crash = np.zeros((len(pos_conditions), len(yaw_conditions), 249)) # 499
    rewards_rl_crash = np.zeros((len(pos_conditions), len(yaw_conditions), 249)) # 499

    pos_errors_rl_COM = np.zeros((len(pos_conditions), len(yaw_conditions), 249)) # 499
    yaw_errors_rl_COM = np.zeros((len(pos_conditions), len(yaw_conditions), 249)) # 499
    rewards_rl_COM = np.zeros((len(pos_conditions), len(yaw_conditions), 249)) # 499


    all_rl_trials = torch.zeros((len(pos_conditions), len(yaw_conditions), 100, 500, 33))
    all_dc_trials = torch.zeros((len(pos_conditions), len(yaw_conditions), 100, 500, 33))
    all_rl_crash_trials = torch.zeros((len(pos_conditions), len(yaw_conditions), 100, 500, 33))
    all_rl_COM_trials = torch.zeros((len(pos_conditions), len(yaw_conditions), 100, 500, 33))


    for i, pos_condition in enumerate(pos_conditions):
        for j, yaw_condition in enumerate(yaw_conditions):
            print("Loading pos_condition: ", pos_condition, " yaw_condition: ", yaw_condition)
            # M x T x 33
            rl_eval = torch.load(os.path.join(rl_base_path, "pos_" + pos_condition + "_yaw_" + yaw_condition + "_eval_full_states.pt"))
            dc_eval = torch.load(os.path.join(dc_base_path, "pos_" + pos_condition + "_yaw_" + yaw_condition + "_eval_full_states.pt"))
            rl_crash_eval = torch.load(os.path.join(rl_crash_base_path, "pos_" + pos_condition + "_yaw_" + yaw_condition + "_eval_full_states.pt"))
            rl_COM_eval = torch.load(os.path.join(rl_COM_base_path, "pos_" + pos_condition + "_yaw_" + yaw_condition + "_eval_full_states.pt"))
            # M x T
            rl_eval_rewards = torch.load(os.path.join(rl_base_path, "pos_" + pos_condition + "_yaw_" + yaw_condition + "_eval_rewards.pt"))
            dc_eval_rewards = torch.load(os.path.join(dc_base_path, "pos_" + pos_condition + "_yaw_" + yaw_condition + "_eval_rewards.pt"))
            rl_crash_eval_rewards = torch.load(os.path.join(rl_crash_base_path, "pos_" + pos_condition + "_yaw_" + yaw_condition + "_eval_rewards.pt"))
            rl_COM_eval_rewards = torch.load(os.path.join(rl_COM_base_path, "pos_" + pos_condition + "_yaw_" + yaw_condition + "_eval_rewards.pt"))

            all_rl_trials[i, j, :, :, :] = rl_eval
            all_dc_trials[i, j, :, :, :] = dc_eval
            all_rl_crash_trials[i, j, :, :, :] = rl_crash_eval
            all_rl_COM_trials[i, j, :, :, :] = rl_COM_eval
            
            pos_error_t = torch.norm(rl_eval[:, :-251, -7:-4] - rl_eval[:, :-251, 13:16], dim=-1).mean(dim=0).cpu().numpy()
            yaw_error_t = math_utils.yaw_error_from_quats(rl_eval[:, :-251, -4:], rl_eval[:, :-251, 16:20], 0).mean(dim=0).cpu().numpy()
            reward_t = rl_eval_rewards[:,:-251].mean(dim=0).cpu().numpy()

            dc_pos_error_t = torch.norm(dc_eval[:, :-251, -7:-4] - dc_eval[:, :-251, 13:16], dim=-1).mean(dim=0).cpu().numpy()
            dc_yaw_error_t = math_utils.yaw_error_from_quats(dc_eval[:, :-251, -4:], dc_eval[:, :-251, 16:20], 0).mean(dim=0).cpu().numpy()
            dc_reward_t = dc_eval_rewards[:,:-251].mean(dim=0).cpu().numpy()

            rl_crash_pos_error_t = torch.norm(rl_crash_eval[:, :-251, -7:-4] - rl_crash_eval[:, :-251, 13:16], dim=-1).mean(dim=0).cpu().numpy()
            rl_crash_yaw_error_t = math_utils.yaw_error_from_quats(rl_crash_eval[:, :-251, -4:], rl_crash_eval[:, :-251, 16:20], 0).mean(dim=0).cpu().numpy()
            rl_crash_reward_t = rl_crash_eval_rewards[:,:-251].mean(dim=0).cpu().numpy()

            rl_COM_pos_error_t = torch.norm(rl_COM_eval[:, :-251, -7:-4] - rl_COM_eval[:, :-251, 13:16], dim=-1).mean(dim=0).cpu().numpy()
            rl_COM_yaw_error_t = math_utils.yaw_error_from_quats(rl_COM_eval[:, :-251, -4:], rl_COM_eval[:, :-251, 16:20], 0).mean(dim=0).cpu().numpy()
            rl_COM_reward_t = rl_COM_eval_rewards[:,:-251].mean(dim=0).cpu().numpy()

            pos_errors[i, j, :] = pos_error_t
            yaw_errors[i, j, :] = yaw_error_t
            rewards[i, j, :] = reward_t

            pos_errors_dc[i, j, :] = dc_pos_error_t
            yaw_errors_dc[i, j, :] = dc_yaw_error_t
            rewards_dc[i, j, :] = dc_reward_t

            pos_errors_rl_crash[i, j, :] = rl_crash_pos_error_t
            yaw_errors_rl_crash[i, j, :] = rl_crash_yaw_error_t
            rewards_rl_crash[i, j, :] = rl_crash_reward_t

            pos_errors_rl_COM[i, j, :] = rl_COM_pos_error_t
            yaw_errors_rl_COM[i, j, :] = rl_COM_yaw_error_t
            rewards_rl_COM[i, j, :] = rl_COM_reward_t

            # find the steady state error, and then compute the time to reach threshold*steady_state_error
            pos_ss_error = pos_error_t[-1]
            yaw_ss_error = yaw_error_t[-1]
            pos_error_threshold = np.max(pos_error_t)  - 0.95 * (np.max(pos_error_t) - pos_ss_error)
            yaw_error_threshold = np.max(yaw_error_t)  - 0.95 * (np.max(yaw_error_t) - yaw_ss_error)
            pos_settling_time_idx = (np.flatnonzero(pos_error_t > pos_error_threshold)[-1] + 1) * 0.02
            # pos_settling_time_idx = ((np.flatnonzero(np.abs(pos_error_t - pos_ss_error) > 0.05*pos_ss_error))[-1] + 1) * 0.02
            yaw_settling_time_idx = (np.flatnonzero(yaw_error_t > yaw_error_threshold)[-1] + 1) * 0.02

            dc_pos_ss_error = dc_pos_error_t[-1]
            dc_yaw_ss_error = dc_yaw_error_t[-1]
            dc_pos_error_threshold = np.max(dc_pos_error_t)  - 0.95 * (np.max(dc_pos_error_t) - dc_pos_ss_error)
            dc_yaw_error_threshold = np.max(dc_yaw_error_t)  - 0.95 * (np.max(dc_yaw_error_t) - dc_yaw_ss_error)

            rl_crash_pos_ss_error = rl_crash_pos_error_t[-1]
            rl_crash_yaw_ss_error = rl_crash_yaw_error_t[-1]
            rl_crash_pos_error_threshold = np.max(rl_crash_pos_error_t)  - 0.95 * (np.max(rl_crash_pos_error_t) - rl_crash_pos_ss_error)
            rl_crash_yaw_error_threshold = np.max(rl_crash_yaw_error_t)  - 0.95 * (np.max(rl_crash_yaw_error_t) - rl_crash_yaw_ss_error)
            # rl_crash_pos_settling_time_idx = (np.flatnonzero(rl_crash_pos_error_t > rl_crash_pos_error_threshold)[-1] + 1) * 0.02
            # rl_crash_yaw_settling_time_idx = (np.flatnonzero(rl_crash_yaw_error_t > rl_crash_yaw_error_threshold)[-1] + 1) * 0.02

            rl_COM_pos_ss_error = rl_COM_pos_error_t[-1]
            rl_COM_yaw_ss_error = rl_COM_yaw_error_t[-1]
            rl_COM_pos_error_threshold = np.max(rl_COM_pos_error_t)  - 0.95 * (np.max(rl_COM_pos_error_t) - rl_COM_pos_ss_error)
            rl_COM_yaw_error_threshold = np.max(rl_COM_yaw_error_t)  - 0.95 * (np.max(rl_COM_yaw_error_t) - rl_COM_yaw_ss_error)


            try:
                dc_pos_settling_time_idx = (np.flatnonzero(dc_pos_error_t > dc_pos_error_threshold)[-1] + 1) * 0.02
            except:
                dc_pos_settling_time_idx = 0
            try:
                dc_yaw_settling_time_idx = (np.flatnonzero(dc_yaw_error_t > dc_yaw_error_threshold)[-1] + 1) * 0.02
            except:
                dc_yaw_settling_time_idx = 0

            try:
                rl_crash_pos_settling_time_idx = (np.flatnonzero(rl_crash_pos_error_t > rl_crash_pos_error_threshold)[-1] + 1) * 0.02
            except:
                rl_crash_pos_settling_time_idx = 0
            try:
                rl_crash_yaw_settling_time_idx = (np.flatnonzero(rl_crash_yaw_error_t > rl_crash_yaw_error_threshold)[-1] + 1) * 0.02
            except:
                rl_crash_yaw_settling_time_idx = 0

            try:
                rl_COM_pos_settling_time_idx = (np.flatnonzero(rl_COM_pos_error_t > rl_COM_pos_error_threshold)[-1] + 1) * 0.02
            except:
                rl_COM_pos_settling_time_idx = 0
            try:
                rl_COM_yaw_settling_time_idx = (np.flatnonzero(rl_COM_yaw_error_t > rl_COM_yaw_error_threshold)[-1] + 1) * 0.02
            except:
                rl_COM_yaw_settling_time_idx = 0

            # append to lists:
            rl_pos_settling_times[i, j] = pos_settling_time_idx
            rl_yaw_settling_times[i, j] = yaw_settling_time_idx
            rl_pos_errors[i, j] = pos_ss_error
            rl_yaw_errors[i, j] = yaw_ss_error

            dc_pos_settling_times[i, j] = dc_pos_settling_time_idx
            dc_yaw_settling_times[i, j] = dc_yaw_settling_time_idx
            dc_pos_errors[i, j] = dc_pos_ss_error
            dc_yaw_errors[i, j] = dc_yaw_ss_error

            rl_crash_pos_settling_times[i, j] = rl_crash_pos_settling_time_idx
            rl_crash_yaw_settling_times[i, j] = rl_crash_yaw_settling_time_idx
            rl_crash_pos_errors[i, j] = rl_crash_pos_ss_error
            rl_crash_yaw_errors[i, j] = rl_crash_yaw_ss_error

            rl_COM_pos_settling_times[i, j] = rl_COM_pos_settling_time_idx
            rl_COM_yaw_settling_times[i, j] = rl_COM_yaw_settling_time_idx
            rl_COM_pos_errors[i, j] = rl_COM_pos_ss_error
            rl_COM_yaw_errors[i, j] = rl_COM_yaw_ss_error

            rl_pos_rmse[i,j] = np.sqrt(np.mean((pos_error_t)**2))
            rl_yaw_rmse[i,j] = np.sqrt(np.mean((yaw_error_t)**2))
            dc_pos_rmse[i,j] = np.sqrt(np.mean((dc_pos_error_t)**2))
            dc_yaw_rmse[i,j] = np.sqrt(np.mean((dc_yaw_error_t)**2))
            rl_crash_pos_rmse[i,j] = np.sqrt(np.mean((rl_crash_pos_error_t)**2))
            rl_crash_yaw_rmse[i,j] = np.sqrt(np.mean((rl_crash_yaw_error_t)**2))
            rl_COM_pos_rmse[i,j] = np.sqrt(np.mean((rl_COM_pos_error_t)**2))
            rl_COM_yaw_rmse[i,j] = np.sqrt(np.mean((rl_COM_yaw_error_t)**2))

    # make scatter plot
    # color will be by controller (blue for now since just RL)
    # position conditions will be circle, triangle, square
    # yaw conditions will be filled none, left, full
    from matplotlib.markers import MarkerStyle
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.5), dpi=1000)
    for i, pos_condition in enumerate(pos_conditions):
        for j, yaw_condition in enumerate(yaw_conditions):
            marker_type = "o" if pos_condition == "none" else "^" if pos_condition == "half" else "s"
            marker_fill = "full" if yaw_condition == "full" else "left" if yaw_condition == "half" else "none"
            marker_options = MarkerStyle(marker=marker_type, fillstyle=marker_fill)
            
            axs[0].scatter(rl_pos_errors[i, j], rl_pos_settling_times[i, j], label="RL", marker=marker_options, color="tab:blue", edgecolors="face")
            axs[1].scatter(rl_yaw_errors[i, j], rl_yaw_settling_times[i, j], label="RL", marker=marker_options, color="tab:blue", edgecolors="face")
            # axs[0].scatter(rl_pos_rmse[i, j], rl_pos_settling_times[i, j], label="RL", marker=marker_options, color="tab:blue", edgecolors="face")
            # axs[1].scatter(rl_yaw_rmse[i, j], rl_yaw_settling_times[i, j], label="RL", marker=marker_options, color="tab:blue", edgecolors="face")

            axs[0].scatter(dc_pos_errors[i, j], dc_pos_settling_times[i, j], label="DC", marker=marker_options, color="tab:orange", edgecolors="face")
            axs[1].scatter(dc_yaw_errors[i, j], dc_yaw_settling_times[i, j], label="DC", marker=marker_options, color="tab:orange", edgecolors="face")
            # axs[0].scatter(dc_pos_rmse[i, j], dc_pos_settling_times[i, j], label="DC", marker=marker_options, color="tab:orange", edgecolors="face")
            # axs[1].scatter(dc_yaw_rmse[i, j], dc_yaw_settling_times[i, j], label="DC", marker=marker_options, color="tab:orange", edgecolors="face")

            # axs[0].scatter(rl_crash_pos_errors[i, j], rl_crash_pos_settling_times[i, j], label="RL Crash", marker=marker_options, color="tab:green", edgecolors="face")
            # axs[1].scatter(rl_crash_yaw_errors[i, j], rl_crash_yaw_settling_times[i, j], label="RL Crash", marker=marker_options, color="tab:green", edgecolors="face")
            # axs[0].scatter(rl_crash_pos_rmse[i, j], rl_crash_pos_settling_times[i, j], label="RL Crash", marker=marker_options, color="tab:green", edgecolors="face")
            # axs[1].scatter(rl_crash_yaw_rmse[i, j], rl_crash_yaw_settling_times[i, j], label="RL Crash", marker=marker_options, color="tab:green", edgecolors="face")

            axs[0].scatter(rl_COM_pos_errors[i, j], rl_COM_pos_settling_times[i, j], label="RL COM", marker=marker_options, color="tab:purple", edgecolors="face")
            axs[1].scatter(rl_COM_yaw_errors[i, j], rl_COM_yaw_settling_times[i, j], label="RL COM", marker=marker_options, color="tab:purple", edgecolors="face")
            # axs[0].scatter(rl_COM_pos_rmse[i, j], rl_COM_pos_settling_times[i, j], label="RL COM", marker=marker_options, color="tab:purple", edgecolors="face")
            # axs[1].scatter(rl_COM_yaw_rmse[i, j], rl_COM_yaw_settling_times[i, j], label="RL COM", marker=marker_options, color="tab:purple", edgecolors="face")
    
    axs[0].set_xlabel("Steady State Position Error (m)")
    # axs[0].set_xlabel("RMSE Position Error (m)")
    axs[0].set_ylabel("Settling Time (s)")
    axs[1].set_xlabel("Steady State Yaw Error (rad)")
    # axs[1].set_xlabel("RMSE Yaw Error (rad)")
    axs[1].set_ylabel("Settling Time (s)")

    legend_elements = [ Line2D([0], [0], marker='o', color='tab:blue', label='RL'),
                        Line2D([0], [0], marker='o', color='tab:orange', label='DC'),
                        # Line2D([0], [0], marker='o', color='tab:green', label='RL Crash'),
                        Line2D([0], [0], marker='o', color='tab:purple', label='RL COM'),
                        # Patch(facecolor='none', edgecolor='none', fill=False, label=''),

                        # Line2D([0], [0], marker='D', color='tab:green', label='No Yaw', fillstyle='none', markerfacecolor='tab:green', markeredgecolor='black'),
                        # Line2D([0], [0], marker='D', color='tab:green', label='Half Yaw', fillstyle='left', markerfacecolor='tab:green', markeredgecolor='black'),
                        # Line2D([0], [0], marker='D', color='tab:green', label='Full Yaw', fillstyle='full', markerfacecolor='tab:green', markeredgecolor='black'),
                        Line2D([0], [0], marker='D', color='tab:gray', label='No Yaw', fillstyle='none', markerfacecolor='tab:gray'),
                        Line2D([0], [0], marker='D', color='tab:gray', label='Half Yaw', fillstyle='left', markerfacecolor='tab:gray'),
                        Line2D([0], [0], marker='D', color='tab:gray', label='Full Yaw', fillstyle='full', markerfacecolor='tab:gray'),
                        
                        Line2D([0], [0], marker='o', color='tab:gray', label='No Pos', markerfacecolor='tab:gray'),
                        Line2D([0], [0], marker='^', color='tab:gray', label='Half Pos', markerfacecolor='tab:gray'),
                        Line2D([0], [0], marker='s', color='tab:gray', label='Full Pos', markerfacecolor='tab:gray')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.25))
    # fig.legend()
    plt.tight_layout()
    plt.savefig("rl_pareto.pdf", dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig("rl_pareto.png", dpi=1000, format='png', bbox_inches='tight')

    fig, axs = plt.subplots(3,3,dpi=1000)
    time_axis = np.arange(0, 249) * 0.02
    for i, pos_condition in enumerate(pos_conditions):
        for j, yaw_condition in enumerate(yaw_conditions):
            axs[i, j].plot(time_axis, pos_errors[i, j, :], label="RL")
            axs[i, j].plot(time_axis, pos_errors_dc[i, j, :], label="DC")
            # axs[i, j].plot(time_axis, pos_errors_rl_crash[i, j, :], label="RL Crash")
            axs[i, j].plot(time_axis, pos_errors_rl_COM[i, j, :], label="RL COM")
        
    axs[0,0].set_ylabel("Pos: None")
    axs[1,0].set_ylabel("Pos: Half")
    axs[2,0].set_ylabel("Pos: Full")
    axs[2,0].set_xlabel("Time (s)")
    axs[2,1].set_xlabel("Time (s)")
    axs[2,2].set_xlabel("Time (s)")
    axs[0,0].set_title("Yaw: None")
    axs[0,1].set_title("Yaw: Half")
    axs[0,2].set_title("Yaw: Full")
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.suptitle("Position Error (m)")
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    plt.savefig("rl_pareto_curves_pos.pdf", dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig("rl_pareto_curves_pos.png", dpi=1000, format='png', bbox_inches='tight')

    fig, axs = plt.subplots(3,3,dpi=1000)
    time_axis = np.arange(0, 249) * 0.02
    for i, pos_condition in enumerate(pos_conditions):
        for j, yaw_condition in enumerate(yaw_conditions):
            axs[i, j].plot(time_axis, yaw_errors[i, j, :], label="RL")
            axs[i, j].plot(time_axis, yaw_errors_dc[i, j, :], label="DC")
            # axs[i, j].plot(time_axis, yaw_errors_rl_crash[i, j, :], label="RL Crash")
            axs[i, j].plot(time_axis, yaw_errors_rl_COM[i, j, :], label="RL COM")
        
    axs[0,0].set_ylabel("Pos: None")
    axs[1,0].set_ylabel("Pos: Half")
    axs[2,0].set_ylabel("Pos: Full")
    axs[2,0].set_xlabel("Time (s)")
    axs[2,1].set_xlabel("Time (s)")
    axs[2,2].set_xlabel("Time (s)")
    axs[0,0].set_title("Yaw: None")
    axs[0,1].set_title("Yaw: Half")
    axs[0,2].set_title("Yaw: Full")
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.suptitle("Yaw Error (rads)")
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    plt.savefig("rl_pareto_curves_yaw.pdf", dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig("rl_pareto_curves_yaw.png", dpi=1000, format='png', bbox_inches='tight')



    # fig, axs = plt.subplots(3, 2, dpi=1000)
    # time_axis = np.arange(0, 499) * 0.02
    # for i, pos_condition in enumerate(pos_conditions):
    #     axs[i, 0].plot(time_axis, pos_errors[i, 0, :], label="Yaw=None")
    #     axs[i, 0].plot(time_axis, pos_errors[i, 1, :], label="Yaw=Half")
    #     axs[i, 0].plot(time_axis, pos_errors[i, 2, :], label="Yaw=Full")

    #     axs[i, 1].plot(time_axis, yaw_errors[i, 0, :], label="Yaw=None")
    #     axs[i, 1].plot(time_axis, yaw_errors[i, 1, :], label="Yaw=Half")
    #     axs[i, 1].plot(time_axis, yaw_errors[i, 2, :], label="Yaw=Full")

    #     axs[i, 0].set_ylabel("pos=" + pos_condition)
    
    # axs[0,0].set_title("Pos Error (m)")
    # axs[0,1].set_title("Yaw Error (rad)")
    # plt.tight_layout()
    # handles, labels = axs[0,0].get_legend_handles_labels()
    # fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.25))
    # plt.savefig("rl_pareto_curves.pdf", dpi=1000, format='pdf', bbox_inches='tight')
    # plt.savefig("rl_pareto_curves.png", dpi=1000, format='png', bbox_inches='tight')

    plt.figure()
    for i, pos_condition in enumerate(pos_conditions):
        for j, yaw_condition in enumerate(yaw_conditions):
            # plt.plot(time_axis, rewards[i, j, :], label="<" + pos_condition + ", " + yaw_condition + ">")
            # plt.plot(time_axis, rewards_dc[i, j, :], label="<" + pos_condition + ", " + yaw_condition + ">")
            plt.plot(time_axis, rewards_rl_crash[i, j, :], label="<" + pos_condition + ", " + yaw_condition + ">")
    plt.xlabel("Time (s)")
    plt.ylabel("Rewards")
    plt.title("<pos_rand, yaw_rand> Rewards")
    plt.legend(loc="best")
    plt.savefig("rl_pareto_rewards.pdf", dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig("rl_pareto_rewards.png", dpi=1000, format='png', bbox_inches='tight')

    
    rl_errors = torch.norm(all_rl_trials[:, :, :, :, -7:-4] - all_rl_trials[:, :, :, :, 13:16], dim=-1)
    dc_errors = torch.norm(all_dc_trials[:, :, :, :, -7:-4] - all_dc_trials[:, :, :, :, 13:16], dim=-1)
    rl_crash_errors = torch.norm(all_rl_crash_trials[:, :, :, :, -7:-4] - all_rl_crash_trials[:, :, :, :, 13:16], dim=-1)
    rl_COM_errors = torch.norm(all_rl_COM_trials[:, :, :, :, -7:-4] - all_rl_COM_trials[:, :, :, :, 13:16], dim=-1)

    rl_yaw_errors = math_utils.yaw_error_from_quats(all_rl_trials[:, :, :, :, -4:], all_rl_trials[:, :, :, :, 16:20], 0)
    dc_yaw_errors = math_utils.yaw_error_from_quats(all_dc_trials[:, :, :, :, -4:], all_dc_trials[:, :, :, :, 16:20], 0)
    rl_crash_yaw_errors = math_utils.yaw_error_from_quats(all_rl_crash_trials[:, :, :, :, -4:], all_rl_crash_trials[:, :, :, :, 16:20], 0)
    rl_COM_yaw_errors = math_utils.yaw_error_from_quats(all_rl_COM_trials[:, :, :, :, -4:], all_rl_COM_trials[:, :, :, :, 16:20], 0)


    time_limit = 4
    time_limit_index = time_limit * 50

    rl_less_than_gc = (dc_errors - rl_errors)
    dc_rl_difference_mean = rl_less_than_gc.mean(dim=2)
    dc_rl_difference_std = rl_less_than_gc.std(dim=2)
    dc_rl_difference_quantiles = torch.quantile(rl_less_than_gc, torch.tensor([0.25, 0.5, 0.75]), dim=2)
    rl_less_than_rl_com = (rl_COM_errors - rl_errors)
    rl_com_rl_difference_mean = rl_less_than_rl_com.mean(dim=2)
    rl_com_rl_difference_std = rl_less_than_rl_com.std(dim=2)
    rl_com_rl_difference_quantiles = torch.quantile(rl_less_than_rl_com, torch.tensor([0.25, 0.5, 0.75]), dim=2)

    print("Quantile: ", dc_rl_difference_quantiles.shape)

    print(rl_errors.shape)
    print(rl_less_than_gc.shape)
    print(dc_rl_difference_mean.shape)
    time_axis = np.arange(0, time_limit_index-1) * 0.02
    fig, axs = plt.subplots(3, 3, dpi=1000)
    for i, pos_condition in enumerate(pos_conditions):
        for j, yaw_condition in enumerate(yaw_conditions):
            axs[i, j].plot(time_axis, dc_rl_difference_mean[i, j, :time_limit_index-1], label="DC - RL")
            axs[i, j].fill_between(time_axis, dc_rl_difference_mean[i, j, :time_limit_index-1] - dc_rl_difference_std[i, j, :time_limit_index-1], dc_rl_difference_mean[i, j, :time_limit_index-1] + dc_rl_difference_std[i, j, :time_limit_index-1], alpha=0.2)
            axs[i, j].plot(time_axis, rl_com_rl_difference_mean[i, j, :time_limit_index-1], label="RL COM - RL")
            axs[i, j].fill_between(time_axis, rl_com_rl_difference_mean[i, j, :time_limit_index-1] - rl_com_rl_difference_std[i, j, :time_limit_index-1], rl_com_rl_difference_mean[i, j, :time_limit_index-1] + rl_com_rl_difference_std[i, j, :time_limit_index-1], alpha=0.2)
    
    axs[0,0].set_ylabel("Pos: None")
    axs[1,0].set_ylabel("Pos: Half")
    axs[2,0].set_ylabel("Pos: Full")
    axs[2,0].set_xlabel("Time (s)")
    axs[2,1].set_xlabel("Time (s)")
    axs[2,2].set_xlabel("Time (s)")
    axs[0,0].set_title("Yaw: None")
    axs[0,1].set_title("Yaw: Half")
    axs[0,2].set_title("Yaw: Full")
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.suptitle("RL Relative Win Pos Error (m)")
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    # plt.plot(time_axis, dc_rl_difference_mean[2, 2, :time_limit_index-1], label = "DC - RL")
    # plt.fill_between(time_axis, dc_rl_difference_mean[2, 2, :time_limit_index-1] - dc_rl_difference_std[2, 2, :time_limit_index-1], dc_rl_difference_mean[2, 2, :time_limit_index-1] + dc_rl_difference_std[2, 2, :time_limit_index-1], alpha=0.2)
    # plt.plot(time_axis, rl_com_rl_difference_mean[2, 2, :time_limit_index-1], label = "RL COM - RL")
    # plt.fill_between(time_axis, rl_com_rl_difference_mean[2, 2, :time_limit_index-1] - rl_com_rl_difference_std[2, 2, :time_limit_index-1], rl_com_rl_difference_mean[2, 2, :time_limit_index-1] + rl_com_rl_difference_std[2, 2, :time_limit_index-1], alpha=0.2)
    # plt.xlabel("Time (s)")
    # plt.ylabel("RL Relative Win Position Error (m)")
    # plt.legend(loc="best")
    plt.savefig("rl_dc_difference_pos.png", dpi=1000, format='png', bbox_inches='tight')

    rl_less_than_gc_yaw = (dc_yaw_errors - rl_yaw_errors)
    dc_rl_difference_mean_yaw = rl_less_than_gc_yaw.mean(dim=2)
    dc_rl_difference_std_yaw = rl_less_than_gc_yaw.std(dim=2)
    dc_rl_difference_quantiles_yaw = torch.quantile(rl_less_than_gc_yaw, torch.tensor([0.25, 0.5, 0.75]), dim=2)
    rl_less_than_rl_com_yaw = (rl_COM_yaw_errors - rl_yaw_errors)
    rl_com_rl_difference_mean_yaw = rl_less_than_rl_com_yaw.mean(dim=2)
    rl_com_rl_difference_std_yaw = rl_less_than_rl_com_yaw.std(dim=2)
    rl_com_rl_difference_quantiles_yaw = torch.quantile(rl_less_than_rl_com_yaw, torch.tensor([0.25, 0.5, 0.75]), dim=2)
    fig, axs = plt.subplots(3, 3, dpi=1000)
    for i, pos_condition in enumerate(pos_conditions):
        for j, yaw_condition in enumerate(yaw_conditions):
            axs[i, j].plot(time_axis, dc_rl_difference_mean_yaw[i, j, :time_limit_index-1], label="DC - RL")
            axs[i, j].fill_between(time_axis, dc_rl_difference_mean_yaw[i, j, :time_limit_index-1] - dc_rl_difference_std_yaw[i, j, :time_limit_index-1], dc_rl_difference_mean_yaw[i, j, :time_limit_index-1] + dc_rl_difference_std_yaw[i, j, :time_limit_index-1], alpha=0.2)
            axs[i, j].plot(time_axis, rl_com_rl_difference_mean_yaw[i, j, :time_limit_index-1], label="RL COM - RL")
            axs[i, j].fill_between(time_axis, rl_com_rl_difference_mean_yaw[i, j, :time_limit_index-1] - rl_com_rl_difference_std_yaw[i, j, :time_limit_index-1], rl_com_rl_difference_mean_yaw[i, j, :time_limit_index-1] + rl_com_rl_difference_std_yaw[i, j, :time_limit_index-1], alpha=0.2)
    
    axs[0,0].set_ylabel("Pos: None")
    axs[1,0].set_ylabel("Pos: Half")
    axs[2,0].set_ylabel("Pos: Full")
    axs[2,0].set_xlabel("Time (s)")
    axs[2,1].set_xlabel("Time (s)")
    axs[2,2].set_xlabel("Time (s)")
    axs[0,0].set_title("Yaw: None")
    axs[0,1].set_title("Yaw: Half")
    axs[0,2].set_title("Yaw: Full")
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.suptitle("RL Relative Win Yaw Error (rad)")
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    plt.savefig("rl_dc_difference_yaw.png", dpi=1000, format='png', bbox_inches='tight')

    # Plot train condition (2, 2) for pos and yaw
    variance_method = "std"
    # variance_method = "iqr"
    fig, axs = plt.subplots(1, 2, dpi=1000)
    
    if variance_method == "std":
        axs[0].plot(time_axis, dc_rl_difference_mean[2, 2, :time_limit_index-1], label = "DC vs. RL(EE)")
        axs[0].fill_between(time_axis, dc_rl_difference_mean[2, 2, :time_limit_index-1] - dc_rl_difference_std[2, 2, :time_limit_index-1], dc_rl_difference_mean[2, 2, :time_limit_index-1] + dc_rl_difference_std[2, 2, :time_limit_index-1], alpha=0.2)
        axs[0].plot(time_axis, rl_com_rl_difference_mean[2, 2, :time_limit_index-1], label = "RL(COM) vs. RL(EE)")
        axs[0].fill_between(time_axis, rl_com_rl_difference_mean[2, 2, :time_limit_index-1] - rl_com_rl_difference_std[2, 2, :time_limit_index-1], rl_com_rl_difference_mean[2, 2, :time_limit_index-1] + rl_com_rl_difference_std[2, 2, :time_limit_index-1], alpha=0.2)
        axs[1].plot(time_axis, dc_rl_difference_mean_yaw[2, 2, :time_limit_index-1], label = "DC vs. RL(EE)")
        axs[1].fill_between(time_axis, dc_rl_difference_mean_yaw[2, 2, :time_limit_index-1] - dc_rl_difference_std_yaw[2, 2, :time_limit_index-1], dc_rl_difference_mean_yaw[2, 2, :time_limit_index-1] + dc_rl_difference_std_yaw[2, 2, :time_limit_index-1], alpha=0.2)
        axs[1].plot(time_axis, rl_com_rl_difference_mean_yaw[2, 2, :time_limit_index-1], label = "RL(COM) vs. RL(EE)")
        axs[1].fill_between(time_axis, rl_com_rl_difference_mean_yaw[2, 2, :time_limit_index-1] - rl_com_rl_difference_std_yaw[2, 2, :time_limit_index-1], rl_com_rl_difference_mean_yaw[2, 2, :time_limit_index-1] + rl_com_rl_difference_std_yaw[2, 2, :time_limit_index-1], alpha=0.2)
    elif variance_method == "iqr":
        axs[0].plot(time_axis, dc_rl_difference_quantiles[1, 2, 2, :time_limit_index-1], label = "DC vs. RL(EE)")
        axs[0].fill_between(time_axis, dc_rl_difference_quantiles[0, 2, 2, :time_limit_index-1], dc_rl_difference_quantiles[2, 2, 2, :time_limit_index-1], alpha=0.2)
        axs[0].plot(time_axis, rl_com_rl_difference_quantiles[1, 2, 2, :time_limit_index-1], label = "RL(COM) vs. RL(EE)")
        axs[0].fill_between(time_axis, rl_com_rl_difference_quantiles[0, 2, 2, :time_limit_index-1], rl_com_rl_difference_quantiles[2, 2, 2, :time_limit_index-1], alpha=0.2)
        axs[1].plot(time_axis, dc_rl_difference_quantiles_yaw[1, 2, 2, :time_limit_index-1], label = "DC vs. RL(EE)")
        axs[1].fill_between(time_axis, dc_rl_difference_quantiles_yaw[0, 2, 2, :time_limit_index-1], dc_rl_difference_quantiles_yaw[2, 2, 2, :time_limit_index-1], alpha=0.2)
        axs[1].plot(time_axis, rl_com_rl_difference_quantiles_yaw[1, 2, 2, :time_limit_index-1], label = "RL(COM) vs. RL(EE)")
        axs[1].fill_between(time_axis, rl_com_rl_difference_quantiles_yaw[0, 2, 2, :time_limit_index-1], rl_com_rl_difference_quantiles_yaw[2, 2, 2, :time_limit_index-1], alpha=0.2)

    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("RL(EE) Relative Win Position Error (m)")
    # axs[0].legend(loc="best")
    
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("RL(EE) Relative Win Yaw Error (rad)")
    # axs[1].legend(loc="best")
    handles, labels = axs[0].get_legend_handles_labels()
    plt.suptitle("Train Condition, shown with: " + variance_method.upper())
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.savefig("rl_dc_difference_pos_yaw_" + variance_method + ".png", dpi=1000, format='png', bbox_inches='tight')

    # Plot raw trials for the (2,2) condition. Left is DC-RL, right is RL(COM)-RL(EE)
    fig, axs = plt.subplots(1, 2, dpi=1000)
    axs[0].plot(time_axis, rl_less_than_gc[2,2,:, :time_limit_index-1].T, alpha=0.1, color="tab:blue")
    axs[1].plot(time_axis, rl_less_than_rl_com[2,2,:, :time_limit_index-1].T, alpha=0.1, color="tab:orange")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("DC - RL(EE) Position Error (m)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("RL(COM) - RL(EE) Position Error (m)")
    plt.tight_layout()
    plt.savefig("rl_dc_difference_pos_raw.png", dpi=1000, format='png', bbox_inches='tight')

    # Plot raw trials for the (2,2) condition. Left is DC-RL, right is RL(COM)-RL(EE)
    fig, axs = plt.subplots(1, 2, dpi=1000)
    axs[0].plot(time_axis, rl_less_than_gc_yaw[2,2,:, :time_limit_index-1].T, alpha=0.1, color="tab:blue")
    axs[1].plot(time_axis, rl_less_than_rl_com_yaw[2,2,:, :time_limit_index-1].T, alpha=0.1, color="tab:orange")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("DC - RL(EE) Yaw Error (rad)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("RL(COM) - RL(EE) Yaw Error (rad)")
    plt.tight_layout()
    plt.savefig("rl_dc_difference_yaw_raw.png", dpi=1000, format='png', bbox_inches='tight')



def load_single_rl_data(rl_eval_path, min_reward = 0.0, max_reward = 15.0):
    generalized_data_path = "eval_full_states.pt"
    generalized_rewards_path = "eval_rewards.pt"
    
    M = 100
    T = 500

    # Load RL Full Data
    generalized_data = torch.zeros(M, T, 33)
    generalized_rewards = torch.zeros(M, T)


    single_trial_generalized = torch.load(os.path.join(rl_eval_path, generalized_data_path))
    single_trial_rewards = torch.load(os.path.join(rl_eval_path, generalized_rewards_path))

    generalized_data[:,:,:] = single_trial_generalized
    generalized_rewards[:,:] = single_trial_rewards

    generalized_rl_pos_norm = torch.norm(generalized_data[:, :-1, -7:-4] - generalized_data[:, :-1, 13:16], dim=-1)
    generalized_rl_yaw_error = math_utils.yaw_error_from_quats(generalized_data[:, :-1, -4:], generalized_data[:, :-1, 16:20], 0)
    
    generalized_rewards = (generalized_rewards[:,:-1] - min_reward) / (max_reward - min_reward)

    generalized_mean_pos_error = generalized_rl_pos_norm.mean(dim=0).cpu().numpy()
    generalized_mean_yaw_error = generalized_rl_yaw_error.mean(dim=0).cpu().numpy()
    generalized_mean_rewards = generalized_rewards.mean(dim=0).cpu().numpy()
    
    return generalized_mean_pos_error, generalized_mean_yaw_error, generalized_mean_rewards


def plot_rl_trials():
    rl_exp_labels = [
                    #  "pos bonus (15)", 
                    #  "DC EE LQR",
                    #  "stay alive (2)", 
                    #  "stay_alive (2) squared", 
                    #  "stay alive (5)", 
                    #  "crash penalty (2)", 
                    #  "crash penalty (5)",
                    # "crash penalty normalized (2)",
                    # "combined exp bonus (0.8)",
                    # "combined exp bonus (0.8) no vel",
                    # "COM crash (2)",
                    # r"AP($\alpha$, tol)=(2, 1.5)",
                    # r"AP($\alpha$, tol)=(10, 0.5)(high vel)",
                    # r"AP($\alpha$, tol)=(10, 0.5)(low vel)",
                    # r"AP($\alpha$, tol)=(10, 0.5)",
                    # r"AP($\alpha$, tol)=(20, 1.0)",
                    # r"AP($\alpha$, tol)=(3, 1.0)",
                    # r"AP($\alpha$, tol)=(50, 0.6)",
                    r"AP($\alpha$, tol)=(3, 1.0)(low vel)",
                    r"AP($\alpha$, tol)=(50, 0.3)",
                    r"AP($\alpha$, tol)=(3, 0.6)(low vel)",
                    r"AP($\alpha$, tol)=(50, 0.3)(low vel)",
                    r"AP($\alpha$, tol)=(1, 0.1)(high vel)",
                    # "combined radius 0.1",
                    # "combined radius 1.0",
                    ]

    rl_seeds_list = []
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-10-01_17-52-33_new_defaults_anneal_50e6_seed_0/")
    # rl_seeds_list.append("../rl/baseline_0dof_ee_lqr_tune/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0_DOF/2024-10-04_12-26-01_LQR_reward_-2_pos_-2_yaw_stay_alive_10/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0_DOF/2024-10-07_12-33-16_LQR_reward_squared_-2_pos_-2_yaw_stay_alive_10/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0_DOF/2024-10-07_13-08-43_LQR_reward_-5_pos_-5_yaw_stay_alive_10/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0_DOF/2024-10-07_13-27-55_LQR_reward_-2_pos_-2_yaw_-10_crash/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0_DOF/2024-10-07_14-44-21_LQR_reward_-5_pos_-5_yaw_-10_crash/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0_DOF/2024-10-07_16-05-19_LQR_reward_normalized_yaw_-2_pos_-2_yaw_-10_crash/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0_DOF/2024-10-07_16-12-06_combined_pos_and_yaw_1.0_radius_0.8/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0_DOF/2024-10-07_16-46-48_combined_pos_and_yaw_1.0_radius_0.8_no_vel/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_COM/2024-10-07_18-59-53_LQR_-2_pos_-2_yaw_-10_crash_COM_all_bodies_reward_change/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_AP/2024-10-15_15-49-27_agile_test_alpha_2_tol_1.5/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_AP/2024-10-15_15-49-25_precision_test_alpha_10_tol_0.5_vel_-0.1_ang_vel_-0.1/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_AP/2024-10-15_16-08-12_precision_test_alpha_10_tol_0.5_vel_-0.05_ang_vel_-0.01/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_AP/2024-10-16_10-58-03_test_alpha_20_tol_1.0/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_AP/2024-10-16_10-58-04_test_alpha_3_tol_1.0/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_AP/2024-10-16_14-08-25_test_alpha_50_tol_0.6/")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_AP/2024-10-16_14-04-37_test_alpha_3_tol_1.0_original_vel_penalties/")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_AP/2024-10-16_15-55-13_test_alpha_50_tol_0.3/")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_AP/2024-10-16_15-50-55_test_alpha_3_tol_0.6_original_vel_penalties/")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_AP/2024-10-16_19-04-09_test_alpha_50_tol_0.3_original_vel_penalties/")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_AP/2024-10-16_19-01-01_test_alpha_1_tol_0.1_vel_-0.1_ang_vel_-0.05//")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0_DOF/2024-10-07_19-00-25_combined_pos_and_yaw_1.0_radius_0.1_jit/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0_DOF/2024-10-07_19-01-14_combined_pos_and_yaw_1.0_radius_1.0_jit/")

    time_axis = np.arange(0, 499) * 0.02


    fig, axs = plt.subplots(1, 2, dpi=1000)
    for i in range(len(rl_seeds_list)):
        if "crash" in rl_seeds_list[i]:
            generalized_mean_pos_error, generalized_mean_yaw_error, generalized_mean_rewards = load_single_rl_data(rl_seeds_list[i], -10.0, 0.0)
        else:
            generalized_mean_pos_error, generalized_mean_yaw_error, generalized_mean_rewards = load_single_rl_data(rl_seeds_list[i])
        axs[0].plot(time_axis, generalized_mean_pos_error, label=rl_exp_labels[i])
        axs[1].plot(time_axis, generalized_mean_yaw_error, label=rl_exp_labels[i])
    
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position Error (m)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Yaw Error (rad)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    # plt.savefig("rl_trials.pdf", dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig("rl_trials.png", dpi=1000, format='png', bbox_inches='tight')

def plot_case_study():
    rl_pos_squared_path = "../rl/logs/rsl_rl/AM_0DOF/2024-11-06_13-59-09_test_paper_params_no_yaw_error_in_obs/"
    rl_pos_not_squared_path = "../rl/logs/rsl_rl/AM_0DOF/2024-11-07_09-22-59_test_paper_params_no_yaw_pos_error_not_squared/"
    rl_com_path = "../rl/logs/rsl_rl/AM_0DOF/2024-11-07_10-32-36_test_paper_params_pos_error_not_squared_COM/"
    dc_path = "../rl/baseline_0dof_ee_reward_tune/"

    rl_pos_squared_data = torch.load(os.path.join(rl_pos_squared_path, "case_study_eval_full_states.pt"))
    rl_pos_not_squared_data = torch.load(os.path.join(rl_pos_not_squared_path, "case_study_eval_full_states.pt"))
    rl_com_data = torch.load(os.path.join(rl_com_path, "case_study_eval_full_states.pt"))
    dc_data = torch.load(os.path.join(dc_path, "case_study_eval_full_states.pt"))

    max_time = 200
    rl_pos_squared_pos_error = torch.norm(rl_pos_squared_data[:, :max_time, -7:-4] - rl_pos_squared_data[:, :max_time, 13:16], dim=-1)
    rl_pos_squared_yaw_error = math_utils.yaw_error_from_quats(rl_pos_squared_data[:, :max_time, -4:], rl_pos_squared_data[:, :max_time, 16:20], 0)
    rl_pos_not_squared_pos_error = torch.norm(rl_pos_not_squared_data[:, :max_time, -7:-4] - rl_pos_not_squared_data[:, :max_time, 13:16], dim=-1)
    rl_pos_not_squared_yaw_error = math_utils.yaw_error_from_quats(rl_pos_not_squared_data[:, :max_time, -4:], rl_pos_not_squared_data[:, :max_time, 16:20], 0)
    rl_com_pos_error = torch.norm(rl_com_data[:, :max_time, -7:-4] - rl_com_data[:, :max_time, 13:16], dim=-1)
    rl_com_yaw_error = math_utils.yaw_error_from_quats(rl_com_data[:, :max_time, -4:], rl_com_data[:, :max_time, 16:20], 0)
    dc_pos_error = torch.norm(dc_data[:, :max_time, -7:-4] - dc_data[:, :max_time, 13:16], dim=-1)
    dc_yaw_error = math_utils.yaw_error_from_quats(dc_data[:, :max_time, -4:], dc_data[:, :max_time, 16:20], 0)

    
    time_axis = np.arange(0, max_time) * 0.02
    fig, axs = plt.subplots(1, 2, dpi=1000)
    axs[0].plot(time_axis, rl_pos_squared_pos_error.mean(dim=0).cpu().numpy(), label="RL-EE", color="tab:blue")
    # axs[0].plot(time_axis, rl_pos_not_squared_pos_error.mean(dim=0).cpu().numpy(), label="RL Pos Not Squared")
    # axs[0].plot(time_axis, rl_com_pos_error.mean(dim=0).cpu().numpy(), label="RL-COM", color="tab:purple")
    axs[0].plot(time_axis, dc_pos_error.mean(dim=0).cpu().numpy(), label="DC", color="tab:orange")
    axs[1].plot(time_axis, rl_pos_squared_yaw_error.mean(dim=0).cpu().numpy(), label="RL-EE", color="tab:blue")
    # axs[1].plot(time_axis, rl_pos_not_squared_yaw_error.mean(dim=0).cpu().numpy(), label="RL Pos Not Squared")
    # axs[1].plot(time_axis, rl_com_yaw_error.mean(dim=0).cpu().numpy(), label="RL-COM", color="tab:purple")
    axs[1].plot(time_axis, dc_yaw_error.mean(dim=0).cpu().numpy(), label="DC", color="tab:orange")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position Error (m)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Yaw Error (rad)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    # plt.savefig("case_study.pdf", dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig("case_study.png", dpi=1000, format='png', bbox_inches='tight')

    
def plot_crazyflie_evals():
    rl_ee_path = "../rl/logs/rsl_rl/CrazyflieManipulator_CTATT/2024-11-18_16-07-09_full_ori_pos_anneal_50e6_prev_action_-0.1/"
    dc_path = "../rl/baseline_cf_0dof/"

if __name__ == "__main__":
    # rl_eval_path = "../rl/runs/AM_0DOF_hover_pos_and_yaw_yaw_error_scale_-0.1_custom_yaw_error_1/"
    # rl_eval_path = "../rl/runs/AM_0DOF_hover_pos_and_yaw_yaw_error_scale_-0.1_custom_yaw_func_anneal_lr_1/"
    # rl_eval_path = "../rl/runs/AM_0DOF_hover_pos_and_yaw_yaw_distance_scale_5_pos_distance_15_smooth_transition_1"

    # rl_eval_path = "../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_14-38-12_rsl_rl_test_default_1024_env_pos_distance_15_yaw_error_-2.0_no_smooth_transition_full_ori/"
    # rl_eval_path = "../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_17-44-44_paper_model_0/"

    # rl_eval_path = "../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-25_15-35-32_pos_radius_anneal_50e6_4096_envs_64_steps_15_pos_distance_-2_yaw_error_full_ori_matrix/"
    
    # rl_eval_path = "../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-28_16-15-34_z_moment_0.025_more_altitude"
    # dc_eval_path = "../rl/baseline_0dof_com_lqr_tune/"
    # plot_data(rl_eval_path, dc_eval_path)
    # plot_data(rl_eval_path, dc_eval_path, "case_study_")
    # exit()

    # plot_gc_tuning_variants()
    # exit()

    # plot_ball_catching(rl_eval_path, dc_eval_path)

    # plot_rl_pareto()
    # plot_rl_trials()
    plot_case_study()
    exit()



    rl_seeds_list = []
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_17-44-44_paper_model_0/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_17-45-04_paper_model_1/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_18-24-39_paper_model_2/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_18-24-40_paper_model_3/")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-14_18-24-50_paper_model_4/")
    
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-25_15-35-32_pos_radius_anneal_50e6_4096_envs_64_steps_15_pos_distance_-2_yaw_error_full_ori_matrix/")
    
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-27_12-02-10_radius_anneal_50e6_seed_0")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-27_12-03-42_radius_anneal_50e6_seed_1")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-09-27_12-02-17_radius_anneal_50e6_seed_2")

    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-10-01_13-44-46_new_defaults_4096_envs_64_steps_seed_0")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-10-01_13-48-06_new_defaults_4096_envs_64_steps_seed_1")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-10-01_13-48-19_new_defaults_4096_envs_64_steps_seed_2")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-10-01_13-47-05_new_defaults_4096_envs_64_steps_seed_3")
    # rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-10-01_13-48-41_new_defaults_4096_envs_64_steps_seed_4")

    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-10-01_17-52-33_new_defaults_anneal_50e6_seed_0")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-10-01_17-52-48_new_defaults_anneal_50e6_seed_1")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-10-01_17-53-03_new_defaults_anneal_50e6_seed_2")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-10-01_17-53-30_new_defaults_anneal_50e6_seed_3")
    rl_seeds_list.append("../rl/logs/rsl_rl/AM_0DOF_Hover/2024-10-01_17-53-40_new_defaults_anneal_50e6_seed_4")

    dc_eval_path = "../rl/baseline_0dof/"
    # dc_eval_path = "../rl/baseline_0dof_com_reward_tune/"
    plot_paper_figs(rl_seeds_list, dc_eval_path)

    plot_settling_time_and_error()

    # rl_eval_path = "../rl/runs/AM_0DOF_hover_pos_and_yaw_yaw_error_scale_-2.0_smooth_transition_rework_10_yaw_error_urdf_match_1"
    # dc_eval_path = "../rl/baseline_0dof/"
    # plot_traj_tracking(rl_eval_path, dc_eval_path, "eval_traj_track_1Hz", seed="0")