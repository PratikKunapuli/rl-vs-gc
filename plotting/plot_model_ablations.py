import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import itertools

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


import omni.isaac.lab.utils.math as isaac_math_utils
import utils.math_utilities as math_utils


from matplotlib import rc
rc('font', size=8)
rc('legend', fontsize=8)
rc('ytick', labelsize=6)
rc('xtick', labelsize=6)
sns.set_context("paper")
sns.set_theme()

import plotting.plotting_utils as plotting_utils
from plotting.plotting_utils import params

model_base_paths = {
    # "RL-EE Hover" : "../rl/logs/rsl_rl/Hover_PaperModels/2025-01-13_09-11-23_new_model_default_params_anneal_50e6/",
    "RL-EE Hover" : "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    "RL-EE TrajTrack" : "../rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/",
    "GC+FF TrajTrack" : "../rl/baseline_0dof_ee_reward_tune_with_ff/",
    "GC+FF Hover": "../rl/baseline_0dof_ee_reward_tune_with_ff_hover/",
    "GC Hover" : "../rl/baseline_0dof_ee_reward_tune_no_ff/",
    "GC PID Hover" : "../rl/baseline_0dof_ee_reward_tune_pid/",
    "GC Hand-Tuned" : "../rl/baseline_0dof_hand_tuned/",
}


def plot_model_ablations():

    # fig, axs = plt.subplots(1, 2, dpi=300)
    fig = plt.figure(layout="constrained", dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        """
    )
    axs = [axd['A'], axd['B'], axd['C'], axd['D'], axd['E'], axd['F']]

    for model_name, model_base_path in model_base_paths.items():
        model_hover_data = torch.load(model_base_path + "Hover_rand_init_eval_traj_track_50Hz_eval_full_states.pt" , weights_only=True).cpu()
        model_hover_rewards = torch.load(model_base_path + "Hover_rand_init_eval_traj_track_50Hz_eval_rewards.pt" , weights_only=True).cpu()

        model_traj_data = torch.load(model_base_path + "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt" , weights_only=True).cpu()
        model_traj_rewards = torch.load(model_base_path + "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_rewards.pt" , weights_only=True).cpu()

        # Normalize rewards between 0 and 1 corresponding to the min and 15.0
        # min_hover_reward = torch.min(model_hover_rewards)
        # min_traj_reward = torch.min(model_traj_rewards)
        # model_hover_rewards = (model_hover_rewards - min_hover_reward) / (15.0 - min_hover_reward)
        # model_traj_rewards = (model_traj_rewards - min_traj_reward) / (15.0 - min_traj_reward)

        hover_pos_error, hover_yaw_error = plotting_utils.get_errors(model_hover_data)
        traj_pos_error, traj_yaw_error = plotting_utils.get_errors(model_traj_data)


        hover_hatch = "" if "Hover" in model_name else "xx"
        traj_hatch = "" if "TrajTrack" in model_name else "xx"

        # Plot the rewards as a bar with error bars
        model_hover_reward_quantiles = plotting_utils.get_quantiles(model_hover_rewards.mean(dim=1), [0.25, 0.5, 0.75])
        model_traj_reward_quantiles = plotting_utils.get_quantiles(model_traj_rewards.mean(dim=1), [0.25, 0.5, 0.75])
        
        hover_reward_err_bars = plotting_utils.get_error_bars_from_quantiles(model_hover_reward_quantiles)
        traj_reward_err_bars = plotting_utils.get_error_bars_from_quantiles(model_traj_reward_quantiles)

        axs[0].bar(model_name, model_hover_reward_quantiles[1], yerr=hover_reward_err_bars, label=model_name, hatch=hover_hatch, capsize=2)
        axs[1].bar(model_name, model_traj_reward_quantiles[1], yerr=traj_reward_err_bars, label=model_name, hatch=traj_hatch, capsize=2)
    
        # Plot the Position RMSE as a bar with error bars
        hover_pos_rmse = plotting_utils.get_RMSE_from_error(hover_pos_error)
        traj_pos_rmse = plotting_utils.get_RMSE_from_error(traj_pos_error)
        hover_pos_rmse_quantiles = plotting_utils.get_quantiles(hover_pos_rmse, [0.25, 0.5, 0.75])
        traj_pos_rmse_quantiles = plotting_utils.get_quantiles(traj_pos_rmse, [0.25, 0.5, 0.75])
        axs[2].bar(model_name, hover_pos_rmse_quantiles[1], yerr=plotting_utils.get_error_bars_from_quantiles(hover_pos_rmse_quantiles), label=model_name, hatch=hover_hatch, capsize=2)
        axs[3].bar(model_name, traj_pos_rmse_quantiles[1], yerr=plotting_utils.get_error_bars_from_quantiles(traj_pos_rmse_quantiles), label=model_name, hatch=traj_hatch, capsize=2)

        # Plot the Yaw RMSE as a bar with error bars
        hover_yaw_rmse = plotting_utils.get_RMSE_from_error(hover_yaw_error)
        traj_yaw_rmse = plotting_utils.get_RMSE_from_error(traj_yaw_error)
        hover_yaw_rmse_quantiles = plotting_utils.get_quantiles(hover_yaw_rmse, [0.25, 0.5, 0.75])
        traj_yaw_rmse_quantiles = plotting_utils.get_quantiles(traj_yaw_rmse, [0.25, 0.5, 0.75])
        axs[4].bar(model_name, hover_yaw_rmse_quantiles[1], yerr=plotting_utils.get_error_bars_from_quantiles(hover_yaw_rmse_quantiles), label=model_name, hatch=hover_hatch, capsize=2)
        axs[5].bar(model_name, traj_yaw_rmse_quantiles[1], yerr=plotting_utils.get_error_bars_from_quantiles(traj_yaw_rmse_quantiles), label=model_name, hatch=traj_hatch, capsize=2)

    axs[0].set_title("Hover")
    axs[1].set_title("Lissajous Tracking")
    axs[0].set_ylabel("Normalized\nReward")
    # axs[1].set_ylabel("Normalized Reward")
    axs[0].set_ylim([0, 15.0])
    axs[0].set_yticks(np.linspace(0, 15.0, 3))
    axs[0].set_yticklabels(np.linspace(0, 1.0, 3))
    axs[1].set_ylim([0, 15.0])
    axs[1].set_yticks(np.linspace(0, 15.0, 3))
    axs[1].set_yticklabels(np.linspace(0, 1.0, 3))
    axs[1].yaxis.set_ticks_position("right")

    axs[2].set_ylabel("Position\nRMSE (m)")
    axs[2].set_ylim([0, 0.8])
    axs[2].set_yticks(np.linspace(0, 0.8, 3))
    axs[2].set_yticklabels(np.linspace(0, 0.8, 3))
    axs[3].set_ylim([0, 0.8])
    axs[3].set_yticks(np.linspace(0, 0.8, 3))
    axs[3].set_yticklabels(np.linspace(0, 0.8, 3))
    axs[3].yaxis.set_ticks_position("right")

    axs[4].set_ylabel("Yaw\nRMSE (rad)")
    axs[4].set_ylim([0, 1.2])
    axs[4].set_yticks(np.linspace(0, 1.2, 3))
    axs[4].set_yticklabels(np.linspace(0, 1.2, 3))
    axs[5].set_ylim([0, 1.2])
    axs[5].set_yticks(np.linspace(0, 1.2, 3))
    axs[5].set_yticklabels(np.linspace(0, 1.2, 3))
    axs[5].yaxis.set_ticks_position("right")

    axs[1].tick_params(axis='y', which='both', left=False, right=False)
    axs[3].tick_params(axis='y', which='both', left=False, right=False)
    axs[5].tick_params(axis='y', which='both', left=False, right=False)



    axs[0].yaxis.set_minor_locator(AutoMinorLocator(2))
    axs[0].grid(which='minor', linestyle=':', linewidth='1.0', color='white')
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(2))
    axs[1].grid(which='minor', linestyle=':', linewidth='1.0', color='white')
    axs[2].yaxis.set_minor_locator(AutoMinorLocator(2))
    axs[2].grid(which='minor', linestyle=':', linewidth='1.0', color='white')
    axs[3].yaxis.set_minor_locator(AutoMinorLocator(2))
    axs[3].grid(which='minor', linestyle=':', linewidth='1.0', color='white')
    axs[4].yaxis.set_minor_locator(AutoMinorLocator(2))
    axs[4].grid(which='minor', linestyle=':', linewidth='1.0', color='white')
    axs[5].yaxis.set_minor_locator(AutoMinorLocator(2))
    axs[5].grid(which='minor', linestyle=':', linewidth='1.0', color='white')


    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    axs[3].grid(True)
    axs[4].grid(True)
    axs[5].grid(True)
    
    #remove x ticks from both axes
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xticks([])
    axs[3].set_xticks([])
    axs[4].set_xticks([])
    axs[5].set_xticks([])

    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3)

    # plt.tight_layout()
    plt.savefig("model_ablations.png", bbox_inches='tight')
    # plt.show()


    


if __name__ == "__main__":
    plot_model_ablations()