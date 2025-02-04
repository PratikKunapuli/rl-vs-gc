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

quad_model_base_paths = {
    "RL-Opt.-Liss.-FF": "../rl/logs/rsl_rl/QuadOnly/2025-01-28_10-25-00_TrajTrack_previous_action/",
    # "RL-Opt.-Liss.-FF": "../rl/logs/rsl_rl/QuadOnly/2025-01-30_20-34-17_TrajTrack_with_FF/",
    "GC-Opt.-Liss.-FF": "../rl/baseline_quad_only_reward_tune/",
    # "RL-Opt.-Liss.-None": "../rl/logs/rsl_rl/QuadOnly/2025-01-30_20-11-54_TrajTrack_previous_action_no_FF/",
}

am_model_base_paths = {
    "RL-Opt.-Liss.-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/",
    # "RL-Opt-Liss.-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_18-50-03_RL_Lissajous_with_FF/",
    # "RL-Opt.-Liss.-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_18-50-03_RL_Lissajous_with_FF/",
    "GC-Opt.-Liss.-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_traj/",
    # "RL-Opt.-Liss.-None": "../rl/logs/rsl_rl/TrajTrack/2025-01-30_19-27-46_RL_Lissajous_no_FF/",
}

def plot_traj_tracking():
    fig = plt.figure(layout="constrained", dpi=500, figsize=(5, 4.5))
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        """
    )
    axs = [axd['A'], axd['B'], axd['C'], axd['D']]

    for model_name, base_paths in quad_model_base_paths.items():
        model_data = torch.load(base_paths + "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt", weights_only=True).cpu()
        model_rewards = torch.load(base_paths + "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_rewards.pt", weights_only=True).cpu()

        pos_error, yaw_error = plotting_utils.get_errors(model_data)
        pos_error_quantiles = plotting_utils.get_quantiles(pos_error, [0.25, 0.5, 0.75])
        yaw_error_quantiles = plotting_utils.get_quantiles(yaw_error, [0.25, 0.5, 0.75])

        axs[0].plot(pos_error_quantiles[1], label=model_name)
        axs[0].fill_between(np.arange(len(pos_error_quantiles[1])), pos_error_quantiles[0], pos_error_quantiles[2], alpha=0.2)
        axs[2].plot(yaw_error_quantiles[1], label=model_name)
        axs[2].fill_between(np.arange(len(yaw_error_quantiles[1])), yaw_error_quantiles[0], yaw_error_quantiles[2], alpha=0.2)

        reward = model_rewards.mean(dim=1)

        print("\nQuadrotor: ")
        print("Model name: ", model_name)
        print("Reward: ", reward.mean(dim=0), " +- ", reward.std(dim=0))
        print("Pos RMSE: ", plotting_utils.get_RMSE_from_error(pos_error[:,:200]).mean(dim=0), " +- ", plotting_utils.get_RMSE_from_error(pos_error[:,:200]).std(dim=0))
        print("Yaw RMSE: ", plotting_utils.get_RMSE_from_error(yaw_error[:,:200]).mean(dim=0), " +- ", plotting_utils.get_RMSE_from_error(yaw_error[:,:200]).std(dim=0))


    for model_name, base_paths in am_model_base_paths.items():
        model_data = torch.load(base_paths + "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt", weights_only=True).cpu()
        model_rewards = torch.load(base_paths + "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_rewards.pt", weights_only=True).cpu()

        # print(model_rewards.mean(dim=1).median(dim=0))

        pos_error, yaw_error = plotting_utils.get_errors(model_data)
        pos_error_quantiles = plotting_utils.get_quantiles(pos_error, [0.25, 0.5, 0.75])
        yaw_error_quantiles = plotting_utils.get_quantiles(yaw_error, [0.25, 0.5, 0.75])

        axs[1].plot(pos_error_quantiles[1], label=model_name)
        axs[1].fill_between(np.arange(len(pos_error_quantiles[1])), pos_error_quantiles[0], pos_error_quantiles[2], alpha=0.2)
        axs[3].plot(yaw_error_quantiles[1], label=model_name)
        axs[3].fill_between(np.arange(len(yaw_error_quantiles[1])), yaw_error_quantiles[0], yaw_error_quantiles[2], alpha=0.2)

        reward = model_rewards.mean(dim=1)

        print("\nAerial Manipulator: ")
        print("Model name: ", model_name)
        print("Reward: ", reward.mean(dim=0), " +- ", reward.std(dim=0))
        print("Pos RMSE: ", plotting_utils.get_RMSE_from_error(pos_error[:,:200]).mean(dim=0), " +- ", plotting_utils.get_RMSE_from_error(pos_error[:,:200]).std(dim=0))
        print("Yaw RMSE: ", plotting_utils.get_RMSE_from_error(yaw_error[:,:200]).mean(dim=0), " +- ", plotting_utils.get_RMSE_from_error(yaw_error[:,:200]).std(dim=0))

    axs[0].set_title("Quadrotor")
    axs[1].set_title("Aerial Manipulator")
    axs[0].set_ylabel("Position\nError (m)")
    axs[2].set_ylabel("Yaw\nError (rad)")
    axs[2].set_xlabel("Time (s)")
    axs[3].set_xlabel("Time (s)")
    axs[1].yaxis.set_ticklabels([])
    axs[3].yaxis.set_ticklabels([])

    axs[0].set_xlim([0, 200])
    axs[1].set_xlim([0, 200])
    axs[2].set_xlim([0, 200])
    axs[3].set_xlim([0, 200])

    axs[0].set_xticks(np.linspace(0, 200, 3))
    axs[0].set_xticklabels([])
    axs[1].set_xticks(np.linspace(0, 200, 3))
    axs[1].set_xticklabels([])
    axs[2].set_xticks(np.linspace(0, 200, 3))
    axs[2].set_xticklabels(["0", "2", "4"])
    axs[3].set_xticks(np.linspace(0, 200, 3))
    axs[3].set_xticklabels(["0", "2", "4"])

    axs[2].set_yticks(np.linspace(0, 2.0, 3))
    axs[2].set_yticklabels(np.linspace(0, 2.0, 3))
    axs[3].set_yticks(np.linspace(0, 2.0, 3))

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    

    plt.savefig("trajectory_tracking.png", dpi=500, bbox_inches='tight')
    plt.savefig("trajectory_tracking.pdf", dpi=500, bbox_inches='tight')




if __name__ == "__main__":
    plot_traj_tracking()