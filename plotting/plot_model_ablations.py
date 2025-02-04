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
    "RL Hover" : "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    "RL TrajTrack" : "../rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/",
    "GC+FF TrajTrack" : "../rl/baseline_0dof_ee_reward_tune_with_ff_traj/",
    "GC+FF Hover": "../rl/baseline_0dof_ee_reward_tune_with_ff_hover/",
    "GC Hover" : "../rl/baseline_0dof_ee_reward_tune_no_ff_hover/",
    "GC PID Hover" : "../rl/baseline_0dof_ee_reward_tune_pid_hover/",
    "GC Hand-Tuned" : "../rl/baseline_0dof_hand_tuned/",
}

all_model_paths = {
    "RL-Opt-Lissajous-None":  "../rl/logs/rsl_rl/TrajTrack/2025-01-31_05-57-57_RL_Lissajous_no_FF/",
    "RL-Opt-Lissajous-FF":  "../rl/logs/rsl_rl/TrajTrack/2025-01-29_18-50-03_RL_Lissajous_with_FF/",
    "RL-Opt-Hover-None":  "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    "RL-Opt-Hover-FF":  "../rl/logs/rsl_rl/TrajTrack/2025-01-29_17-35-58_RL_Hover_FF/",
    "GC-Opt-Lissajous-None":  "../rl/baseline_0dof_ee_reward_tune_no_ff_traj/",
    "GC-Opt-Lissajous-FF":  "../rl/baseline_0dof_ee_reward_tune_with_ff_traj/",
    "GC-Opt-Lissajous-PID":  "../rl/baseline_0dof_ee_reward_tune_pid_traj/",
    "GC-Opt-Hover-None":  "../rl/baseline_0dof_ee_reward_tune_no_ff_hover/",
    "GC-Opt-Hover-FF":  "../rl/baseline_0dof_ee_reward_tune_with_ff_hover/",
    "GC-Opt-Hover-PID":  "../rl/baseline_0dof_ee_reward_tune_pid_hover/",
    "GC-Man-Lissajous-FF":  "../rl/baseline_0dof_hand_tuned_with_ff/",
    "GC-Man-Hover-FF":  "../rl/baseline_0dof_hand_tuned_with_ff/",
    "GC-Man-Lissajous-None":  "../rl/baseline_0dof_hand_tuned/",
    "GC-Man-Hover-None":  "../rl/baseline_0dof_hand_tuned/",
}

model_paths_v2 = {
    # 'Plot 1': {
    # "RL-Man-Lissajous-FF": "",
    # "RL-Opt-Lissajous-FF":  "../rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/",
    # "GC-Man-Lissajous-FF": "../rl/baseline_0dof_hand_tuned_with_ff/",
    # # "GC-No-Lissajous-FF": "../rl/baseline_0dof_hand_tuned/",
    # "GC-Opt-Lissajous-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_traj/",
    # },
    'Plot 1': {
    "RL-Man-Lissajous-None": "",
    "RL-Opt-Lissajous-None":  "../rl/logs/rsl_rl/TrajTrack/2025-01-29_16-50-47_RL_Lissajous_No_FF/",
    "GC-Man-Lissajous-None": "../rl/baseline_0dof_hand_tuned/",
    # "GC-No-Lissajous-None": "../rl/baseline_0dof_hand_tuned/",
    "GC-Opt-Lissajous-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_traj/",
    },

    # 'Plot 2': {
    # # "RL-Opt-Hover-FF": "",
    # "RL-Opt-Hover-FF": "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    # "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/",
    # "GC-Opt-Hover-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_hover/",
    # "GC-Opt-Lissajous-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_traj/",
    # },

    'Plot 2': {
    # "RL-Opt-Hover-FF": "",
    "RL-Opt-Hover-None": "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    "RL-Opt-Lissajous-None": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_16-50-47_RL_Lissajous_No_FF/",
    "GC-Opt-Hover-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_hover/",
    "GC-Opt-Lissajous-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_traj/",
    },

    'Plot 3': {
    "RL-Opt-Lissajous-None": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_16-50-47_RL_Lissajous_No_FF/",
    # "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/",
    "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-30_07-23-27_RL_Lissajous_with_FF_ang_vel_-0.05/",
    "GC-Opt-Lissajous-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_traj/",
    "GC-Opt-Lissajous-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_traj/",
    "GC-Opt-Lissajous-PID": "../rl/baseline_0dof_ee_reward_tune_pid_traj/",
    },

    

    # 'Plot 4': {
    # "RL-Man-Hover-FF": "",
    # # "RL-Opt-Hover-FF":  "",
    # "RL-Opt-Hover-FF":  "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    # "GC-Man-Hover-FF": "../rl/baseline_0dof_hand_tuned_with_ff/",
    # "GC-Opt-Hover-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_hover/",
    # },

    'Plot 4': {
    "RL-Man-Hover-None": "",
    # "RL-Opt-Hover-None":  "",
    "RL-Opt-Hover-None":  "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    "GC-Man-Hover-None": "../rl/baseline_0dof_hand_tuned/",
    "GC-Opt-Hover-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_hover/",
    },

    # 'Plot 5': {
    # "RL-Opt-Hover-FF": "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    # "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/",
    # "GC-Opt-Hover-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_hover/",
    # "GC-Opt-Lissajous-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_traj/",
    # },

    'Plot 5': {
    "RL-Opt-Hover-None": "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    "RL-Opt-Lissajous-None": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_16-50-47_RL_Lissajous_No_FF/",
    "GC-Opt-Hover-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_hover/",
    "GC-Opt-Lissajous-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_traj/",
    },

    'Plot 6': {
    "RL-Opt-Hover-None": "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    "RL-Opt-Hover-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_17-35-58_RL_Hover_FF/",
    "GC-Opt-Hover-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_hover/",
    "GC-Opt-Hover-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_hover/",
    "GC-Opt-Hover-PID": "../rl/baseline_0dof_ee_reward_tune_pid_hover/",
    },
}

model_paths_v3 = {
    'Plot 1': {
    "RL-Man-Lissajous-FF": "",
    "RL-Opt-Lissajous-FF":  "../rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/",
    # "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-30_07-23-27_RL_Lissajous_with_FF_ang_vel_-0.05/",
    # "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_18-50-03_RL_Lissajous_with_FF/",
    "GC-Man-Lissajous-FF": "../rl/baseline_0dof_hand_tuned_with_ff/",
    # "GC-No-Lissajous-FF": "../rl/baseline_0dof_hand_tuned/",
    "GC-Opt-Lissajous-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_traj/",
    },

    'Plot 2': {
    # "RL-Opt-Hover-FF": "",
    "RL-Opt-Hover-FF": "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/",
    # "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-30_07-23-27_RL_Lissajous_with_FF_ang_vel_-0.05/",
    # "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_18-50-03_RL_Lissajous_with_FF/",
    "GC-Opt-Hover-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_hover/",
    "GC-Opt-Lissajous-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_traj/",
    },

    # 'Plot 2': {
    # # "RL-Opt-Hover-FF": "",
    # "RL-Opt-Hover-None": "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    # "RL-Opt-Lissajous-None": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_16-50-47_RL_Lissajous_No_FF/",
    # "GC-Opt-Hover-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_hover/",
    # "GC-Opt-Lissajous-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_traj/",
    # },

    'Plot 3': {
    # "RL-Opt-Lissajous-None": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_16-50-47_RL_Lissajous_No_FF/",
    # "RL-Opt-Lissajous-None": "../rl/logs/rsl_rl/TrajTrack/2025-01-30_19-27-46_RL_Lissajous_no_FF/",
    "RL-Opt-Lissajous-None": "../rl/logs/rsl_rl/TrajTrack/2025-01-31_05-57-57_RL_Lissajous_no_FF/",
    # "RL-Opt-Lissajous-None": "../rl/logs/rsl_rl/TrajTrack/2025-01-30_09-09-37_RL_Lissajous_no_FF_ang_vel_-0.05/",
    "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/",
    # "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_18-50-03_RL_Lissajous_with_FF/",
    "GC-Opt-Lissajous-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_traj/",
    "GC-Opt-Lissajous-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_traj/",
    "GC-Opt-Lissajous-PID": "../rl/baseline_0dof_ee_reward_tune_pid_traj/",
    },

    

    'Plot 4': {
    "RL-Man-Hover-FF": "",
    # "RL-Opt-Hover-FF":  "",
    "RL-Opt-Hover-FF":  "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    "GC-Man-Hover-FF": "../rl/baseline_0dof_hand_tuned_with_ff/",
    "GC-Opt-Hover-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_hover/",
    },

    # 'Plot 4': {
    # "RL-Man-Hover-None": "",
    # # "RL-Opt-Hover-None":  "",
    # "RL-Opt-Hover-None":  "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    # "GC-Man-Hover-None": "../rl/baseline_0dof_hand_tuned/",
    # "GC-Opt-Hover-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_hover/",
    # },

    'Plot 5': {
    "RL-Opt-Hover-FF": "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/",
    # "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-30_07-23-27_RL_Lissajous_with_FF_ang_vel_-0.05/",
    # "RL-Opt-Lissajous-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_18-50-03_RL_Lissajous_with_FF/",
    "GC-Opt-Hover-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_hover/",
    "GC-Opt-Lissajous-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_traj/",
    },

    # 'Plot 5': {
    # "RL-Opt-Hover-None": "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    # "RL-Opt-Lissajous-None": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_16-50-47_RL_Lissajous_No_FF/",
    # "GC-Opt-Hover-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_hover/",
    # "GC-Opt-Lissajous-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_traj/",
    # },

    'Plot 6': {
    "RL-Opt-Hover-None": "../rl/logs/rsl_rl/PaperModels_Hover/2025-01-21_17-44-46_Hover_ee_traj_env_fixed_init/",
    "RL-Opt-Hover-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_17-35-58_RL_Hover_FF/",
    "GC-Opt-Hover-None": "../rl/baseline_0dof_ee_reward_tune_no_ff_hover/",
    "GC-Opt-Hover-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff_hover/",
    "GC-Opt-Hover-PID": "../rl/baseline_0dof_ee_reward_tune_pid_hover/",
    },
}

model_colors = {
    "RL-Man-Lissajous-FF": "none",
    "RL-Man-Hover-FF": "none",
    "RL-Man-Lissajous-None": "none",
    "RL-Man-Hover-None": "none",
    "RL-Opt-Lissajous-FF": 0,
    "RL-Opt-Hover-FF": 1,
    "RL-Opt-Lissajous-None": 2,
    "RL-Opt-Hover-None": 3,
    "GC-Opt-Lissajous-FF": 4,
    "GC-Opt-Hover-FF": 5,
    "GC-Opt-Lissajous-None": 6,
    "GC-Opt-Hover-None": 7,
    "GC-Opt-Lissajous-PID": 8,
    "GC-Opt-Hover-PID": 9,
    "GC-Man-Lissajous-FF": 10,
    "GC-Man-Hover-FF": 11,
    "GC-Man-Lissajous-None": 12,
    "GC-Man-Hover-None": 13,
}

ablation_legend_elements = [
                    Patch(facecolor="C0", edgecolor="C0", fill=True, label='RL Hover'),
                    Patch(facecolor="C1", edgecolor="C1", fill=True, label='RL TrajTrack'),
                    Patch(facecolor="C2", edgecolor="C2", fill=True, label='GC+FF TrajTrack'),
                    Patch(facecolor="C3", edgecolor="C3", fill=True, label='GC+FF Hover'),
                    Patch(facecolor="C4", edgecolor="C4", fill=True, label='GC Hover'),
                    Patch(facecolor="C5", edgecolor="C5", fill=True, label='GC PID Hover'),
                    Patch(facecolor="C6", edgecolor="C6", fill=True, label='GC Hand-Tuned'),
                    # Line2D([0], [0], marker='o', color='tab:blue', label='RL')
                ] 


def plot_model_ablations(fig=None, axs=None):

    if axs is None:
        single_column = True
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
    else:
        single_column = False

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

    if single_column:
        axs[0].set_ylabel("Normalized\nReward")
    else:
        axs[0].set_ylabel("Normalized Reward")
        axs[2].set_title("Hover")
        axs[3].set_title("Lissajous Tracking")




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
    axs[1].yaxis.set_ticklabels([])
    axs[3].yaxis.set_ticklabels([])
    axs[5].yaxis.set_ticklabels([])




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

    
    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3)

    if single_column:
        fig.legend(handles=ablation_legend_elements, loc='lower center', bbox_to_anchor=(0.563, -0.16), ncol=3)
    else:
        fig.legend(handles=ablation_legend_elements, loc='lower center', bbox_to_anchor=(0.563, -0.2), ncol=3)


    # plt.tight_layout()
    save_suffix = "single_column" if single_column else "double_column"
    plt.savefig("model_ablations_"+save_suffix+".png", bbox_inches='tight', dpi=500)
    plt.savefig("model_ablations_"+save_suffix+".pdf", bbox_inches='tight', dpi=500)
    # plt.show()

def get_hatch(i, optimized, dataset, feed_forward):
    if i%3 == 0: # Checking if optimized
        return "" if "Opt" in optimized else "xx"
    elif i%3 == 1: # Checking if dataset
        if i < 3:
            return "" if dataset == "Lissajous" else "xx"
        else:
            return "" if dataset == "Hover" else "xx"
    else:
        return "" if feed_forward == "FF" else "xx"

def plot_model_ablations_v2(version=2):
    import distinctipy
    colors = distinctipy.get_colors(15, pastel_factor=0.6)

    fig = plt.figure(layout="constrained", dpi=300, figsize=(11,6.5)) #figsize=(8,5)
    axd = fig.subplot_mosaic(
        """
        ABC
        EFG
        """
    , gridspec_kw=dict(wspace=0.2))
    axs = [axd['A'], axd['B'], axd['C'], axd['E'], axd['F'], axd['G']]

    baseline_median_lissajous = None
    baseline_median_hover = None

    print("Version: ", version)
    model_paths = model_paths_v2 if version == 2 else model_paths_v3

    for i, plot_name in enumerate(model_paths.keys()):
        print("Making subplot: ", i)
        plot_index = 0
        for model_name in model_paths[plot_name].keys():
            base_path = model_paths[plot_name][model_name]
            controller = model_name.split("-")[0]
            optimized = model_name.split("-")[1]
            dataset = model_name.split("-")[2]
            feed_forward = model_name.split("-")[3]

            print("Plotting: ", model_name)

            if i < 3:
                data_path = base_path + "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
                reward_path = base_path + "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_rewards.pt"
            else:
                data_path = base_path + "Hover_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
                reward_path = base_path + "Hover_rand_init_eval_traj_track_50Hz_eval_rewards.pt"
            
            try:
                data = torch.load(data_path, weights_only=True).cpu()
                rewards_orig = torch.load(reward_path, weights_only=True).cpu()
                rewards = 15.0 - rewards_orig
                reward_quantiles = plotting_utils.get_quantiles(rewards.mean(dim=1), [0.25, 0.5, 0.75])
                reward_err_bars = plotting_utils.get_error_bars_from_quantiles(reward_quantiles)

                pos_error, yaw_error = plotting_utils.get_errors(data)
                pos_rmse = plotting_utils.get_RMSE_from_error(pos_error)
                yaw_rmse = plotting_utils.get_RMSE_from_error(yaw_error)
                print(model_name, " & ", np.round(rewards_orig.mean(dim=1).mean(dim=0).item(), 3)," $\pm$ ",np.round(rewards_orig.mean(dim=1).std(dim=0).item(), 3), " & ", np.round(pos_rmse.mean(dim=0).item(), 3)," $\pm$ ",np.round(pos_rmse.std(dim=0).item(), 3),  " & ", np.round(yaw_rmse.mean(dim=0).item(), 3)," $\pm$ ",np.round(yaw_rmse.std(dim=0).item(), 3))

            except:
                reward_quantiles = [0, 0, 0]
                reward_err_bars = np.array([0, 0]).reshape((2,1))

            # print("Reward Quantiles: ", reward_quantiles)


            if i == 0:
                baseline_median_lissajous = reward_quantiles[1] if (optimized == "Man" and dataset == "Lissajous") else baseline_median_lissajous
            elif i == 3:
                baseline_median_hover = reward_quantiles[1] if (optimized == "Man" and dataset == "Hover") else baseline_median_hover

            if plot_index == 2:
                plot_index = 3

            # bar_color = "none" if model_colors[model_name] == "none" else colors[model_colors[model_name]] 
            bar_color = "none" if model_colors[model_name] == "none" else colors[0] if 'RL' in model_name else colors[1]
            axs[i].bar(plot_index, reward_quantiles[1], yerr=reward_err_bars, label=model_name, hatch=get_hatch(i, optimized, dataset, feed_forward), color=bar_color, capsize=2)
            plot_index += 1

    legend_elements = []
    for model_name in model_colors.keys():
        bar_color = "none" if model_colors[model_name] == "none" else colors[model_colors[model_name]]
        legend_elements.append(Patch(facecolor=bar_color, edgecolor=bar_color, fill=True, label=model_name))

    
    # axs[2].yaxis.set_ticklabels([])
    # axs[4].yaxis.set_ticklabels([])
    # axs[5].yaxis.set_ticklabels([])


    if version == 2:
        axs[0].set_title("Asymmetry: Optimization\n<Manual> vs <Optimized>")
        axs[0].set_ylabel("Lissajous Tracking\nNormalized Gap to Optimal Reward")
        # axs[0].set_ylim([0, 15.0])
        # axs[0].set_yticks(np.linspace(0, 15.0, 3))
        # axs[0].set_yticklabels(np.linspace(0, 1.0, 3))
        axs[0].set_ylim([0, 9.0])
        axs[0].set_yticks(np.linspace(0, 9.0, 4))
        axs[0].set_yticklabels(np.round(np.linspace(0, 9.0/15.0, 4), 2))
        axs[0].set_xticks([0, 1, 2, 3, 4], labels=["Man.", "Opt.", "", "Man.", "Opt."])
        second_x = axs[0].secondary_xaxis(location=0)
        # second_x.set_xticks([0.5, 2.5], labels= ["\nRL-?-Lissajous-FF", "\nGC-?-Lissajous-FF"])
        second_x.set_xticks([0.5, 3.5], labels= ["\nRL-?-Liss.-None", "\nGC-?-Liss.-None"])
        second_x.tick_params(axis='x', which='both', bottom=False, top=False)

        axs[1].set_title("Asymmetry: Dataset\n<Hover> vs <Lissajous>")
        # axs[1].set_ylim([0, 15.0])
        # axs[1].set_yticks(np.linspace(0, 15.0, 3))
        # axs[1].set_yticklabels(np.round(np.linspace(0, 1.0, 3))
        axs[1].set_ylim([0, 9.0])
        axs[1].set_yticks(np.linspace(0, 9.0, 4))
        axs[1].set_yticklabels(np.round(np.linspace(0, 9.0/15.0, 4), 2))
        axs[1].tick_params(axis='y', which='both', left=False, right=False)
        # axs[1].yaxis.set_ticklabels([])
        axs[1].set_xticks([0, 1, 2, 3, 4], labels=["Hover", "Liss.", "", "Hover", "Liss."])
        second_x = axs[1].secondary_xaxis(location=0)
        # second_x.set_xticks([0.5, 2.5], labels= ["\nRL-Opt-?-FF", "\nGC-Opt-?-FF"])
        second_x.set_xticks([0.5, 3.5], labels= ["\nRL-Opt-?-None", "\nGC-Opt-?-None"])
        second_x.tick_params(axis='x', which='both', bottom=False, top=False)

        axs[2].set_title("Asymmetry: Feed Forward\n<None> vs <Feed Forward> vs <PID>")
        # axs[2].axhline(baseline_median_Liss., color="grey", linestyle="--")
        axs[2].set_ylim([0, 5.5])
        axs[2].set_yticks(np.linspace(0, 5.5, 4))
        axs[2].set_yticklabels(np.round(np.linspace(0, 5.5/15.0, 4), 2))
        axs[2].tick_params(axis='y', which='both', left=False, right=False)
        # axs[2].yaxis.set_ticklabels([])
        axs[2].set_xticks([0, 1, 2, 3, 4, 5], labels=["None", "FF", "", "None", "FF", "PID"])
        second_x = axs[2].secondary_xaxis(location=0)
        second_x.set_xticks([0.5, 4], labels= ["\nRL-Opt-Liss.-?", "\nGC-Opt-Liss.-?"])
        second_x.tick_params(axis='x', which='both', bottom=False, top=False)

        axs[3].set_ylabel("Hover\nNormalized Gap to Optimal Reward")
        # axs[3].set_ylim([0, 15.0])
        # axs[3].set_yticks(np.linspace(0, 15.0, 3))
        # axs[3].set_yticklabels(np.round(np.linspace(0, 1.0, 3))
        axs[3].set_ylim([0, 2.0])
        axs[3].set_yticks(np.linspace(0, 2.0, 4))
        axs[3].set_yticklabels(np.round(np.linspace(0, 2.0/15.0, 4), 2))
        axs[3].set_xticks([0, 1, 2, 3, 4], labels=["Man.", "Opt.", "", "Man.", "Opt."])
        second_x = axs[3].secondary_xaxis(location=0)
        second_x.set_xticks([0.5, 3.5], labels= ["\nRL-?-Hover-None", "\nGC-?-Hover-None"])
        second_x.tick_params(axis='x', which='both', bottom=False, top=False)

        # axs[4].set_ylim([0, 15.0])
        # axs[4].set_yticks(np.linspace(0, 15.0, 3))
        # axs[4].set_yticklabels(np.round(np.linspace(0, 1.0, 3))
        axs[4].set_ylim([0, 5.0])
        axs[4].set_yticks(np.linspace(0, 5.0, 4))
        axs[4].set_yticklabels(np.round(np.linspace(0, 5.0/15.0, 4), 2))
        axs[4].tick_params(axis='y', which='both', left=False, right=False)
        # axs[4].yaxis.set_ticklabels([])
        axs[4].set_xticks([0, 1, 2, 3, 4], labels=["Hover", "Liss.", "", "Hover", "Liss."])
        second_x = axs[4].secondary_xaxis(location=0)
        second_x.set_xticks([0.5, 3.5], labels= ["\nRL-Opt-?-None", "\nGC-Opt-?-None"])
        second_x.tick_params(axis='x', which='both', bottom=False, top=False)

        # axs[5].axhline(baseline_median_hover, color="grey", linestyle="--")
        axs[5].set_ylim([0, 7.0])
        axs[5].set_yticks(np.linspace(0, 7.0, 4))
        axs[5].set_yticklabels(np.round(np.linspace(0, 7.0/15.0, 4),2))
        axs[5].tick_params(axis='y', which='both', left=False, right=False)
        # axs[5].yaxis.set_ticklabels([])
        axs[5].set_xticks([0, 1, 2, 3, 4, 5], labels=["None", "FF", "", "None", "FF", "PID"])
        second_x = axs[5].secondary_xaxis(location=0)
        second_x.set_xticks([0.5, 4], labels= ["\nRL-Opt-Hover-?", "\nGC-Opt-Hover-?"])
        second_x.tick_params(axis='x', which='both', bottom=False, top=False)
    else:
        axs[0].set_title("Asymmetry: Optimization\n<Manual> vs <Optimized>")
        axs[0].set_ylabel("Lissajous Tracking\nNormalized Gap to Max Reward")
        axs[0].set_ylim([0, 2.0])
        axs[0].set_yticks(np.linspace(0, 2.0, 4))
        axs[0].set_yticklabels(np.round(np.linspace(0, 2.0/15.0, 4), 2))
        axs[0].set_xticks([0, 1, 2, 3, 4], labels=["Man.", "Opt.", "", "Man.", "Opt."])
        second_x = axs[0].secondary_xaxis(location=0)
        second_x.set_xticks([0.5, 3.5], labels= ["\nRL-?-Liss.-FF", "\nGC-?-Liss.-FF"])
        second_x.tick_params(axis='x', which='both', bottom=False, top=False)

        axs[1].set_title("Asymmetry: Dataset\n<Hover> vs <Lissajous>")
        # axs[1].set_ylim([0, 15.0])
        # axs[1].set_yticks(np.linspace(0, 15.0, 3))
        # axs[1].set_yticklabels(np.round(np.linspace(0, 1.0, 3))
        axs[1].set_ylim([0, 2.5])
        axs[1].set_yticks(np.linspace(0, 2.5, 4))
        axs[1].set_yticklabels(np.round(np.linspace(0, 2.5/15.0, 4), 2))
        axs[1].tick_params(axis='y', which='both', left=False, right=False)
        # axs[1].yaxis.set_ticklabels([])
        axs[1].set_xticks([0, 1, 2, 3, 4], labels=["Hover", "Liss.", "", "Hover", "Liss."])
        second_x = axs[1].secondary_xaxis(location=0)
        second_x.set_xticks([0.5, 3.5], labels= ["\nRL-Opt-?-FF", "\nGC-Opt-?-FF"])
        second_x.tick_params(axis='x', which='both', bottom=False, top=False)

        axs[2].set_title("Asymmetry: Feed Forward\n<None> vs <Feed Forward> vs <PID>")
        # axs[2].axhline(baseline_median_Liss., color="grey", linestyle="--")
        axs[2].set_ylim([0, 5.5])
        axs[2].set_yticks(np.linspace(0, 5.5, 4))
        axs[2].set_yticklabels(np.round(np.linspace(0, 5.5/15.0, 4), 2))
        axs[2].tick_params(axis='y', which='both', left=False, right=False)
        # axs[2].yaxis.set_ticklabels([])
        axs[2].set_xticks([0, 1, 2, 3, 4, 5], labels=["None", "FF", "", "None", "FF", "PID"])
        second_x = axs[2].secondary_xaxis(location=0)
        second_x.set_xticks([0.5, 4], labels= ["\nRL-Opt-Liss.-?", "\nGC-Opt-Liss.-?"])
        second_x.tick_params(axis='x', which='both', bottom=False, top=False)

        axs[3].set_ylabel("Hover\nNormalized Gap to Max Reward")
        # axs[3].set_ylim([0, 15.0])
        # axs[3].set_yticks(np.linspace(0, 15.0, 3))
        # axs[3].set_yticklabels(np.round(np.linspace(0, 1.0, 3))
        axs[3].set_ylim([0, 2.5])
        axs[3].set_yticks(np.linspace(0, 2.5, 4))
        axs[3].set_yticklabels(np.round(np.linspace(0, 2.5/15.0, 4), 2))
        axs[3].set_xticks([0, 1, 2, 3, 4], labels=["Man.", "Opt.", "", "Man.", "Opt."])
        second_x = axs[3].secondary_xaxis(location=0)
        second_x.set_xticks([0.5, 3.5], labels= ["\nRL-?-Hover-FF", "\nGC-?-Hover-FF"])
        second_x.tick_params(axis='x', which='both', bottom=False, top=False)

        # axs[4].set_ylim([0, 15.0])
        # axs[4].set_yticks(np.linspace(0, 15.0, 3))
        # axs[4].set_yticklabels(np.round(np.linspace(0, 1.0, 3))
        axs[4].set_ylim([0, 3.5])
        axs[4].set_yticks(np.linspace(0, 3.5, 4))
        axs[4].set_yticklabels(np.round(np.linspace(0, 3.5/15.0, 4), 2))
        axs[4].tick_params(axis='y', which='both', left=False, right=False)
        # axs[4].yaxis.set_ticklabels([])
        axs[4].set_xticks([0, 1, 2, 3, 4], labels=["Hover", "Liss.", "", "Hover", "Liss."])
        second_x = axs[4].secondary_xaxis(location=0)
        second_x.set_xticks([0.5, 3.5], labels= ["\nRL-Opt-?-FF", "\nGC-Opt-?-FF"])
        second_x.tick_params(axis='x', which='both', bottom=False, top=False)

        # axs[5].axhline(baseline_median_hover, color="grey", linestyle="--")
        axs[5].set_ylim([0, 7.0])
        axs[5].set_yticks(np.linspace(0, 7.0, 4))
        axs[5].set_yticklabels(np.round(np.linspace(0, 7.0/15.0, 4),2))
        axs[5].tick_params(axis='y', which='both', left=False, right=False)
        # axs[5].yaxis.set_ticklabels([])
        axs[5].set_xticks([0, 1, 2, 3, 4, 5], labels=["None", "FF", "", "None", "FF", "PID"])
        second_x = axs[5].secondary_xaxis(location=0)
        second_x.set_xticks([0.5, 4], labels= ["\nRL-Opt-Hover-?", "\nGC-Opt-Hover-?"])
        second_x.tick_params(axis='x', which='both', bottom=False, top=False)

    # fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4)

    plt.savefig("model_ablations_v"+str(version)+".png", bbox_inches='tight', dpi=500)
    plt.savefig("model_ablations_v"+str(version)+".pdf", bbox_inches='tight', dpi=500)


def plot_model_components():
    fig = plt.figure(layout="constrained", dpi=300, figsize=(5, 7))
    axd = fig.subplot_mosaic(
        """
        A
        B
        """
    )
    axs = [axd['A'], axd['B']]

    lissajous_models = ["GC-Opt-Lissajous-FF", "GC-Opt-Lissajous-None", "GC-Opt-Hover-None", "GC-Man-Lissajous-None", "GC-Opt-Lissajous-PID", "GC-Man-Lissajous-FF"]
    hover_models = ["GC-Opt-Hover-FF", "GC-Opt-Hover-None", "GC-Opt-Lissajous-None", "GC-Man-Hover-None", "GC-Opt-Hover-PID", "GC-Man-Hover-FF"]

    for i, model_name in enumerate(lissajous_models):
        base_path = all_model_paths[model_name]
        data_path = base_path + "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
        reward_path = base_path + "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_rewards.pt"
        data = torch.load(data_path, weights_only=True).cpu()
        rewards = torch.load(reward_path, weights_only=True).cpu()

        reward_quartiles = plotting_utils.get_quantiles(rewards.mean(dim=1), [0.25, 0.5, 0.75])
        reward_err_bars = plotting_utils.get_error_bars_from_quantiles(reward_quartiles)

        plot_idx = i if i < 1 else i+1
        axs[0].bar(plot_idx, reward_quartiles[1], yerr=reward_err_bars, label=model_name, capsize=2)
    axs[0].set_xticks([0, 1, 2, 3, 4, 5, 6], labels=["GC-Opt-Liss.-FF", "", "- FF", "- Data", "- Opt", "- FF + PID", "- Opt + FF"], rotation=45)
    axs[0].set_title("Lissajous Tracking")
    axs[0].set_ylabel("Normalized Reward")

    for i, model_name in enumerate(hover_models):
        base_path = all_model_paths[model_name]
        data_path = base_path + "Hover_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
        reward_path = base_path + "Hover_rand_init_eval_traj_track_50Hz_eval_rewards.pt"
        data = torch.load(data_path, weights_only=True).cpu()
        rewards = torch.load(reward_path, weights_only=True).cpu()

        reward_quartiles = plotting_utils.get_quantiles(rewards.mean(dim=1), [0.25, 0.5, 0.75])
        reward_err_bars = plotting_utils.get_error_bars_from_quantiles(reward_quartiles)

        plot_idx = i if i < 1 else i+1
        axs[1].bar(plot_idx, reward_quartiles[1], yerr=reward_err_bars, label=model_name, capsize=2)
    axs[1].set_xticks([0, 1, 2, 3, 4, 5, 6], labels=["GC-Opt-Hover-FF", "", "- FF", "- Data", "- Opt", "- FF + PID", "- Opt + FF"], rotation=45)
    axs[1].set_title("Hover")
    axs[1].set_ylabel("Normalized Reward")

    plt.savefig("model_components.png", bbox_inches='tight', dpi=500)
    # plt.savefig("model_components.pdf", bbox_inches='tight', dpi=500)





if __name__ == "__main__":
    # fig, axs = plt.subplots(1, 2, dpi=300)
    # fig = plt.figure(layout="constrained", dpi=300, figsize=(8, 4))
    # axd = fig.subplot_mosaic(
    #     """
    #     ABCD
    #     ABEF
    #     """
    # )
    # axs = [axd['A'], axd['B'], axd['C'], axd['D'], axd['E'], axd['F']]

    # plot_model_ablations() # single column
    # plot_model_ablations(fig, axs) # double column

    plot_model_ablations_v2(3)
    plot_model_components()