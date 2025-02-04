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


rl_ee_base_path = "../rl/logs/rsl_rl/Hover_PaperModels/2025-01-13_09-11-23_new_model_default_params_anneal_50e6/"
rl_com_base_path = "../rl/logs/rsl_rl/Hover_PaperModels/2025-01-13_10-33-26_new_model_default_params_anneal_50e6_COM/"
gc_base_path = "../rl/baseline_0dof_ee_reward_tune/"

error_legend_elements = [ 
                    Line2D([0], [0], color=params["rl_ee_color"], label='RL-EE'),
                    Line2D([0], [0], color=params["rl_com_color"], label='RL-COM'),
                    Line2D([0], [0], color=params["gc_color"], label='GC'),
                    # Patch(facecolor='none', edgecolor='none', fill=False, label=''),
                    # Line2D([0], [0], marker='o', color='tab:blue', label='RL')
                ]

violin_legend_elements_v1 = [
                    Patch(facecolor=params["violin_color_1"], edgecolor=params["violin_color_1"], fill=True, label='Settling Time'),
                    Patch(facecolor=params["violin_color_2"], edgecolor=params["violin_color_2"], fill=True, label='RMSE'),
                    Patch(facecolor='none', edgecolor='none', fill=False, label=''),
                    # Line2D([0], [0], marker='o', color='tab:blue', label='RL')
                ]

violin_legend_elements_v2 = [
                    Patch(facecolor=params["violin_color_1"], edgecolor=params["violin_color_1"], fill=True, label='Position'),
                    Patch(facecolor=params["violin_color_2"], edgecolor=params["violin_color_2"], fill=True, label='Yaw'),
                    Patch(facecolor='none', edgecolor='none', fill=False, label=''),
                    # Line2D([0], [0], marker='o', color='tab:blue', label='RL')
                ] 

violin_legend_elements_COM_ablation = [
                    Patch(facecolor=params["violin_color_1"], edgecolor=params["violin_color_1"], fill=True, label='RL-EE'),
                    Patch(facecolor=params["violin_color_2"], edgecolor=params["violin_color_2"], fill=True, label='GC'),
] 



def load_data():
    rl_ee_path = os.path.join(rl_ee_base_path, "eval_full_states.pt")
    rl_com_path = os.path.join(rl_com_base_path, "eval_full_states.pt")
    gc_path = os.path.join(gc_base_path, "hover_eval_full_states.pt")

    rl_ee_data = torch.load(rl_ee_path, weights_only=True)
    rl_com_data = torch.load(rl_com_path, weights_only=True)
    gc_data = torch.load(gc_path, weights_only=True)

    return rl_ee_data, rl_com_data, gc_data

@torch.no_grad()
def plot_error_pos_yaw(axs=None):
    rl_ee_data, rl_com_data, gc_data = load_data()

    N = rl_ee_data.shape[0]
    T = rl_ee_data.shape[1]-1

    rl_ee_pos_quantiles, rl_ee_yaw_quantiles = plotting_utils.get_quantiles_error(rl_ee_data, [0.25, 0.5, 0.75])
    rl_com_pos_quantiles, rl_com_yaw_quantiles = plotting_utils.get_quantiles_error(rl_com_data, [0.25, 0.5, 0.75])
    gc__pos_quantiles, gc_yaw_quantiles = plotting_utils.get_quantiles_error(gc_data, [0.25, 0.5, 0.75])

    # Make a seaborn line plot where the x-axis is the time step and the y-axis is the error, showing the Percentile Interval of 50% (IQR)
    if axs==None:
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 5), dpi=300)
    x_axis = np.arange(T) * 0.02
    plot_clip_time = 4
    sns.lineplot(x=x_axis, y=rl_ee_pos_quantiles[1], ax=axs[0], label="RL-EE", color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=rl_com_pos_quantiles[1], ax=axs[0], label="RL-COM", color=params["rl_com_color"], legend=False)
    sns.lineplot(x=x_axis, y=gc__pos_quantiles[1], ax=axs[0], label="GC", color=params["gc_color"], legend=False)
    axs[0].fill_between(x_axis, rl_ee_pos_quantiles[0], rl_ee_pos_quantiles[2], alpha=0.2, color=params["rl_ee_color"])
    axs[0].fill_between(x_axis, rl_com_pos_quantiles[0], rl_com_pos_quantiles[2], alpha=0.2, color=params["rl_com_color"])
    axs[0].fill_between(x_axis, gc__pos_quantiles[0], gc__pos_quantiles[2], alpha=0.2, color=params["gc_color"])
    axs[0].set_ylabel("Position Error (m)")
    plt.setp(axs[0].get_xticklabels(), visible=False) # hide x axis ticks for top plot
    # axs[0].set_xlabel("Time (s)")
    axs[0].set_xlim(0, plot_clip_time)
    sns.lineplot(x=x_axis, y=rl_ee_yaw_quantiles[1], ax=axs[1], label="RL-EE", color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=rl_com_yaw_quantiles[1], ax=axs[1], label="RL-COM", color=params["rl_com_color"], legend=False)
    sns.lineplot(x=x_axis, y=gc_yaw_quantiles[1], ax=axs[1], label="GC", color=params["gc_color"], legend=False)
    axs[1].fill_between(x_axis, rl_ee_yaw_quantiles[0], rl_ee_yaw_quantiles[2], alpha=0.2, color=params["rl_ee_color"])
    axs[1].fill_between(x_axis, rl_com_yaw_quantiles[0], rl_com_yaw_quantiles[2], alpha=0.2, color=params["rl_com_color"])
    axs[1].fill_between(x_axis, gc_yaw_quantiles[0], gc_yaw_quantiles[2], alpha=0.2, color=params["gc_color"])
    axs[1].set_ylabel("Yaw Error (rad)")
    # axs[1].yaxis.set_label_position("right")
    # axs[1].yaxis.tick_right()
    axs[1].set_xlabel("Time (s)")
    axs[1].set_xlim(0, plot_clip_time)

    # fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    # plt.tight_layout()

@torch.no_grad()
def compute_settling_time(errors, tolerance=0.05, dt=1.0):
    """
    Compute the settling time for each trajectory using PyTorch.

    Parameters:
    - errors: torch.Tensor of shape (1000, 500), errors over time for trajectories.
    - tolerance: float, percentage tolerance for settling (e.g., 0.05 for 5%).
    - dt: float, time step for each frame.

    Returns:
    - settling_times: torch.Tensor of shape (1000,), settling time for each trajectory.
    """
    # Final values for each trajectory (last column)
    initial_values = errors[:, 0].unsqueeze(1)  # Shape: (1000, 1)
    final_values = errors[:, -1].unsqueeze(1)  # Shape: (1000, 1)

    # Compute tolerance bounds
    error_range = (initial_values - final_values).abs()  # Absolute difference
    lower_bound = final_values - (tolerance * error_range)
    upper_bound = final_values + (tolerance * error_range)

    # Create a mask for values within the tolerance band
    within_band = (errors >= lower_bound) & (errors <= upper_bound)  # Shape: (1000, 500)

    # Create a mask for values outside the tolerance band
    outside_band = (errors < lower_bound) | (errors > upper_bound)  # Shape: (1000, 500)

    # Convert boolean mask to integer type for argmax
    outside_band_int = outside_band.int()  # Convert to int for argmax

    # Walk backwards to find the last index where the error is outside the band
    last_outside_index = outside_band.size(1) - torch.argmax(outside_band_int.flip(dims=[1]), dim=1) - 1

    # Handle cases where the error never goes outside the tolerance band
    never_outside_mask = ~outside_band.any(dim=1)
    last_outside_index[never_outside_mask] = -1  # Assign -1 to indicate never outside

    # Compute settling times (add 1 because indices are zero-based)
    settling_times = (last_outside_index + 1).float() * dt
    settling_times[last_outside_index == -1] = 0.0  # If never outside, settling time is 0
    return settling_times

def plot_violin_rmse_settling_time(axs=None):
    rl_ee_data, rl_com_data, gc_data = load_data()

    N = rl_ee_data.shape[0]
    T = rl_ee_data.shape[1]-1

    rl_ee_pos_error, rl_ee_yaw_error = plotting_utils.get_errors(rl_ee_data)
    rl_com_pos_error, rl_com_yaw_error = plotting_utils.get_errors(rl_com_data)
    gc_pos_error, gc_yaw_error = plotting_utils.get_errors(gc_data)

    rl_ee_pos_rmse = torch.sqrt(torch.mean(rl_ee_pos_error**2, dim=1)).cpu()
    rl_com_pos_rmse = torch.sqrt(torch.mean(rl_com_pos_error**2, dim=1)).cpu()
    gc_pos_rmse = torch.sqrt(torch.mean(gc_pos_error**2, dim=1)).cpu()
    rl_ee_yaw_rmse = torch.sqrt(torch.mean(rl_ee_yaw_error**2, dim=1)).cpu()
    rl_com_yaw_rmse = torch.sqrt(torch.mean(rl_com_yaw_error**2, dim=1)).cpu()
    gc_yaw_rmse = torch.sqrt(torch.mean(gc_yaw_error**2, dim=1)).cpu()

    rl_ee_pos_settling_time = compute_settling_time(rl_ee_pos_error, tolerance=0.05, dt=0.02).cpu()
    rl_com_pos_settling_time = compute_settling_time(rl_com_pos_error, tolerance=0.05, dt=0.02).cpu()
    gc_pos_settling_time = compute_settling_time(gc_pos_error, tolerance=0.05, dt=0.02).cpu()
    rl_ee_yaw_settling_time = compute_settling_time(rl_ee_yaw_error, tolerance=0.05, dt=0.02).cpu()
    rl_com_yaw_settling_time = compute_settling_time(rl_com_yaw_error, tolerance=0.05, dt=0.02).cpu()
    gc_yaw_settling_time = compute_settling_time(gc_yaw_error, tolerance=0.05, dt=0.02).cpu()

    data = pd.DataFrame({
        "Settling Time": torch.cat([rl_ee_pos_settling_time, rl_ee_yaw_settling_time, rl_com_pos_settling_time, rl_com_yaw_settling_time, gc_pos_settling_time, gc_yaw_settling_time]),
        "RMSE": torch.cat([rl_ee_pos_rmse, rl_ee_yaw_rmse, rl_com_pos_rmse, rl_com_yaw_rmse, gc_pos_rmse, gc_yaw_rmse]),
        "Type": (["Position"] * N + ["Yaw"] * N + ["Position"] * N + ["Yaw"] * N + ["Position"] * N + ["Yaw"] * N),
        "Method": (["RL-EE"] * 2*N + ["RL-COM"] * 2*N + ["GC"] * 2*N)
    })


    combined_data = pd.DataFrame({
        "Data": torch.cat([rl_ee_pos_settling_time, rl_ee_yaw_settling_time, rl_com_pos_settling_time, rl_com_yaw_settling_time, gc_pos_settling_time, gc_yaw_settling_time, rl_ee_pos_rmse, rl_ee_yaw_rmse, rl_com_pos_rmse, rl_com_yaw_rmse, gc_pos_rmse, gc_yaw_rmse]),
        "Metric": (["Settling Time"] * 6*N + ["RMSE"] * 6*N),
        "Type": (6*(["Position"] * N + ["Yaw"] * N)),
        "Method": (2*(["RL-EE"] * 2*N + ["RL-COM"] * 2*N + ["GC"] * 2*N)),
    })

    # Just plot the settling time for positions for now
    if axs==None:
        fig, axs = plt.subplots(1, 2, figsize=(3.5, 3.5), dpi=300)
    sns.violinplot(data=data, x="Method", y="Settling Time", hue="Type", inner="quart", split=True, palette=[params['violin_color_1'], params['violin_color_2']], legend=False, ax=axs[0])
    axs[0].set_ylabel("Settling Time (s)")
    sns.violinplot(data=data, x="Method", y="RMSE", hue="Type", inner="quart", split=True, palette=[params['violin_color_1'], params['violin_color_2']], legend=False, ax=axs[1])
    axs[1].set_ylabel("RMSE (m/rad)")


    # Plot position data on left, yaw data on right
    # fig, axs = plt.subplots(1, 2, figsize=(3.5, 3.5), dpi=300)
    # sns.violinplot(data=combined_data[combined_data["Type"] == "Position"], x="Method", y="Data", hue="Metric", inner="quart", split=True, palette=[params['violin_color_1'], params['violin_color_2']], legend=False, ax=axs[0])
    # axs[0].set_ylabel("Position")
    # sns.violinplot(data=combined_data[combined_data["Type"] == "Yaw"], x="Method", y="Data", hue="Metric", inner="quart", split=True, palette=[params['violin_color_1'], params['violin_color_2']], legend=False, ax=axs[1])
    # axs[1].set_ylabel("Yaw")

    # fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    # plt.tight_layout()
def plot_COM_ablation():
    gc_com_v_base_path = "../rl/baseline_0dof_small_arm_com_v_tuned/"
    gc_com_middle_base_path = "../rl/baseline_0dof_small_arm_com_middle_tuned/"
    gc_com_ee_base_path = "../rl/baseline_0dof_small_arm_com_ee_tuned/"
    rl_com_v_base_path = "../rl/logs/rsl_rl/COM_Ablation_Hover/2025-01-15_08-21-13_COM_V/"
    rl_com_middle_base_path = "../rl/logs/rsl_rl/COM_Ablation_Hover/2025-01-15_08-43-17_COM_Middle/"
    rl_com_ee_base_path = "../rl/logs/rsl_rl/COM_Ablation_Hover/2025-01-15_08-21-42_COM_EE/"

    gc_com_v_data = torch.load(os.path.join(gc_com_v_base_path, "Hover_eval_full_states.pt"), weights_only=True)
    gc_com_middle_data = torch.load(os.path.join(gc_com_middle_base_path, "Hover_eval_full_states.pt"), weights_only=True)
    gc_com_ee_data = torch.load(os.path.join(gc_com_ee_base_path, "Hover_eval_full_states.pt"), weights_only=True)
    rl_com_v_data = torch.load(os.path.join(rl_com_v_base_path, "Hover_eval_full_states.pt"), weights_only=True)
    rl_com_middle_data = torch.load(os.path.join(rl_com_middle_base_path, "Hover_eval_full_states.pt"), weights_only=True)
    rl_com_ee_data = torch.load(os.path.join(rl_com_ee_base_path, "Hover_eval_full_states.pt"), weights_only=True)

    gc_com_v_pos_error, gc_com_v_yaw_error = plotting_utils.get_errors(gc_com_v_data)
    gc_com_middle_pos_error, gc_com_middle_yaw_error = plotting_utils.get_errors(gc_com_middle_data)
    gc_com_ee_pos_error, gc_com_ee_yaw_error = plotting_utils.get_errors(gc_com_ee_data)
    rl_com_v_pos_error, rl_com_v_yaw_error = plotting_utils.get_errors(rl_com_v_data)
    rl_com_middle_pos_error, rl_com_middle_yaw_error = plotting_utils.get_errors(rl_com_middle_data)
    rl_com_ee_pos_error, rl_com_ee_yaw_error = plotting_utils.get_errors(rl_com_ee_data)

    gc_com_v_pos_rmse = torch.sqrt(torch.mean(gc_com_v_pos_error**2, dim=1)).cpu()
    gc_com_middle_pos_rmse = torch.sqrt(torch.mean(gc_com_middle_pos_error**2, dim=1)).cpu()
    gc_com_ee_pos_rmse = torch.sqrt(torch.mean(gc_com_ee_pos_error**2, dim=1)).cpu()
    gc_com_v_yaw_rmse = torch.sqrt(torch.mean(gc_com_v_yaw_error**2, dim=1)).cpu()
    gc_com_middle_yaw_rmse = torch.sqrt(torch.mean(gc_com_middle_yaw_error**2, dim=1)).cpu()
    gc_com_ee_yaw_rmse = torch.sqrt(torch.mean(gc_com_ee_yaw_error**2, dim=1)).cpu()
    
    rl_com_v_pos_rmse = torch.sqrt(torch.mean(rl_com_v_pos_error**2, dim=1)).cpu()
    rl_com_middle_pos_rmse = torch.sqrt(torch.mean(rl_com_middle_pos_error**2, dim=1)).cpu()
    rl_com_ee_pos_rmse = torch.sqrt(torch.mean(rl_com_ee_pos_error**2, dim=1)).cpu()
    rl_com_v_yaw_rmse = torch.sqrt(torch.mean(rl_com_v_yaw_error**2, dim=1)).cpu()
    rl_com_middle_yaw_rmse = torch.sqrt(torch.mean(rl_com_middle_yaw_error**2, dim=1)).cpu()
    rl_com_ee_yaw_rmse = torch.sqrt(torch.mean(rl_com_ee_yaw_error**2, dim=1)).cpu()

    N_gc = gc_com_v_data.shape[0]
    N_rl = rl_com_v_data.shape[0]
    T = gc_com_v_data.shape[1]-1

    gc_data = pd.DataFrame({
        "Position RMSE": torch.cat([gc_com_v_pos_rmse, gc_com_middle_pos_rmse, gc_com_ee_pos_rmse, rl_com_v_pos_rmse, rl_com_middle_pos_rmse, rl_com_ee_pos_rmse]),
        "Yaw RMSE": torch.cat([gc_com_v_yaw_rmse, gc_com_middle_yaw_rmse, gc_com_ee_yaw_rmse, rl_com_v_yaw_rmse, rl_com_middle_yaw_rmse, rl_com_ee_yaw_rmse]),
        "COM Location": (["Vehicle"] * N_gc + ["Middle"] * N_gc + ["End-Effector"] * N_gc + ["Vehicle"] * N_rl + ["Middle"] * N_rl + ["End-Effector"] * N_rl),
        "Controller": (["GC"] *3*N_gc + ["RL-EE"] * 3*N_rl),
    })

    fig = plt.figure(layout="constrained", dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        """
    )
    axs = [axd["A"], axd["B"]]
    sns.violinplot(data=gc_data, x="COM Location", y="Position RMSE", inner="quart", hue="Controller", split=True, palette=[params['violin_color_2'], params['violin_color_1']], legend=False, ax=axs[0])
    axs[0].set_ylabel("Position RMSE (m)")
    sns.violinplot(data=gc_data, x="COM Location", y="Yaw RMSE", inner="quart", hue="Controller", split=True, palette=[params['violin_color_2'], params['violin_color_1']], legend=False, ax=axs[1])
    axs[1].set_ylabel("Yaw RMSE (rad)")

    fig.legend(handles=violin_legend_elements_COM_ablation, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
    plt.savefig("COM_ablation_hover_violin.png", bbox_inches='tight', dpi=500, format='png')

    rl_com_v_pos_quantiles, rl_com_v_yaw_quantiles = plotting_utils.get_quantiles_error(rl_com_v_data, [0.25, 0.5, 0.75])
    rl_com_middle_pos_quantiles, rl_com_middle_yaw_quantiles = plotting_utils.get_quantiles_error(rl_com_middle_data, [0.25, 0.5, 0.75])
    rl_com_ee_pos_quantiles, rl_com_ee_yaw_quantiles = plotting_utils.get_quantiles_error(rl_com_ee_data, [0.25, 0.5, 0.75])
    gc_com_v_pos_quantiles, gc_com_v_yaw_quantiles = plotting_utils.get_quantiles_error(gc_com_v_data, [0.25, 0.5, 0.75])
    gc_com_middle_pos_quantiles, gc_com_middle_yaw_quantiles = plotting_utils.get_quantiles_error(gc_com_middle_data, [0.25, 0.5, 0.75])
    gc_com_ee_pos_quantiles, gc_com_ee_yaw_quantiles = plotting_utils.get_quantiles_error(gc_com_ee_data, [0.25, 0.5, 0.75])


    x_axis = np.arange(T) * 0.02
    plot_clip_time=4
    fig = plt.figure(layout="constrained", dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        """
    )
    axs = [axd["A"], axd["B"], axd["C"], axd["D"], axd["E"], axd["F"]]
    sns.lineplot(x=x_axis, y=rl_com_v_pos_quantiles[1], ax=axs[0], label="RL-EE", color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=gc_com_v_pos_quantiles[1], ax=axs[0], label="GC", color=params["gc_color"], legend=False)
    axs[0].fill_between(x_axis, rl_com_v_pos_quantiles[0], rl_com_v_pos_quantiles[2], alpha=0.2, color=params["rl_ee_color"])
    axs[0].fill_between(x_axis, gc_com_v_pos_quantiles[0], gc_com_v_pos_quantiles[2], alpha=0.2, color=params["gc_color"])
    axs[0].set_ylabel("COM-V\nPosition Error (m)")
    plt.setp(axs[0].get_xticklabels(), visible=False) # hide x axis ticks for top plot
    axs[0].set_xlim(0, plot_clip_time)
    sns.lineplot(x=x_axis, y=rl_com_v_yaw_quantiles[1], ax=axs[1], label="RL-EE", color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=gc_com_v_yaw_quantiles[1], ax=axs[1], label="GC", color=params["gc_color"], legend=False)
    axs[1].fill_between(x_axis, rl_com_v_yaw_quantiles[0], rl_com_v_yaw_quantiles[2], alpha=0.2, color=params["rl_ee_color"])
    axs[1].fill_between(x_axis, gc_com_v_yaw_quantiles[0], gc_com_v_yaw_quantiles[2], alpha=0.2, color=params["gc_color"])
    axs[1].set_ylabel("Yaw Error (rad)")
    axs[1].set_xlim(0, plot_clip_time)
    plt.setp(axs[1].get_xticklabels(), visible=False) # hide x axis ticks for top plot
    # axs[1].yaxis.set_label_position("right")
    # axs[1].yaxis.tick_right()
    sns.lineplot(x=x_axis, y=rl_com_middle_pos_quantiles[1], ax=axs[2], label="RL-EE", color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=gc_com_middle_pos_quantiles[1], ax=axs[2], label="GC", color=params["gc_color"], legend=False)
    axs[2].fill_between(x_axis, rl_com_middle_pos_quantiles[0], rl_com_middle_pos_quantiles[2], alpha=0.2, color=params["rl_ee_color"])
    axs[2].fill_between(x_axis, gc_com_middle_pos_quantiles[0], gc_com_middle_pos_quantiles[2], alpha=0.2, color=params["gc_color"])
    axs[2].set_ylabel("COM-Middle\nPosition Error (m)")
    plt.setp(axs[2].get_xticklabels(), visible=False) # hide x axis ticks for top plot
    axs[2].set_xlim(0, plot_clip_time)
    sns.lineplot(x=x_axis, y=rl_com_middle_yaw_quantiles[1], ax=axs[3], label="RL-EE", color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=gc_com_middle_yaw_quantiles[1], ax=axs[3], label="GC", color=params["gc_color"], legend=False)
    axs[3].fill_between(x_axis, rl_com_middle_yaw_quantiles[0], rl_com_middle_yaw_quantiles[2], alpha=0.2, color=params["rl_ee_color"])
    axs[3].fill_between(x_axis, gc_com_middle_yaw_quantiles[0], gc_com_middle_yaw_quantiles[2], alpha=0.2, color=params["gc_color"])
    axs[3].set_ylabel("Yaw Error (rad)")
    axs[3].set_xlim(0, plot_clip_time)
    plt.setp(axs[3].get_xticklabels(), visible=False) # hide x axis ticks for top plot
    # axs[1].yaxis.set_label_position("right")
    # axs[1].yaxis.tick_right()
    sns.lineplot(x=x_axis, y=rl_com_ee_pos_quantiles[1], ax=axs[4], label="RL-EE", color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=gc_com_ee_pos_quantiles[1], ax=axs[4], label="GC", color=params["gc_color"], legend=False)
    axs[4].fill_between(x_axis, rl_com_ee_pos_quantiles[0], rl_com_ee_pos_quantiles[2], alpha=0.2, color=params["rl_ee_color"])
    axs[4].fill_between(x_axis, gc_com_ee_pos_quantiles[0], gc_com_ee_pos_quantiles[2], alpha=0.2, color=params["gc_color"])
    axs[4].set_ylabel("COM-EE\nPosition Error (m)")
    axs[4].set_xlim(0, plot_clip_time)
    axs[4].set_xlabel("Time (s)")
    sns.lineplot(x=x_axis, y=rl_com_ee_yaw_quantiles[1], ax=axs[5], label="RL-EE", color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=gc_com_ee_yaw_quantiles[1], ax=axs[5], label="GC", color=params["gc_color"], legend=False)
    axs[5].fill_between(x_axis, rl_com_ee_yaw_quantiles[0], rl_com_ee_yaw_quantiles[2], alpha=0.2, color=params["rl_ee_color"])
    axs[5].fill_between(x_axis, gc_com_ee_yaw_quantiles[0], gc_com_ee_yaw_quantiles[2], alpha=0.2, color=params["gc_color"])
    axs[5].set_ylabel("Yaw Error (rad)")
    axs[5].set_xlim(0, plot_clip_time)
    axs[5].set_xlabel("Time (s)")

    fig.legend(handles=error_legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.savefig("COM_ablation_hover_error.png", bbox_inches='tight', dpi=500, format='png')

def plot_rewards():
    rl_ee_path = os.path.join(rl_ee_base_path, "eval_rewards.pt")
    rl_com_path = os.path.join(rl_com_base_path, "eval_rewards.pt")
    gc_path = os.path.join(gc_base_path, "hover_eval_rewards.pt")

    rl_ee_rewards_raw = torch.load(rl_ee_path, weights_only=True).cpu()
    rl_com_rewards_raw = torch.load(rl_com_path, weights_only=True).cpu()
    gc_rewards_raw = torch.load(gc_path, weights_only=True).cpu()

    N = rl_ee_rewards_raw.shape[0]
    T = rl_ee_rewards_raw.shape[1]

    max_reward = 15.0
    min_reward = torch.min(torch.cat([rl_ee_rewards_raw, rl_com_rewards_raw, gc_rewards_raw])).item()
    
    # normalize rewards
    rl_ee_rewards = (rl_ee_rewards_raw - min_reward) / (max_reward - min_reward)
    rl_com_rewards = (rl_com_rewards_raw - min_reward) / (max_reward - min_reward)
    gc_rewards = (gc_rewards_raw - min_reward) / (max_reward - min_reward)

    quantiles = torch.tensor([0.25, 0.5, 0.75], device=rl_ee_rewards.device)
    rl_ee_quantiles = torch.quantile(rl_ee_rewards, quantiles, dim=0).cpu()
    rl_com_quantiles = torch.quantile(rl_com_rewards, quantiles, dim=0).cpu()
    gc_quantiles = torch.quantile(gc_rewards, quantiles, dim=0).cpu()

    x_axis = np.arange(T) * 0.02
    plot_clip_time = 4
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), dpi=300)
    sns.lineplot(x=x_axis, y=rl_ee_quantiles[1], ax=ax, label="RL-EE", color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=rl_com_quantiles[1], ax=ax, label="RL-COM", color=params["rl_com_color"], legend=False)
    sns.lineplot(x=x_axis, y=gc_quantiles[1], ax=ax, label="GC", color=params["gc_color"], legend=False)
    ax.fill_between(x_axis, rl_ee_quantiles[0], rl_ee_quantiles[2], alpha=0.2, color=params["rl_ee_color"])
    ax.fill_between(x_axis, rl_com_quantiles[0], rl_com_quantiles[2], alpha=0.2, color=params["rl_com_color"])
    ax.fill_between(x_axis, gc_quantiles[0], gc_quantiles[2], alpha=0.2, color=params["gc_color"])
    ax.set_ylabel("Normalized Reward")
    ax.set_xlim(0, plot_clip_time)
    ax.set_xlabel("Time (s)")
    fig.legend(handles=error_legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.savefig("hover_rewards.png", bbox_inches='tight', dpi=500, format='png')

    rl_ee_avg_reward = torch.mean(rl_ee_rewards, dim=1).cpu()
    rl_com_avg_reward = torch.mean(rl_com_rewards, dim=1).cpu()
    gc_avg_reward = torch.mean(gc_rewards, dim=1).cpu()

    data = pd.DataFrame({
        "Average Reward": torch.cat([rl_ee_avg_reward, rl_com_avg_reward, gc_avg_reward]),
        "Method": (["RL-EE"] * N + ["RL-COM"] * N + ["GC"] * N)
    })

    # Make bar plot with error bars
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), dpi=300)
    sns.barplot(data=data, x="Method", y="Average Reward", errorbar="sd", hue="Method", palette=[params["rl_ee_color"], params["rl_com_color"], params["gc_color"]], ax=ax)
    ax.set_ylabel("Average Normalized Reward")
    ax.set_ylim(0.9, 1.0)
    plt.savefig("hover_avg_rewards.png", bbox_inches='tight', dpi=500, format='png')

    rl_ee_accumulated_reward = torch.sum(rl_ee_rewards_raw, dim=1).cpu() / (T*max_reward)
    rl_com_accumulated_reward = torch.sum(rl_com_rewards_raw, dim=1).cpu()  / (T*max_reward)
    gc_accumulated_reward = torch.sum(gc_rewards_raw, dim=1).cpu()  / (T*max_reward)

    data = pd.DataFrame({
        "Accumulated Reward": torch.cat([rl_ee_accumulated_reward, rl_com_accumulated_reward, gc_accumulated_reward]),
        "Method": (["RL-EE"] * N + ["RL-COM"] * N + ["GC"] * N)
    })

    # Make bar plot with error bars
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), dpi=300)
    sns.barplot(data=data, x="Method", y="Accumulated Reward", errorbar="sd", hue="Method", palette=[params["rl_ee_color"], params["rl_com_color"], params["gc_color"]], ax=ax)
    ax.set_ylabel("Accumulated Normalized Reward")
    ax.set_ylim(0.9, 1.0)
    plt.savefig("hover_accumulated_rewards.png", bbox_inches='tight', dpi=500, format='png')





def gen_combined_layout():
    fig = plt.figure(layout="constrained", dpi=300, figsize=(7, 4))
    axd = fig.subplot_mosaic(
        """
        ACD
        BCD
        """
    )
    plot_error_pos_yaw([axd["A"], axd["B"]])
    plot_violin_rmse_settling_time([axd["C"], axd["D"]])
    # legend_elements = error_legend_elements + violin_legend_elements
    # legend_elements = list(itertools.chain.from_iterable(zip(error_legend_elements, violin_legend_elements_v1)))
    legend_elements = list(itertools.chain.from_iterable(zip(error_legend_elements, violin_legend_elements_v2)))
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    # plt.suptitle("V1: Grouped by Pos/Yaw")
    # plt.suptitle("V2: Grouped by Metric")
    plt.savefig("hover_error_violin_v2.png", bbox_inches='tight', dpi=500, format='png')
    plt.savefig("hover_error_violin_v2.pdf", bbox_inches='tight', dpi=500, format='pdf')

def gen_error_plot_only():
    fig = plt.figure(layout="constrained", dpi=300, figsize=(3.5, 4))
    axd = fig.subplot_mosaic(
        """
        A
        B
        """
    )
    plot_error_pos_yaw([axd["A"], axd["B"]])
    legend_elements = error_legend_elements
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    plt.savefig("hover_error_plot.png", bbox_inches='tight', dpi=500, format='png')
    plt.savefig("hover_error_plot.pdf", bbox_inches='tight', dpi=500, format='pdf')


if __name__ == "__main__":
    # gen_combined_layout()
    # gen_error_plot_only()
    
    # plot_COM_ablation()
    plot_rewards()
    # plt.show()