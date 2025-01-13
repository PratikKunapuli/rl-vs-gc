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

from plotting_params import params


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

# violin_legend_elements = [
#                     Patch(facecolor=params["violin_color_1"], edgecolor=params["violin_color_1"], fill=True, label='Settling Time'),
#                     Patch(facecolor=params["violin_color_2"], edgecolor=params["violin_color_2"], fill=True, label='RMSE'),
#                     Patch(facecolor='none', edgecolor='none', fill=False, label=''),
#                     # Line2D([0], [0], marker='o', color='tab:blue', label='RL')
#                 ]

violin_legend_elements = [
                    Patch(facecolor=params["violin_color_1"], edgecolor=params["violin_color_1"], fill=True, label='Position'),
                    Patch(facecolor=params["violin_color_2"], edgecolor=params["violin_color_2"], fill=True, label='Yaw'),
                    Patch(facecolor='none', edgecolor='none', fill=False, label=''),
                    # Line2D([0], [0], marker='o', color='tab:blue', label='RL')
                ]  

@torch.no_grad()
def get_quantiles_error(data, quantiles):
    N = data.shape[0]
    T = data.shape[1]-1

    pos_error = torch.norm(data[:, :T, params["goal_pos_slice"]] - data[:, :T, params["ee_pos_slice"]], dim=-1)
    yaw_error = math_utils.yaw_error_from_quats(data[:,:T,params["goal_ori_slice"]], data[:,:T,params["ee_ori_slice"]], 0)

    pos_quantiles = torch.quantile(pos_error, torch.tensor(quantiles, device=data.device), dim=0).cpu()
    yaw_quantiles = torch.quantile(yaw_error, torch.tensor(quantiles, device=data.device), dim=0).cpu()

    return pos_quantiles, yaw_quantiles

@torch.no_grad()
def get_errors(data):
    N = data.shape[0]
    T = data.shape[1]-1

    pos_error = torch.norm(data[:, :T, params["goal_pos_slice"]] - data[:, :T, params["ee_pos_slice"]], dim=-1)
    yaw_error = math_utils.yaw_error_from_quats(data[:,:T,params["goal_ori_slice"]], data[:,:T,params["ee_ori_slice"]], 0)

    return pos_error, yaw_error

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

    rl_ee_pos_quantiles, rl_ee_yaw_quantiles = get_quantiles_error(rl_ee_data, [0.25, 0.5, 0.75])
    rl_com_pos_quantiles, rl_com_yaw_quantiles = get_quantiles_error(rl_com_data, [0.25, 0.5, 0.75])
    gc__pos_quantiles, gc_yaw_quantiles = get_quantiles_error(gc_data, [0.25, 0.5, 0.75])

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

    rl_ee_pos_error, rl_ee_yaw_error = get_errors(rl_ee_data)
    rl_com_pos_error, rl_com_yaw_error = get_errors(rl_com_data)
    gc_pos_error, gc_yaw_error = get_errors(gc_data)

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


if __name__ == "__main__":
    fig = plt.figure(layout="constrained", dpi=300, figsize=(7, 5))
    axd = fig.subplot_mosaic(
        """
        ACD
        BCD
        """
    )
    plot_error_pos_yaw([axd["A"], axd["B"]])
    plot_violin_rmse_settling_time([axd["C"], axd["D"]])
    # legend_elements = error_legend_elements + violin_legend_elements
    legend_elements = list(itertools.chain.from_iterable(zip(error_legend_elements, violin_legend_elements)))
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    plt.suptitle("V2: Grouped by Metric")
    plt.savefig("hover_error_violin_v2.png", bbox_inches='tight', dpi=500, format='png')
    plt.savefig("hover_error_violin_v2.pdf", bbox_inches='tight', dpi=500, format='pdf')
