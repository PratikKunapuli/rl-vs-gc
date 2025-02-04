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
# sns.set_context("paper")
# sns.set_theme()

import plotting.plotting_utils as plotting_utils
from plotting.plotting_utils import params

model_paths = {
    # "RL-Opt.-Liss.-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/",
    "RL-Opt.-Liss.-FF": "../rl/logs/rsl_rl/TrajTrack/2025-01-29_18-50-03_RL_Lissajous_with_FF/",
    "GC-Opt.-Liss.-FF": "../rl/baseline_0dof_ee_reward_tune_with_ff/",
}

model_colors = {
    "RL-Opt.-Liss.-FF": "C0",
    "GC-Opt.-Liss.-FF": "C1",
}

def plot_traj():
    figure = plt.figure()

    reference = None
    traj_index = 0
    time_limit = 499

    # background_img = plt.imread("./frame_extracts/frame_0.04.png")
    # plt.imshow(background_img, extent=[-0.960, 0.960, -0.540, 0.540])
    # # plt.imshow(background_img)
    # plt.savefig("background_img.png")

    for model_name, model_path in model_paths.items():
        model_data = torch.load(model_path + "Viz_eval_traj_track_50Hz_eval_full_states.pt", weights_only=True).cpu()

        if reference is None:
            reference = model_data[traj_index, :time_limit, params["goal_pos_slice"]]
            print(reference[:2, :])
            plt.scatter(reference[:,0], reference[:,1], label="Reference", alpha =0.5, color="black")

        pos = model_data[traj_index, :time_limit, params["ee_pos_slice"]]

        plt.scatter(pos[:,0], pos[:,1], label=model_name, alpha=0.5, color=model_colors[model_name])

        # Draw a line between the reference and the end effector scatter points in the same color as the model scatter
        for i in range(len(pos)):
            plt.plot([reference[i,0], pos[i,0]], [reference[i,1], pos[i,1]], color=model_colors[model_name], alpha=0.3)

    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.title("Trajectory Tracking over 2 seconds")
    plt.legend()
    # plt.show()

    plt.savefig("example_traj_track.png")
    plt.savefig("example_traj_track.svg", format="svg", dpi=1200)

def extract_frame():
    import moviepy.editor as mpy

    clip = mpy.VideoFileClip(model_paths["RL-Opt.-Liss.-FF"] + "videos/eval/Viz_eval_traj_track_50Hz__eval_video_robot_0_viz_robot-step-0.mp4")
    # clip = mpy.VideoFileClip(model_paths["GC-Opt.-Liss.-FF"] + "Viz_eval_traj_track_50Hz__eval_video_robot_0_viz_viz-step-0.mp4")

    for t in np.arange(0, 0.2, 0.02):
        clip.save_frame("./frame_extracts/frame_" + str(t) + ".png", t=t)

if __name__ == "__main__":
    # extract_frame()
    plot_traj()