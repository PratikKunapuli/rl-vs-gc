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

rl_models = ["2025-03-22_14-44-55_CTBM_TrajTrack_DR_all_latency_0ms", "2025-03-22_03-06-34_CTBM_TrajTrack_DR_all_latency_20ms", "2025-03-22_03-07-10_CTBM_TrajTrack_DR_all_latency_40ms"]
gc_models = ["baseline_cf_ctbm_DR", "baseline_cf_ctbm_DR_20ms", "baseline_cf_ctbm_DR_40ms"]
eval_settings = ["control_delay_0ms", "control_delay_20ms", "control_delay_40ms"]

@torch.no_grad()
def print_avg_rewards():
    for rl_model in rl_models:
        print("-"*20)
        print("RL Model: {}".format(rl_model))
        for eval_setting in eval_settings:
            load_path = "../rl/logs/rsl_rl/BrushlessCrazyflie_DR/{}/{}eval_rewards.pt".format(rl_model, eval_setting)
            eval_rewards = torch.load(load_path, weights_only=True) / 0.02
            eval_rewards = torch.mean(eval_rewards, dim=1)
            avg_reward = torch.mean(eval_rewards)
            std_reward = torch.std(eval_rewards)
            print("{}: {:.3f} $\pm$ {:.2f} ".format(eval_setting, avg_reward, std_reward))

    for gc_model in gc_models:
        print("-"*20)
        print("GC Model: {}".format(gc_model))
        for eval_setting in eval_settings:
            load_path = "../rl/{}/{}eval_rewards.pt".format(gc_model, eval_setting)
            eval_rewards = torch.load(load_path, weights_only=True) / 0.02
            eval_rewards = torch.mean(eval_rewards, dim=1)
            avg_reward = torch.mean(eval_rewards)
            std_reward = torch.std(eval_rewards)
            print("{}: {:.3f} $\pm$ {:.2f} ".format(eval_setting, avg_reward, std_reward))

@torch.no_grad()
def print_rmse():
    for rl_model in rl_models:
        print("-"*20)
        print("RL Model: {}".format(rl_model))
        for eval_setting in eval_settings:
            load_path = "../rl/logs/rsl_rl/BrushlessCrazyflie_DR/{}/{}eval_full_states.pt".format(rl_model, eval_setting)
            data = torch.load(load_path, weights_only=True)
            pos_error, yaw_error = plotting_utils.get_errors(data)
            pos_rmse = plotting_utils.get_RMSE_from_error(pos_error)
            yaw_rmse = plotting_utils.get_RMSE_from_error(yaw_error)
            pos_rmse_mean = torch.mean(pos_rmse)
            pos_rmse_std = torch.std(pos_rmse)
            yaw_rmse_mean = torch.mean(yaw_rmse)
            yaw_rmse_std = torch.std(yaw_rmse)
            print("{}: pos: {:.3f} $\pm$ {:.2f}, yaw: {:.3f} $\pm$ {:.2f}".format(eval_setting, pos_rmse_mean, pos_rmse_std, yaw_rmse_mean, yaw_rmse_std))

    for gc_model in gc_models:
        print("-"*20)
        print("GC Model: {}".format(gc_model))
        for eval_setting in eval_settings:
            load_path = "../rl/{}/{}eval_full_states.pt".format(gc_model, eval_setting)
            data = torch.load(load_path, weights_only=True)
            pos_error, yaw_error = plotting_utils.get_errors(data)
            pos_rmse = plotting_utils.get_RMSE_from_error(pos_error)
            yaw_rmse = plotting_utils.get_RMSE_from_error(yaw_error)
            pos_rmse_mean = torch.mean(pos_rmse)
            pos_rmse_std = torch.std(pos_rmse)
            yaw_rmse_mean = torch.mean(yaw_rmse)
            yaw_rmse_std = torch.std(yaw_rmse)
            print("{}: pos: {:.3f} $\pm$ {:.2f}, yaw: {:.3f} $\pm$ {:.2f}".format(eval_setting, pos_rmse_mean, pos_rmse_std, yaw_rmse_mean, yaw_rmse_std))
            



if __name__ == "__main__":
    # print_avg_rewards()
    print_rmse()