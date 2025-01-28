import torch

import utils.math_utilities as math_utils
import omni.isaac.lab.utils.math as isaac_math_utils

# Description: Parameters for plotting
params = {
    # Data indicies
    "quad_pos_slice": slice(0,3),
    "quad_ori_slice" : slice(3,7),
    "ee_pos_slice" : slice(13,16),
    "ee_ori_slice" : slice(16,20),
    "goal_pos_slice" : slice(26,29),
    "goal_ori_slice" : slice(29,33),

    # Colors
    "rl_ee_color": "#56B4E9",
    "rl_com_color": "#CC79A7",
    "gc_color": "#E69F00",
    "violin_color_1": "#009E73",
    "violin_color_2": "#0072B2",
}

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
def get_quantiles(data, quantiles):
    quantiles = torch.tensor(quantiles, device=data.device)
    return torch.quantile(data, quantiles, dim=0).cpu()

def get_error_bars_from_quantiles(quantiles):
    return torch.abs(quantiles[::2] - quantiles[1]).numpy().reshape((2,1))

@torch.no_grad()
def get_errors(data):
    N = data.shape[0]
    T = data.shape[1]-1

    pos_error = torch.norm(data[:, :T, params["goal_pos_slice"]] - data[:, :T, params["ee_pos_slice"]], dim=-1)
    yaw_error = math_utils.yaw_error_from_quats(data[:,:T,params["goal_ori_slice"]], data[:,:T,params["ee_ori_slice"]], 0)

    return pos_error, yaw_error

@torch.no_grad()
def get_RMSE_from_error(error):
    # Assumes error is of shape (N, T)
    return torch.sqrt(torch.mean(error**2, dim=1))