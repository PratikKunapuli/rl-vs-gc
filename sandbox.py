import torch
from typing import Tuple
import utils.math_utilities as math_utils
import utils.trajectory_utilities as traj_utils
import matplotlib.pyplot as plt
import time
import omni.isaac.lab.utils.math as isaac_math_utils


def check_traj_gen():
    t_points = torch.arange(0, 51) * 0.02
    # print("T: ", t_points.shape, t_points)
    amps = torch.tensor([1, 2, 3, 4])
    freqs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    phases = torch.tensor([0.0, 0.0, 0.0, 0.0])
    offsets = torch.tensor([0.0, 0.0, 0.0, 0.0])

    start = time.time()
    pos_traj, yaw_traj = traj_utils.eval_lissajous_curve(t_points, amps, freqs, phases, offsets, derivatives=4)
    print("Time taken: ", time.time() - start)
    print("Pos: ", pos_traj.shape)
    print("Yaw: ", yaw_traj.shape)

    parallel_amps = torch.randn((4096, 4))
    parallel_freqs = torch.randn((4096, 4))
    parallel_phases = torch.randn((4096, 4))
    parallel_offsets = torch.randn((4096, 4))

    start = time.time()
    parallel_pos_traj, parallel_yaw_traj = traj_utils.eval_lissajous_curve(t_points, parallel_amps, parallel_freqs, parallel_phases, parallel_offsets, derivatives=4)
    print("Time taken: ", time.time() - start)

    print("Pos: ", parallel_pos_traj.shape)
    print("Yaw: ", parallel_yaw_traj.shape)

    pos_traj = pos_traj.detach().cpu().numpy()
    yaw_traj = yaw_traj.detach().cpu().numpy()

    fig, axs = plt.subplots(2,1, dpi=500)
    axs[0].plot(t_points, pos_traj[0,0, 0,:], label="X")
    axs[0].plot(t_points, pos_traj[0,0, 1,:], label="Y")
    axs[0].plot(t_points, pos_traj[0,0, 2,:], label="Z")
    axs[0].set_ylabel("Position")
    axs[0].legend()
    axs[1].plot(t_points, yaw_traj[0,0, :], label="Yaw")
    axs[1].set_ylabel("Yaw")
    axs[1].legend()
    plt.savefig("lissajous_curve.png")

def check_eval_rollout():
    data_path = "./rl/logs/rsl_rl/CrazyflieManip_CTATT/2024-11-14_11-01-24_default_params_no_squared_pos/"
    data = torch.load(data_path + "eval_full_states.pt", weights_only=True)

    robot_number = 1

    data = data[robot_number, :, :]
    pos_w = data[:-1, 0:3].detach().cpu().numpy()
    quat_w = data[:-1, 3:7]
    lin_vel_w = data[:-1, 7:10].detach().cpu().numpy()
    ang_vel_w = data[:-1, 10:13].detach().cpu().numpy()
    goal_pos_w = data[:-1, 13:16].detach().cpu().numpy()
    goal_ori_w = data[:-1, 16:20]

    yaw = math_utils.yaw_from_quat(quat_w)
    yaw = yaw.detach().cpu().numpy()
    goal_yaw = math_utils.yaw_from_quat(goal_ori_w)
    goal_yaw = goal_yaw.detach().cpu().numpy()

    fig, axs = plt.subplots(4, 1, dpi=500)
    axs[0].plot(pos_w[:, 0], label="X")
    axs[0].plot(goal_pos_w[:, 0], label="Goal X")
    axs[0].set_ylabel("X")
    axs[0].legend()
    axs[1].plot(pos_w[:, 1], label="Y")
    axs[1].plot(goal_pos_w[:, 1], label="Goal Y")
    axs[1].set_ylabel("Y")
    axs[1].legend()
    axs[2].plot(pos_w[:, 2], label="Z")
    axs[2].plot(goal_pos_w[:, 2], label="Goal Z")
    axs[2].set_ylabel("Z")
    axs[2].legend()
    axs[3].plot(yaw, label="Yaw")
    axs[3].plot(goal_yaw, label="Goal Yaw")
    axs[3].set_ylabel("Yaw")
    axs[3].legend()
    plt.savefig("eval_rollout.png")

def check_poly_traj():
    # make x, y, z polynomial curves be 0 for all time, and yaw be a linear function of time
    t = torch.arange(0, 51) * 0.02
    n_envs = 4096
    coeffs = torch.zeros((n_envs, 4, 4)) # 2 since 1 would be constant term and 1 would be linear term
    coeffs[:, -1, 1] = 1.0 # make the linear term 1.0 for the yaw curve only
    coeffs[:, -1, 2] = 3.0 # make the linear term 1.0 for the yaw curve only


    start_time = time.time()
    pos_traj, yaw_traj = traj_utils.eval_polynomial_curve(t, coeffs, derivatives=4)
    print("Time taken: ", time.time() - start_time)
    print("Pos: ", pos_traj.shape)
    print("Yaw: ", yaw_traj.shape)

    pos_traj = pos_traj.detach().cpu().numpy()
    yaw_traj = yaw_traj.detach().cpu().numpy()

    fig, axs = plt.subplots(2,1, dpi=500)
    axs[0].plot(t, pos_traj[0,0, 0,:], label="X")
    axs[0].plot(t, pos_traj[0,0, 1,:], label="Y")
    axs[0].plot(t, pos_traj[0,0, 2,:], label="Z")
    axs[0].set_ylabel("Position")
    axs[0].legend()
    axs[1].plot(t, yaw_traj[0,0, :], label="Yaw")
    axs[1].set_ylabel("Yaw")
    axs[1].legend()
    plt.savefig("polynomial_curve.png")

def check_combined_traj():
    t = torch.arange(0, 101) * 0.02
    n_envs = 4096
    # amps = torch.tensor([1, 1, 0, 0])
    # freqs = torch.tensor([3.14159, 3.14159, 0.0, 0.0])
    # phases = torch.tensor([0.0, 1.5707, 0.0, 0.0])
    # offsets = torch.tensor([0.0, 0.0, 0.5, 0.0])
    amps = torch.tensor([0, 0, 0, 0])
    freqs = torch.tensor([0, 0, 0.0, 0.0])
    phases = torch.tensor([0.0, 0, 0.0, 0.0])
    offsets = torch.tensor([0.0, 0.0, 0.5, 0.0])
    coeffs = torch.zeros((n_envs, 4, 2))
    coeffs[:, -1, 0] = 0.0
    coeffs[:, -1, 1] = 0.0

    pos_lis, yaw_lis = traj_utils.eval_lissajous_curve(t, amps, freqs, phases, offsets, derivatives=4)
    pos_poly, yaw_poly = traj_utils.eval_polynomial_curve(t, coeffs, derivatives=4)
    pos = pos_lis + pos_poly
    yaw = yaw_lis + yaw_poly
    yaw[0,0,:] = isaac_math_utils.wrap_to_pi(yaw[0,0,:])

    pos = pos.detach().cpu().numpy()
    yaw = yaw.detach().cpu().numpy()

    fig, axs = plt.subplots(2,1, dpi=500)
    axs[0].plot(t, pos[0,0, 0,:], label="X")
    axs[0].plot(t, pos[0,0, 1,:], label="Y")
    axs[0].plot(t, pos[0,0, 2,:], label="Z")
    axs[0].set_ylabel("Position")
    axs[0].legend()
    axs[1].plot(t, yaw[0,0, :], label="Yaw")
    axs[1].set_ylabel("Yaw")
    axs[1].legend()
    plt.savefig("combined_curve.png")


def check_traj_data():
    # gc_run_path = "./rl/baseline_0dof_ee_reward_tune/eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path = "./rl/logs/rsl_rl/TrajTrack/2024-12-13_12-31-41_4d_lissajous_amplitudes_2m_frequencies_2.0_yaw_1.0_initial_offset_0.5m_anneal_39e6_ang_vel_-0.05/eval_traj_track_50Hz_eval_full_states.pt"
    
    # Base Paths:
    gc_run_path = "./rl/baseline_0dof_ee_reward_tune/"
    # gc_run_path = "./rl/baseline_0dof_long_arm_com_middle_tuned/"

    # rl_run_path = "./rl/logs/rsl_rl/TrajTrack/2024-12-14_16-33-12_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel/"
    # rl_run_path = "./rl/logs/rsl_rl/TrajTrack/2024-12-18_15-57-47_4d_lissajous_init_change_rand_pos_and_vel_traj_ori_as_yaw_long/"
    # rl_run_path = "./rl/logs/rsl_rl/TrajTrack/2025-01-06_12-10-13_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6/"
    rl_run_path = "./rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/"

    # rl_run_path = "./rl/logs/rsl_rl/TrajTrack/2025-01-21_13-29-33_LongArm_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_39e6_yaw_error_-4/"

    # 2D lissajous Eval:
    # traj_name = "2d_lissajous"
    # gc_run_path += "2d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "2d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # 4D lissajous Eval:
    # traj_name = "4d_lissajous"
    # gc_run_path += "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # 4D lissajous out of distribution Eval:
    # traj_name = "4d_lissajous_ood"
    # gc_run_path += "4d_lissajous_ood_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "4d_lissajous_ood_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # 3D Lissajous (no yaw)
    # traj_name = "3d_lissajous"
    # gc_run_path += "3d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "3d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # 4D Lissajous Fast Yaw
    # traj_name = "4d_lissajous_fast_yaw"
    # gc_run_path += "4d_lissajous_fast_yaw_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "4d_lissajous_fast_yaw_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # 4D lissajous fast yaw 6
    # traj_name = "4d_lissajous_fast_yaw_freq_6"
    # gc_run_path += "4d_lissajous_fast_yaw_6_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "4d_lissajous_fast_yaw_6_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # 4D lissajous fast yaw freq 6 amp 6
    # traj_name = "4d_lissajous_fast_yaw_freq_6_amp_6"
    # gc_run_path += "4d_lissajous_fast_yaw_6_wrap_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "4d_lissajous_fast_yaw_6_wrap_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # 3D lissajous polynomial yaw
    # traj_name = "3d_lissajous_poly_yaw"
    # gc_run_path += "3d_lissajous_polynomial_yaw_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "3d_lissajous_polynomial_yaw_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # 3D lissajous polynomial yaw fast
    # traj_name = "3d_lissajous_poly_yaw_fast"
    # gc_run_path += "3d_lissajous_polynomial_yaw_fast_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "3d_lissajous_polynomial_yaw_fast_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # 3D lissajous polynomial yaw very fast
    # traj_name = "3d_lissajous_poly_yaw_very_fast"
    # gc_run_path += "3d_lissajous_polynomial_yaw_very_fast_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "3d_lissajous_polynomial_yaw_very_fast_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # Circle Slow Tangent
    traj_name = "circle_slow_tangent"
    gc_run_path += "circle_traj_tangent_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "circle_traj_tangent_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    rl_run_path += "circle_traj_rand_init_eval_traj_track_50Hz_eval_full_states.pt"


    # Circle Slow Eval:
    # traj_name = "circle_slow"
    # gc_run_path += "circle_traj_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "circle_traj_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # Circle Fast Eval:
    # traj_name = "circle_fast"
    # gc_run_path += "circle_traj_fast_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # rl_run_path += "circle_traj_fast_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    gc_data = torch.load(gc_run_path, weights_only=True)
    rl_data = torch.load(rl_run_path, weights_only=True)

    print("GC: ", gc_data.shape)
    print("RL: ", rl_data.shape)

    traj_pos = gc_data[:, :-1, 26:29].detach().cpu()
    gc_pos = gc_data[:, :-1, 13:16].detach().cpu()
    rl_pos = rl_data[:, :-1, 13:16].detach().cpu()
    traj_yaw = math_utils.yaw_from_quat(gc_data[:, :-1, 29:]).detach().cpu()
    gc_ori = math_utils.yaw_from_quat(gc_data[:, :-1, 16:20]).detach().cpu()
    rl_ori = math_utils.yaw_from_quat(rl_data[:, :-1, 16:20]).detach().cpu()

    reset_pos = gc_data[:, 0, 13:16].detach().cpu() # last position is the reset position for each environment
    traj_reset_pos = gc_data[:, 0, 26:29].detach().cpu()

    rl_pos_error = torch.norm(rl_pos - traj_pos, dim=-1)
    gc_pos_error = torch.norm(gc_pos - traj_pos, dim=-1)
    rl_ori_error = torch.abs(isaac_math_utils.wrap_to_pi(rl_ori - traj_yaw))
    gc_ori_error = torch.abs(isaac_math_utils.wrap_to_pi(gc_ori - traj_yaw))

    # get median and 2 quartiles for the errors
    quantiles = torch.tensor([0.25, 0.5, 0.75])
    gc_pos_error = torch.quantile(gc_pos_error, quantiles, dim=0)
    rl_pos_error = torch.quantile(rl_pos_error, quantiles, dim=0)
    gc_ori_error = torch.quantile(gc_ori_error, quantiles, dim=0)
    rl_ori_error = torch.quantile(rl_ori_error, quantiles, dim=0)

    # Each environment has a different offset in the world, so we need to make them relative to their first timestep for plotting
    gc_pos = gc_pos - reset_pos.unsqueeze(1)
    rl_pos = rl_pos - reset_pos.unsqueeze(1)
    traj_pos = traj_pos - traj_reset_pos.unsqueeze(1)
    traj_pos = traj_pos[0]


    # print("GC Pos: ", gc_pos.shape)
    # print("RL Pos: ", rl_pos.shape)
    # print("Traj Pos: ", traj_pos.shape)
    # print("GC Yaw: ", gc_ori.shape)
    # print("RL Yaw: ", rl_ori.shape)
    # print("Traj Yaw: ", traj_yaw.shape)
    # print("GC Pos Error: ", gc_pos_error.shape)
    # print("RL Pos Error: ", rl_pos_error.shape)
    # print("GC Yaw Error: ", gc_ori_error.shape)
    # print("RL Yaw Error: ", rl_ori_error.shape)

    # Plot the errors as shaded regions for the median and 2 quartiles, left side position, right side yaw
    fig, axs = plt.subplots(1, 2, dpi=900)
    time = torch.arange(0, gc_pos_error.shape[1]) * 0.02
    axs[0].plot(time, gc_pos_error[1], label="GC", color="tab:orange")
    axs[0].fill_between(time, gc_pos_error[0], gc_pos_error[2], color="tab:orange", alpha=0.5)
    axs[0].plot(time, rl_pos_error[1], label="RL", color="tab:blue")
    axs[0].fill_between(time, rl_pos_error[0], rl_pos_error[2], color="tab:blue", alpha=0.5)
    axs[0].set_ylabel("Position Error")
    axs[0].set_xlabel("Time (s)")
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(time, gc_ori_error[1], label="GC", color="tab:orange")
    axs[1].fill_between(time, gc_ori_error[0], gc_ori_error[2], color="tab:orange", alpha=0.5)
    axs[1].plot(time, rl_ori_error[1], label="RL", color="tab:blue")
    axs[1].fill_between(time, rl_ori_error[0], rl_ori_error[2], color="tab:blue", alpha=0.5)
    axs[1].set_ylabel("Yaw Error")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid()
    axs[1].legend()
    plt.tight_layout()
    plt.suptitle("Traj: " + traj_name)
    plt.savefig("traj_errors_" + traj_name + ".png")
    return

    # Plot the trajectory in X-Y plane and show left side all rollouts for GC and right side all rollouts for RL
    fig, axs = plt.subplots(2, 2, dpi=900)
    axs[0,0].plot(traj_pos[:, 0], traj_pos[:, 1], label="Trajectory", color="tab:green")
    axs[0,0].plot(gc_pos[:, :, 0], gc_pos[:, :, 1], label="GC", color="tab:orange", alpha=0.02)
    axs[0,0].set_ylabel("Y")
    axs[0,0].set_xlabel("X")
    axs[0,0].set_title("GC")
    # axs[0,0].legend()

    axs[0,1].plot(traj_pos[:, 0], traj_pos[:, 1], label="Trajectory", color="tab:green")
    axs[0,1].plot(rl_pos[:, :, 0], rl_pos[:, :, 1], label="RL", color="tab:blue", alpha=0.02)
    axs[0,1].set_ylabel("Y")
    axs[0,1].set_xlabel("X")
    axs[0,1].set_title("RL")
    # axs[0,1].legend()
    # plt.tight_layout()

    # Show the z trajectory for GC and RL as function of time
    axs[1,0].plot(time, traj_pos[:, 2], label="Trajectory", color="tab:green")
    axs[1,0].plot(time, gc_pos[:, :, 2].T, color="tab:orange", alpha=0.02)
    axs[1,0].set_ylabel("Z")
    axs[1,0].set_xlabel("Time (s)")

    axs[1,1].plot(time, traj_pos[:, 2], label="Trajectory", color="tab:green")
    axs[1,1].plot(time, rl_pos[:, :, 2].T, color="tab:blue", alpha=0.02)
    axs[1,1].set_ylabel("Z")
    axs[1,1].set_xlabel("Time (s)")
    plt.tight_layout()

    plt.savefig("traj_xy_" + traj_name + ".png")

def investigate_lissajous_vs_circle():
    gc_base_path = "./rl/baseline_0dof_ee_reward_tune/"
    rl_base_path = "./rl/logs/rsl_rl/TrajTrack/2025-01-08_17-07-17_4d_lissajous_init_change_rand_pos_vel_yaw_and_ang_vel_anneal_50e6_yaw_error_-4/"

    # 4d lissajous
    traj_name = "4d_lissajous"
    gc_lissajous = gc_base_path + "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    rl_lissajous = rl_base_path + "4d_lissajous_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # Circle Slow
    traj_name = "circle_slow"
    gc_circle_slow = gc_base_path + "circle_traj_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    rl_circle_slow = rl_base_path + "circle_traj_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # Circle Fast
    traj_name = "circle_fast"
    gc_circle_fast = gc_base_path + "circle_traj_fast_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    rl_circle_fast = rl_base_path + "circle_traj_fast_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # 100 x 500 x 33
    gc_lissajous_data = torch.load(gc_lissajous, weights_only=True)
    rl_lissajous_data = torch.load(rl_lissajous, weights_only=True)
    gc_circle_slow_data = torch.load(gc_circle_slow, weights_only=True)
    rl_circle_slow_data = torch.load(rl_circle_slow, weights_only=True)
    gc_circle_fast_data = torch.load(gc_circle_fast, weights_only=True)
    rl_circle_fast_data = torch.load(rl_circle_fast, weights_only=True)


    def parse_trial(data):
        traj_pos = data[:, :-1, 26:29].detach().cpu()
        traj_vel = (data[:, 1:, 26:29] - data[:, :-1, 26:29]).cpu() / 0.02
        traj_ori = (data[:, :-1, 29:]).detach().cpu()
        traj_yaw = math_utils.yaw_from_quat(traj_ori)
        ee_pos = data[:, :-1, 13:16].detach().cpu()
        ee_ori = (data[:, :-1, 16:20]).detach().cpu()
        ee_yaw = math_utils.yaw_from_quat(ee_ori)
        traj_vel_b = isaac_math_utils.quat_rotate_inverse(ee_ori[:,:], traj_vel)
        pos_error = torch.norm(ee_pos - traj_pos, dim=-1)
        ori_error = torch.abs(isaac_math_utils.wrap_to_pi(ee_yaw - traj_yaw))

        return traj_pos, traj_vel, traj_ori, traj_yaw, ee_pos, ee_ori, ee_yaw, traj_vel_b, pos_error, ori_error


    # gc_lissajous_traj_pos = gc_lissajous_data[:, :-1, 26:29].detach().cpu()
    # gc_lissajous_traj_vel = (gc_lissajous_data[:, 1:, 26:29] - gc_lissajous_data[:, :-1, 26:29]).cpu() / 0.02
    # gc_lissajous_traj_ori = (gc_lissajous_data[:, :-1, 29:]).detach().cpu()
    # gc_lissajous_traj_yaw = math_utils.yaw_from_quat(gc_lissajous_traj_ori)
    # gc_lissajous_ee_pos = gc_lissajous_data[:, :-1, 13:16].detach().cpu()
    # gc_lissajous_ee_ori = (gc_lissajous_data[:, :-1, 16:20]).detach().cpu()
    # gc_lissajous_ee_yaw = math_utils.yaw_from_quat(gc_lissajous_ee_ori)
    # gc_lissajous_traj_vel_b = isaac_math_utils.quat_rotate_inverse(gc_lissajous_ee_ori[:,:], gc_lissajous_traj_vel)
    # gc_lissajous_pos_error = torch.norm(gc_lissajous_ee_pos - gc_lissajous_traj_pos, dim=-1)
    # gc_lissajous_ori_error = torch.abs(isaac_math_utils.wrap_to_pi(gc_lissajous_ee_yaw - gc_lissajous_traj_yaw))

    gc_lissajous_traj_pos, gc_lissajous_traj_vel, gc_lissajous_traj_ori, gc_lissajous_traj_yaw, gc_lissajous_ee_pos, gc_lissajous_ee_ori, gc_lissajous_ee_yaw, gc_lissajous_traj_vel_b, gc_lissajous_pos_error, gc_lissajous_ori_error = parse_trial(gc_lissajous_data)
    gc_circle_slow_traj_pos, gc_circle_slow_traj_vel, gc_circle_slow_traj_ori, gc_circle_slow_traj_yaw, gc_circle_slow_ee_pos, gc_circle_slow_ee_ori, gc_circle_slow_ee_yaw, gc_circle_slow_traj_vel_b, gc_circle_slow_pos_error, gc_circle_slow_ori_error = parse_trial(gc_circle_slow_data)
    gc_circle_fast_traj_pos, gc_circle_fast_traj_vel, gc_circle_fast_traj_ori, gc_circle_fast_traj_yaw, gc_circle_fast_ee_pos, gc_circle_fast_ee_ori, gc_circle_fast_ee_yaw, gc_circle_fast_traj_vel_b, gc_circle_fast_pos_error, gc_circle_fast_ori_error = parse_trial(gc_circle_fast_data)
    

    def plot_position_error_vs_velocity(pos_error, velocity, title):
        import numpy as np
        flat_velocity = velocity.reshape(-1, 3).numpy()
        flat_position_error = pos_error.flatten().numpy()

        # Bin velocity components
        num_bins = 100
        velocity_bins = [
            np.linspace(flat_velocity[:, i].min(), flat_velocity[:, i].max(), num_bins + 1) for i in range(3)
        ]

        # Assign velocity values to bins
        bin_indices = [
            np.digitize(flat_velocity[:, i], velocity_bins[i]) - 1 for i in range(3)
        ]
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # Aggregate position error for each velocity bin
        position_error_hist = np.zeros((num_bins, num_bins, num_bins))
        bin_counts = np.zeros((num_bins, num_bins, num_bins))

        for i in range(len(flat_position_error)):
            x_bin, y_bin, z_bin = bin_indices[0][i], bin_indices[1][i], bin_indices[2][i]
            position_error_hist[x_bin, y_bin, z_bin] += flat_position_error[i]
            bin_counts[x_bin, y_bin, z_bin] += 1

        # Compute average position error per bin
        avg_position_error = np.divide(position_error_hist, bin_counts, out=np.zeros_like(position_error_hist), where=bin_counts > 0)
        # Aggregate over axes for subplots
        avg_error_x = np.mean(avg_position_error, axis=(1, 2))  # Aggregate over y, z
        avg_error_y = np.mean(avg_position_error, axis=(0, 2))  # Aggregate over x, z
        avg_error_z = np.mean(avg_position_error, axis=(0, 1))  # Aggregate over x, y

        # Centers for plotting
        x_centers = (velocity_bins[0][:-1] + velocity_bins[0][1:]) / 2
        y_centers = (velocity_bins[1][:-1] + velocity_bins[1][1:]) / 2
        z_centers = (velocity_bins[2][:-1] + velocity_bins[2][1:]) / 2

        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=False)

        # X-axis plot
        axs[0].plot(x_centers, avg_error_x, label="Avg Position Error", color="blue")
        axs[0].set_title("Position Error vs Velocity X")
        axs[0].set_xlabel("Velocity X")
        axs[0].set_ylabel("Avg Position Error")
        axs[0].grid()

        # Y-axis plot
        axs[1].plot(y_centers, avg_error_y, label="Avg Position Error", color="green")
        axs[1].set_title("Position Error vs Velocity Y")
        axs[1].set_xlabel("Velocity Y")
        axs[1].set_ylabel("Avg Position Error")
        axs[1].grid()

        # Z-axis plot
        axs[2].plot(z_centers, avg_error_z, label="Avg Position Error", color="red")
        axs[2].set_title("Position Error vs Velocity Z")
        axs[2].set_xlabel("Velocity Z")
        axs[2].set_ylabel("Avg Position Error")
        axs[2].grid()

        plt.tight_layout()
        plt.savefig("position_error_vs_velocity_" + title + ".png")

        return avg_position_error, velocity_bins

    # I want to see the position error as a function of the components of the velocity in the body frame, as a histogram
    print(gc_lissajous_traj_vel.shape)
    print(gc_lissajous_pos_error.shape)
    print(gc_lissajous_traj_yaw.shape)
    # make a 3x1 plot of histograms of the velocity components
    # fig, axs = plt.subplots(3, 1, dpi=300)
    # axs[0].hist(gc_lissajous_traj_vel_b[:, :, 0].flatten(), bins=100)
    # axs[0].set_ylabel("X Vel")
    # axs[1].hist(gc_lissajous_traj_vel_b[:, :, 1].flatten(), bins=100)
    # axs[1].set_ylabel("Y Vel")
    # axs[2].hist(gc_lissajous_traj_vel_b[:, :, 2].flatten(), bins=100)
    # axs[2].set_ylabel("Z Vel")
    # plt.tight_layout()
    # plt.show()

    combined_vel_b_yaw_lissajous = gc_lissajous_traj_vel_b * gc_lissajous_traj_yaw.unsqueeze(-1)
    combined_vel_b_yaw_circle_slow = gc_circle_slow_traj_vel_b * gc_circle_slow_traj_yaw.unsqueeze(-1)
    combined_vel_b_yaw_circle_fast = gc_circle_fast_traj_vel_b * gc_circle_fast_traj_yaw.unsqueeze(-1)
    # print(combined_vel_b_yaw_lissajous.shape)

    avg_error_lissajous, velocity_bins_lissajous = plot_position_error_vs_velocity(gc_lissajous_pos_error, combined_vel_b_yaw_lissajous, "gc_lissajous")
    avg_error_circle_slow, velocity_bins_circle_slow = plot_position_error_vs_velocity(gc_circle_slow_pos_error, combined_vel_b_yaw_circle_slow, "gc_circle_slow")
    avg_error_circle_fast, velocity_bins_circle_fast = plot_position_error_vs_velocity(gc_circle_fast_pos_error, combined_vel_b_yaw_circle_fast, "gc_circle_fast")

    import numpy as np
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=False)

    # X-axis plot
    axs[0].plot((velocity_bins_lissajous[0][:-1] + velocity_bins_lissajous[0][1:]) / 2, np.mean(avg_error_lissajous, axis=(1,2)), label="Lissajous")
    axs[0].plot((velocity_bins_circle_slow[0][:-1] + velocity_bins_circle_slow[0][1:]) / 2, np.mean(avg_error_circle_slow, axis=(1,2)), label="Circle Slow")
    axs[0].plot((velocity_bins_circle_fast[0][:-1] + velocity_bins_circle_fast[0][1:]) / 2, np.mean(avg_error_circle_fast, axis=(1,2)), label="Circle Fast")
    # axs[0].set_title("Position Error vs Velocity X")
    axs[0].set_xlabel("Velocity X")
    axs[0].set_ylabel("Avg Position Error")
    axs[0].grid()
    axs[0].legend()

    # Y-axis plot
    axs[1].plot((velocity_bins_lissajous[1][:-1] + velocity_bins_lissajous[1][1:]) / 2, np.mean(avg_error_lissajous, axis=(0,2)), label="Lissajous")
    axs[1].plot((velocity_bins_circle_slow[1][:-1] + velocity_bins_circle_slow[1][1:]) / 2, np.mean(avg_error_circle_slow, axis=(0,2)), label="Circle Slow")
    axs[1].plot((velocity_bins_circle_fast[1][:-1] + velocity_bins_circle_fast[1][1:]) / 2, np.mean(avg_error_circle_fast, axis=(0,2)), label="Circle Fast")
    # axs[1].set_title("Position Error vs Velocity Y")
    axs[1].set_xlabel("Velocity Y")
    axs[1].set_ylabel("Avg Position Error")
    axs[1].grid()
    axs[1].legend()

    # Z-axis plot
    axs[2].plot((velocity_bins_lissajous[2][:-1] + velocity_bins_lissajous[2][1:]) / 2, np.mean(avg_error_lissajous, axis=(0,1)), label="Lissajous")
    axs[2].plot((velocity_bins_circle_slow[2][:-1] + velocity_bins_circle_slow[2][1:]) / 2, np.mean(avg_error_circle_slow, axis=(0,1)), label="Circle Slow")
    axs[2].plot((velocity_bins_circle_fast[2][:-1] + velocity_bins_circle_fast[2][1:]) / 2, np.mean(avg_error_circle_fast, axis=(0,1)), label="Circle Fast")
    # axs[2].set_title("Position Error vs Velocity Z")
    axs[2].set_xlabel("Velocity Z")
    axs[2].set_ylabel("Avg Position Error")
    axs[2].grid()
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("position_error_vs_velocity_comparison.png")
    

    position_error = gc_lissajous_pos_error.numpy()
    trajectory_velocity = combined_vel_b_yaw_lissajous.numpy()

    # Compute average velocity per trial for each component
    avg_velocity_per_trial = trajectory_velocity.mean(axis=1)  # Shape: (100, 3)
    position_rmse = np.sqrt(np.mean(position_error ** 2, axis=1))

    # Find the indices of the top 5 trials for each velocity component
    top_trials_x = np.argsort(avg_velocity_per_trial[:, 0])[-5:][::-1]
    top_trials_y = np.argsort(avg_velocity_per_trial[:, 1])[-5:][::-1]
    top_trials_z = np.argsort(avg_velocity_per_trial[:, 2])[-5:][::-1]

    top_trials_rmse = np.argsort(position_rmse)[-5:][::-1]

    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Plot position error for top trials based on Velocity X
    for trial_idx in top_trials_x:
        axs[0].plot(position_error[trial_idx], label=f"Trial {trial_idx}")
    axs[0].set_title("Position Error Over Time for Top Trials (Velocity X)")
    axs[0].set_ylabel("Position Error")
    axs[0].legend()
    axs[0].grid()

    # Plot position error for top trials based on Velocity Y
    for trial_idx in top_trials_y:
        axs[1].plot(position_error[trial_idx], label=f"Trial {trial_idx}")
    axs[1].set_title("Position Error Over Time for Top Trials (Velocity Y)")
    axs[1].set_ylabel("Position Error")
    axs[1].legend()
    axs[1].grid()

    # Plot position error for top trials based on Velocity Z
    for trial_idx in top_trials_z:
        axs[2].plot(position_error[trial_idx], label=f"Trial {trial_idx}")
    axs[2].set_title("Position Error Over Time for Top Trials (Velocity Z)")
    axs[2].set_xlabel("Time Steps")
    axs[2].set_ylabel("Position Error")
    axs[2].legend()
    axs[2].grid()

    for trial_idx in top_trials_rmse:
        axs[3].plot(position_error[trial_idx], label=f"Trial {trial_idx}")
    axs[3].set_title("Position Error Over Time for Top Trials (RMSE)")
    axs[3].set_xlabel("Time Steps")
    axs[3].set_ylabel("Position Error")
    axs[3].legend()
    axs[3].grid()

    # Adjust layout
    plt.tight_layout()
    plt.savefig("position_error_top_trials.png")
    

    

    # # Visualize the average position error as a 3D heatmap projection
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # # Generate grid for plotting
    # x_centers = (velocity_bins[0][:-1] + velocity_bins[0][1:]) / 2
    # y_centers = (velocity_bins[1][:-1] + velocity_bins[1][1:]) / 2
    # z_centers = (velocity_bins[2][:-1] + velocity_bins[2][1:]) / 2
    # x, y, z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

    # # Plot scatter points for bins with data
    # valid_bins = bin_counts > 0
    # scatter = ax.scatter(
    #     x[valid_bins], y[valid_bins], z[valid_bins], 
    #     c=avg_position_error[valid_bins], cmap="viridis", s=50
    # )
    # plt.colorbar(scatter, label="Avg Position Error")
    # ax.set_xlabel("Velocity X")
    # ax.set_ylabel("Velocity Y")
    # ax.set_zlabel("Velocity Z")
    # plt.title("Position Error as a Function of Velocity")
    # plt.savefig("position_error_vs_velocity.png")




    rl_lissajous_traj_pos = rl_lissajous_data[:, :-1, 26:29].detach().cpu()
    rl_lissajous_traj_vel = (rl_lissajous_data[:, 1:, 26:29] - rl_lissajous_data[:, :-1, 26:29]) / 0.02
    rl_lissajous_ee_pos = rl_lissajous_data[:, :-1, 13:16].detach().cpu()
    rl_lissajous_ee_ori = (rl_lissajous_data[:, :-1, 16:20]).detach().cpu()



def check_COM_ablation():
    gc_com_v_path = "./rl/baseline_0dof_small_arm_com_v_tuned/"
    gc_com_middle_path = "./rl/baseline_0dof_small_arm_com_middle_tuned/"
    gc_com_ee_path = "./rl/baseline_0dof_small_arm_com_ee_tuned/"

    # Circle (slow)
    # traj_name = "circle_slow"
    # gc_com_v_path += "circle_traj_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # gc_com_middle_path += "circle_traj_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    # gc_com_ee_path += "circle_traj_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    # Circle (fast)
    traj_name = "circle_fast"
    gc_com_v_path += "circle_traj_fast_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    gc_com_middle_path += "circle_traj_fast_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    gc_com_ee_path += "circle_traj_fast_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    gc_com_v_data = torch.load(gc_com_v_path, weights_only=True)
    gc_com_middle_data = torch.load(gc_com_middle_path, weights_only=True)
    gc_com_ee_data = torch.load(gc_com_ee_path, weights_only=True)

    print("GC COM V: ", gc_com_v_data.shape)
    print("GC COM Middle: ", gc_com_middle_data.shape)
    print("GC COM EE: ", gc_com_ee_data.shape)

    traj_pos = gc_com_v_data[:, :-1, 26:29].detach().cpu()
    gc_com_v_pos = gc_com_v_data[:, :-1, 13:16].detach().cpu()
    gc_com_middle_pos = gc_com_middle_data[:, :-1, 13:16].detach().cpu()
    gc_com_ee_pos = gc_com_ee_data[:, :-1, 13:16].detach().cpu()
    traj_yaw = math_utils.yaw_from_quat(gc_com_v_data[:, :-1, 29:]).detach().cpu()
    gc_com_v_ori = math_utils.yaw_from_quat(gc_com_v_data[:, :-1, 16:20]).detach().cpu()
    gc_com_middle_ori = math_utils.yaw_from_quat(gc_com_middle_data[:, :-1, 16:20]).detach().cpu()
    gc_com_ee_ori = math_utils.yaw_from_quat(gc_com_ee_data[:, :-1, 16:20]).detach().cpu()

    gc_com_v_pos_error = torch.norm(gc_com_v_pos - gc_com_v_data[:, :-1, 26:29].detach().cpu(), dim=-1)
    gc_com_middle_pos_error = torch.norm(gc_com_middle_pos - gc_com_middle_data[:, :-1, 26:29].detach().cpu(), dim=-1)
    gc_com_ee_pos_error = torch.norm(gc_com_ee_pos - gc_com_ee_data[:, :-1, 26:29].detach().cpu(), dim=-1)
    gc_com_v_ori_error = torch.abs(isaac_math_utils.wrap_to_pi(gc_com_v_ori - traj_yaw))
    gc_com_middle_ori_error = torch.abs(isaac_math_utils.wrap_to_pi(gc_com_middle_ori - traj_yaw))
    gc_com_ee_ori_error = torch.abs(isaac_math_utils.wrap_to_pi(gc_com_ee_ori - traj_yaw))

    # get median and 2 quartiles for the errors
    quantiles = torch.tensor([0.25, 0.5, 0.75])
    gc_com_v_pos_error = torch.quantile(gc_com_v_pos_error, quantiles, dim=0)
    gc_com_middle_pos_error = torch.quantile(gc_com_middle_pos_error, quantiles, dim=0)
    gc_com_ee_pos_error = torch.quantile(gc_com_ee_pos_error, quantiles, dim=0)
    gc_com_v_ori_error = torch.quantile(gc_com_v_ori_error, quantiles, dim=0)
    gc_com_middle_ori_error = torch.quantile(gc_com_middle_ori_error, quantiles, dim=0)
    gc_com_ee_ori_error = torch.quantile(gc_com_ee_ori_error, quantiles, dim=0)

    fig, axs = plt.subplots(1,2, dpi=300)
    time = torch.arange(0, gc_com_v_pos_error.shape[1]) * 0.02
    axs[0].plot(time, gc_com_v_pos_error[1], label="GC COM V", color="tab:orange")
    axs[0].plot(time, gc_com_middle_pos_error[1], label="GC COM Middle", color="tab:blue")
    axs[0].plot(time, gc_com_ee_pos_error[1], label="GC COM EE", color="tab:green")
    axs[0].fill_between(time, gc_com_v_pos_error[0], gc_com_v_pos_error[2], color="tab:orange", alpha=0.2)
    axs[0].fill_between(time, gc_com_middle_pos_error[0], gc_com_middle_pos_error[2], color="tab:blue", alpha=0.2)
    axs[0].fill_between(time, gc_com_ee_pos_error[0], gc_com_ee_pos_error[2], color="tab:green", alpha=0.2)
    axs[0].set_ylabel("Position Error (m)")
    axs[0].set_xlabel("Time (s)")
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(time, gc_com_v_ori_error[1], label="GC COM V", color="tab:orange")
    axs[1].plot(time, gc_com_middle_ori_error[1], label="GC COM Middle", color="tab:blue")
    axs[1].plot(time, gc_com_ee_ori_error[1], label="GC COM EE", color="tab:green")
    axs[1].fill_between(time, gc_com_v_ori_error[0], gc_com_v_ori_error[2], color="tab:orange", alpha=0.2)
    axs[1].fill_between(time, gc_com_middle_ori_error[0], gc_com_middle_ori_error[2], color="tab:blue", alpha=0.2)
    axs[1].fill_between(time, gc_com_ee_ori_error[0], gc_com_ee_ori_error[2], color="tab:green", alpha=0.2)
    axs[1].set_ylabel("Yaw Error (rad)")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid()
    axs[1].legend()
    plt.tight_layout()
    plt.suptitle("Traj: " + traj_name)
    plt.savefig("com_ablation_" + traj_name + ".png")

def check_finite_difference():
    t = torch.arange(0, 101) * 0.02
    n_envs = 4096
    amps = torch.tensor([0, 0, 0, 0])
    freqs = torch.tensor([0, 0, 0.0, 0.0])
    phases = torch.tensor([0.0, 0, 0.0, 0.0])
    offsets = torch.tensor([0.0, 0.0, 0.5, 0.0])
    coeffs = torch.zeros((n_envs, 4, 2))
    coeffs[:, -1, 0] = -3.14159
    coeffs[:, -1, 1] = -1.570794

    out = traj_utils.eval_polynomial_curve(t, coeffs, derivatives=0)
    print(out)
    yaw_poly[0,0,:] = isaac_math_utils.wrap_to_pi(yaw_poly[0,0,:])

    yaw_dot = torch.zeros_like(yaw_poly)
    yaw_dot[0,0,1:] = (yaw_poly[0,0,1:] - yaw_poly[0,0,:-1]) / 0.02

    # plot yaw and yaw_dot 
    yaw_poly = yaw_poly.detach().cpu().numpy()
    yaw_dot = yaw_dot.detach().cpu().numpy()
    plt.figure()
    plt.plot(t, yaw_poly[0,0,:], label="Yaw")
    plt.plot(t, yaw_dot[0,0,:], label="Yaw Dot")
    plt.legend()
    plt.show()

def check_gc_shape():
    t_points = (torch.arange(0, 500) * 0.02).unsqueeze(0)
    print("T: ", t_points.shape)
    amps = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    freqs = torch.tensor([[1.570796, 1.570796, 0.0, 0.0]])
    phases = torch.tensor([[0.0, 1.570796, 0.0, 0.0]])
    offsets = torch.tensor([[0.0, 0.0, 3.0, 0.0]])

    com_offset = torch.tensor([0.0000e+00, -2.0072e-01, -1.5983e-04])

    ref_pos, ref_yaw = traj_utils.eval_lissajous_curve(t_points, amps, freqs, phases, offsets, derivatives=4)
    print(ref_pos[0,0,:,:].shape) # (num_derivatives + 1, n_envs, 3, n_samples)
    # add the COM offset to the ref_pos
    ref_pos_com = ref_pos[0,0,:,:] + com_offset.unsqueeze(1)
    print(ref_pos_com.shape)


    s_buffer = torch.load("s_buffer.pt", weights_only=True).squeeze()
    s_dot_buffer = torch.load("s_dot_buffer.pt", weights_only=True).squeeze()
    s_des_buffer = torch.load("s_des_buffer.pt", weights_only=True).squeeze()
    s_dot_des_buffer = torch.load("s_dot_des_buffer.pt", weights_only=True).squeeze()
    ref_buffer = torch.load("ref_pos_buffer.pt", weights_only=True).squeeze()
    pos_buffer = torch.load("pos_buffer.pt", weights_only=True).squeeze()
    ref_buffer = ref_buffer - pos_buffer
    ref_buffer = isaac_math_utils.normalize(ref_buffer)
    
    plt.figure(dpi=900, figsize=(10, 10))
    plt.subplot(3,2,1)
    # Plot X values for buffers
    plt.plot(s_buffer[:, 0], label="s")
    plt.plot(s_des_buffer[:, 0], label="s_des")
    plt.plot(s_dot_buffer[:, 0], label="s_dot")
    plt.plot(s_dot_des_buffer[:, 0], label="s_dot_des")
    plt.ylabel("X")
    plt.xlabel("timesteps")
    plt.legend()
    plt.subplot(3,2,3)
    # Plot Y values for buffers
    plt.plot(s_buffer[:, 1], label="s")
    plt.plot(s_des_buffer[:, 1], label="s_des")
    plt.plot(s_dot_buffer[:, 1], label="s_dot")
    plt.plot(s_dot_des_buffer[:, 1], label="s_dot_des")
    plt.ylabel("Y")
    plt.xlabel("timesteps")
    plt.legend()
    plt.subplot(3,2,5)
    plt.plot(s_buffer[:, 2], label="s")
    plt.plot(s_des_buffer[:, 2], label="s_des")
    plt.plot(s_dot_buffer[:, 2], label="s_dot")
    plt.plot(s_dot_des_buffer[:, 2], label="s_dot_des")
    plt.ylabel("Z")
    plt.xlabel("timesteps")
    plt.legend()


    plt.subplot(3,2,2)
    plt.plot(s_buffer[:, 0], label="s")
    plt.plot(s_des_buffer[:, 0], label="s_des")
    plt.plot(ref_buffer[:, 0], label="ref_gc")
    # plt.plot(ref_pos_com[0,:], label="ref (COM)")
    # plt.plot(ref_pos[0,0,0,:], label="ref (EE)")
    # plt.plot(ref_pos[1,0,0,:], label="ref_dot (EE)")
    plt.xlabel("timesteps")
    plt.ylabel("X")
    plt.legend()
    plt.subplot(3,2,4)
    plt.plot(s_buffer[:, 1], label="s")
    plt.plot(s_des_buffer[:, 1], label="s_des")
    plt.plot(ref_buffer[:, 1], label="ref_gc")
    # plt.plot(ref_pos_com[1,:], label="ref (COM)")
    # plt.plot(ref_pos[0,0,1,:], label="ref (EE)")
    # plt.plot(ref_pos[1,0,1,:], label="ref_dot (EE)")
    plt.xlabel("timesteps")
    plt.ylabel("Y")
    plt.legend()
    plt.subplot(3,2,6)
    plt.plot(s_des_buffer[:, 2], label="s_des")
    plt.plot(s_buffer[:, 2], label="s")
    plt.plot(ref_buffer[:, 2], label="ref_gc")
    # plt.plot(ref_pos_com[2,:], label="ref (COM)")
    # plt.plot(ref_pos[0,0,2,:], label="ref (EE)")
    # plt.plot(ref_pos[1,0,2,:], label="ref_dot (EE)")
    plt.xlabel("timesteps")
    plt.ylabel("Z")
    plt.legend()

    plt.savefig("gc_shape_no_grav.png")

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


def check_circle_trajs():
    circle_tangent_path = "./rl/baseline_0dof_ee_reward_tune/circle_traj_tangent_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    circle_slow_path = "./rl/baseline_0dof_ee_reward_tune/circle_traj_extra_info_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    circle_fast_path = "./rl/baseline_0dof_ee_reward_tune/circle_traj_fast_extra_info_rand_init_eval_traj_track_50Hz_eval_full_states.pt"
    lissajous_path = "./rl/baseline_0dof_ee_reward_tune/4d_lissajous_extra_info_rand_init_eval_traj_track_50Hz_eval_full_states.pt"

    circle_tangent_data = torch.load(circle_tangent_path, weights_only=True)
    circle_slow_data = torch.load(circle_slow_path, weights_only=True)
    circle_fast_data = torch.load(circle_fast_path, weights_only=True)
    lissajous_data = torch.load(lissajous_path, weights_only=True)

    lissajous_pos_error = torch.norm(lissajous_data[:, :, 26:29] - lissajous_data[:, :, 13:16], dim=-1).cpu()
    lissajous_rmse = torch.sqrt(torch.mean(lissajous_pos_error ** 2, dim=-1)).cpu()
    lissajous_settling_time = compute_settling_time(lissajous_pos_error, tolerance=0.05, dt=0.02).cpu()
    print("Lissajous Settling Time: ", lissajous_settling_time.shape)
    highest_rmse_env = torch.argmax(lissajous_rmse)
    lowest_rmse_env = torch.argmin(lissajous_rmse)

    print("Highest RMSE: ", highest_rmse_env)
    print("Lowest RMSE: ", lowest_rmse_env)

    circle_tangent_pos_traj = circle_tangent_data[0, :, 33:42].cpu()
    circle_tangent_yaw_traj = circle_tangent_data[0, :, 42:44].cpu()
    circle_slow_pos_traj = circle_slow_data[0, :, 33:42].cpu()
    circle_slow_yaw_traj = circle_slow_data[0, :, 42:44].cpu()
    circle_fast_pos_traj = circle_fast_data[0, :, 33:42].cpu()
    circle_fast_yaw_traj = circle_fast_data[0, :, 42:44].cpu()
    lissajous_pos_traj = lissajous_data[highest_rmse_env, :, 33:42].cpu()
    lissajous_yaw_traj = lissajous_data[highest_rmse_env, :, 42:44].cpu()
    lissajous_pos_traj_low = lissajous_data[lowest_rmse_env, :, 33:42].cpu()
    lissajous_yaw_traj_low = lissajous_data[lowest_rmse_env, :, 42:44].cpu()
    lissajous_pos_traj_all = lissajous_data[:, :, 33:42].cpu()
    lissajous_yaw_traj_all = lissajous_data[:, :, 42:44].cpu()

    
    fig, axs = plt.subplots(1,2)
    axs[0].boxplot(lissajous_settling_time)
    axs[0].set_title("Lissajous Settling Time")
    axs[1].boxplot(lissajous_rmse)
    axs[1].set_title("Lissajous RMSE")



    print(circle_fast_pos_traj.shape)
    print(circle_fast_pos_traj[0,:])


    # fig, axs = plt.subplots(4,1)
    time = torch.arange(0, circle_tangent_data.shape[1]) * 0.02
    # axs[0].plot(time, circle_tangent_pos_traj[:,0], label="Pos")
    # axs[0].plot(time, circle_tangent_pos_traj[:,3], label="Vel")
    # axs[0].plot(time, circle_tangent_pos_traj[:,6], label="Acc")
    # axs[0].set_ylabel("X")
    # axs[0].legend()

    # axs[1].plot(time, circle_tangent_pos_traj[:,1], label="Pos")
    # axs[1].plot(time, circle_tangent_pos_traj[:,4], label="Vel")
    # axs[1].plot(time, circle_tangent_pos_traj[:,7], label="Acc")
    # axs[1].set_ylabel("Y")
    # axs[1].legend()

    # axs[2].plot(time, circle_tangent_pos_traj[:,2], label="Pos")
    # axs[2].plot(time, circle_tangent_pos_traj[:,5], label="Vel")
    # axs[2].plot(time, circle_tangent_pos_traj[:,8], label="Acc")
    # axs[2].set_ylabel("Z")
    # axs[2].legend()

    # axs[3].plot(time, isaac_math_utils.wrap_to_pi(circle_tangent_yaw_traj[:,0]), label="Yaw")
    # axs[3].plot(time, circle_tangent_yaw_traj[:,1], label="Yaw Dot")
    # axs[3].set_ylabel("Yaw")
    # axs[3].legend()
    # axs[3].set_xlabel("Time (s)")
    # plt.tight_layout()

    # find the angle between the velocity vector and the yaw vector. 
    # This should be zero for the tangent traj
    tangent_vel = circle_tangent_pos_traj[:,3:6]
    tangent_acc = circle_tangent_pos_traj[:,6:9]
    slow_vel = circle_slow_pos_traj[:,3:6]
    slow_acc = circle_slow_pos_traj[:,6:9]
    fast_vel = circle_fast_pos_traj[:,3:6]
    fast_acc = circle_fast_pos_traj[:,6:9]
    lissajous_vel = lissajous_pos_traj[:,3:6]
    lissajous_acc = lissajous_pos_traj[:,6:9]
    lissajous_vel_low = lissajous_pos_traj_low[:,3:6]
    lissajous_acc_low = lissajous_pos_traj_low[:,6:9]

    lissajous_vel_all = lissajous_pos_traj_all[:,:,3:6]
    lissajous_acc_all = lissajous_pos_traj_all[:,:,6:9]


    tangent_yaw = isaac_math_utils.wrap_to_pi(circle_tangent_yaw_traj[:,0])
    slow_yaw = isaac_math_utils.wrap_to_pi(circle_slow_yaw_traj[:,0])
    fast_yaw = isaac_math_utils.wrap_to_pi(circle_fast_yaw_traj[:,0])
    lissajous_yaw = isaac_math_utils.wrap_to_pi(lissajous_yaw_traj[:,0])
    lissajous_yaw_low = isaac_math_utils.wrap_to_pi(lissajous_yaw_traj_low[:,0])
    lissajous_yaw_all = isaac_math_utils.wrap_to_pi(lissajous_yaw_traj_all[:,:,0])

    # vel = vel / torch.norm(vel, dim=-1, keepdim=True)
    tangent_vel_norm = isaac_math_utils.normalize(tangent_vel)
    tangent_acc_norm = isaac_math_utils.normalize(tangent_acc)
    tangent_yaw_dir = torch.stack([torch.cos(tangent_yaw), torch.sin(tangent_yaw), torch.zeros_like(tangent_yaw)], dim=-1)
    slow_vel_norm = isaac_math_utils.normalize(slow_vel)
    slow_acc_norm = isaac_math_utils.normalize(slow_acc)
    slow_yaw_dir = torch.stack([torch.cos(slow_yaw), torch.sin(slow_yaw), torch.zeros_like(slow_yaw)], dim=-1)
    fast_vel_norm = isaac_math_utils.normalize(fast_vel)
    fast_acc_norm = isaac_math_utils.normalize(fast_acc)
    fast_yaw_dir = torch.stack([torch.cos(fast_yaw), torch.sin(fast_yaw), torch.zeros_like(fast_yaw)], dim=-1)
    lissajous_vel_norm = isaac_math_utils.normalize(lissajous_vel)
    lissajous_acc_norm = isaac_math_utils.normalize(lissajous_acc)
    lissajous_yaw_dir = torch.stack([torch.cos(lissajous_yaw), torch.sin(lissajous_yaw), torch.zeros_like(lissajous_yaw)], dim=-1)
    lissajous_vel_norm_low = isaac_math_utils.normalize(lissajous_vel_low)
    lissajous_acc_norm_low = isaac_math_utils.normalize(lissajous_acc_low)
    lissajous_yaw_dir_low = torch.stack([torch.cos(lissajous_yaw_low), torch.sin(lissajous_yaw_low), torch.zeros_like(lissajous_yaw_low)], dim=-1)

    dot_yaw_tangent_vel = torch.sum(tangent_vel_norm * tangent_yaw_dir, dim=-1)
    dot_yaw_tangent_acc = torch.sum(tangent_acc_norm * tangent_yaw_dir, dim=-1)
    dot_yaw_slow_vel = torch.sum(slow_vel_norm * slow_yaw_dir, dim=-1)
    dot_yaw_slow_acc = torch.sum(slow_acc_norm * slow_yaw_dir, dim=-1)
    dot_yaw_fast_vel = torch.sum(fast_vel_norm * fast_yaw_dir, dim=-1)
    dot_yaw_fast_acc = torch.sum(fast_acc_norm * fast_yaw_dir, dim=-1)
    dot_yaw_lissajous_vel = torch.sum(lissajous_vel_norm * lissajous_yaw_dir, dim=-1)
    dot_yaw_lissajous_acc = torch.sum(lissajous_acc_norm * lissajous_yaw_dir, dim=-1)
    dot_yaw_lissajous_vel_low = torch.sum(lissajous_vel_norm_low * lissajous_yaw_dir_low, dim=-1)
    dot_yaw_lissajous_acc_low = torch.sum(lissajous_acc_norm_low * lissajous_yaw_dir_low, dim=-1)

    angle_yaw_tangent_vel = torch.acos(dot_yaw_tangent_vel.clamp(-1.0, 1.0))
    angle_yaw_tangent_acc = torch.acos(dot_yaw_tangent_acc.clamp(-1.0, 1.0))
    angle_yaw_slow_vel = torch.acos(dot_yaw_slow_vel.clamp(-1.0, 1.0))
    angle_yaw_slow_acc = torch.acos(dot_yaw_slow_acc.clamp(-1.0, 1.0))
    angle_yaw_fast_vel = torch.acos(dot_yaw_fast_vel.clamp(-1.0, 1.0))
    angle_yaw_fast_acc = torch.acos(dot_yaw_fast_acc.clamp(-1.0, 1.0))
    angle_yaw_lissajous_vel = torch.acos(dot_yaw_lissajous_vel.clamp(-1.0, 1.0))
    angle_yaw_lissajous_acc = torch.acos(dot_yaw_lissajous_acc.clamp(-1.0, 1.0))
    angle_yaw_lissajous_vel_low = torch.acos(dot_yaw_lissajous_vel_low.clamp(-1.0, 1.0))
    angle_yaw_lissajous_acc_low = torch.acos(dot_yaw_lissajous_acc_low.clamp(-1.0, 1.0))

    print(angle_yaw_tangent_vel[0])
    print(angle_yaw_tangent_acc[0])
    print(angle_yaw_slow_vel[0])
    print(angle_yaw_slow_acc[0])
    print(angle_yaw_fast_vel[0])
    print(angle_yaw_fast_acc[0])

    plt.figure()
    # plt.plot(time, angle_yaw_tangent_vel, label="circle_tangent vel")
    # plt.plot(time, angle_yaw_tangent_acc, label="circle_tangent acc")
    # plt.plot(time, angle_yaw_slow_vel, label="circle_slow vel")
    # plt.plot(time, angle_yaw_slow_acc, label="circle_slow acc")
    # plt.plot(time, angle_yaw_fast_vel, label="circle_fast vel")
    # plt.plot(time, angle_yaw_fast_acc, label="circle_fast acc")
    plt.plot(time, angle_yaw_lissajous_vel, label="highest RMSE vel")
    plt.plot(time, angle_yaw_lissajous_acc, label="highest RMSE acc")
    plt.plot(time, angle_yaw_lissajous_vel_low, label="lowest RMSE vel")
    plt.plot(time, angle_yaw_lissajous_acc_low, label="lowest RMSE acc")
    plt.legend()
    plt.ylabel("Angle (rad)")
    plt.xlabel("Time (s)")

    # print(lissajous_acc_norm.shape)

    avg_lissajous_vel_norm = torch.mean(lissajous_vel_norm, dim=0)
    avg_lissajous_acc_norm = torch.mean(lissajous_acc_norm, dim=0)
    avg_lissajous_acc_norm_low = torch.mean(lissajous_acc_norm_low, dim=0)
    avg_lissajous_vel_norm_low = torch.mean(lissajous_vel_norm_low, dim=0)

    avg_lissajous_vel_norm_all = torch.mean(isaac_math_utils.normalize(lissajous_vel_all), dim=1)
    avg_lissajous_acc_norm_all = torch.mean(isaac_math_utils.normalize(lissajous_acc_all), dim=1)
    
    print("----------")
    print(lissajous_vel_all.shape)
    print(avg_lissajous_vel_norm_all.shape)

    # color_metric = -lissajous_rmse
    color_metric = -lissajous_settling_time


    fig, axs = plt.subplots(3, 1)
    # plt.scatter(avg_lissajous_vel_norm, avg_lissajous_acc_norm, label="All")
    axs[0].scatter(avg_lissajous_vel_norm_all[:,0], avg_lissajous_acc_norm_all[:,0], c=color_metric, cmap="winter", label="Lissajous All")
    axs[0].scatter(avg_lissajous_vel_norm[0], avg_lissajous_acc_norm[0], marker="*", label="Highest RMSE")
    axs[0].scatter(avg_lissajous_vel_norm_low[0], avg_lissajous_acc_norm_low[0], marker="*", label="Lowest RMSE")
    axs[0].set_ylabel("X Vel")
    axs[0].set_xlabel("X Acc")
    axs[0].legend()

    axs[1].scatter(avg_lissajous_vel_norm_all[:,1], avg_lissajous_acc_norm_all[:,1], c=color_metric, cmap="winter", label="Lissajous All")
    axs[1].scatter(avg_lissajous_vel_norm[1], avg_lissajous_acc_norm[1], marker="*", label="Highest RMSE")
    axs[1].scatter(avg_lissajous_vel_norm_low[1], avg_lissajous_acc_norm_low[1], marker="*", label="Lowest RMSE")
    axs[1].set_ylabel("Y Vel")
    axs[1].set_xlabel("Y Acc")
    axs[1].legend()

    axs[2].scatter(avg_lissajous_vel_norm_all[:,2], avg_lissajous_acc_norm_all[:,2], c=color_metric, cmap="winter", label="Lissajous All")
    axs[2].scatter(avg_lissajous_vel_norm[2], avg_lissajous_acc_norm[2], marker="*", label="Highest RMSE")
    axs[2].scatter(avg_lissajous_vel_norm_low[2], avg_lissajous_acc_norm_low[2], marker="*", label="Lowest RMSE")
    axs[2].set_ylabel("Z Vel")
    axs[2].set_xlabel("Z Acc")
    axs[2].legend()
    fig.suptitle("Trajectory Vel vs. Acc World Frame")


    # Rotate vel and acc in to body frame
    lissajous_ee_ori = math_utils.quat_from_yaw(lissajous_yaw_all)
    lissajous_vel_all_b = isaac_math_utils.quat_rotate_inverse(lissajous_ee_ori, lissajous_vel_all)
    lissajous_acc_all_b = isaac_math_utils.quat_rotate_inverse(lissajous_ee_ori, lissajous_acc_all)

    circle_slow_ee_ori = math_utils.quat_from_yaw(slow_yaw)
    circle_tangent_ee_ori = math_utils.quat_from_yaw(tangent_yaw)
    circle_fast_ee_ori = math_utils.quat_from_yaw(fast_yaw)
    circle_slow_vel_b = isaac_math_utils.quat_rotate_inverse(circle_slow_ee_ori, slow_vel)
    circle_slow_acc_b = isaac_math_utils.quat_rotate_inverse(circle_slow_ee_ori, slow_acc)
    circle_tangent_vel_b = isaac_math_utils.quat_rotate_inverse(circle_tangent_ee_ori, tangent_vel)
    circle_tangent_acc_b = isaac_math_utils.quat_rotate_inverse(circle_tangent_ee_ori, tangent_acc)
    circle_fast_vel_b = isaac_math_utils.quat_rotate_inverse(circle_fast_ee_ori, fast_vel)
    circle_fast_acc_b = isaac_math_utils.quat_rotate_inverse(circle_fast_ee_ori, fast_acc)

    avg_lissajous_vel_norm_all_b = torch.mean(isaac_math_utils.normalize(lissajous_vel_all_b), dim=1)
    avg_lissajous_acc_norm_all_b = torch.mean(isaac_math_utils.normalize(lissajous_acc_all_b), dim=1)

    avg_circle_slow_vel_norm_b = torch.mean(isaac_math_utils.normalize(circle_slow_vel_b), dim=0)
    avg_circle_slow_acc_norm_b = torch.mean(isaac_math_utils.normalize(circle_slow_acc_b), dim=0)
    avg_circle_tangent_vel_norm_b = torch.mean(isaac_math_utils.normalize(circle_tangent_vel_b), dim=0)
    avg_circle_tangent_acc_norm_b = torch.mean(isaac_math_utils.normalize(circle_tangent_acc_b), dim=0)
    avg_circle_fast_vel_norm_b = torch.mean(isaac_math_utils.normalize(circle_fast_vel_b), dim=0)
    avg_circle_fast_acc_norm_b = torch.mean(isaac_math_utils.normalize(circle_fast_acc_b), dim=0)

    fig, axs = plt.subplots(3, 1)
    
    axs[0].scatter(avg_lissajous_vel_norm_all_b[:,0], avg_lissajous_acc_norm_all_b[:,0], c=color_metric, cmap="winter", label="Lissajous All")
    axs[0].scatter(avg_circle_slow_vel_norm_b[0], avg_circle_slow_acc_norm_b[0], marker='s', label="Circle Slow")
    axs[0].scatter(avg_circle_tangent_vel_norm_b[0], avg_circle_tangent_acc_norm_b[0], marker='s', label="Circle Tangent")
    axs[0].scatter(avg_circle_fast_vel_norm_b[0], avg_circle_fast_acc_norm_b[0], marker='s', c='r', label="Circle Fast")
    axs[0].set_ylabel("X Vel")
    axs[0].set_xlabel("X Acc")
    axs[0].legend()

    axs[1].scatter(avg_lissajous_vel_norm_all_b[:,1], avg_lissajous_acc_norm_all_b[:,1], c=color_metric, cmap="winter", label="Lissajous All")
    axs[1].scatter(avg_circle_slow_vel_norm_b[1], avg_circle_slow_acc_norm_b[1], marker='s', label="Circle Slow")
    axs[1].scatter(avg_circle_tangent_vel_norm_b[1], avg_circle_tangent_acc_norm_b[1], marker='s', label="Circle Tangent")
    axs[1].scatter(avg_circle_fast_vel_norm_b[1], avg_circle_fast_acc_norm_b[1], marker='s', c='r',label="Circle Fast")
    axs[1].set_ylabel("Y Vel")
    axs[1].set_xlabel("Y Acc")
    axs[1].legend()

    axs[2].scatter(avg_lissajous_vel_norm_all_b[:,2], avg_lissajous_acc_norm_all_b[:,2], c=color_metric, cmap="winter", label="Lissajous All")
    axs[2].scatter(avg_circle_slow_vel_norm_b[2], avg_circle_slow_acc_norm_b[2], marker='s', label="Circle Slow")
    axs[2].scatter(avg_circle_tangent_vel_norm_b[2], avg_circle_tangent_acc_norm_b[2], marker='s', label="Circle Tangent")
    axs[2].scatter(avg_circle_fast_vel_norm_b[2], avg_circle_fast_acc_norm_b[2], marker='s', c='r', label="Circle Fast")
    axs[2].set_ylabel("Z Vel")
    axs[2].set_xlabel("Z Acc")
    axs[2].legend()

    fig.suptitle("Trajectory Vel vs. Acc Body Frame")

    plt.show()



if __name__ == "__main__":
    # check_traj_gen()
    # check_eval_rollout()
    # check_poly_traj()
    # check_combined_traj()
    
    # check_traj_data()
    # investigate_lissajous_vs_circle()
    # check_COM_ablation()

    check_circle_trajs()

    # check_finite_difference()
    # check_gc_shape()