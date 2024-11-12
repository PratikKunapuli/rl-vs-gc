# Launch Sim window
import argparse
# from isaacsim import SimulationApp
from omni.isaac.lab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run demo with Isaac Sim")
parser.add_argument("--video", action="store_true", help="Record video")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=100, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-AerialManipulator-0DOF-Debug-Hover-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

args_cli = parser.parse_args()


# simulation_app = SimulationApp(vars(args_cli))

# args_cli.headless=False
args_cli.headless=True
args_cli.enable_cameras=True
# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab.utils.math as isaac_math_utils
import utils.math_utilities as math_utils

import gymnasium as gym
import torch
import numpy as np
# from envs.hover import hover_env
import envs
import envs.hover
# from AerialManipulation.envs.hover import hover_env

from controllers.decoupled_controller import DecoupledController
from controllers.nmpc import NMPC

def main():

    # Tasks are: 
    # "Isaac-AerialManipulator-2DOF-Hover-v0"
    # "Isaac-AerialManipulator-1DOF-Hover-v0"
    # "Isaac-AerialManipulator-0DOF-Hover-v0"

    # Can also specify the task body
    # env_cfg.task_body = "root" or "vehicle" or "endeffector"

    # Can also specify the task goal
    # env_cfg.task_goal = "rand" or "fixed" or "initial"

    env_cfg = parse_env_cfg(args_cli.task, num_envs= args_cli.num_envs, use_fabric=not args_cli.disable_fabric)

    # env_cfg.viewer.eye = (5.0, 2.0, 2.0)
    # env_cfg.viewer.resolution = (1920, 1080)
    # env_cfg.viewer.lookat = (0.0, 1.5, 0.5)
    # env_cfg.viewer.origin_type = "env"
    # env_cfg.viewer.env_index = 0
    
    env_cfg.viewer.eye = (-0.25, 0.25, 0.25)
    env_cfg.viewer.resolution = (1280, 720)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.asset_name = "robot"

    # env_cfg.viewer.eye = (-0.75, -0.5, 3.75)
    # env_cfg.viewer.resolution = (1280, 720)
    # env_cfg.viewer.lookat = (0.0, -0.5, 3.0)
    # env_cfg.viewer.origin_type = "env"
    # env_cfg.viewer.env_index = 0

    print("Dropping into Demo_Env")
    import code; code.interact(local=locals())

    print(env_cfg.robot.spawn)

    env_cfg.seed = 1
    env_cfg.goal_cfg = "rand" # "rand" or "fixed"
    # env_cfg.goal_pos = [-0.2007, -0.2007, 0.5]
    # env_cfg.goal_pos = [-0.15, -0.15, 0.5]
    env_cfg.goal_pos = [0.0, -1.0, 3.0]
    
    env_cfg.goal_ori = [1.0, 0.0, 0.0, 0.0] # 0 deg z axis
    # env_cfg.goal_ori = [0.7071068, 0.0, 0.0, 0.7071068] # pi/2 deg z axis
    # env_cfg.goal_ori = [0.0871557, 0.0, 0.0, 0.9961947] # 170 deg z axis

    
    
    # env_cfg.sim_rate_hz = 100
    # env_cfg.policy_rate_hz = 50
    # env_cfg.sim.dt = 1/env_cfg.sim_rate_hz
    # env_cfg.decimation = env_cfg.sim_rate_hz // env_cfg.policy_rate_hz
    # env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.eval_mode = True
    # env_cfg.init_cfg = "fixed"

    env_cfg.task_body = "root"
    env_cfg.goal_body = "COM"

    env_cfg.gc_mode = True
    env_cfg.control_mode = "CTATT"

    
    if "Traj" in args_cli.task:
        env_cfg.trajectory_params["x_amp"] = 1.01
        env_cfg.trajectory_params["y_amp"] = 0.0
        env_cfg.trajectory_params["z_amp"] = 0.0
        env_cfg.trajectory_params["z_offset"] = 0.5
        env_cfg.trajectory_params["yaw_amp"] = 1.0
        env_cfg.trajectory_params["yaw_freq"] = 1.0
        env_cfg.traj_update_dt = 0.02

        # env_cfg.viewer.eye = (3.0, 3.0, 1.25)
        # env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
        # env_cfg.viewer.origin_type = "asset_root"
        # env_cfg.viewer.env_index = 0
        # env_cfg.viewer.asset_name = "robot"

        print("Lissajous Params: ", env_cfg.trajectory_params)


    # Turn gravity off
    env_cfg.robot.spawn.rigid_props.disable_gravity = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # Get mass from env
    vehicle_mass = env.vehicle_mass # this is pulled from the body "vehicle" in the USD file
    # vehicle_mass = torch.tensor([0.706028], device=env.device)
    arm_mass = env.arm_mass 
    # arm_mass = env.arm_mass - vehicle_mass
    inertia =  env.quad_inertia
    arm_offset = env.arm_offset
    pos_offset = env.position_offset
    ori_offset = env.orientation_offset
    print("Vehicle Mass: ", vehicle_mass)
    print("Arm Mass: ", arm_mass)
    print("Mass: ", vehicle_mass + arm_mass)
    print("Inertia: ", inertia)
    print("Arm Offset: ", arm_offset)
    print("Pos Offset: ", pos_offset)
    print("Ori Offset: ", ori_offset)

    # input("Press Enter to continue...")

    # gc = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=env.device,
    #                                 kp_pos_gain_xy=6.5, kp_pos_gain_z=15.0, kd_pos_gain_xy=4.0, kd_pos_gain_z=9.0,
    #                                 kp_att_gain_xy=544, kp_att_gain_z=544, kd_att_gain_xy=46.64, kd_att_gain_z=46.64, 
    #                                 skip_precompute=True, vehicle="Crazyflie", control_mode="CTATT")

    # gc = DecoupledController(args_cli.num_envs, 0, vehicle_mass, arm_mass, inertia, arm_offset, ori_offset, com_pos_w=env.com_pos_w, device=env.device)
    
    # gc = DecoupledController(args_cli.num_envs, 0, vehicle_mass, arm_mass, inertia, arm_offset, ori_offset, print_debug=True, com_pos_w=None, device=env.device,
    #                          use_full_obs=False)
    gc = DecoupledController(args_cli.num_envs, 0, vehicle_mass, arm_mass, inertia, arm_offset, ori_offset, print_debug=True, com_pos_w=None, device=env.device,
                             kp_pos_gain_xy=43.507, kp_pos_gain_z=24.167, kd_pos_gain_xy=9.129, kd_pos_gain_z=6.081,
                             kp_att_gain_xy=998.777, kp_att_gain_z=18.230, kd_att_gain_xy=47.821, kd_att_gain_z=8.818)
    
    # nmpc = NMPC(args_cli.num_envs, (vehicle_mass+arm_mass).detach().cpu().numpy(), inertia.detach().cpu().numpy())
    
    # print("Quad in EE Frame: ", gc.quad_pos_ee_frame)
    # print("COM in EE Frame: ", gc.com_pos_ee_frame)


    video_kwargs = {
        "video_folder": "videos",
        "step_trigger": lambda step: step == 0,
        # "episode_trigger": lambda episode: episode == 0,
        "video_length": 501,
        "name_prefix": "crazyflie_srt_ctatt"
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)


    obs_dict, info = env.reset()

    gc_obs = obs_dict["gc"]
    # init_states = np.zeros((args_cli.num_envs, 18, 1))
    # goals = np.zeros((args_cli.num_envs, 12, 1))
    # init_states[:,:3,0] = gc_obs[:,:3].detach().cpu().numpy()
    # init_states[:,3:12,0] = isaac_math_utils.matrix_from_quat(gc_obs[:,3:7]).detach().cpu().numpy().reshape(-1, 9)
    # init_states[:,12:15,0] = gc_obs[:,7:10].detach().cpu().numpy()
    # init_states[:,15:18,0] = gc_obs[:,10:13].detach().cpu().numpy()
    # goals[:,:3,0] = gc_obs[:,13:16].detach().cpu().numpy()
    # goals[:,3:12,0] = isaac_math_utils.matrix_from_quat(math_utils.quat_from_yaw(gc_obs[:,16])).detach().cpu().numpy().reshape(-1, 9)
    # nmpc.initialize(init_states, goals)




    done = False
    done_count = 0
    ee_omega_list = []
    quad_omega_list = []
    ee_pos_list = []
    
    # import code; code.interact(local=locals())
    
    while simulation_app.is_running():
        while done_count < 1:
            obs_tensor = obs_dict["policy"]
            # action = env.action_space.sample()
            # action = torch.zeros_like(torch.from_numpy(env.action_space.sample()))
            # action[0]= -1.0/3.0 # nominal hover action with gravity enabled 
            action = torch.tensor([-1.0/3.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # nominal hover action with gravity enabled.
            # action = torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # nominal hover action with gravity disabled.

            # action = torch.tensor([-1.0/1.9, 0.0, 0.0, 0.0])
            action = torch.tile(action, (args_cli.num_envs, 1)).to(obs_tensor.device)


            # full_state = obs_dict["full_state"]
            # action_gc = gc.get_action(full_state)
            obs = obs_dict["gc"]
            action_gc = gc.get_action(obs)
            print("Action GC: ", action_gc)

            print("Pos: ", obs[:,:3])
            print("Goal pos: ", obs[:,13:16])

            # init_states[:,:3,0] = obs[:,:3].detach().cpu().numpy()
            # init_states[:,3:12,0] = isaac_math_utils.matrix_from_quat(obs[:,3:7]).detach().cpu().numpy().reshape(-1, 9)
            # init_states[:,12:15,0] = obs[:,7:10].detach().cpu().numpy()
            # init_states[:,15:18,0] = obs[:,10:13].detach().cpu().numpy()
            # action_nmpc = nmpc.get_action(init_states).squeeze()
            # action = torch.from_numpy(action_nmpc).to(obs_tensor.device)
            # print("action: ", action)


            # print("Action shape: ", action_gc.shape)

            # goal_pos_w = full_state[:, 26:26 + 3]
            # goal_ori_w = full_state[:, 26 + 3:26 + 7]
            # ee_pos = full_state[:, 13:16]
            # ee_ori_quat = full_state[:, 16:20]
            # ee_vel = full_state[:, 20:23]
            # ee_omega = full_state[:, 23:26]
            # quad_pos = full_state[:, :3]
            # quad_ori_quat = full_state[:, 3:7]
            # quad_vel = full_state[:, 7:10]
            # quad_omega = full_state[:, 10:13]
            # batch_size = full_state.shape[0]

            # print("[Debug] Quad Omega: ", quad_omega)
            # print("[Debug] EE Omega: ", ee_omega)
            # print("[Debug] Omega equal?: ", torch.allclose(quad_omega, ee_omega))

            # ee_omega_list.append(ee_omega.detach().cpu().numpy())
            # quad_omega_list.append(quad_omega.detach().cpu().numpy())
            # ee_pos_list.append(ee_pos.detach().cpu().numpy())


            action = action_gc.to(obs_tensor.device)

            obs_dict, reward, terminated, truncated, info = env.step(action)
            done_count += terminated.sum().item() + truncated.sum().item()
            print("Done count: ", done_count)
            print("Reward: ", reward)
            print()
            # input()
        print("Final info: ", info)
        # import code; code.interact(local=locals())
        # print("Catch count: ", info['log']['Episode Reward/catch'].item())
        # print("Max possible catches: ", args_cli.num_envs * 5.0)
        # print("Catch percentage: ", info['log']['Episode Reward/catch'].item() / (args_cli.num_envs * 5.0))

        print("Done!")
        # import numpy as np
        # ee_pos = np.array(ee_pos_list).squeeze()
        # print("EE Pos: ", ee_pos.shape)
        # np.save("ee_pos_mass_com.npy", ee_pos)


        # import matplotlib.pyplot as plt
        # import numpy as np

        # ee_omega = np.array(ee_omega_list)
        # quad_omega = np.array(quad_omega_list)
        # print("EE Omega: ", ee_omega.shape)
        # print("Quad Omega: ", quad_omega.shape)

        # plt.figure()
        # plt.plot(np.linalg.norm(ee_omega, axis=1), label="EE Omega")
        # plt.plot(np.linalg.norm(quad_omega, axis=1), label="Quad Omega")
        # plt.legend()
        # plt.show()


        env.close()
        simulation_app.close()

    


if __name__ == "__main__":
    main()

    simulation_app.close()