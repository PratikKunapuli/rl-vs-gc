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
import gymnasium as gym
import torch
# from envs.hover import hover_env
import envs
import envs.hover
# from AerialManipulation.envs.hover import hover_env

from rl.policies import DecoupledController

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
    env_cfg.viewer.eye = (3.0, 3.0, 0.2)
    env_cfg.viewer.resolution = (1920, 1080)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0

    # import code; code.interact(local=locals())

    print(env_cfg.robot.spawn)

    env_cfg.goal_cfg = "rand" # "rand" or "fixed"
    env_cfg.goal_pos = [1.0, 1.0, 0.5]
    env_cfg.goal_ori = [0.7071068, 0.0, 0.0, 0.7071068]
    # env_cfg.goal_ori = [0.7071068, 0.0, 0.0, 0.7071068]
    env_cfg.sim_rate_hz = 100
    env_cfg.policy_rate_hz = 50
    env_cfg.sim.dt = 1/env_cfg.sim_rate_hz
    env_cfg.decimation = env_cfg.sim_rate_hz // env_cfg.policy_rate_hz
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.eval_mode = True
    # env_cfg.init_cfg = "fixed"


    # Turn gravity off
    env_cfg.robot.spawn.rigid_props.disable_gravity = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # Get mass from env
    vehicle_mass = env.vehicle_mass
    arm_mass = env.arm_mass
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

    # gc = DecoupledController(args_cli.num_envs, 0, vehicle_mass, arm_mass, inertia, arm_offset, ori_offset, com_pos_w=env.com_pos_w, device=env.device)
    gc = DecoupledController(args_cli.num_envs, 0, vehicle_mass, arm_mass, inertia, arm_offset, ori_offset, com_pos_w=None, device=env.device)
    print("Quad in EE Frame: ", gc.quad_pos_ee_frame)
    print("COM in EE Frame: ", gc.com_pos_ee_frame)


    video_kwargs = {
        "video_folder": "videos",
        # "step_trigger": lambda step: step == 0,
        "episode_trigger": lambda episode: episode == 0,
        # "video_length": args_cli.video_length,
        "name_prefix": "GC_Tuning"
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)


    obs_dict, info = env.reset()
    done = False
    done_count = 0
    
    import code; code.interact(local=locals())
    
    while simulation_app.is_running():
        while done_count < 2:
            obs_tensor = obs_dict["policy"]
            # action = env.action_space.sample()
            # action = torch.zeros_like(torch.from_numpy(env.action_space.sample()))
            # action[0]= -1.0/3.0 # nominal hover action with gravity enabled 
            action = torch.tensor([1.0/3.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # nominal hover action with gravity enabled.
            # action = torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # nominal hover action with gravity disabled.

            # action = torch.tensor([-1.0/1.9, 0.0, 0.0, 0.0])
            action = torch.tile(action, (args_cli.num_envs, 1)).to(obs_tensor.device)


            full_state = obs_dict["full_state"]
            action_gc = gc.get_action(full_state)
            print("Action GC: ", action_gc)
            # print("Action shape: ", action_gc.shape)


            action = action_gc.to(obs_tensor.device)

            obs_dict, reward, terminated, truncated, info = env.step(action)
            done_count += terminated.sum().item() + truncated.sum().item()
            print("Done count: ", done_count)
            print()
            # input()
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()

    simulation_app.close()