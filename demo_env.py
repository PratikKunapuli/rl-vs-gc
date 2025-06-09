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

from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config

from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab.utils.math as isaac_math_utils
import utils.math_utilities as math_utils

import gymnasium as gym
import torch
import numpy as np
import envs

from controllers.geometric_controller import GeometricController


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    env_cfg.seed = 1
    
    env_cfg.eval_mode = True


    # Create the environment configuration
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    
    # Record a video
    video_kwargs = {
        "video_folder": "videos",
        "step_trigger": lambda step: step == 0,
        # "episode_trigger": lambda episode: episode == 0,
        "video_length": 501,
        "name_prefix": "demo_env"
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)


    obs_dict, info = env.reset()

    done_count = 0

    while simulation_app.is_running():
        while done_count < 1:
            
            action = torch.rand(env.num_envs, 4, device=env.device) * 2 - 1  # Random actions for testing

            obs_dict, reward, terminated, truncated, info = env.step(action)
            done_count += terminated.sum().item() + truncated.sum().item()



        env.close()
        simulation_app.close()

    


if __name__ == "__main__":
    main()

    simulation_app.close()