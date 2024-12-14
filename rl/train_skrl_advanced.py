# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

from utils import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append skrl cli arguments
cli_args.add_skrl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import envs
from networks import SKRL_Shared_MLP, SKRL_Shared_CNN_MLP


import skrl
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.3.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # Update logging directory from CLI args
    agent_cfg["agent"]["experiment"]["directory"] = args_cli.experiment_name
    
    if args_cli.run_name:
        agent_cfg["agent"]["experiment"]["experiment_name"] = args_cli.run_name
    else:
        agent_cfg["agent"]["experiment"]["experiment_name"] = ""

    if args_cli.num_steps_per_env:
        agent_cfg["agent"]["rollouts"] = args_cli.num_steps_per_env

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework, wrapper="isaaclab-single-agent")  # same as: `wrap_env(env, wrapper="auto")`

    # instantiate a memory as rollout buffer (any memory can be used for this)
    device = env_cfg.sim.device

    memory = RandomMemory(memory_size=agent_cfg["agent"]["rollouts"], num_envs=env.num_envs, device=device)


    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    if args_cli.model_type.lower() == "mlp":
        models = {}
        models["policy"] = SKRL_Shared_MLP(env.observation_space, env.action_space, device)
        models["value"] = models["policy"]  # same instance: shared model
        print("[INFO] Using MLP model.")
    elif "cnn" in args_cli.model_type.lower():
        horizon_length = env_cfg.trajectory_horizon
        use_yaw_traj = not args_cli.ignore_yaw_traj
        models = {}
        models["policy"] = SKRL_Shared_CNN_MLP(env.observation_space, env.action_space, device, use_yaw_traj=use_yaw_traj, horizon_length=horizon_length)
        models["value"] = models["policy"]
        print("[INFO] Using CNN model, with horizon length:", horizon_length)
        print("[INFO] Using yaw trajectory:", use_yaw_traj)
    else:
        raise ValueError(f"Unsupported model type: {args_cli.model_type}")

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = agent_cfg["agent"]["rollouts"]  # memory_size
    cfg["learning_epochs"] = agent_cfg["agent"]["learning_epochs"]
    cfg["mini_batches"] = agent_cfg["agent"]["mini_batches"]  # 16 * 1024 / 4096
    cfg["discount_factor"] = agent_cfg["agent"]["discount_factor"]
    cfg["lambda"] = agent_cfg["agent"]["lambda"]
    cfg["learning_rate"] = agent_cfg["agent"]["learning_rate"]
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": agent_cfg["agent"]["learning_rate_scheduler_kwargs"]["kl_threshold"]}
    cfg["random_timesteps"] = agent_cfg["agent"]["random_timesteps"]
    cfg["learning_starts"] = agent_cfg["agent"]["learning_starts"]
    cfg["grad_norm_clip"] = agent_cfg["agent"]["grad_norm_clip"]
    cfg["ratio_clip"] = agent_cfg["agent"]["ratio_clip"]
    cfg["value_clip"] = agent_cfg["agent"]["value_clip"]
    cfg["clip_predicted_values"] = agent_cfg["agent"]["clip_predicted_values"]
    cfg["entropy_loss_scale"] = agent_cfg["agent"]["entropy_loss_scale"]
    cfg["value_loss_scale"] = agent_cfg["agent"]["value_loss_scale"]
    cfg["kl_threshold"] = agent_cfg["agent"]["kl_threshold"]
    cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * agent_cfg["agent"]["rewards_shaper_scale"]
    cfg["time_limit_bootstrap"] = agent_cfg["agent"]["time_limit_bootstrap"]
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    # cfg["state_preprocessor_kwargs"] = agent_cfg["agent"]["state_preprocessor_kwargs"]
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # cfg["value_preprocessor_kwargs"] = agent_cfg["agent"]["value_preprocessor_kwargs"]
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = agent_cfg["agent"]["experiment"]["write_interval"]
    cfg["experiment"]["checkpoint_interval"] = agent_cfg["agent"]["experiment"]["checkpoint_interval"]
    cfg["experiment"]["directory"] = agent_cfg["agent"]["experiment"]["directory"]
    cfg["experiment"]["experiment_name"] = agent_cfg["agent"]["experiment"]["experiment_name"]

    agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": agent_cfg["trainer"]["timesteps"], "headless": True, "close_environment_at_exit": False}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()

    # close the simulator
    env.close()

    print("\n\nPlease run the following line to evaluate the trained model: ")
    # print(f"python eval_skrl_advanced.py --video --task {args_cli.task} --num_envs 100 --experiment_name {agent_cfg["agent"]["experiment"]["directory"]} --load_run {agent_cfg["agent"]["experiment"]["experiment_name"]} --model_type {args_cli.model_type} --use_yaw_traj {env_cfg.use_yaw_traj} --horizon_length {env_cfg.trajectory_horizon}")
    print("\n Hydra args: ")
    print(hydra_args)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()