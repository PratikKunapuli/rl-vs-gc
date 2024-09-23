# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg


def add_ppo_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("ppo", description="Arguments for PPO agent.")
    # -- experiment arguments
    arg_group.add_argument("--exp_name", type=str, help="the name of this experiment")
    arg_group.add_argument("--torch_deterministic", type=bool, default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    arg_group.add_argument("--cuda", type=bool, default=True, help="if toggled, cuda will be enabled by default")
    arg_group.add_argument("--track", type=bool, default=False, help="if toggled, this experiment will be tracked with Weights and Biases")
    arg_group.add_argument("--wandb_project_name", type=str, default="cleanRL", help="the wandb's project name")
    arg_group.add_argument("--wandb_entity", type=str, default=None, help="the entity (team) of wandb's project")
    arg_group.add_argument("--capture_video", type=bool, default=False, help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    arg_group.add_argument("--env_id", type=str, default="Isaac-AerialManipulator-Hover-v0", help="the id of the environment")
    arg_group.add_argument("--total_timesteps", type=int, default=15000000, help="total timesteps of the experiments")
    arg_group.add_argument("--learning_rate", type=float, default=0.003, help="the learning rate of the optimizer")
    arg_group.add_argument("--num_steps", type=int, default=16, help="the number of steps to run in each environment per policy rollout")
    arg_group.add_argument("--anneal_lr", type=bool, default=False, help="Toggle learning rate annealing for policy and value networks")
    arg_group.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    arg_group.add_argument("--gae_lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    arg_group.add_argument("--num_minibatches", type=int, default=2, help="the number of mini-batches")
    arg_group.add_argument("--update_epochs", type=int, default=4, help="the K epochs to update the policy")
    arg_group.add_argument("--norm_adv", type=bool, default=True, help="Toggles advantages normalization")
    arg_group.add_argument("--clip_coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    arg_group.add_argument("--clip_vloss", type=bool, default=False, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    arg_group.add_argument("--ent_coef", type=float, default=0.0, help="coefficient of the entropy")
    arg_group.add_argument("--vf_coef", type=float, default=2, help="coefficient of the value function")
    arg_group.add_argument("--max_grad_norm", type=float, default=1, help="the maximum norm for the gradient clipping")
    arg_group.add_argument("--target_kl", type=float, default=None, help="the target KL divergence threshold")
    arg_group.add_argument("--reward_scaler", type=float, default=1, help="the scale factor applied to the reward during training")
    arg_group.add_argument("--record_video_step_frequency", type=int, default=512*200, help="the frequency at which to record the videos")

    # Miscellaneous
    arg_group.add_argument("--save_interval", type=int, default=500000, help="interval at which to save the model")
    arg_group.add_argument("--save_model", type=bool, default=True, help="if toggled, the model will be saved")

    # Task specific arguments
    arg_group.add_argument("--goal_task", type=str, default="rand", help="Goal task for the environment.")
    arg_group.add_argument("--frame", type=str, default="root", help="Frame of the task.")
    arg_group.add_argument("--sim_rate_hz", type=int, default=100, help="Simulation rate in Hz.")
    arg_group.add_argument("--policy_rate_hz", type=int, default=50, help="Policy rate in Hz.")
    arg_group.add_argument("--pos_radius", type=float, default=0.8, help="Position radius for the task.")

    # To be filled in runtime (placeholders for runtime)
    arg_group.add_argument("--batch_size", type=int, default=0, help="the batch size (computed in runtime)")
    arg_group.add_argument("--minibatch_size", type=int, default=0, help="the mini-batch size (computed in runtime)")
    arg_group.add_argument("--num_iterations", type=int, default=0, help="the number of iterations (computed in runtime)")

def parse_ppo_cfg(args_cli: argparse.Namespace, agent_cfg: argparse.Namespace) -> argparse.Namespace:
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    agent_cfg.exp_name = args_cli.exp_name
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    agent_cfg.torch_deterministic = args_cli.torch_deterministic
    agent_cfg.cuda = args_cli.cuda
    agent_cfg.track = args_cli.track
    agent_cfg.wandb_project_name = args_cli.wandb_project_name
    agent_cfg.wandb_entity = args_cli.wandb_entity
    agent_cfg.capture_video = args_cli.capture_video
    agent_cfg.env_id = args_cli.env_id
    agent_cfg.total_timesteps = args_cli.total_timesteps
    agent_cfg.learning_rate = args_cli.learning_rate
    agent_cfg.num_envs = args_cli.num_envs
    agent_cfg.num_steps = args_cli.num_steps
    agent_cfg.anneal_lr = args_cli.anneal_lr
    agent_cfg.gamma = args_cli.gamma
    agent_cfg.gae_lambda = args_cli.gae_lambda
    agent_cfg.num_minibatches = args_cli.num_minibatches
    agent_cfg.update_epochs = args_cli.update_epochs
    agent_cfg.norm_adv = args_cli.norm_adv
    agent_cfg.clip_coef = args_cli.clip_coef
    agent_cfg.clip_vloss = args_cli.clip_vloss
    agent_cfg.ent_coef = args_cli.ent_coef
    agent_cfg.vf_coef = args_cli.vf_coef
    agent_cfg.max_grad_norm = args_cli.max_grad_norm
    agent_cfg.target_kl = args_cli.target_kl
    agent_cfg.reward_scaler = args_cli.reward_scaler
    agent_cfg.record_video_step_frequency = args_cli.record_video_step_frequency
    agent_cfg.save_interval = args_cli.save_interval
    agent_cfg.video_length = args_cli.video_length
    agent_cfg.save_model = args_cli.save_model
    agent_cfg.batch_size = args_cli.batch_size
    agent_cfg.minibatch_size = args_cli.minibatch_size
    agent_cfg.num_iterations = args_cli.num_iterations

    return agent_cfg

def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr("args_cli", "seed") and args_cli.seed is not None:
        print("Assigning seed: " + str(args_cli.seed))
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
