import argparse
import sys
from omni.isaac.lab.app import AppLauncher
from utils import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CleanRL. ")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=250, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# parser.add_argument(
#     "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
# )
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# parser.add_argument("--capture_video", action="store_true", default=False, help="Capture video of the agent performance.")
# parser.add_argument("--exp_name", type=str, default="cleanrl_test", help="Name of the experiment.")
# parser.add_argument("--anneal_lr", action="store_true", default=False, help="Anneal the learning rate.")
# parser.add_argument("--learning_rate", type=float, default=0.0026, help="Learning rate of the optimizer.")
# parser.add_argument("--total_timesteps", type=int, default=15e6, help="Total timesteps of the experiments.")
# parser.add_argument("--goal_task", type=str, default="rand", help="Goal task for the environment.")
# parser.add_argument("--frame", type=str, default="root", help="Frame of the task.")
# parser.add_argument("--sim_rate_hz", type=int, default=500, help="Simulation rate in Hz.")
# parser.add_argument("--policy_rate_hz", type=int, default=100, help="Policy rate in Hz.")
# parser.add_argument("--pos_radius", type=float, default=0.8, help="Position radius for the task.")
# parser.add_argument("--device", type=int, default="0", help="Device to run the training on.")

cli_args.add_ppo_args(parser)

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
# args_cli = parser.parse_args()
args_cli, hydra_args = parser.parse_known_args() # adding hydra config

args_cli.enable_cameras = True
args_cli.headless = True # make false to see the simulation

# clear out the sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import envs
from policies import Agent

from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, rewards, dones, truncateds, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - (torch.logical_or(dones, truncateds)).long()
        self.episode_lengths *= 1 - (torch.logical_or(dones, truncateds)).long()
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            truncateds,
            infos,
        )


class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["policy"]
    
@hydra_task_config(args_cli.task, "cleanrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    agent_cfg = argparse.Namespace(**agent_cfg) # convert to namespace so we can access and modify the attributes like without Hydra
    
    # print("Default env_cfg:\n", env_cfg)
    # print("Default agent_cfg:\n", agent_cfg)
    # input("Check these...")
    args = cli_args.parse_ppo_cfg(args_cli, agent_cfg)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    run_name = args.exp_name
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Save args into a file
    with open(f"runs/{run_name}/args.txt", "w") as f:
        f.write(str(vars(args)))
    
    with open(f"runs/{run_name}/args_cli.txt", "w") as f:
        f.write(str(vars(args_cli)))

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device(args_cli.device)

    # Any environment specific configuration goes here such as camera placement
    if "Quadcopter" not in args_cli.task:
        env_cfg.goal_cfg = args_cli.goal_task
        env_cfg.task_body = args_cli.frame
        env_cfg.sim_rate_hz = args_cli.sim_rate_hz
        env_cfg.policy_rate_hz = args_cli.policy_rate_hz
        env_cfg.sim.dt = 1/env_cfg.sim_rate_hz
        env_cfg.decimation = env_cfg.sim_rate_hz // env_cfg.policy_rate_hz
        env_cfg.sim.render_interval = env_cfg.decimation
        if env_cfg.use_yaw_representation:
            # env_cfg.num_observations += 4
            env_cfg.num_observations += 1
        
        if env_cfg.use_full_ori_matrix:
            env_cfg.num_observations += 6

        # These are now modifyable in the CLI with hydra 
        # you simply need to add env.pos_radius=0.8 in the CLI
        # env_cfg.pos_radius = args_cli.pos_radius
        # env_cfg.joint_vel_reward_scale = 0.0
        # env_cfg.joint_vel_reward_scale = -10.0
        # env_cfg.action_norm_reward_scale = 0.0
        # env_cfg.action_norm_reward_scale = -0.1
        # env_cfg.ori_error_reward_scale = -0.5

    # env_cfg.viewer.eye = (-2.0, 2.0, 2.0)
    # env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    # env_cfg.viewer.origin_type = "env"
    # env_cfg.viewer.env_index = 0

    print(env_cfg)
    # input("Please check env cfg and press Enter to continue...")

    # create environment
    envs = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    video_kwargs = {
        "video_folder": f"runs/{run_name}",
        "step_trigger": lambda step: step % args_cli.video_interval == 0,
        # "episode_trigger": lambda episode: (episode % args.save_interval) == 0,
        "video_length": args_cli.video_length,
        "name_prefix": "training_video"
    }
    envs = gym.wrappers.RecordVideo(envs, **video_kwargs)
    envs = ExtractObsWrapper(envs)
    envs = RecordEpisodeStatisticsTorch(envs, device)
    if args.capture_video:
        envs.is_vector_env = True
        print(f"record_video_step_frequency={args.record_video_step_frequency}")
        envs = gym.wrappers.RecordVideo(
            envs,
            f"videos/{run_name}",
            step_trigger=lambda step: step % args.record_video_step_frequency == 0,
            video_length=1000,  # for each video record up to 10 seconds
        )
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    envs.single_action_space_shape = (np.array(envs.single_action_space.shape[1]).prod(),)
    envs.single_observation_space_shape = (np.array(envs.single_observation_space.shape[1]).prod(),)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space_shape, dtype=torch.float).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space_shape, dtype=torch.float).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    save_steps = 0
    save_iterations = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)

    max_episode_return = -np.inf


    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            save_steps += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards[step], next_done, next_truncated, info = envs.step(action)
            # if 0 <= step <= 2:
            for idx, d in enumerate(next_done):
                if d or next_truncated[idx]:
                    episodic_return = info["r"][idx].item()
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

                    if "Quadcopter" in args_cli.task: # original isaac lab env
                        writer.add_scalar("charts/episodic_summed_vel_reward", info["log"]["Episode Reward/lin_vel"].item(), global_step)
                        writer.add_scalar("charts/episodic_summed_ang_vel_reward", info["log"]["Episode Reward/ang_vel"].item(), global_step)
                        writer.add_scalar("charts/episodic_summed_pos_reward", info["log"]["Episode Reward/distance_to_goal"].item(), global_step)
                        writer.add_scalar("charts/episode_final_distance_to_goal", info["log"]["Metrics/final_distance_to_goal"], global_step)
                    else: 
                        # Rewards
                        writer.add_scalar("charts/episodic_summed_vel_reward", info["log"]["Episode Reward/endeffector_lin_vel"].item(), global_step)
                        writer.add_scalar("charts/episodic_summed_ang_vel_reward", info["log"]["Episode Reward/endeffector_ang_vel"].item(), global_step)
                        writer.add_scalar("charts/episodic_summed_pos_reward", info["log"]["Episode Reward/endeffector_pos_distance"].item(), global_step)
                        writer.add_scalar("charts/episodic_summed_ori_reward", info["log"]["Episode Reward/endeffector_ori_error"].item(), global_step)
                        writer.add_scalar("charts/episodic_summed_yaw_error_reward", info["log"]["Episode Reward/endeffector_yaw_error"].item(), global_step)
                        writer.add_scalar("charts/episodic_summed_yaw_distance_reward", info["log"]["Episode Reward/endeffector_yaw_distance"].item(), global_step)
                        writer.add_scalar("charts/episodic_summed_joint_vel_reward", info["log"]["Episode Reward/joint_vel"].item(), global_step)
                        writer.add_scalar("charts/episode_final_distance_to_goal", info["log"]["Metrics/Final Distance to Goal"], global_step)
                        writer.add_scalar("charts/episode_final_orientation_error", info["log"]["Metrics/Final Orientation Error to Goal"], global_step)
                        writer.add_scalar("charts/episode_final_yaw_error", info["log"]["Metrics/Final Yaw Error to Goal"], global_step)
                        # Errors
                        writer.add_scalar("charts/episode_summed_pos_error", info["log"]["Episode Error/pos_error"].item(), global_step)
                        writer.add_scalar("charts/episode_summed_pos_distance", info["log"]["Episode Error/pos_distance"].item(), global_step)
                        writer.add_scalar("charts/episode_summed_vel_error", info["log"]["Episode Error/lin_vel"].item(), global_step)
                        writer.add_scalar("charts/episode_summed_ang_vel_error", info["log"]["Episode Error/ang_vel"].item(), global_step)
                        writer.add_scalar("charts/episode_summed_ori_error", info["log"]["Episode Error/ori_error"].item(), global_step)
                        writer.add_scalar("charts/episode_summed_yaw_error", info["log"]["Episode Error/yaw_error"].item(), global_step)
                        writer.add_scalar("charts/episode_summed_yaw_distance", info["log"]["Episode Error/yaw_distance"].item(), global_step)
                        writer.add_scalar("charts/episode_summed_joint_vel_error", info["log"]["Episode Error/joint_vel"].item(), global_step)
                        writer.add_scalar("charts/episode_summed_action_norm", info["log"]["Episode Error/action_norm"].item(), global_step)
                    
                    # save model on best episodic return
                    if episodic_return > max_episode_return:
                        max_episode_return = episodic_return
                        torch.save(agent.state_dict(), f"runs/{run_name}/best_cleanrl_model.pt")
                        
                    if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                        writer.add_scalar(
                            "charts/consecutive_successes", info["consecutive_successes"].item(), global_step
                        )
                    break
            
            # Save the model every 500k global steps but since there's many environments it could step over 500k
            if save_steps >= args.save_interval:
                save_steps = 0
                model_save_number = (save_iterations+1) * args.save_interval
                torch.save(agent.state_dict(), f"runs/{run_name}/model_{model_save_number}_steps.pt")
                save_iterations += 1


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.long()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                # newlogprob = newlogprob.sum(1)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)



    if args.save_model:
        model_path = f"runs/{run_name}/model_final.pt"
        torch.save(agent.state_dict(), model_path)

    # envs.close()
    writer.close()


if __name__ == "__main__":
    run_name = f"{args_cli.exp_name}_1"
    # If run exists, increment run number
    run_number = 1
    while os.path.exists(f"runs/{run_name}"):
        run_number += 1
        run_name = f"{args_cli.exp_name}_{run_number}"
    args_cli.exp_name = run_name
    sys.argv.append(f"hydra.run.dir=runs/{run_name}")
    # run the main function
    main()
    # close sim app
    simulation_app.close()