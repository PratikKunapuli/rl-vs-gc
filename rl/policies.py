import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        num_obs = np.array(envs.single_observation_space.shape[1]).prod()
        num_acts = np.array(envs.single_action_space.shape[1]).prod()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_obs, 256)),
            # nn.Tanh(),
            nn.ELU(),
            layer_init(nn.Linear(256, 256)),
            # nn.Tanh(),
            nn.ELU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(num_obs, 256)),
            # nn.Tanh(),
            nn.ELU(),
            layer_init(nn.Linear(256, 256)),
            # nn.Tanh(),
            nn.ELU(),
            layer_init(nn.Linear(256, num_acts), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_acts))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        assert torch.all(action_std >= 0), f"std: {action_std} \n logstd: {action_logstd}"
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def predict(self, x, deterministic=True):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        assert torch.all(action_std >= 0), f"std: {action_std} \n logstd: {action_logstd}"
        probs = Normal(action_mean, action_std)
        if deterministic:
            return action_mean
        else:
            return probs.sample()