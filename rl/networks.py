import torch
import torch.nn as nn

# SKRL Imports
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

# define shared model (stochastic and deterministic models) using mixins
class SKRL_Shared_MLP(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU(),)

        self.mean_layer = nn.Linear(256, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(256, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}

class SKRL_Shared_CNN_MLP(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, horizon_length = 10, use_yaw_traj=True,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.horizon_length=horizon_length
        self.cnn_output_shape = (8, self.horizon_length-6)
        self.traj_encoded_dim = self.cnn_output_shape[0] * self.cnn_output_shape[1]
        self.use_yaw_traj = use_yaw_traj
        self.input_channels = 7 if self.use_yaw_traj else 4
        self.state_dim = self.observation_space.shape[0] - self.input_channels*self.horizon_length

        # We want a 1d convolution over the future trajectory points, and then concatenate with the current state
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.input_channels, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=8, kernel_size=3),
            nn.ReLU()
        )

        self.net = nn.Sequential(nn.Linear(self.state_dim + self.traj_encoded_dim, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU(),)
        self.mean_layer = nn.Linear(256, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(256, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role not in {"policy", "value"}:
            return
        
        # Last 4*horizon length are ori_traj, then the previous 3*horizon length are the position_traj, then current state
        if self.use_yaw_traj:
            ori_traj = inputs["states"][:,-4*self.horizon_length:].view(-1, self.horizon_length, 4)
            pos_traj = inputs["states"][:,-7*self.horizon_length:-4*self.horizon_length].view(-1, self.horizon_length, 3)
            current_state = inputs["states"][:,:-7*self.horizon_length]
        else:
            ori_traj = inputs["states"][:,-self.horizon_length:].view(-1, self.horizon_length, 1)
            pos_traj = inputs["states"][:,-4*self.horizon_length:-self.horizon_length].view(-1, self.horizon_length, 3)
            current_state = inputs["states"][:,:-4*self.horizon_length]
            # traj = pos_traj.permute(0,2,1)
        traj = torch.cat([pos_traj, ori_traj], dim=-1).permute(0,2,1) # (batch, 7, horizon_length) or (batch, 4, horizon_length)
        traj_encoded = self.cnn(traj).view(-1, self.traj_encoded_dim)
        state = torch.cat([current_state, traj_encoded], dim=-1)
        
        if role == "policy":
            self._shared_output = self.net(state)
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(state) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}