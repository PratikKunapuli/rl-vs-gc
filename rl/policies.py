import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

import omni.isaac.lab.utils.math as isaac_math_utils
from utils.math import vee_map

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
        
class DecoupledController():
    def __init__(self, num_dofs, mass, inertia_tensor, pos_offset, ori_offset, device='cpu'):
        self.num_dofs = num_dofs
        # self.arm_offset = arm_offset
        self.mass = mass
        self.inertia_tensor = inertia_tensor
        self.position_offset = pos_offset
        self.orientation_offset = ori_offset
        self.moment_scale_xy = 0.5
        self.moment_scale_z = 0.025 
        self.thrust_to_weight = 3.0
        self.device = torch.device(device)
        self.inertia_tensor.to(self.device)
        
        self.initial_yaw_offset = torch.tensor([[0.7071, 0, 0, -0.7071]], device=self.device)

        self.gravity = torch.tensor([0.0, 0.0, 9.81], device=self.device)
        # self.gravity = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        # self.kp_pos = torch.tensor([1.0, 1.0, 5.0], device=self.device)
        # self.kd_pos = torch.tensor([0.5, 0.5, 1.0], device=self.device)
        self.kp_pos = torch.tensor([6.5, 6.5, 20.0], device=self.device)
        self.kd_pos = torch.tensor([6.0, 6.0, 9.0], device=self.device)
        # self.kp_pos = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        # self.kd_pos = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        self.kp_att = torch.tensor([400.0, 400.0, 2.0], device=self.device) 
        # self.kp_att = 20.0
        # self.kp_att = 544.0
        self.kd_att = torch.tensor([50.0, 50.0, 2.0], device=self.device)
        # self.kd_att = 7.0
        # self.kd_att = 0.0
        # self.kd_att = 46.64


        # self.kp_pos = 0.0
        # self.kd_pos = 0.0
        # self.kp_att = 0.0
        # self.kd_att = 0.0
        
    def compute_desired_pose(self, obs):
        goal_pos = obs[:, 26+self.num_dofs*2:26+self.num_dofs*2 + 3]
        goal_ori = obs[:, 26+self.num_dofs*2 + 3:26+self.num_dofs*2 + 7]

        # quad_pos = obs[:, :3]
        # quad_ori_quat = obs[:, 3:7]
        # ee_pos = obs[:, 13:16]
        # ee_ori_quat = obs[:, 16:20]

        # position_offset = quad_pos - ee_pos
        # orientation_offset = isaac_math_utils.quat_mul(quad_ori_quat, isaac_math_utils.quat_conjugate(ee_ori_quat))
        print("Position Offset: ", self.position_offset)
        print("Orientation Offset: ", self.orientation_offset)
        # quad_pos_ee_frame, quad_ori_ee_frame = isaac_math_utils.subtract_frame_transforms(quad_pos, quad_ori_quat, ee_pos, ee_ori_quat)
        # # quad_pos_ee_frame, quad_ori_ee_frame = isaac_math_utils.subtract_frame_transforms(ee_pos, ee_ori_quat, quad_pos, quad_ori_quat)
        # print("Quad Pos in EE Frame: ", quad_pos_ee_frame)
        # print("arm_offset: ", self.arm_offset)
        # print("quad ori in EE Frame: ", quad_ori_ee_frame)

        # if initial_yaw_offset is not None:
        #     desired_ori = isaac_math_utils.quat_mul(goal_ori, initial_yaw_offset)
        # else:
        #     desired_ori = isaac_math_utils.yaw_quat(goal_ori) 


        print("Goal Pos: ", goal_pos)
        print("Goal Ori: ", goal_ori)
        desired_ori = isaac_math_utils.yaw_quat(goal_ori)
        print("Goal Ori (yaw only): ", desired_ori)
        quad_ori_desired = isaac_math_utils.quat_mul(desired_ori, self.orientation_offset)
        print("Quad Ori Desired: ", quad_ori_desired)
        quad_pos_desired = goal_pos + self.position_offset
        print("Quad Pos Desired: ", quad_pos_desired)


        # arm_offset = self.arm_offset.tile(obs.shape[0], 1).unsqueeze(1) # [batch_size, 1, 3]
        # desired_pos = isaac_math_utils.transform_points(arm_offset, goal_pos, goal_ori)
        # desired_pos_new, desired_ori_new = isaac_math_utils.combine_frame_transforms(goal_pos, desired_ori, quad_pos_ee_frame, quad_ori_ee_frame)
        # desired_pos_new, desired_ori_new = isaac_math_utils.combine_frame_transforms(quad_pos_ee_frame, quad_ori_ee_frame, goal_pos, desired_ori)

        # print("Desired Pos New: ", desired_pos_new)
        # print("Desired Ori New: ", desired_ori_new)

        return quad_pos_desired, quad_ori_desired
    
    def isolated_desired_pose(self, obs):
        """
        Make the quadrotor go to the origin with Identity orientation
        """
        quad_pos_desired = torch.tensor([1.0, 0.0, 0.5], device=self.device).tile(obs.shape[0], 1)
        # quad_ori_desired = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).tile(obs.shape[0], 1)
        quad_ori_desired = self.orientation_offset.tile(obs.shape[0], 1)
        return quad_pos_desired, quad_ori_desired
    
    def compute_desired_joint_angles(self, obs):
        return None

    
    def rescale_command(self, command, min_val, max_val):
        """
        We want to rescale the command to be between -1 and 1, where the original command is between min_val and max_val
        """
        return 2.0 * (command - min_val) / (max_val - min_val) - 1.0

    def get_action(self, obs):
        print("-"*15)
        # Find virtual setpoints
        # desired_pos, desired_ori = self.compute_desired_pose(obs)
        desired_pos, desired_ori = self.isolated_desired_pose(obs)
        # desired_ori = isaac_math_utils.quat_from_matrix(torch.tensor([[1.0, 0., 0.],
        #                                                              [0.,1.,0.],
        #                                                              [0., 0., 1.]], device=self.device))
        desired_joint_angles = self.compute_desired_joint_angles(obs)

        quad_ori_quat= obs[:, 3:7]
        ee_ori_quat = obs[:, 16:20]

        # print("Quad yaw quat: ", isaac_math_utils.yaw_quat(quad_ori_quat))
        # print("quat yaw: ", 2 * torch.arctan2(quad_ori_quat[:, 3], quad_ori_quat[:, 0]))
        # print("EE yaw quat: ", isaac_math_utils.yaw_quat(ee_ori_quat))
        # print("EE yaw: ", 2 * torch.arctan2(ee_ori_quat[:, 3], ee_ori_quat[:, 0]))


        # print("Goal: ", obs[:, 26+self.num_dofs*2:26+self.num_dofs*2 + 3])
        # print("Desired Pos: ", desired_pos)
        # print("Desired Ori: ", desired_ori)
        desired_yaw = 2 * torch.arctan2(desired_ori[:, 3], desired_ori[:, 0])
        print("Desired Yaw: ", desired_yaw)
        # _, _, desired_yaw  = isaac_math_utils.euler_xyz_from_quat(desired_ori)

        quad_pos = obs[:, :3]
        quad_ori_quat = obs[:, 3:7]
        quad_vel = obs[:, 7:10]
        quad_omega = obs[:, 10:13].to(self.device)

        # Rotate omega into body frame from world frame
        quad_omega = isaac_math_utils.quat_rotate(isaac_math_utils.quat_conjugate(quad_ori_quat), quad_omega)

        print("Quad Pos: ", quad_pos)

        pos_error = quad_pos - desired_pos
        vel_error = quad_vel - torch.zeros_like(quad_vel)

        print("Pos Error: ", pos_error)
        print("Vel Error: ", vel_error)

        # Compute desired Force (batch_size, 3)
        F_des = self.mass * (-self.kp_pos * pos_error + \
                             -self.kd_pos * vel_error + \
                            self.gravity.tile(obs.shape[0], 1)) # (N, 3)
        
        print("F_des: ", F_des)
        
        batch_size = obs.shape[0]
        quad_ori_matrix = isaac_math_utils.matrix_from_quat(quad_ori_quat) # (batch_size, 3, 3)
        quad_b3 = quad_ori_matrix[:, :, 2] # (batch_size, 3)
        print("b3: ", quad_b3)
        # collective_thrust = torch.bmm(F_des.view(batch_size, 1, 3), quad_b3.view(batch_size, 3, 1)).squeeze(2)
        collective_thrust = (F_des * quad_b3).sum(dim=-1)
        print("Collective Thrust: ", collective_thrust)

        # Compute the desired orientation
        b3_des = (F_des / torch.linalg.norm(F_des, dim=1)).squeeze(1) #(N, 3)
        # b3_des = torch.tensor([0., 0., 1.0], device=self.device).tile(batch_size, 1)
        print("b3_des: ", b3_des)
        yaw_des = desired_yaw
        c1_des = torch.tensor([torch.cos(yaw_des), torch.sin(yaw_des), 0.0], device=self.device).view(batch_size, 3) # (N, 3)
        print("c1_des: ", c1_des)
        b2_des_unnormalized = torch.cross(b3_des, c1_des, dim=1)
        print("b2_des_unnormalized: ", b2_des_unnormalized)
        b2_des = b2_des_unnormalized / torch.linalg.norm(b2_des_unnormalized, dim=1)
        print("b2_des: ", b2_des)
        b1_des = torch.cross(b2_des, b3_des, dim=1)
        print("b1_des: ", b1_des)
        R_des = torch.stack([b1_des, b2_des, b3_des], dim=2) # (batch_size, 3, 3)



        print("R_des being set manually!")
        collective_thrust = torch.zeros(batch_size, 1, device=self.device)
        R_des = torch.tensor([[  1.0000000,  0.0000000,  0.0000000],
   [0.0000000,  1.0000000,  0.0000000],
   [0.0000000,  0.0000000,  1.0000000 ]], device=self.device).tile(batch_size, 1, 1)
        print("R_des: ", R_des)

        # Compute orientation error
        print("Quad Ori Matrix: ", quad_ori_matrix)
        S_err = 0.5 * (torch.bmm(R_des.transpose(-2, -1), quad_ori_matrix) - torch.bmm(quad_ori_matrix.transpose(-2, -1), R_des)) # (batch_size, 3, 3)
        print("S_err: ", S_err)
        att_err = vee_map(S_err) # (batch_size, 3)
        print("Attitude Error: ", att_err)
        if torch.any(torch.isnan(att_err)):
            print("Nan detected in attitude error!", att_err)
            att_err = torch.zeros_like(att_err, device=self.device)
        omega_err = quad_omega - torch.zeros_like(quad_omega, device=self.device) # Omega des is 0. (batch_size, 3)
        print("Omega Error: ", omega_err)
        # Compute desired moments
        # M = I @ (-kp_att * att_err - kd_att * omega_err) + omega x I @ omega
        inertia = self.inertia_tensor.unsqueeze(0).tile(batch_size, 1, 1).to(self.device)
        att_pd = -self.kp_att * att_err - self.kd_att * omega_err
        I_omega = torch.bmm(inertia.view(batch_size, 3, 3), quad_omega.unsqueeze(2)).squeeze(2).to(self.device)

        M_des = torch.bmm(inertia.view(batch_size, 3, 3), att_pd.unsqueeze(2)).squeeze(2) 
                # torch.cross(quad_omega, I_omega, dim=1)
        
        # print("Thrust: ", collective_thrust.shape)
        # print("M_des: ", M_des.shape)

        print("Thrust: ", collective_thrust)
        print("M_des (world frame): ", M_des)

        # Rotate M_des to body frame
        # M_des = isaac_math_utils.quat_rotate(quad_ori_quat, M_des)
        # print("M_des (body frame): ", M_des)
        
        u1 = self.rescale_command(collective_thrust, 0.0, self.thrust_to_weight * 9.81*self.mass).view(batch_size, 1)
        u2 = self.rescale_command(M_des[:, 0], -self.moment_scale_xy, self.moment_scale_xy).view(batch_size, 1)
        u3 = self.rescale_command(M_des[:, 1], -self.moment_scale_xy, self.moment_scale_xy).view(batch_size, 1)
        u4 = self.rescale_command(M_des[:, 2], -self.moment_scale_z, self.moment_scale_z).view(batch_size, 1)

        # import code; code.interact(local=locals())

        return torch.stack([u1, u2, u3, u4], dim=1).view(batch_size, 4)


