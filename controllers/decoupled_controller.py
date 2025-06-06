import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from typing import Tuple

import omni.isaac.lab.utils.math as isaac_math_utils
from utils.math_utilities import vee_map, yaw_from_quat, quat_from_yaw, matrix_log
import utils.flatness_utilities as flat_utils
import utils.math_utilities as math_utils
        

@torch.jit.script
def get_point_state_from_ee_transform_w(ee_pos_w, ee_ori_quat_w, ee_vel_w, ee_omega_w, point_pos_ee_frame):
    point_pos_w, _ = isaac_math_utils.combine_frame_transforms(ee_pos_w, ee_ori_quat_w, point_pos_ee_frame)
    point_vel_w = ee_vel_w + torch.cross(ee_omega_w, isaac_math_utils.quat_rotate(ee_ori_quat_w, point_pos_ee_frame), dim=1)

    
    return point_pos_w, point_vel_w

@torch.jit.script
def rescale_command(command: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """
    We want to rescale the command to be between -1 and 1, where the original command is between min_val and max_val
    """
    return 2.0 * torch.div((command - min_val),(max_val - min_val)) - torch.ones_like(command)

@torch.jit.script
def compute_ff_terms(obs: torch.Tensor, policy_dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given a batch of observations, compute the feed forward terms for position and yaw.

    Observations are horizon waypoints in position and yaw quaternion, which can be finitely differentiated to get velocity, acceleration, jerk, and snap.
    """
    # first 17 terms are part of state, then horizon*3 future positions, then horizon*4 future quaternions
    batch_size = obs.shape[0]
    num_obs = obs.shape[1]
    horizon = (num_obs - 17) // 7 # 3 for position and 4 for quaternion
    futures = obs[:, -7*horizon:]

    # Position Feed Forwards
    future_com_pos_w = futures[:, :3*horizon].reshape(batch_size, horizon, 3)
    feed_forward_velocities = (future_com_pos_w[:, 1:] - future_com_pos_w[:, :-1]) / policy_dt
    feed_forward_accelerations = (feed_forward_velocities[:, 1:] - feed_forward_velocities[:, :-1]) / policy_dt
    feed_forward_jerks = (feed_forward_accelerations[:, 1:] - feed_forward_accelerations[:, :-1]) / policy_dt
    feed_forward_snaps = (feed_forward_jerks[:, 1:] - feed_forward_jerks[:, :-1]) / policy_dt
    ff_pos = future_com_pos_w[:, 0]
    ff_vel = feed_forward_velocities[:, 0]
    ff_acc = feed_forward_accelerations[:, 0]
    ff_jerk = feed_forward_jerks[:, 0]
    ff_snap = feed_forward_snaps[:, 0]

    # Yaw Feed Forwards
    future_com_ori_w = futures[:, 3*horizon:].reshape(batch_size, horizon, 4)
    feed_forward_yaws = yaw_from_quat(future_com_ori_w)
    feed_forward_yaws_dot = ((feed_forward_yaws[:, 1:] - feed_forward_yaws[:, :-1] + 3*torch.pi) % (2*torch.pi) - torch.pi) / policy_dt

    feed_forward_yaws_ddot = (feed_forward_yaws_dot[:, 1:] - feed_forward_yaws_dot[:, :-1]) / policy_dt
    ff_yaw = feed_forward_yaws[:, 0]
    ff_yaw_dot = feed_forward_yaws_dot[:, 0]
    ff_yaw_ddot = feed_forward_yaws_ddot[:, 0]

    return ff_pos, ff_vel, ff_acc, ff_jerk, ff_snap, ff_yaw, ff_yaw_dot, ff_yaw_ddot

class DecoupledController():
    def __init__(self, num_envs, num_dofs, vehicle_mass, arm_mass, inertia_tensor, pos_offset, ori_offset, print_debug=False, com_pos_w=None, device='cpu',
                  kp_pos_gain_xy=10.0, kp_pos_gain_z=20.0, kd_pos_gain_xy=7.0, kd_pos_gain_z=9.0, 
                  kp_att_gain_xy=400.0, kp_att_gain_z=2.0, kd_att_gain_xy=70.0, kd_att_gain_z=2.0,
                  ki_pos_gain_xy=0.0, ki_pos_gain_z=0.0, ki_att_gain_xy=0.0, ki_att_gain_z=0.0,
                  tuning_mode=False, vehicle="AM", control_mode="CTBM", policy_dt=0.02,
                  feed_forward=False, use_integral = False, disable_gravity=False, track_buffers = False, **kwargs):
        self.num_envs = num_envs
        self.num_dofs = num_dofs
        self.print_debug = print_debug
        self.tuning_mode = tuning_mode
        self.track_buffers = track_buffers

        self.vehicle_mass = vehicle_mass
        self.arm_mass = arm_mass
        self.mass = vehicle_mass + arm_mass
        if len(self.mass.shape) < 1:
            self.mass = torch.tensor([self.mass], device=device)


        self.com_pos_w = com_pos_w
        self.policy_dt = policy_dt
        self.feed_forward = feed_forward
        self.use_integral = use_integral

        self.control_mode = control_mode

        self.inertia_tensor = inertia_tensor
        self.position_offset = pos_offset
        self.orientation_offset = ori_offset

        if vehicle == "AM":
            self.moment_scale_xy = 0.5
            self.moment_scale_z = 0.025 #0.025 # 0.1
            self.thrust_to_weight = 3.0
        else:
            #Crazyflie
            # self.thrust_to_weight = 1.8 # brushed crazyflie
            self.thrust_to_weight = 3.5 # brushelss crazyflie
            self.moment_scale_xy = 0.01
            self.moment_scale_z = 0.01


            self.attitude_scale_z = torch.pi
            self.attitude_scale_xy = 0.2
            self.attitude_scale = torch.pi/6 # 30 degrees
            self.body_rate_scale_xy = 30.0 # rad/s
            self.body_rate_scale_z = 5.0 # rad/s

        self.device = torch.device(device)
        self.inertia_tensor.to(self.device)
        
        self.initial_yaw_offset = torch.tensor([[0.7071, 0, 0, -0.7071]], device=self.device)

        self.gravity = torch.tensor([0.0, 0.0, 9.81], device=self.device)
        if disable_gravity:
            self.gravity = torch.tensor([0.0, 0.0, 0.0], device=self.device)


        self.kp_pos = torch.tensor([kp_pos_gain_xy, kp_pos_gain_xy, kp_pos_gain_z], device=self.device)
        self.kd_pos = torch.tensor([kd_pos_gain_xy, kd_pos_gain_xy, kd_pos_gain_z], device=self.device)
        self.kp_att = torch.tensor([kp_att_gain_xy, kp_att_gain_xy, kp_att_gain_z], device=self.device)
        self.kd_att = torch.tensor([kd_att_gain_xy, kd_att_gain_xy, kd_att_gain_z], device=self.device)
        self.ki_pos = torch.tensor([ki_pos_gain_xy, ki_pos_gain_xy, ki_pos_gain_z], device=self.device)
        self.ki_att = torch.tensor([ki_att_gain_xy, ki_att_gain_xy, ki_att_gain_z], device=self.device)

        self.pos_error_integral = torch.zeros(num_envs, 3, device=self.device)
        self.att_error_integral = torch.zeros(num_envs, 3, device=self.device)


        # Logging buffers
        if self.track_buffers: 
            self.s_buffer = []
            self.s_des_buffer = []
            self.s_dot_buffer = []
            self.s_dot_des_buffer = []
            self.ref_pos_buffer = []
            self.pos_buffer = []

    
    def reset_integral_terms(self, env_mask):
        reset_envs = env_mask.nonzero(as_tuple=False).squeeze(1)
        self.pos_error_integral[reset_envs] = torch.zeros(env_mask.sum(), 3, device=self.device)
        self.att_error_integral[reset_envs] = torch.zeros(env_mask.sum(), 3, device=self.device)
    
    # Reset the mass, inertia, and thrust-to-weight ratio of the vehicle
    def reset_dr_terms(self, env_mask, new_mass, new_inertia, new_ttw):
        self.mass = new_mass
        self.inertia_tensor = new_inertia
        self.thrust_to_weight = new_ttw


    def SE3_Control(self, desired_pos, desired_yaw, 
                    com_pos, com_ori_quat, com_vel, com_omega, 
                    obs):

        batch_size = obs.shape[0]

        if self.feed_forward:
            desired_pos, ff_vel, ff_acc, ff_jerk, ff_snap, ff_yaw, ff_yaw_dot, ff_yaw_ddot = compute_ff_terms(obs, self.policy_dt)

            quad_omega = isaac_math_utils.quat_rotate(isaac_math_utils.quat_conjugate(com_ori_quat), com_omega) # Rotate into body frame
            gravity_vec = self.gravity.tile(com_pos.shape[0], 1) # (N, 3)
            Id_3 = torch.eye(3, device=self.device).unsqueeze(0).tile(com_pos.shape[0], 1, 1) # (N, 3, 3)
        
            x_ddot_des = -self.kp_pos*(com_pos - desired_pos) - self.kd_pos*(com_vel - ff_vel) + ff_acc # (N, 3)
            R_actual = isaac_math_utils.matrix_from_quat(com_ori_quat)
            yaw_actual = yaw_from_quat(com_ori_quat)
            s_actual = flat_utils.getShapeFromRotationAndYaw(R_actual, yaw_actual) #(N, 3)


            if len(self.mass.shape) < 1:
                self.mass = self.mass.tile(batch_size).to(self.device)
            collective_thrust = self.mass.unsqueeze(1) * (s_actual.unsqueeze(-1).transpose(-2, -1) @ (x_ddot_des.unsqueeze(-1) + gravity_vec.unsqueeze(-1))).squeeze(-1)


            # Project x_ddot_des + gravity onto the s_actual direction. 
            # s, x_ddot_des, gravity are all (N, 3) vectors
            projected_x_ddot_des = (s_actual.unsqueeze(-1).transpose(-2, -1) @ (x_ddot_des.unsqueeze(-1) + gravity_vec.unsqueeze(-1))).squeeze(-1) / torch.linalg.norm(s_actual, dim=1).unsqueeze(1) * s_actual

            # Compute desired accelerations and derivatives
            x_ddot = -gravity_vec + projected_x_ddot_des
            x_dddot_des = -self.kp_pos*(com_vel - ff_vel) - self.kd_pos*(x_ddot - ff_acc) + ff_jerk
            
            s_dot = torch.bmm(R_actual, math_utils.hat_map(quad_omega))[:,:,-1].view(batch_size, 3)
    
            x_dddot = (s_dot.unsqueeze(-1).transpose(-2, -1) @ (x_ddot_des.unsqueeze(-1) + gravity_vec.unsqueeze(-1))).squeeze(-1) * s_dot + (s_actual.unsqueeze(-1).transpose(-2, -1) @ x_dddot_des.unsqueeze(-1)).squeeze(-1) * s_actual
            x_ddddot_des = -self.kp_pos*(x_ddot - ff_acc) - self.kd_pos*(x_dddot - ff_jerk) + ff_snap
            

            # Compute desired shapes and derivatives
            denom = torch.linalg.norm(x_ddot_des + gravity_vec, dim=1).unsqueeze(1)
            s_des = (x_ddot_des + gravity_vec) / denom # (N, 3)

            s_dot_des = (torch.bmm(Id_3 - s_des.unsqueeze(-1) * s_des.unsqueeze(1), x_dddot_des.unsqueeze(-1))).squeeze(-1) / denom # (N, 3) # TODO: Check if this should be s_actual or s_des
            num1 = torch.bmm(Id_3 - s_des.unsqueeze(-1) * s_des.unsqueeze(1), x_ddddot_des.unsqueeze(-1)).squeeze(-1) # TODO: Check if this should be s_actual or s_des
            num2 = torch.bmm(2*s_dot_des.unsqueeze(-1)*s_des.unsqueeze(1) + s_des.unsqueeze(-1)*s_dot_des.unsqueeze(1), x_dddot_des.unsqueeze(-1)).squeeze(-1) # TODO: Check if this should be s_actual or s_des
            s_ddot_des = (num1 - num2) / denom

            # Compute desired rotations and derivatives
            R_des = flat_utils.getRotationFromShape(s_des, ff_yaw) # (N, 3, 3)
            R_dot_des = flat_utils.getRotationDotFromShape(s_des, s_dot_des, ff_yaw, ff_yaw_dot) # (N, 3, 3)
            R_ddot_des = flat_utils.getRotationDDotFromShape(s_des, s_dot_des, s_ddot_des, ff_yaw, ff_yaw_dot, ff_yaw_ddot) # (N, 3, 3)

            # Compute feed forward omega
            omega_hat_des = R_actual.transpose(-2, -1) @ R_dot_des # (N, 3, 3)
            omega_dot_hat_des = R_actual.transpose(-2, -1) @ R_ddot_des  - torch.bmm(omega_hat_des, omega_hat_des) # (N, 3, 3)
            omega_des = vee_map(omega_hat_des) # (N, 3) [Body Frame]
            omega_dot_des = vee_map(omega_dot_hat_des) # (N, 3) [Body Frame]

            ff_part_1 = torch.bmm(torch.bmm(torch.bmm(math_utils.hat_map(quad_omega), R_actual.transpose(-2, -1)), R_des), omega_des.unsqueeze(-1)).squeeze(-1) # (N, 3)
            ff_part_2 = torch.bmm(torch.bmm(R_actual.transpose(-2, -1), R_des), omega_dot_des.unsqueeze(-1)).squeeze(-1) # (N, 3)
            feed_forward_angular_acceleration = ff_part_1 - ff_part_2

            if self.track_buffers: 
                self.ref_pos_buffer.append(desired_pos)
                self.pos_buffer.append(com_pos)
                self.s_buffer.append(s_actual)
                self.s_dot_buffer.append(s_dot)
                self.s_des_buffer.append(s_des)
                self.s_dot_des_buffer.append(s_dot_des)

        else: # Not using feed forward - no future positions
            ff_vel = torch.zeros_like(com_vel)
            ff_acc = torch.zeros_like(com_vel)
            com_omega = isaac_math_utils.quat_rotate(isaac_math_utils.quat_conjugate(com_ori_quat), com_omega)

            pos_error = com_pos - desired_pos
            vel_error = com_vel - ff_vel

            self.pos_error_integral += pos_error * self.policy_dt

            if self.use_integral:
                pos_error_integral = self.pos_error_integral
            else:
                pos_error_integral = torch.zeros_like(self.pos_error_integral)

            F_des = self.mass.unsqueeze(1) * (-self.kp_pos * pos_error + \
                             -self.kd_pos * vel_error + \
                                -self.ki_pos * pos_error_integral + \
                             ff_acc + \
                            self.gravity.tile(com_pos.shape[0], 1)) # (N, 3)
            
            quad_ori_matrix = isaac_math_utils.matrix_from_quat(com_ori_quat) # (batch_size, 3, 3)
            R_actual = quad_ori_matrix
            quad_omega = com_omega # (batch_size, 3)
            quad_b3 = quad_ori_matrix[:, :, 2] # (batch_size, 3)
            
            collective_thrust = (F_des * quad_b3).sum(dim=-1)

            # Compute the desired orientation
            b3_des = isaac_math_utils.normalize(F_des)
            yaw_des = desired_yaw.view(batch_size,1) # (N,)
            c1_des = torch.stack([torch.cos(yaw_des), torch.sin(yaw_des), torch.zeros(self.num_envs, 1, device=self.device)],dim=1).view(batch_size, 3) # (N, 3)
            b2_des = torch.cross(b3_des, c1_des, dim=1)
            b2_des = isaac_math_utils.normalize(b2_des)
            b1_des = torch.cross(b2_des, b3_des, dim=1)
            R_des = torch.stack([b1_des, b2_des, b3_des], dim=2) # (batch_size, 3, 3)

            omega_des = torch.zeros_like(quad_omega, device=self.device) # Omega des is 0. (batch_size, 3)
            omega_dot_des = torch.zeros_like(quad_omega, device=self.device) # Omega des is 0. (batch_size, 3)
            feed_forward_angular_acceleration = torch.zeros_like(quad_omega, device=self.device) # Omega des is 0. (batch_size, 3)


        # Compute orientation error
        S_err = 0.5 * (torch.bmm(R_des.transpose(-2, -1), R_actual) - torch.bmm(R_actual.transpose(-2, -1), R_des)) # (batch_size, 3, 3)
        att_err = vee_map(S_err) # (batch_size, 3)
        self.att_error_integral += att_err * self.policy_dt
        if torch.any(torch.isnan(att_err)):
            print("Nan detected in attitude error!", att_err)
            att_err = torch.zeros_like(att_err, device=self.device)
        omega_err = quad_omega - omega_des # Omega des is 0. (batch_size, 3)
        
        if self.use_integral:
            att_err_integral = self.att_error_integral
        else:
            att_err_integral = torch.zeros_like(self.att_error_integral)

        # Compute desired moments
        # M = I @ (-kp_att * att_err - kd_att * omega_err) + omega x I @ omega
        if len(self.inertia_tensor.shape) < 3:
            inertia = self.inertia_tensor.unsqueeze(0).tile(batch_size, 1, 1).to(self.device)
        else:
            inertia = self.inertia_tensor.to(self.device)
        att_pd = -self.kp_att * att_err - self.kd_att * omega_err  - self.ki_att * att_err_integral
        I_omega = torch.bmm(inertia.view(batch_size, 3, 3), quad_omega.unsqueeze(2)).squeeze(2).to(self.device)

        M_des = torch.bmm(inertia.view(batch_size, 3, 3), att_pd.unsqueeze(2)).squeeze(2) + \
                torch.cross(quad_omega, I_omega, dim=1) - \
                torch.bmm(inertia.view(batch_size, 3, 3), feed_forward_angular_acceleration.unsqueeze(2)).squeeze(2)
        
        if self.control_mode == "CTBM":
            return collective_thrust, M_des

        elif self.control_mode == "CTATT":
            # att_des = flat_utils.getAttitudeFromRotationAndYaw(R_des, yaw_des)

            roll_des, pitch_des, yaw_des = isaac_math_utils.euler_xyz_from_quat(isaac_math_utils.quat_from_matrix(R_des))
            att_des = torch.stack([roll_des, pitch_des, yaw_des], dim=1) # (batch_size, 3)

            return collective_thrust, att_des
        elif self.control_mode == "CTBR":
                # We already have collective thrust, we need to compute the desired body rates
                # br_des = 0.01 * (-self.kp_att*att_err - self.kd_att*omega_err)
                br_des = att_pd

                return collective_thrust, br_des
        else:
            raise NotImplementedError("Control mode not implemented!")

        

    def shift_CTBM_to_rigid_frame(self, collective_thrust, M_des, com_in_local_frame):
        """
        Helper method to implement the Wrench shift from the COM location (provided by SE3 controller) to any frame on the same rigid body.
        """
        f_vec = torch.zeros(collective_thrust.shape[0], 3, device=self.device)
        f_vec[:, 2] = collective_thrust

        M_des = M_des + torch.cross(com_in_local_frame, f_vec, dim=1)

        return collective_thrust, M_des

    def get_action(self, obs):
        
        batch_size = obs.shape[0]
        num_obs = obs.shape[1]
        com_pos = obs[:, :3]
        com_ori_quat = obs[:, 3:7]
        com_vel = obs[:, 7:10]
        com_omega = obs[:, 10:13]
        desired_pos = obs[:, 13:16]
        desired_yaw = obs[:, 16:17]

        
        collective_thrust, M_des = self.SE3_Control(desired_pos, desired_yaw, com_pos, com_ori_quat, com_vel, com_omega, obs)
        ones = torch.ones(batch_size, device=self.device)    

        if self.control_mode == "CTBM":
            ct_shape = collective_thrust.shape
            # print("Collective thrust shape:", ct_shape)

            mass_reshape = self.mass.reshape(ct_shape) if type(self.mass) == float else self.mass
            # print("Mass reshape shape:", mass_reshape.shape)
            u1 = rescale_command(collective_thrust.squeeze(), torch.zeros_like(mass_reshape), self.thrust_to_weight * 9.81*mass_reshape).view(batch_size, 1)
            u2 = rescale_command(M_des[:, 0], -self.moment_scale_xy * ones, self.moment_scale_xy * ones).view(batch_size, 1)
            u3 = rescale_command(M_des[:, 1], -self.moment_scale_xy  * ones, self.moment_scale_xy * ones).view(batch_size, 1)
            u4 = rescale_command(M_des[:, 2], -self.moment_scale_z * ones, self.moment_scale_z * ones).view(batch_size, 1)
        elif self.control_mode == "CTATT":
            u1 = rescale_command(collective_thrust, torch.zeros_like(self.mass), self.thrust_to_weight * 9.81*self.mass).view(batch_size, 1)
            # u2 = rescale_command(M_des[:, 0], -self.attitude_scale_xy, self.attitude_scale_xy).view(batch_size, 1)
            # u3 = rescale_command(M_des[:, 1], -self.attitude_scale_xy, self.attitude_scale_xy).view(batch_size, 1)
            # u4 = rescale_command(M_des[:, 2], -self.attitude_scale_z, self.attitude_scale_z).view(batch_size, 1)

            u2 = rescale_command(M_des[:, 0], -self.attitude_scale * ones, self.attitude_scale * ones).view(batch_size, 1)
            u3 = rescale_command(M_des[:, 1], -self.attitude_scale * ones, self.attitude_scale * ones).view(batch_size, 1)
            u4 = rescale_command(M_des[:, 2], -self.attitude_scale * ones, self.attitude_scale * ones).view(batch_size, 1)
        elif self.control_mode == "CTBR":
            u1 = rescale_command(collective_thrust, torch.zeros_like(self.mass), self.thrust_to_weight * 9.81*self.mass).view(batch_size, 1)
            u2 = rescale_command(M_des[:, 0], -self.body_rate_scale_xy * ones, self.body_rate_scale_xy * ones).view(batch_size, 1)
            u3 = rescale_command(M_des[:, 1], -self.body_rate_scale_xy * ones, self.body_rate_scale_xy * ones).view(batch_size, 1)
            u4 = rescale_command(M_des[:, 2], -self.body_rate_scale_z * ones, self.body_rate_scale_z * ones).view(batch_size, 1)
            

        return torch.stack([u1, u2, u3, u4], dim=1).view(batch_size, 4)

    def log_buffers(self):
        if self.track_buffers:
            self.s_buffer = torch.stack(self.s_buffer, dim=0)
            self.s_dot_buffer = torch.stack(self.s_dot_buffer, dim=0)
            self.s_des_buffer = torch.stack(self.s_des_buffer, dim=0)
            self.s_dot_des_buffer = torch.stack(self.s_dot_des_buffer, dim=0)
            self.ref_pos_buffer = torch.stack(self.ref_pos_buffer, dim=0)
            self.pos_buffer = torch.stack(self.pos_buffer, dim=0)

            self.s_buffer = self.s_buffer.cpu().detach()
            self.s_dot_buffer = self.s_dot_buffer.cpu().detach()
            self.s_des_buffer = self.s_des_buffer.cpu().detach()
            self.s_dot_des_buffer = self.s_dot_des_buffer.cpu().detach()
            self.ref_pos_buffer = self.ref_pos_buffer.cpu().detach()
            self.pos_buffer = self.pos_buffer.cpu().detach()

            torch.save(self.s_buffer, "s_buffer.pt")
            torch.save(self.s_dot_buffer, "s_dot_buffer.pt")
            torch.save(self.s_des_buffer, "s_des_buffer.pt")
            torch.save(self.s_dot_des_buffer, "s_dot_des_buffer.pt")
            torch.save(self.ref_pos_buffer, "ref_pos_buffer.pt")
            torch.save(self.pos_buffer, "pos_buffer.pt")
