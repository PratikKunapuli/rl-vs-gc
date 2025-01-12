import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

import omni.isaac.lab.utils.math as isaac_math_utils
from utils.math_utilities import vee_map, yaw_from_quat, quat_from_yaw, matrix_log
import utils.flatness_utilities as flat_utils
import utils.math_utilities as math_utils

def compute_2d_rotation_matrix(thetas: torch.tensor):
    """
    Given theta (N,) angles, return (N, 3, 3) matricies corresponding to Rz(theta)
    """
    N = thetas.shape[0]
    zeros = torch.zeros(N, device=thetas.device)
    ones = torch.ones(N, device=thetas.device)
    R = torch.stack([torch.stack([torch.cos(thetas), -torch.sin(thetas), zeros], dim=1),
                     torch.stack([torch.sin(thetas), torch.cos(thetas), zeros], dim=1),
                     torch.stack([zeros, zeros, ones], dim=1)], dim=1)
    return R
        
def compute_desired_pose_0dof(goal_pos_w, goal_ori_w, pos_transform, ori_transform):
    # Find b2 in the ori frame, set z component to 0 and the desired yaw is the atan2 of the x and y components
    b2 = isaac_math_utils.quat_rotate(goal_ori_w, torch.tensor([[0.0, 1.0, 0.0]], device=goal_ori_w.device).tile(goal_ori_w.shape[0], 1))
    b2[:, 2] = 0.0
    b2 = isaac_math_utils.normalize(b2)
     
    # Yaw is the angle between b2 and the y-axis
    yaw_desired = torch.atan2(b2[:, 1], b2[:, 0]) - torch.pi/2
    yaw_desired = isaac_math_utils.wrap_to_pi(yaw_desired)

    # Position desired is the pos_transform along -b2 direction
    pos_desired = goal_pos_w + torch.bmm(torch.linalg.norm(pos_transform, dim=1).view(-1, 1, 1), -1*b2.unsqueeze(1)).squeeze(1)

    # We want to find the position desired where we go in the -b2 direction by the pos_transform. 
    # pos_transform is (N,3), -b2 is (N,3), we want to find the position desired (N,3)
    # pos_desired = goal_pos_w + isaac_math_utils.quat_rotate(quat_from_yaw(yaw_desired), pos_transform)


    # r_z_theta = compute_2d_rotation_matrix(yaw_desired)
    # pos_desired = goal_pos_w + pos_transform * b2

    # r_z_theta = compute_2d_rotation_matrix(yaw_desired)
    # minus_y_axis = torch.zeros(goal_pos_w.shape[0], 3, device=goal_pos_w.device)
    # minus_y_axis[:, 1] = -1.0 * torch.linalg.norm(pos_transform, dim=1)
    # # local_offset = torch.bmm(r_z_theta, pos_transform.unsqueeze(2)).squeeze(2)
    # local_offset = torch.bmm(r_z_theta, minus_y_axis.unsqueeze(2)).squeeze(2)


    return pos_desired, yaw_desired, b2

def compute_desired_pose_1dof(goal_pos_w, goal_ori_w, pos_transform):
    b2 = isaac_math_utils.quat_rotate(goal_ori_w, torch.tensor([[0.0, 1.0, 0.0]], device=goal_ori_w.device).tile(goal_ori_w.shape[0], 1))
    b2 = isaac_math_utils.normalize(b2)
     
    # Yaw is the angle between b2 and the y-axis
    yaw_desired = torch.atan2(b2[:, 1], b2[:, 0]) - torch.pi/2
    yaw_desired = isaac_math_utils.wrap_to_pi(yaw_desired)

    # Position desired is the pos_transform norm along -b2 direction
    pos_desired = goal_pos_w + torch.bmm(torch.linalg.norm(pos_transform, dim=1).view(-1, 1, 1), -1*b2.unsqueeze(1)).squeeze(1)

    theta_des = torch.arcsin(b2[:, 2])

    return pos_desired, yaw_desired, theta_des

@torch.jit.script
def get_point_state_from_ee_transform_w(ee_pos_w, ee_ori_quat_w, ee_vel_w, ee_omega_w, point_pos_ee_frame):
    point_pos_w, _ = isaac_math_utils.combine_frame_transforms(ee_pos_w, ee_ori_quat_w, point_pos_ee_frame)
    point_vel_w = ee_vel_w + torch.cross(ee_omega_w, isaac_math_utils.quat_rotate(ee_ori_quat_w, point_pos_ee_frame), dim=1)
    
    return point_pos_w, point_vel_w

class DecoupledController():
    def __init__(self, num_envs, num_dofs, vehicle_mass, arm_mass, inertia_tensor, pos_offset, ori_offset, print_debug=False, com_pos_w=None, device='cpu',
                  kp_pos_gain_xy=10.0, kp_pos_gain_z=20.0, kd_pos_gain_xy=7.0, kd_pos_gain_z=9.0, 
                  kp_att_gain_xy=400.0, kp_att_gain_z=2.0, kd_att_gain_xy=70.0, kd_att_gain_z=2.0,
                  tuning_mode=False, use_full_obs=False, skip_precompute=False, vehicle="AM", control_mode="CTBM", policy_dt=0.02,
                  feed_forward=False, disable_gravity=False):
        self.num_envs = num_envs
        self.num_dofs = num_dofs
        self.print_debug = print_debug
        self.tuning_mode = tuning_mode
        self.use_full_obs = use_full_obs
        # self.arm_offset = arm_offset
        self.vehicle_mass = vehicle_mass
        self.arm_mass = arm_mass
        self.mass = vehicle_mass + arm_mass
        self.com_pos_w = com_pos_w
        self.policy_dt = policy_dt
        self.feed_forward = feed_forward

        self.control_mode = control_mode

        print("\n\n[Debug] Total Mass: ", self.mass, "\n\n")

        self.inertia_tensor = inertia_tensor
        self.position_offset = pos_offset
        self.orientation_offset = ori_offset

        if vehicle == "AM":
            self.moment_scale_xy = 0.5
            self.moment_scale_z = 0.025 #0.025 # 0.1
            self.thrust_to_weight = 3.0
        else:
            #Crazyflie
            self.thrust_to_weight = 1.8
            self.moment_scale_xy = 0.01
            self.moment_scale_z = 0.01
            # self.attitude_scale = torch.pi/6.0
            self.attitude_scale_z = torch.pi
            self.attitude_scale_xy = 0.2

        self.device = torch.device(device)
        self.inertia_tensor.to(self.device)
        
        self.initial_yaw_offset = torch.tensor([[0.7071, 0, 0, -0.7071]], device=self.device)

        self.gravity = torch.tensor([0.0, 0.0, 9.81], device=self.device)
        if disable_gravity:
            self.gravity = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        # Tested defaults gains
        # self.kp_pos = torch.tensor([10.0, 10.0, 20.0], device=self.device)
        # self.kd_pos = torch.tensor([7.0, 7.0, 9.0], device=self.device)
        # self.kp_att = torch.tensor([400.0, 400.0, 2.0], device=self.device) 
        # self.kd_att = torch.tensor([70.0, 70.0, 2.0], device=self.device)

        self.kp_pos = torch.tensor([kp_pos_gain_xy, kp_pos_gain_xy, kp_pos_gain_z], device=self.device)
        self.kd_pos = torch.tensor([kd_pos_gain_xy, kd_pos_gain_xy, kd_pos_gain_z], device=self.device)
        self.kp_att = torch.tensor([kp_att_gain_xy, kp_att_gain_xy, kp_att_gain_z], device=self.device)
        self.kd_att = torch.tensor([kd_att_gain_xy, kd_att_gain_xy, kd_att_gain_z], device=self.device)


        # self.kp_pos = torch.tensor([7.5, 15.0, 20.0], device=self.device)
        # self.kd_pos = torch.tensor([15.0, 8.0, 9.0], device=self.device)

        # self.kp_att = torch.tensor([400.0, 200.0, 2.0], device=self.device) 
        # self.kd_att = torch.tensor([50.0, 200.0, 2.0], device=self.device)

        self.s_buffer = []
        self.s_des_buffer = []
        self.s_dot_buffer = []
        self.s_dot_des_buffer = []
        self.ref_pos_buffer = []
        self.pos_buffer = []


        if not skip_precompute:
            self.precompute_transforms()

    def precompute_transforms(self):
        quad_pos_w = self.position_offset
        quad_ori_quat_w = self.orientation_offset
        ee_pos_w = torch.tensor([0.0, 0.0, 0.5], device=self.device).reshape(1, 3)
        ee_ori_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).reshape(1, 4)

        # com_pos_w = (quad_pos_w * self.vehicle_mass + ee_pos_w * self.arm_mass) / self.mass

        self.com_pos_ee_frame, self.com_ori_ee_frame = isaac_math_utils.subtract_frame_transforms(ee_pos_w, ee_ori_quat_w, quad_pos_w, quad_ori_quat_w)
        # self.quad_pos_ee_frame, self.quad_ori_ee_frame = isaac_math_utils.subtract_frame_transforms( quad_pos_w, quad_ori_quat_w, ee_pos_w, ee_ori_quat_w)
        self.quad_pos_ee_frame, self.quad_ori_ee_frame = isaac_math_utils.subtract_frame_transforms(ee_pos_w, ee_ori_quat_w, quad_pos_w, quad_ori_quat_w)
        quad_pos_w, _ = isaac_math_utils.combine_frame_transforms(ee_pos_w, ee_ori_quat_w, self.position_offset)
        self.quad_pos_ee_frame = (self.position_offset).unsqueeze(0)
        # print("COM Pos in EE Frame: ", self.com_pos_ee_frame)
        # print("Quad Pos in EE Frame: ", self.quad_pos_ee_frame)
        if self.com_pos_w is not None:
            self.com_pos_ee_frame, self.com_ori_ee_frame = isaac_math_utils.subtract_frame_transforms(ee_pos_w, ee_ori_quat_w, self.com_pos_w, quad_ori_quat_w)
            vehicle_com_offset_local_frame = torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(0)
            arm_com_offset_local_frame = torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(0)
            self.com_pos_w = torch.zeros(1, 3, device=self.device)
            self.com_pos_w += self.vehicle_mass * (quad_pos_w + isaac_math_utils.quat_rotate(quad_ori_quat_w, vehicle_com_offset_local_frame))
            self.com_pos_w += self.arm_mass * (ee_pos_w + isaac_math_utils.quat_rotate(ee_ori_quat_w, arm_com_offset_local_frame))
            self.com_pos_w /= self.mass
            self.com_pos_ee_frame, self.com_ori_ee_frame = isaac_math_utils.subtract_frame_transforms(ee_pos_w, ee_ori_quat_w, self.com_pos_w, quad_ori_quat_w)
            self.com_pos_v_frame, _ = isaac_math_utils.subtract_frame_transforms(self.quad_pos_ee_frame, self.quad_ori_ee_frame, self.com_pos_ee_frame)

        else:
            self.com_pos_ee_frame = torch.tensor([0.00000000e+00, -2.00715814e-01, -1.59835415e-04], device=self.device).reshape(1, 3) # pulled from Pinocchio
            # self.com_pos_ee_frame = torch.tensor([0.0, -0.2, 0], device=self.device).reshape(1, 3) # pulled from Pinocchio
            # print("[Debug] Quad ori ee_frame = ", self.quad_ori_ee_frame)
            self.com_ori_ee_frame = self.quad_ori_ee_frame
            # self.com_ori_ee_frame = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).reshape(1, 4)
            # self.com_pos_w = self.com_pos_ee_frame + ee_pos_w
            self.com_pos_v_frame = torch.zeros(1, 3, device=self.device)
            self.com_pos_w, _ = isaac_math_utils.combine_frame_transforms(ee_pos_w, ee_ori_quat_w, self.com_pos_ee_frame)
            self.com_pos_v_frame, self.com_ori_v_frame = isaac_math_utils.subtract_frame_transforms(quad_pos_w, quad_ori_quat_w, self.com_pos_w, quad_ori_quat_w)
            if self.print_debug:
                print("COM Pos in V Frame: ", self.com_pos_v_frame)

            #get the com offset from the ee using the com pos in v frame
            com_pos_ee_frame_from_v, _ = isaac_math_utils.combine_frame_transforms(self.quad_pos_ee_frame, self.quad_ori_ee_frame, self.com_pos_v_frame)
            # print("COM Pos in EE Frame from V Frame: ", com_pos_ee_frame_from_v)

        # print("quad_pos_w: ", quad_pos_w)
        if self.print_debug:
            print("COM Pos in World Frame: ", self.com_pos_w)
            print("COM Ori in World Frame: ", self.com_ori_ee_frame)

        des_com_ori_w = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], device=self.device).reshape(1, 4)
        ee_pos_com_frame = -1.0 * self.com_pos_ee_frame
        ee_ori_com_frame = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).reshape(1, 4)
        if self.print_debug:
            print("Desired COM pos in World Frame: ", self.com_pos_w)
            print("Desired COM ori in World Frame: ", des_com_ori_w)

        des_ee_pos_w, des_ee_ori_w = isaac_math_utils.combine_frame_transforms(self.com_pos_w, des_com_ori_w, ee_pos_com_frame, ee_ori_com_frame)
        if self.print_debug:
            print("Desired EE pos in World Frame: ", des_ee_pos_w)
        # print("Desired EE ori in World Frame: ", des_ee_ori_w)
        # print("COM Pos in EE Frame: ", self.com_pos_ee_frame)

        # self.com_pos_ee_frame = self.quad_pos_ee_frame + torch.tensor([0.0, 0.0, 0.0], device=self.device).reshape(1, 3) # pulled from Pinocchio

        # self.com_pos_v_frame = torch.tensor([0.0, 0.0, 0.0], device=self.device).reshape(1, 3)
        # print("COM Pos in V Frame: ", self.com_pos_v_frame)
        # print("Check COM Pos in V Frame: ", self.com_pos_v_frame)

        self.com_pos_v_frame = self.com_pos_v_frame.tile(self.num_envs, 1)
        self.com_pos_ee_frame = self.com_pos_ee_frame.tile(self.num_envs, 1)
        self.quad_pos_ee_frame = self.quad_pos_ee_frame.tile(self.num_envs, 1)
        self.com_ori_ee_frame = self.com_ori_ee_frame.tile(self.num_envs, 1)
        self.quad_ori_ee_frame = self.quad_ori_ee_frame.tile(self.num_envs, 1)

        self.yaw_offset = yaw_from_quat(self.quad_ori_ee_frame)
        # print("Yaw Offset: ", self.yaw_offset)

        # import code; code.interact(local=locals())
    
    def compute_desired_joint_angles(self, obs):
        return None

    
    def rescale_command(self, command, min_val, max_val):
        """
        We want to rescale the command to be between -1 and 1, where the original command is between min_val and max_val
        """
        return 2.0 * (command - min_val) / (max_val - min_val) - 1.0
    
    def compute_ff_terms(self, obs):
        # first 17 terms are part of state, then horizon*3 future positions, then horizon*4 future quaternions
        batch_size = obs.shape[0]
        num_obs = obs.shape[1]
        horizon = (num_obs - 17) // 7 # 3 for position and 4 for quaternion
        futures = obs[:, -7*horizon:]

        # Position Feed Forwards
        future_com_pos_w = futures[:, :3*horizon].reshape(batch_size, horizon, 3)
        feed_forward_velocities = (future_com_pos_w[:, 1:] - future_com_pos_w[:, :-1]) / self.policy_dt
        feed_forward_accelerations = (feed_forward_velocities[:, 1:] - feed_forward_velocities[:, :-1]) / self.policy_dt
        feed_forward_jerks = (feed_forward_accelerations[:, 1:] - feed_forward_accelerations[:, :-1]) / self.policy_dt
        feed_forward_snaps = (feed_forward_jerks[:, 1:] - feed_forward_jerks[:, :-1]) / self.policy_dt
        ff_pos = future_com_pos_w[:, 0]
        ff_vel = feed_forward_velocities[:, 0]
        ff_acc = feed_forward_accelerations[:, 0]
        ff_jerk = feed_forward_jerks[:, 0]
        ff_snap = feed_forward_snaps[:, 0]

        # Yaw Feed Forwards
        future_com_ori_w = futures[:, 3*horizon:].reshape(batch_size, horizon, 4)
        feed_forward_yaws = yaw_from_quat(future_com_ori_w)
        # feed_forward_yaws_dot = (feed_forward_yaws[:, 1:] - feed_forward_yaws[:, :-1]) / self.policy_dt
        feed_forward_yaws_dot = ((feed_forward_yaws[:, 1:] - feed_forward_yaws[:, :-1] + 3*torch.pi) % (2*torch.pi) - torch.pi) / self.policy_dt
        # print("Feed Forward Yaws: ", feed_forward_yaws[0,0])
        # print("Feed Forward Yaws Dot: ", feed_forward_yaws_dot[0,0])
        # print("Smoothed: ", smoothed[0,0])

        feed_forward_yaws_ddot = (feed_forward_yaws_dot[:, 1:] - feed_forward_yaws_dot[:, :-1]) / self.policy_dt
        ff_yaw = feed_forward_yaws[:, 0]
        ff_yaw_dot = feed_forward_yaws_dot[:, 0]
        ff_yaw_ddot = feed_forward_yaws_ddot[:, 0]
    
        return ff_pos, ff_vel, ff_acc, ff_jerk, ff_snap, ff_yaw, ff_yaw_dot, ff_yaw_ddot

    def SE3_Control_FF(self, desired_pos, desired_yaw,
                    com_pos, com_ori_quat, com_vel, com_omega,
                    obs):
        ff_vel, ff_acc, ff_jerk, ff_snap, ff_yaw, ff_yaw_dot, ff_yaw_ddot = self.compute_ff_terms(obs)
        



    def SE3_Control(self, desired_pos, desired_yaw, 
                    com_pos, com_ori_quat, com_vel, com_omega, 
                    obs):
        if self.use_full_obs:
            ee_pos = obs[:, 13:16]
            ee_ori_quat = obs[:, 16:20]
            ee_vel = obs[:, 20:23]
            ee_omega = obs[:, 23:26]

            # Use these if "vehicle" is the body in the USD file
            # quad_pos = obs[:, :3]
            # quad_ori_quat = obs[:, 3:7]
            # quad_vel = obs[:, 7:10]
            # quad_omega = obs[:, 10:13].to(self.device)

            # Use these if "COM" is the body in the USD file
            com_pos = obs[:, :3]
            com_ori_quat = obs[:, 3:7]
            com_vel = obs[:, 7:10]
            com_omega = obs[:, 10:13].to(self.device)

        batch_size = obs.shape[0]

        if self.feed_forward:
            desired_pos, ff_vel, ff_acc, ff_jerk, ff_snap, ff_yaw, ff_yaw_dot, ff_yaw_ddot = self.compute_ff_terms(obs)
            self.ref_pos_buffer.append(desired_pos)
            self.pos_buffer.append(com_pos)

            # print("Input and Traj close: ", torch.allclose(input_des_pos, desired_pos, atol=1e-5))
            quad_omega = isaac_math_utils.quat_rotate(isaac_math_utils.quat_conjugate(com_ori_quat), com_omega) # Rotate into body frame
            gravity_vec = self.gravity.tile(com_pos.shape[0], 1) # (N, 3)
            Id_3 = torch.eye(3, device=self.device).unsqueeze(0).tile(com_pos.shape[0], 1, 1) # (N, 3, 3)
        
            x_ddot_des = -self.kp_pos*(com_pos - desired_pos) - self.kd_pos*(com_vel - ff_vel) + ff_acc # (N, 3)
            R_actual = isaac_math_utils.matrix_from_quat(com_ori_quat)
            yaw_actual = yaw_from_quat(com_ori_quat)
            s_actual = flat_utils.getShapeFromRotationAndYaw(R_actual, yaw_actual) #(N, 3)
            self.s_buffer.append(s_actual)

            collective_thrust = self.mass * (s_actual.unsqueeze(-1).transpose(-2, -1) @ (x_ddot_des.unsqueeze(-1) + gravity_vec.unsqueeze(-1))).squeeze(-1)

            if self.print_debug:
                print("Gravity vec (ge3): ", gravity_vec)
                print("s_actual: ", s_actual)
                print("x_ddot_des: ", x_ddot_des)
                print("sT (x_ddot_des + gravity): ", (s_actual.unsqueeze(-1).transpose(-2, -1) @ (x_ddot_des.unsqueeze(-1) + gravity_vec.unsqueeze(-1))).squeeze(-1))
            # print("s @ (x_ddot_des + gravity): ", (s_actual.unsqueeze(-1) @ (x_ddot_des.unsqueeze(-1) + gravity_vec.unsqueeze(-1))).squeeze(-1))

            # Project x_ddot_des + gravity onto the s_actual direction. 
            # Use the vector projection formula but make it batched
            # projected = s <dot> (x_ddot_des + gravity) / ||s||^2 * (x_ddot_des + gravity)
            # s, x_ddot_des, gravity are all (N, 3) vectors
            projected_x_ddot_des = (s_actual.unsqueeze(-1).transpose(-2, -1) @ (x_ddot_des.unsqueeze(-1) + gravity_vec.unsqueeze(-1))).squeeze(-1) / torch.linalg.norm(s_actual, dim=1).unsqueeze(1) * s_actual




            # Compute desired accelerations and derivatives
            # x_ddot = -gravity_vec + (s_actual.unsqueeze(-1).transpose(-2, -1) @ (x_ddot_des.unsqueeze(-1) + gravity_vec.unsqueeze(-1))).squeeze(-1)
            x_ddot = -gravity_vec + projected_x_ddot_des
            x_dddot_des = -self.kp_pos*(com_vel - ff_vel) - self.kd_pos*(x_ddot - ff_acc) + ff_jerk
            
            # x_dddot = s_dot.T(x_ddot_des + ge3) + sT(x_dddot_des)
            # s_dot comes from the hat_map(R^T @ omega) last column
            s_dot = torch.bmm(R_actual, math_utils.hat_map(quad_omega))[:,:,-1].view(batch_size, 3)
            self.s_dot_buffer.append(s_dot)
            if self.print_debug:
                print("s_dot: ", s_dot.shape)
                print("x_ddot_des: ", x_ddot_des.shape)
                print("x_ddot: ", x_ddot.shape)
                print("x_dddot_des: ", x_dddot_des.shape)
                print("s_actual: ", s_actual.shape)
                print("gravity_vec: ", gravity_vec.shape)
            # s_dot is (N,3), then (N,3,1), then (N,1,3) @ (N,3,1) = (N,1,1) -> (N)
            # I want x_dddot to be (N,3) by the end. 
            x_dddot = (s_dot.unsqueeze(-1).transpose(-2, -1) @ (x_ddot_des.unsqueeze(-1) + gravity_vec.unsqueeze(-1))).squeeze(-1) * s_dot + (s_actual.unsqueeze(-1).transpose(-2, -1) @ x_dddot_des.unsqueeze(-1)).squeeze(-1) * s_actual
            x_ddddot_des = -self.kp_pos*(x_ddot - ff_acc) - self.kd_pos*(x_dddot - ff_jerk) + ff_snap
            
            if self.print_debug:
                print("X_dddot: ", x_dddot.shape)
                print("X_ddddot_des: ", x_ddddot_des.shape)


            if self.print_debug:
                print("Projected x_ddot_des: ", projected_x_ddot_des)
                print("X_ddot_des: ", x_ddot_des)
                print("X_ddot: ", x_ddot)

            # Compute desired shapes and derivatives
            denom = torch.linalg.norm(x_ddot_des + gravity_vec, dim=1).unsqueeze(1)
            s_des = (x_ddot_des + gravity_vec) / denom # (N, 3)
            self.s_des_buffer.append(s_des)
            if self.print_debug:
                print("S_des: ", s_des)
            s_dot_des = (torch.bmm(Id_3 - s_des.unsqueeze(-1) * s_des.unsqueeze(1), x_dddot_des.unsqueeze(-1))).squeeze(-1) / denom # (N, 3) # TODO: Check if this should be s_actual or s_des
            self.s_dot_des_buffer.append(s_dot_des)
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
            if self.print_debug:
                print("R actual: ", R_actual.shape)
                print("R des: ", R_des.shape)
                print("quad_omega_hat: ", math_utils.hat_map(quad_omega).shape)
                print("omega_des: ", omega_des.shape)
                print("omega_dot_des: ", omega_dot_des.shape)

            ff_part_1 = torch.bmm(torch.bmm(torch.bmm(math_utils.hat_map(quad_omega), R_actual.transpose(-2, -1)), R_des), omega_des.unsqueeze(-1)).squeeze(-1) # (N, 3)
            ff_part_2 = torch.bmm(torch.bmm(R_actual.transpose(-2, -1), R_des), omega_dot_des.unsqueeze(-1)).squeeze(-1) # (N, 3)
            feed_forward_angular_acceleration = ff_part_1 - ff_part_2
            # print("Feed Forward Angular Acceleration: ", feed_forward_angular_acceleration)
        else:
            ff_vel = torch.zeros_like(com_vel)
            ff_acc = torch.zeros_like(com_vel)
            com_omega = isaac_math_utils.quat_rotate(isaac_math_utils.quat_conjugate(com_ori_quat), com_omega)

            pos_error = com_pos - desired_pos
            vel_error = com_vel - ff_vel
            F_des = self.mass * (-self.kp_pos * pos_error + \
                             -self.kd_pos * vel_error + \
                             ff_acc + \
                            self.gravity.tile(com_pos.shape[0], 1)) # (N, 3)
        
            if self.print_debug:
                print("[SE3] Pos Error norm: ", torch.linalg.norm(pos_error,dim=1))
        
            # print("F_des: ", F_des)
            
            batch_size = com_pos.shape[0]
            # quad_ori_matrix = isaac_math_utils.matrix_from_quat(quad_ori_quat) # (batch_size, 3, 3)
            quad_ori_matrix = isaac_math_utils.matrix_from_quat(com_ori_quat) # (batch_size, 3, 3)
            R_actual = quad_ori_matrix
            quad_omega = com_omega # (batch_size, 3)
            quad_b3 = quad_ori_matrix[:, :, 2] # (batch_size, 3)
            # print("b3: ", quad_b3)
            # collective_thrust = torch.bmm(F_des.view(batch_size, 1, 3), quad_b3.view(batch_size, 3, 1)).squeeze(2)
            collective_thrust = (F_des * quad_b3).sum(dim=-1)
            # print("Collective Thrust: ", collective_thrust)

            # Compute the desired orientation
            b3_des = isaac_math_utils.normalize(F_des)
            yaw_des = desired_yaw.view(batch_size,1) # (N,)
            c1_des = torch.stack([torch.cos(yaw_des), torch.sin(yaw_des), torch.zeros(self.num_envs, 1, device=self.device)],dim=1).view(batch_size, 3) # (N, 3)
            # print("c1_des: ", c1_des)
            b2_des = torch.cross(b3_des, c1_des, dim=1)
            b2_des = isaac_math_utils.normalize(b2_des)
            b1_des = torch.cross(b2_des, b3_des, dim=1)
            R_des = torch.stack([b1_des, b2_des, b3_des], dim=2) # (batch_size, 3, 3)

            omega_des = torch.zeros_like(quad_omega, device=self.device) # Omega des is 0. (batch_size, 3)
            omega_dot_des = torch.zeros_like(quad_omega, device=self.device) # Omega des is 0. (batch_size, 3)
            feed_forward_angular_acceleration = torch.zeros_like(quad_omega, device=self.device) # Omega des is 0. (batch_size, 3)



        if self.print_debug:
            print("Pos Error: ", com_pos - desired_pos)
            print("Vel Error: ", com_vel - ff_vel)
            print("Yaw error: ", math_utils.yaw_from_quat(com_ori_quat) - desired_yaw)
            print("Collective Thrust: ", collective_thrust)
            print("R_des: ", R_des)
            print("omega_des: ", omega_des)
            print("omega_dot_des: ", omega_dot_des)

        # import code; code.interact(local=locals())
        
        # print("Vel Error: ", vel_error)

        # Compute desired Force (batch_size, 3)

        # Compute orientation error
        S_err = 0.5 * (torch.bmm(R_des.transpose(-2, -1), R_actual) - torch.bmm(R_actual.transpose(-2, -1), R_des)) # (batch_size, 3, 3)
        att_err = vee_map(S_err) # (batch_size, 3)
        if torch.any(torch.isnan(att_err)):
            print("Nan detected in attitude error!", att_err)
            att_err = torch.zeros_like(att_err, device=self.device)
        omega_err = quad_omega - omega_des # Omega des is 0. (batch_size, 3)
        
        # Compute desired moments
        # M = I @ (-kp_att * att_err - kd_att * omega_err) + omega x I @ omega
        inertia = self.inertia_tensor.unsqueeze(0).tile(batch_size, 1, 1).to(self.device)
        att_pd = -self.kp_att * att_err - self.kd_att * omega_err 
        I_omega = torch.bmm(inertia.view(batch_size, 3, 3), quad_omega.unsqueeze(2)).squeeze(2).to(self.device)

        M_des = torch.bmm(inertia.view(batch_size, 3, 3), att_pd.unsqueeze(2)).squeeze(2) + \
                torch.cross(quad_omega, I_omega, dim=1) - \
                torch.bmm(inertia.view(batch_size, 3, 3), feed_forward_angular_acceleration.unsqueeze(2)).squeeze(2)
        
        if self.control_mode == "CTBM":
            return collective_thrust, M_des

        elif self.control_mode == "CTATT":
            # print("DC R des: ", R_des)
            # roll, pitch, yaw  = isaac_math_utils.euler_xyz_from_quat(isaac_math_utils.quat_from_matrix(R_des))
            # print(roll.shape)
            # att_des = torch.stack([isaac_math_utils.wrap_to_pi(roll), isaac_math_utils.wrap_to_pi(pitch), isaac_math_utils.wrap_to_pi(yaw)], dim=1)
            # att_des = att_des.clamp(-self.attitude_scale, self.attitude_scale)
            # print(att_des.shape)

            # Convert the SO(3) matrix R_des to a tangent element by taking the log map and then vee mapping
            # R_des_log = matrix_log(R_des)
            # att_des = vee_map(R_des_log)

            # Flatness based approach
            # H2_s = torch.bmm(R_des, flat_utils.H1(yaw_des).transpose(-2, -1))
            # s_des = H2_s[:, :, 2]
            # x_des, y_des = flat_utils.inv_s2_projection(s_des)
            # att_des = torch.stack([x_des, y_des, yaw_des], dim=1)
            att_des = flat_utils.getAttitudeFromRotationAndYaw(R_des, yaw_des)

            # print("DC Attitude: ", att_des)
            # roll, pitch, yaw  = isaac_math_utils.euler_xyz_from_quat(isaac_math_utils.quat_from_matrix(R_des))
            # att_des = torch.stack([isaac_math_utils.wrap_to_pi(roll), isaac_math_utils.wrap_to_pi(pitch), isaac_math_utils.wrap_to_pi(yaw)], dim=1)
            # att_des = att_des.clamp(-self.attitude_scale, self.attitude_scale)

            return collective_thrust, att_des
        
        else:
            raise NotImplementedError("Control mode not implemented!")

        # print("Thrust: ", collective_thrust) # (n, 1)
        # print("M_des (COM frame): ", M_des)

        

    def shift_CTBM_to_rigid_frame(self, collective_thrust, M_des, com_in_local_frame):
        """
        Helper method to implement the Wrench shift from the COM location (provided by SE3 controller) to any frame on the same rigid body.
        """
        f_vec = torch.zeros(collective_thrust.shape[0], 3, device=self.device)
        f_vec[:, 2] = collective_thrust

        M_des = M_des + torch.cross(com_in_local_frame, f_vec, dim=1)

        return collective_thrust, M_des

    def get_action(self, obs):
        if self.use_full_obs:
            goal_pos_w = obs[:, 26+self.num_dofs*2:26+self.num_dofs*2 + 3]
            goal_ori_w = obs[:, 26+self.num_dofs*2 + 3:26+self.num_dofs*2 + 7]
            ee_pos = obs[:, 13:16]
            ee_ori_quat = obs[:, 16:20]
            ee_vel = obs[:, 20:23]
            ee_omega = obs[:, 23:26]
            com_pos = obs[:, :3]
            com_ori_quat = obs[:, 3:7]
            com_vel = obs[:, 7:10]
            com_omega = obs[:, 10:13].to(self.device)
            batch_size = obs.shape[0]
        else:
            batch_size = obs.shape[0]
            num_obs = obs.shape[1]
            com_pos = obs[:, :3]
            com_ori_quat = obs[:, 3:7]
            com_vel = obs[:, 7:10]
            com_omega = obs[:, 10:13]
            desired_pos = obs[:, 13:16]
            desired_yaw = obs[:, 16:17]

            # if num_obs > 17: # future trajectory is included. 
            #     horizon = (num_obs - 17) // 3
            #     future_com_pos_w = obs[:, 17:].reshape(batch_size, horizon, 3)
            #     feed_forward_velocities = (future_com_pos_w[:, 1:] - future_com_pos_w[:, :-1]) / self.policy_dt
            #     feed_forward_accelerations = (feed_forward_velocities[:, 1:] - feed_forward_velocities[:, :-1]) / self.policy_dt
            #     ff_vel = feed_forward_velocities[:, 0]
            #     ff_acc = feed_forward_accelerations[:, 0]
            # else:
            #     ff_vel = torch.zeros(batch_size, 3, device=self.device)
            #     ff_acc = torch.zeros(batch_size, 3, device=self.device)
                
        
        

        # print("[Debug] Quad Omega: ", quad_omega)
        # print("[Debug] EE Omega: ", ee_omega)

        # ee_pos_error = goal_pos_w - ee_pos
        # print("EE pos error: ", torch.linalg.norm(ee_pos_error, dim=1))

        # print("Goal Pos: ", goal_pos_w)
        # print("com_offset: ", self.com_pos_ee_frame)
        # print("EE Vel: ", ee_vel)
        # print("EE Omega: ", ee_omega)

        # Find virtual setpoints
        # print("COM pos in EE frame: ", self.com_pos_ee_frame)
        # print("COM ori in EE frame: ", self.com_ori_ee_frame)
        # print("Quad ori in EE frame: ", self.quad_ori_ee_frame)
        if self.num_dofs == 0 and self.use_full_obs:
            # desired_pos, desired_yaw = compute_desired_pose_old(goal_pos_w, goal_ori_w, self.com_pos_ee_frame, self.com_ori_ee_frame)
            # print("COM pos in EE frame: ", self.com_pos_ee_frame.shape, " ", self.com_pos_ee_frame)


            desired_pos, desired_yaw, _ = compute_desired_pose_0dof(goal_pos_w, goal_ori_w, self.com_pos_ee_frame, self.com_ori_ee_frame)
            
            # if self.tuning_mode:
            #     desired_pos = goal_pos_w # overwrite the desired pos if we're using the task body as the COM position 
            # desired_yaw = isaac_math_utils.wrap_to_pi(desired_yaw + self.yaw_offset)
            
            # desired_pos, desired_yaw, _ = compute_desired_pose_0dof(goal_pos_w, goal_ori_w, torch.zeros_like(self.com_pos_ee_frame, device=self.device), self.com_ori_ee_frame)
            # desired_pos, desired_yaw = compute_desired_pose_0dof(goal_pos_w, goal_ori_w, self.quad_pos_ee_frame, self.quad_ori_ee_frame)
            # desired_yaw = isaac_math_utils.wrap_to_pi(desired_yaw + self.yaw_offset)
            # desired_pos, desired_yaw, _ = compute_desired_pose_0dof(goal_pos_w, goal_ori_w, self.quad_pos_ee_frame, self.quad_ori_ee_frame)
        # if self.print_debug:
        #     print("Desired Pos: ", desired_pos)
        #     print("Desired Yaw: ", desired_yaw)

        # com_pos_w, _ = isaac_math_utils.combine_frame_transforms(ee_pos, ee_ori_quat, self.com_pos_ee_frame)
        # goal_com_pos_w, goal_com_ori_w = isaac_math_utils.combine_frame_transforms(goal_pos_w, goal_ori_w, self.com_pos_ee_frame, self.com_ori_ee_frame)
        # print("[Debug] COM Pos in World Frame: ", com_pos_w)
        # print("[Debug] Goal COM Pos in World Frame: ", goal_com_pos_w)
        # print("[Debug] Goal COM Ori in World Frame: ", goal_com_ori_w)

        # offset = goal_pos_w - desired_pos
        # if self.print_debug:
        #     print("Offset norm: ", torch.linalg.norm(offset, dim=1))
        #     print("Norm of COM offset: ", torch.linalg.norm(self.com_pos_ee_frame, dim=1))
        #     print("EE Error: ", torch.linalg.norm(ee_pos - goal_pos_w, dim=1))

        if self.print_debug:
            print("Desired Pos: ", desired_pos)
            print("Desired Yaw: ", desired_yaw)
            # if not self.use_full_obs:
            print("COM Pos: ", com_pos)
            print("COM Ori: ", com_ori_quat)

        # desired_pos_quad, desired_ori_quad = compute_desired_pose(goal_pos_w, goal_ori_w, self.quad_pos_ee_frame, self.quad_ori_ee_frame)
        # print("Quad Desired Pos: ", desired_pos_quad)

        desired_joint_angles = self.compute_desired_joint_angles(obs)

        
        collective_thrust, M_des = self.SE3_Control(desired_pos, desired_yaw, com_pos, com_ori_quat, com_vel, com_omega, obs)
        # print("M_des pre transform: ", M_des)
        # M_des[:,0] = 0.0
        # M_des[:,1] = 0.0

        # Shift CTBM to rigid body frame
        # collective_thrust, M_des = self.shift_CTBM_to_rigid_frame(collective_thrust, M_des, self.com_pos_v_frame)
        # print("M_des (body frame): ", M_des)
        # if self.print_debug:
        # print(M_des.shape)

        if self.control_mode == "CTBM":
            u1 = self.rescale_command(collective_thrust, 0.0, self.thrust_to_weight * 9.81*self.mass).view(batch_size, 1)
            u2 = self.rescale_command(M_des[:, 0], -self.moment_scale_xy, self.moment_scale_xy).view(batch_size, 1)
            u3 = self.rescale_command(M_des[:, 1], -self.moment_scale_xy, self.moment_scale_xy).view(batch_size, 1)
            u4 = self.rescale_command(M_des[:, 2], -self.moment_scale_z, self.moment_scale_z).view(batch_size, 1)
        elif self.control_mode == "CTATT":
            u1 = self.rescale_command(collective_thrust, 0.0, self.thrust_to_weight * 9.81*self.mass).view(batch_size, 1)
            u2 = self.rescale_command(M_des[:, 0], -self.attitude_scale_xy, self.attitude_scale_xy).view(batch_size, 1)
            u3 = self.rescale_command(M_des[:, 1], -self.attitude_scale_xy, self.attitude_scale_xy).view(batch_size, 1)
            u4 = self.rescale_command(M_des[:, 2], -self.attitude_scale_z, self.attitude_scale_z).view(batch_size, 1)
            # u2 = M_des[:, 0].view(batch_size, 1)
            # u3 = M_des[:, 1].view(batch_size, 1)
            # u4 = M_des[:, 2].view(batch_size, 1)

        # import code; code.interact(local=locals())

        return torch.stack([u1, u2, u3, u4], dim=1).view(batch_size, 4)

    def log_buffers(self):
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
