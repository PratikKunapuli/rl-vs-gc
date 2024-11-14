import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

import omni.isaac.lab.utils.math as isaac_math_utils
from utils.math_utilities import vee_map, yaw_from_quat, quat_from_yaw, matrix_log
import utils.flatness_utilities as flat_utils

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
                  tuning_mode=False, use_full_obs=False, skip_precompute=False, vehicle="AM", control_mode="CTBM"):
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
            self.thrust_to_weight = 1.6
            self.moment_scale_xy = 0.01
            self.moment_scale_z = 0.01
            # self.attitude_scale = torch.pi/6.0
            self.attitude_scale_z = torch.pi
            self.attitude_scale_xy = 0.2

        self.device = torch.device(device)
        self.inertia_tensor.to(self.device)
        
        self.initial_yaw_offset = torch.tensor([[0.7071, 0, 0, -0.7071]], device=self.device)

        self.gravity = torch.tensor([0.0, 0.0, 9.81], device=self.device)
        # self.gravity = torch.tensor([0.0, 0.0, 0.0], device=self.device)

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
    
    def SE3_Control(self, desired_pos, desired_yaw, 
                    com_pos, com_ori_quat, com_vel, com_omega, obs):
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


        # desired_yaw = 2 * torch.arctan2(desired_ori[:, 3], desired_ori[:, 0])
        # desired_yaw = yaw_from_quat(desired_yaw)
        # print("Desired Yaw: ", desired_yaw)
        # _, _, desired_yaw  = isaac_math_utils.euler_xyz_from_quat(desired_ori)

        # print("Quad ori: ", quad_ori_quat)  

        # quad_yaw = yaw_from_quat(quad_ori_quat)
        # com_yaw = yaw_from_quat(com_ori_quat)
        
        
        # print("Quad yaw: ", quad_yaw)
        # print("Quad yaw: ", 2 * torch.arctan2(quad_ori_quat[:, 3], quad_ori_quat[:, 0]))

        # Rotate omega into body frame from world frame
        # quad_omega = isaac_math_utils.quat_rotate(isaac_math_utils.quat_conjugate(quad_ori_quat), quad_omega)
        com_omega = isaac_math_utils.quat_rotate(isaac_math_utils.quat_conjugate(com_ori_quat), com_omega)

        # com_pos, com_ori = isaac_math_utils.combine_frame_transforms(ee_pos, ee_ori_quat, self.com_pos_ee_frame, self.com_ori_ee_frame)

        # com_pos, com_vel = get_point_state_from_ee_transform_w(ee_pos, ee_ori_quat, ee_vel, ee_omega, self.com_pos_ee_frame)
        # quad_pos_computed, quad_vel_computed = get_point_state_from_ee_transform_w(ee_pos, ee_ori_quat, ee_vel, ee_omega, self.quad_pos_ee_frame)

        # print("COM pos: ", com_pos)
        # print("EE pos: ", ee_pos)
        # print("Quad Pos (isaac) : ", quad_pos)
        # print("Quad Pos computed: ", quad_pos_computed)

        # print("COM Vel: ", com_vel)
        # print("Quad Vel (isaac) : ", quad_vel)
        # print("Quad Vel computed: ", quad_vel_computed)
        # print("Quad vel computed (refactor): ", ee_vel + torch.cross(ee_omega, self.quad_pos_ee_frame, dim=1))
        # com_pos = quad_pos
        # com_vel = quad_vel


        pos_error = com_pos - desired_pos
        vel_error = com_vel - torch.zeros_like(com_vel)

        # pos_error = quad_pos - desired_pos
        # vel_error = quad_vel - torch.zeros_like(quad_vel)

        # pos_error = quad_pos_computed - desired_pos
        # vel_error = quad_vel_computed - torch.zeros_like(quad_vel_computed)

        if self.print_debug:
            print("[SE3] Pos Error norm: ", torch.linalg.norm(pos_error,dim=1))
        # print("Vel Error: ", vel_error)

        # Compute desired Force (batch_size, 3)
        F_des = self.mass * (-self.kp_pos * pos_error + \
                             -self.kd_pos * vel_error + \
                            self.gravity.tile(com_pos.shape[0], 1)) # (N, 3)
        
        # print("F_des: ", F_des)
        
        batch_size = com_pos.shape[0]
        # quad_ori_matrix = isaac_math_utils.matrix_from_quat(quad_ori_quat) # (batch_size, 3, 3)
        quad_ori_matrix = isaac_math_utils.matrix_from_quat(com_ori_quat) # (batch_size, 3, 3)
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


        # Compute orientation error
        S_err = 0.5 * (torch.bmm(R_des.transpose(-2, -1), quad_ori_matrix) - torch.bmm(quad_ori_matrix.transpose(-2, -1), R_des)) # (batch_size, 3, 3)
        att_err = vee_map(S_err) # (batch_size, 3)
        if torch.any(torch.isnan(att_err)):
            print("Nan detected in attitude error!", att_err)
            att_err = torch.zeros_like(att_err, device=self.device)
        omega_err = quad_omega - torch.zeros_like(quad_omega, device=self.device) # Omega des is 0. (batch_size, 3)
        
        # Compute desired moments
        # M = I @ (-kp_att * att_err - kd_att * omega_err) + omega x I @ omega
        inertia = self.inertia_tensor.unsqueeze(0).tile(batch_size, 1, 1).to(self.device)
        att_pd = -self.kp_att * att_err - self.kd_att * omega_err
        I_omega = torch.bmm(inertia.view(batch_size, 3, 3), quad_omega.unsqueeze(2)).squeeze(2).to(self.device)

        M_des = torch.bmm(inertia.view(batch_size, 3, 3), att_pd.unsqueeze(2)).squeeze(2) + \
                torch.cross(quad_omega, I_omega, dim=1) 
        
        if self.control_mode == "CTBM":
            return collective_thrust, M_des

        elif self.control_mode == "CTATT":
            print("DC R des: ", R_des)
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
            com_pos = obs[:, :3]
            com_ori_quat = obs[:, 3:7]
            com_vel = obs[:, 7:10]
            com_omega = obs[:, 10:13]
            desired_pos = obs[:, 13:16]
            desired_yaw = obs[:, 16:17]
        
        

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