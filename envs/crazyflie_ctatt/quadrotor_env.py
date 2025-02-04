# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, random_yaw_orientation, matrix_from_quat, matrix_from_euler, quat_rotate_inverse, quat_rotate, normalize, wrap_to_pi, quat_apply
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from utils.assets import MODELS_PATH
from configs.aerial_manip_asset import CRAZYFLIE_MANIPULATOR_0DOF_CFG, CRAZYFLIE_MANIPULATOR_0DOF_LONG_CFG
from utils.math_utilities import yaw_from_quat, yaw_error_from_quats, quat_from_yaw, compute_desired_pose_from_transform, vee_map, exp_so3, hat_map
import utils.flatness_utilities as flatness_utils

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadrotorEnvWindow(BaseEnvWindow):
    """Window manager for the Quadrotor environment."""

    def __init__(self, env: QuadrotorEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadrotorEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    action_space = 4
    observation_space = 17
    state_space = 0
    debug_vis = True
    sim_rate_hz = 1000
    policy_rate_hz = 50
    pd_loop_rate_hz = 100
    decimation = sim_rate_hz // policy_rate_hz

    num_actions = action_space
    num_observations = observation_space

    ui_window_class_type = QuadrotorEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_rate_hz,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    end_effector_mass = float(1e-9)
    thrust_to_weight = 1.8
    # thrust_to_weight = 1.9
    moment_scale = 0.01
    # attitude_scale = (3.14159/180.0) * 30.0
    # attitude_scale = (3.14159)/2.0
    attitude_scale = 3.14159
    attitude_scale_z = torch.pi - 1e-6
    attitude_scale_xy = 0.2
    has_end_effector = False

    control_mode = "CTATT" # "CTBM" or "CTATT"
    pd_loop_decimation = sim_rate_hz // pd_loop_rate_hz # decimation from sim physics rate

    # reward scales
    pos_distance_reward_scale = 15.0
    pos_radius = 0.8
    pos_radius_curriculum = 50000000
    pos_error_reward_scale= 0.0
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    yaw_error_reward_scale = -2.0
    previous_thrust_reward_scale = -0.1
    previous_attitude_reward_scale = -0.1
    action_norm_reward_scale = 0.0
    stay_alive_reward = 0.0
    crash_penalty = 0.0
    scale_reward_with_time = True

    # observation modifiers
    use_yaw_representation = False
    use_full_ori_matrix = True

    eval_mode = False
    gc_mode = False
    goal_cfg= "rand"
    goal_pos = [0.0, 0.0, 3.0]
    goal_ori = [1.0, 0.0, 0.0, 0.0]

    task_body = "body"
    goal_body = "body"
    reward_task_body = "body"
    reward_goal_body = "body"
    visualization_body= "body"


    seed = 0


    # Motor dynamics
    arm_length = 0.043
    k_eta = 2.3e-8
    k_m = 7.8e-10
    tau_m = 0.005
    motor_speed_min = 0.0
    motor_speed_max = 2500.0

    kp_att = 1575 # 544
    kd_att = 229.93 # 46.64

    # Domain Randomization
    dr_dict = {}

@configclass
class QuadrotorManipulatorEnvCfg(QuadrotorEnvCfg):
    # robot
    robot: ArticulationCfg = CRAZYFLIE_MANIPULATOR_0DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = (CRAZYFLIE_CFG.spawn.replace(usd_path=f"{MODELS_PATH}/crazyflie_manipulator.usd")).replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = (CRAZYFLIE_CFG.replace(usd_path=f"{MODELS_PATH}/crazyflie_manipulator.usd")).replace(prim_path="/World/envs/env_.*/Robot")

    has_end_effector=True
    task_body = "endeffector"
    goal_body = "endeffector"
    reward_task_body = "endeffector"
    reward_goal_body = "endeffector"
    visualization_body= "endeffector"


    dr_dict = {'thrust_to_weight':  False}

@configclass
class QuadrotorManipulatorLongEnvCfg(QuadrotorEnvCfg):
    # robot
    robot: ArticulationCfg = CRAZYFLIE_MANIPULATOR_0DOF_LONG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = (CRAZYFLIE_CFG.spawn.replace(usd_path=f"{MODELS_PATH}/crazyflie_manipulator.usd")).replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = (CRAZYFLIE_CFG.replace(usd_path=f"{MODELS_PATH}/crazyflie_manipulator.usd")).replace(prim_path="/World/envs/env_.*/Robot")

    has_end_effector=True
    task_body = "endeffector"
    goal_body = "endeffector"
    reward_task_body = "endeffector"
    reward_goal_body = "endeffector"
    visualization_body= "endeffector"


    dr_dict = {'thrust_to_weight':  False}

class QuadrotorEnv(DirectRLEnv):
    cfg: QuadrotorEnvCfg

    def __init__(self, cfg: QuadrotorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        torch.manual_seed(self.cfg.seed)

        # Total thrust and moment applied to the base of the quadrotor
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._wrench_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._thrust_to_weight = self.cfg.thrust_to_weight * torch.ones(self.num_envs, device=self.device)
        self._hover_thrust = 2.0 / self.cfg.thrust_to_weight - 1.0
        self._nominal_action = torch.tensor([self._hover_thrust, 0.0, 0.0, 0.0], device=self.device).tile((self.num_envs, 1))

        # Things necessary for motor dynamics
        r2o2 = math.sqrt(2.0) / 2.0
        self._rotor_positions = torch.cat(
            [
                self.cfg.arm_length * torch.tensor([[r2o2, r2o2, 0]]),
                self.cfg.arm_length * torch.tensor([[r2o2, -r2o2, 0]]),
                self.cfg.arm_length * torch.tensor([[-r2o2, -r2o2, 0]]),
                self.cfg.arm_length * torch.tensor([[-r2o2, r2o2, 0]]),
            ],
            dim=0, 
        ).to(self.device)
        self._rotor_directions = torch.tensor([1, -1, 1, -1], device=self.device)
        self.k = self.cfg.k_m / self.cfg.k_eta

        self.f_to_TM = torch.cat(
            [
                torch.tensor([[1, 1, 1, 1]], device=self.device),
                torch.cat(
                    [
                        torch.linalg.cross(self._rotor_positions[i], torch.tensor([0.0, 0.0, 1.0], device=self.device)).view(-1, 1)[0:2] for i in range(4)
                    ], 
                    dim=1
                ).to(self.device),
                self.k * self._rotor_directions.view(1, -1),
            ],
            dim=0
        )
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)


        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_ori_w = torch.zeros(self.num_envs, 4, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "pos_distance",
                "pos_error",
                "yaw_error",
                "previous_thrust",
                "previous_attitude",
                "action_norm",
                "crash_penalty",
                "stay_alive",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        if self.cfg.has_end_effector:
            self._ee_id = self._robot.find_bodies("endeffector")[0]
            default_masses = self._robot.root_physx_view.get_masses()[0]
            # default_masses[self._ee_id] = 1e-9 # 0 gram end effector
            default_masses[self._ee_id] = self.cfg.end_effector_mass # 5 gram end effector
            new_masses = default_masses.tile((self.num_envs, 1))
            self._robot.root_physx_view.set_masses(new_masses, torch.arange(self.num_envs))

            ee_pos = self._robot.data.body_pos_w[0, self._ee_id]
            ee_ori = self._robot.data.body_quat_w[0, self._ee_id]
            quad_pos = self._robot.data.body_pos_w[0, self._body_id]
            quad_ori = self._robot.data.body_quat_w[0, self._body_id]
            self.body_pos_ee_frame, self.body_ori_ee_frame = subtract_frame_transforms(ee_pos, ee_ori, quad_pos, quad_ori)
            # self.body_pos_ee_frame = ee_pos - quad_pos
            
            # print("ee_pos: ", ee_pos)
            # print("quad_pos: ", quad_pos)
            # print("Body pos in EE frame: ", self.body_pos_ee_frame)
            # print("Body ori in EE frame: ", self.body_ori_ee_frame)
            # import code; code.interact(local=locals())
            self.body_pos_ee_frame = self.body_pos_ee_frame.tile((self.num_envs, 1))

        ## INTRODUCED FIXES TO GET THE SRT HOVER EXAMPLE TO WORK
        else:
            quad_pos = self._robot.data.body_pos_w[0, self._body_id]
            quad_ori = self._robot.data.body_quat_w[0, self._body_id]
            self.body_pos_ee_frame, self.body_ori_ee_frame = subtract_frame_transforms(quad_pos, quad_ori, quad_pos, quad_ori)
            self.body_pos_ee_frame = self.body_pos_ee_frame.tile((self.num_envs, 1))

        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._grav_vector_unit = torch.tensor([0.0, 0.0, -1.0], device=self.device).tile((self.num_envs, 1))
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self._frame_positions = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._frame_orientations = torch.zeros(self.num_envs, 2, 4, device=self.device)

        self.inertia_tensor = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).tile(self.num_envs, 1, 1).to(self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)


        self.vehicle_mass = self._robot_mass
        self.arm_mass = 0.0
        self.quad_inertia =  self.inertia_tensor[0]
        self.arm_offset = torch.zeros(3, device=self.device)
        self.position_offset = torch.zeros(3, device=self.device)
        self.orientation_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        # print("Crazyflie mass: ", self._robot_mass)
        # print("Crazyflie inertia: ", self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).squeeze())
        # print("TM_to_f: \n", self.TM_to_f)
        # print("f_to_TM: \n", self.f_to_TM)
        # import code; code.interact(local=locals())


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_motor_speeds(self, wrench_des):
        f_des = torch.matmul(self.TM_to_f, wrench_des.t()).t()
        # print("Desired force: ", f_des[0])
        motor_speed_squared = f_des / self.cfg.k_eta
        # print("Desired motor speed squared: ", motor_speed_squared[0])
        motor_speeds_des = torch.sign(motor_speed_squared) * torch.sqrt(torch.abs(motor_speed_squared))
        # print("Desired motor speed: ", motor_speeds_des[0])
        motor_speeds_des = motor_speeds_des.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)
        # print("Clamped desired motor speed: ", motor_speeds_des[0])
        return motor_speeds_des
    
    def _get_moment_from_ctatt(self, actions):
        ori_matrix = matrix_from_quat(self._robot.data.root_quat_w)
        # old version with euler angles
        # euler_des = actions[:, 1:] * self.cfg.attitude_scale
        # ori_des_matrix = matrix_from_euler(euler_des, "XYZ")
        # print("Env R des: ", ori_des_matrix)

        # Exp Hat Map
        # ori_des_matrix = exp_so3(hat_map(actions[:, 1:] * self.cfg.attitude_scale))
        a = []
        # Flatness based control
        shape_des = flatness_utils.s2_projection(actions[:, 1]* self.cfg.attitude_scale_xy, actions[:, 2]* self.cfg.attitude_scale_xy)
        psi_des = actions[:,3] * self.cfg.attitude_scale_z
        ori_des_matrix = flatness_utils.getRotationFromShape(shape_des, psi_des)


        S_err = 0.5 * (torch.bmm(ori_des_matrix.transpose(-2, -1), ori_matrix) - torch.bmm(ori_matrix.transpose(-2, -1), ori_des_matrix)) # (n_envs, 3, 3)
        att_err = vee_map(S_err) # (n_envs, 3)
        omega_des = torch.zeros(self.num_envs, 3, device=self.device)
        # omega_des[:, 2] = self._actions[3] * self.cfg.moment_scale
        omega_err = self._robot.data.root_ang_vel_b - omega_des # (n_envs, 3)

        att_pd = -self.cfg.kp_att * att_err - self.cfg.kd_att * omega_err
        I_omega = torch.bmm(self.inertia_tensor, self._robot.data.root_ang_vel_b.unsqueeze(2)).squeeze(2).to(self.device)
        cmd_moment = torch.bmm(self.inertia_tensor, att_pd.unsqueeze(2)).squeeze(2) + \
                    torch.cross(self._robot.data.root_ang_vel_b, I_omega, dim=1) 
        return cmd_moment

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        if self.cfg.control_mode == "CTBM":
            self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * (self._robot_weight * self._thrust_to_weight)
            self._wrench_des[:, 1:] = self.cfg.moment_scale * self._actions[:, 1:]
        elif self.cfg.control_mode == "CTATT":
            # 0th action is collective thrust
            # 1st and 2nd action are desired attitude for pitch and roll
            # 3rd action is desired yaw rate
            self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * (self._robot_weight * self._thrust_to_weight)
            
            # compute wrench from desired attitude and current attitude using PD controller
            self._wrench_des[:,1:] = self._get_moment_from_ctatt(self._actions)
            
        else:
            raise NotImplementedError(f"Control mode {self.cfg.control_mode} is not implemented.")

        self._motor_speeds_des = self._compute_motor_speeds(self._wrench_des)
        self.pd_loop_counter = 0
        # print("Desired Motor Speeds: ", self._motor_speeds_des[0])
        # print("Current Motor Speeds: ", self._motor_speeds[0])

    def _apply_action(self):
        # Update PD loop at a lower rate
        if self.pd_loop_counter % self.cfg.pd_loop_decimation == 0 and self.cfg.control_mode == "CTATT":
            self._wrench_des[:,1:] = self._get_moment_from_ctatt(self._actions)
            self._motor_speeds_des = self._compute_motor_speeds(self._wrench_des)
        self.pd_loop_counter += 1
        # print("--------------------")
        # print("Input wrench: ", self._wrench_des[0])
        # print("Motor speed des: ", self._motor_speeds_des[0])
        # print("Current motor speed (pre update): ", self._motor_speeds[0])
        motor_accel = (1/self.cfg.tau_m) * (self._motor_speeds_des - self._motor_speeds)
        self._motor_speeds += motor_accel * self.physics_dt
        self._motor_speeds = self._motor_speeds.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max) # Motor saturation
        # self._motor_speeds = self._motor_speeds_des # assume no delay to simplify the simulation
        motor_forces = self.cfg.k_eta * self._motor_speeds ** 2
        wrench = torch.matmul(self.f_to_TM, motor_forces.t()).t()
        
        # print("Motor acceleration: ", motor_accel[0])
        # print("Current motor speed (post update): ", self._motor_speeds[0])
        # print("Wrench resconstruction error: ", torch.norm(wrench[0] - self._wrench_des[0]))
        # print("Output wrench: ", wrench[0])
        self._thrust[:, 0, 2] = wrench[:, 0]
        self._moment[:, 0, :] = wrench[:, 1:]
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _apply_curriculum(self, total_timesteps):
        """
        Apply the curriculum to the environment.
        """
        # print("[Isaac Env: Curriculum] Total Timesteps: ", total_timesteps, " Pos Radius: ", self.cfg.pos_radius)
        if self.cfg.pos_radius_curriculum > 0:
            # half the pos radius every pos_radius_curriculum timesteps
            self.cfg.pos_radius = 0.8 * (0.25 ** (total_timesteps // self.cfg.pos_radius_curriculum))
            # self.cfg.pos_radius = 0.8 * (0.5 ** (total_timesteps // self.cfg.pos_radius_curriculum))

    def _get_observations(self) -> dict:
        self._apply_curriculum(self.common_step_counter * self.num_envs)

        pos_w, ori_w, lin_vel_w, ang_vel_w = self.get_body_state_by_name(self.cfg.task_body)
        goal_pos_w, goal_ori_w = self.get_goal_state_from_task(self.cfg.task_body)

        # print("Num envs: ", self.num_envs)
        # print("Pos: ", pos_w.shape)
        # print("Ori: ", ori_w.shape)
        # print("Lin vel: ", lin_vel_w.shape)
        # print("Ang vel: ", ang_vel_w.shape)
        

        pos_error_b, ori_error_b = subtract_frame_transforms(
            pos_w, ori_w, goal_pos_w, goal_ori_w
        )

        if self.cfg.use_full_ori_matrix:
            ori_error_b = matrix_from_quat(ori_error_b).view(-1, 9)

        yaw_error = yaw_error_from_quats(self._robot.data.root_quat_w, goal_ori_w, 0)

        lin_vel_b = quat_rotate_inverse(ori_w, lin_vel_w)
        ang_vel_b = quat_rotate_inverse(ori_w, ang_vel_w)
        grav_vector_b = quat_rotate_inverse(ori_w, self._grav_vector_unit)

        obs = torch.cat(
            [
                lin_vel_b, # 3
                ang_vel_b, # 3
                grav_vector_b, # 3
                pos_error_b, # 3
                ori_error_b, # 4 or 9 if use_full_ori_matrix
                yaw_error.unsqueeze(-1), # 1
                self._previous_action, # 4
            ],
            dim=-1,
        )

        if self.cfg.gc_mode:
            pos_w, ori_w, lin_vel_w, ang_vel_w = self.get_body_state_by_name(self.cfg.task_body)
            gc_obs = torch.cat(
                [
                    pos_w,
                    ori_w,
                    lin_vel_w,
                    ang_vel_w,
                    goal_pos_w,
                    yaw_from_quat(goal_ori_w).unsqueeze(1),
                ],
                dim=-1
            )
        else:
            gc_obs = None

        if self.cfg.eval_mode:
            quad_pos_w, quad_ori_w, quad_lin_vel_w, quad_ang_vel_w = self.get_body_state_by_name("body")
            ## added in to allow for SRT hover task
            if self.cfg.has_end_effector:
                ee_pos_w, ee_ori_w, ee_lin_vel_w, ee_ang_vel_w = self.get_body_state_by_name("endeffector")
            else:
                ee_pos_w, ee_ori_w, ee_lin_vel_w, ee_ang_vel_w = torch.tensor([], device=self.device), torch.tensor([], device=self.device), torch.tensor([], device=self.device), torch.tensor([], device=self.device)
            full_state = torch.cat(
                [
                    quad_pos_w,                                 # (num_envs, 3) [0-3]
                    quad_ori_w,                                 # (num_envs, 4) [3-7]
                    quad_lin_vel_w,                             # (num_envs, 3) [7-10]
                    quad_ang_vel_w,                             # (num_envs, 3) [10-13]
                    ee_pos_w,                                   # (num_envs, 3) [13-16]
                    ee_ori_w,                                   # (num_envs, 4) [16-20]
                    ee_lin_vel_w,                               # (num_envs, 3) [20-23]
                    ee_ang_vel_w,                               # (num_envs, 3) [23-26]
                    self._desired_pos_w,                        # (num_envs, 3) [26-29]
                    self._desired_ori_w,                        # (num_envs, 4) [29-33]
                ],
                dim=-1,
            )
        else:
            full_state = None

        observations = {"policy": obs, "gc": gc_obs, "full_state": full_state}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        pos_w, ori_w, lin_vel_w, ang_vel_w = self.get_body_state_by_name(self.cfg.reward_task_body)
        lin_vel_b = quat_rotate_inverse(ori_w, lin_vel_w)
        ang_vel_b = quat_rotate_inverse(ori_w, ang_vel_w)


        ## implementing LQR style rewards
        pos_penalty = [-5000.] * 3
        ori_penalty = [-1.] * 4
        lin_vel_penalty = [-1.] * 3
        ang_vel_penalty = [-0.1] * 3
        # pen_array = pos_penalty + ori_penalty + lin_vel_penalty + ang_vel_penalty
        # Q = torch.diag(torch.tensor(pen_array))
        # Q = Q.to(device=self.device)

        pos_penalty = torch.diag(torch.tensor(pos_penalty)) / 1000.
        pos_penalty = pos_penalty.to(device=self.device)
        ori_penalty = torch.diag(torch.tensor(ori_penalty)) / 1000.
        ori_penalty = ori_penalty.to(device=self.device)
        lin_vel_penalty = torch.diag(torch.tensor(lin_vel_penalty)) / 1000.
        lin_vel_penalty = lin_vel_penalty.to(device=self.device)
        ang_vel_penalty = torch.diag(torch.tensor(ang_vel_penalty)) / 1000.
        ang_vel_penalty = ang_vel_penalty.to(device=self.device)

        # full_world_state = torch.hstack((pos_w, ori_w, lin_vel_w, ang_vel_w))
        goal_pos, goal_ori = self.get_goal_state_from_task("body")
        n = goal_pos.shape[0]
        pos_penalty = (goal_pos - pos_w) @ pos_penalty @ (goal_pos - pos_w).T
        pos_penalty = torch.diag(pos_penalty)
        ori_penalty = (goal_ori - ori_w) @ ori_penalty @ (goal_ori - ori_w).T
        ori_penalty = torch.diag(ori_penalty)
        lin_vel_penalty = lin_vel_w @ lin_vel_penalty @ lin_vel_w.T
        lin_vel_penalty = torch.diag(lin_vel_penalty)
        ang_vel_penalty = ang_vel_w @ ang_vel_penalty @ ang_vel_w.T
        ang_vel_penalty = torch.diag(ang_vel_penalty)
        
        # goal_state = torch.hstack((goal_pos, goal_ori, torch.zeros((n, 3), device=self.device), torch.zeros((n, 3), device=self.device)))
        # state_penalty = (full_world_state - goal_state) @ Q @ (full_world_state - goal_state).T
        # state_penalty = torch.diag(state_penalty)
        # state_penalty = torch.sum(state_penalty)

        R = torch.diag(torch.tensor([-1, -0.1, -0.1, -0.1])) / 1000.
        R = R.to(device=self.device)
        action_penalty = (self._actions - self._nominal_action) @ R @ (self._actions - self._nominal_action).T
        action_penalty = torch.diag(action_penalty)
        # action_penalty = torch.sum(action_penalty)

                               


        lin_vel = torch.sum(torch.square(lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - pos_w, dim=1)
        # distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        # distance_to_goal_mapped = torch.exp(- (distance_to_goal**2) / 0.8)
        distance_to_goal_mapped = torch.exp(- (distance_to_goal) / self.cfg.pos_radius)

        ori_error = yaw_error_from_quats(ori_w, self._desired_ori_w, 0) # Yaw error'

        # action_error = torch.sum(torch.square(self._actions - self._previous_action), dim=1)
        action_thrust_error = torch.square(self._actions[:, 0] - self._previous_action[:, 0])
        action_att_error = torch.sum(torch.square(self._actions[:, 1:] - self._previous_action[:, 1:]), dim=1)
        action_norm_error = torch.sum(torch.square(self._actions - self._nominal_action), dim=1)

        self._previous_action = self._actions.clone()
        crash_penalty_time = self.cfg.crash_penalty * (self.max_episode_length - self.episode_length_buf)

        if self.cfg.scale_reward_with_time:
            time_scale = self.step_dt
        else:
            time_scale = 1.0

        # rewards = {
        #     "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * time_scale,
        #     "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * time_scale,
        #     "pos_distance": distance_to_goal_mapped * self.cfg.pos_distance_reward_scale * time_scale,
        #     "pos_error": distance_to_goal * self.cfg.pos_error_reward_scale * time_scale,
        #     "yaw_error": ori_error * self.cfg.yaw_error_reward_scale * time_scale,
        #     "previous_thrust": action_thrust_error * self.cfg.previous_thrust_reward_scale * time_scale,
        #     "previous_attitude": action_att_error * self.cfg.previous_attitude_reward_scale * time_scale,
        #     "action_norm": action_norm_error * self.cfg.action_norm_reward_scale * time_scale,
        #     "crash_penalty": self.reset_terminated[:].float() * crash_penalty_time * time_scale,
        #     "stay_alive": torch.ones_like(distance_to_goal) * self.cfg.stay_alive_reward * time_scale,
        # }
        # for key, value in rewards.items():
        #     print(key, value.shape)

        rewards = {
            "lin_vel": lin_vel_penalty,
            "ang_vel": ang_vel_penalty,
            "pos_distance": distance_to_goal_mapped * self.cfg.pos_distance_reward_scale * time_scale * 0,
            "pos_error": pos_penalty,
            "yaw_error": ori_penalty,
            "previous_thrust": action_thrust_error * self.cfg.previous_thrust_reward_scale * time_scale * 0,
            "previous_attitude": action_att_error * self.cfg.previous_attitude_reward_scale * time_scale * 0,
            "action_norm": action_penalty,
            "crash_penalty": self.reset_terminated[:].float() * crash_penalty_time * time_scale,
            "stay_alive": torch.ones_like(distance_to_goal) * self.cfg.stay_alive_reward * time_scale,
        }
        # for key, value in rewards.items():
        #     print(key, value.shape)

        ## names have to match above
        # rewards = {
        #     "lin_vel" : state_penalty,
        #     "action_norm" : action_penalty
        # }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # print("[Isaac] pos error: ", distance_to_goal)
        # print("[Isaac] pos error reward: ", rewards["pos_error"])
        # print("[Isaac] yaw error: ", rewards["yaw_error"])
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 5.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        final_yaw_error = yaw_error_from_quats(
            self._robot.data.root_quat_w[env_ids], self._desired_ori_w[env_ids], 0
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Metrics/final_yaw_error_to_goal"] = final_yaw_error.item()
        extras["Metrics/pos_radius"] = self.cfg.pos_radius
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and not self.cfg.eval_mode:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        elif self.cfg.eval_mode:
            self.episode_length_buf[env_ids] = 0

        self._actions[env_ids] = 0.0
        # Sample new commands
        if self.cfg.goal_cfg == "rand":
            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(2.0, 4.0)
            self._desired_ori_w[env_ids] = random_yaw_orientation(len(env_ids), device=self.device) 
        elif self.cfg.goal_cfg == "fixed":
            self._desired_pos_w[env_ids] = torch.tensor(self.cfg.goal_pos, device=self.device).tile((env_ids.size(0), 1))
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_ori_w[env_ids] = torch.tensor(self.cfg.goal_ori, device=self.device).tile((env_ids.size(0), 1))

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, 2] = 3.0 # start at 3m height
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._motor_speeds[env_ids] = 1788.53 * torch.ones_like(self._motor_speeds[env_ids])
        
        # Domain Randomization
        self.domain_randomization(env_ids)

    def domain_randomization(self, env_ids: torch.Tensor | None):
        if env_ids is None or env_ids.shape[0] == 0:
            return
        
        # randomize thrust to weight ratio
        if self.cfg.dr_dict.get("thrust_to_weight", False):
            self._thrust_to_weight[env_ids] = torch.zeros_like(self._thrust_to_weight[env_ids]).normal_(mean=0.0, std=0.4) + self.cfg.thrust_to_weight

    def get_body_state_by_name(self, body_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if body_name == "body":
            pos = self._robot.data.body_pos_w[:, self._body_id].squeeze(1)
            ori = self._robot.data.body_quat_w[:, self._body_id].squeeze(1)
            vel = self._robot.data.body_lin_vel_w[:, self._body_id].squeeze(1)
            ang_vel = self._robot.data.body_ang_vel_w[:, self._body_id].squeeze(1)
        elif body_name == "root":
            pos = self._robot.data.root_pos_w
            ori = self._robot.data.root_quat_w
            vel = self._robot.data.root_lin_vel_w
            ang_vel = self._robot.data.root_ang_vel_w
        elif body_name == "endeffector":
            pos = self._robot.data.body_pos_w[:, self._ee_id].squeeze(1)
            ori = self._robot.data.body_quat_w[:, self._ee_id].squeeze(1)
            vel = self._robot.data.body_lin_vel_w[:, self._ee_id].squeeze(1)
            ang_vel = self._robot.data.body_ang_vel_w[:, self._ee_id].squeeze(1)
        else:
            raise NotImplementedError(f"Body name {body_name} is not implemented.")
        return pos, ori, vel, ang_vel
    
    def compute_desired_pose_from_transform(self, goal_pos_w, goal_ori_w, pos_transform):
        # Find b2 in the ori frame, set z component to 0 and the desired yaw is the atan2 of the x and y components
        b2 = quat_rotate(goal_ori_w, torch.tensor([[0.0, 1.0, 0.0]], device=goal_ori_w.device).tile(goal_ori_w.shape[0], 1))
        if self.cfg.num_joints == 0:
            b2[:, 2] = 0.0
        b2 = normalize(b2)
        
        # Yaw is the angle between b2 and the y-axis
        yaw_desired = torch.atan2(b2[:, 1], b2[:, 0]) - torch.pi/2
        yaw_desired = wrap_to_pi(yaw_desired)

        # Position desired is the pos_transform along -b2 direction
        pos_desired = goal_pos_w + torch.bmm(torch.linalg.norm(pos_transform, dim=1).tile(self.num_envs, 1, 1), -1*b2.unsqueeze(1)).squeeze(1)

        
        return pos_desired, yaw_desired

    def get_goal_state_from_task(self, goal_body:str) -> tuple[torch.Tensor, torch.Tensor]:
        if goal_body == "root":
            goal_pos_w = self._desired_pos_w
            goal_ori_w = self._desired_ori_w
        elif goal_body == "endeffector":
            goal_pos_w = self._desired_pos_w
            goal_ori_w = self._desired_ori_w
        elif goal_body == "COM":
            # desired_pos, desired_yaw = self.compute_desired_pose_from_transform(self._desired_pos_w, self._desired_ori_w, self.com_pos_e)
            desired_pos, desired_yaw = compute_desired_pose_from_transform(self._desired_pos_w, self._desired_ori_w, self.com_pos_e, 0)
            goal_pos_w = desired_pos
            goal_ori_w = quat_from_yaw(desired_yaw)
        elif goal_body == "body":
            # desired_pos, desired_yaw = self.compute_desired_pose_from_transform(self._desired_pos_w, self._desired_ori_w, self.com_pos_e)
            # desired_pos, desired_yaw = compute_desired_pose_from_transform(self._desired_pos_w, self._desired_ori_w, self.body_pos_ee_frame, 0)
            # desired_pos_w is the ee in world frame, we want the corresponding body pos in world frame
            desired_pos = self._desired_pos_w + quat_apply(self._desired_ori_w, self.body_pos_ee_frame)
            # desired_pos = self._desired_pos_w + self.body_pos_ee_frame
            # desired_pos = self._desired_pos_w
            goal_pos_w = desired_pos
            goal_ori_w = self._desired_ori_w
        else:
            raise ValueError("Invalid goal body: ", self.cfg.goal_body)

        return goal_pos_w, goal_ori_w

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Markers",
                                        markers={
                                        "frame": sim_utils.UsdFileCfg(
                                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                                            scale=(0.03, 0.03, 0.03),
                                        ),})
                self.frame_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        pos, ori, _, _ = self.get_body_state_by_name(self.cfg.visualization_body)
        # body_pos, body_ori, _, _ = self.get_body_state_by_name("body")
        # ee_pos, ee_ori, _, _ = self.get_body_state_by_name("endeffector")

        # Update frame positions for debug visualization
        self._frame_positions[:, 0] = pos
        # self._frame_positions[:, 0] = body_pos
        self._frame_positions[:, 1] = self._desired_pos_w
        # self._frame_positions[:, 2] = ee_pos
        # self._frame_positions[:, 2] = self._robot.data.body_pos_w[:, self._body_id].squeeze(1)
        # self._frame_positions[:, 2] = com_pos_w
        self._frame_orientations[:, 0] = ori
        # self._frame_orientations[:, 0] = body_ori
        self._frame_orientations[:, 1] = self._desired_ori_w
        # self._frame_orientations[:, 2] = ee_ori
        # self._frame_orientations[:, 2] = self._robot.data.body_quat_w[:, self._body_id].squeeze(1)
        # self._frame_orientations[:, 2] = com_ori_w
        self.frame_visualizer.visualize(self._frame_positions.flatten(0, 1), self._frame_orientations.flatten(0,1))
