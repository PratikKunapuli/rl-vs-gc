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
from omni.isaac.lab.utils.math import subtract_frame_transforms, random_yaw_orientation, matrix_from_quat, matrix_from_euler
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

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
    thrust_to_weight = 1.6
    moment_scale = 0.01
    # attitude_scale = (3.14159/180.0) * 30.0
    # attitude_scale = (3.14159)/2.0
    attitude_scale = 3.14159
    attitude_scale_z = torch.pi - 1e-6
    attitude_scale_xy = 0.2

    control_mode = "CTBM"
    pd_loop_decimation = sim_rate_hz // pd_loop_rate_hz # decimation from sim physics rate

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    yaw_error_reward_scale = -1.0

    # observation modifiers
    use_yaw_representation = False
    use_full_ori_matrix = False

    eval_mode = False
    gc_mode = False
    goal_cfg="rand"
    goal_pos = [0.0, 0.0, 3.0]
    goal_ori = [1.0, 0.0, 0.0, 0.0]

    seed = 0


    # Motor dynamics
    arm_length = 0.043
    k_eta = 2.3e-8
    k_m = 7.8e-10
    tau_m = 0.005
    motor_speed_min = 0.0
    motor_speed_max = 2500.0

    kp_att = 1000.0 # 544
    kd_att = 100.0 # 46.64


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
                "pos_error",
                "yaw_error",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
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
            self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * (self._robot_weight * self.cfg.thrust_to_weight)
            self._wrench_des[:, 1:] = self.cfg.moment_scale * self._actions[:, 1:]
        elif self.cfg.control_mode == "CTATT":
            # 0th action is collective thrust
            # 1st and 2nd action are desired attitude for pitch and roll
            # 3rd action is desired yaw rate
            self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * (self._robot_weight * self.cfg.thrust_to_weight)
            
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

    def _get_observations(self) -> dict:
        pos_error_b, ori_error_b = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w, self._desired_ori_w
        )

        yaw_error = yaw_error_from_quats(self._robot.data.root_quat_w, self._desired_ori_w, 0)

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b, # 3
                self._robot.data.root_ang_vel_b, # 3
                self._robot.data.projected_gravity_b, # 3
                pos_error_b, # 3
                ori_error_b, # 4
                yaw_error.unsqueeze(-1), # 1
            ],
            dim=-1,
        )

        if self.cfg.gc_mode:
            gc_obs = torch.cat(
                [
                    self._robot.data.root_pos_w,
                    self._robot.data.root_quat_w,
                    self._robot.data.root_lin_vel_w,
                    self._robot.data.root_ang_vel_w,
                    self._desired_pos_w,
                    yaw_from_quat(self._desired_ori_w).unsqueeze(1),
                ],
                dim=-1
            )
        else:
            gc_obs = None

        full_state = torch.cat(
            [
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_w,
                self._desired_pos_w,
                self._desired_ori_w,
            ],
            dim=-1,
        )

        observations = {"policy": obs, "gc": gc_obs, "full_state": full_state}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        # distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        distance_to_goal_mapped = torch.exp(- (distance_to_goal **2) / 0.8)

        ori_error = yaw_error_from_quats(self._robot.data.root_quat_w, self._desired_ori_w, 0) # Yaw error

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "pos_error": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "yaw_error": ori_error * self.cfg.yaw_error_reward_scale * self.step_dt,
        }
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

        

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Markers",
                                        markers={
                                        "frame": sim_utils.UsdFileCfg(
                                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                                            scale=(0.05, 0.05, 0.05),
                                        ),})
                self.frame_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        # update the markers
        # Update frame positions for debug visualization
        self._frame_positions[:, 0] = self._robot.data.root_pos_w
        self._frame_positions[:, 1] = self._desired_pos_w
        # self._frame_positions[:, 2] = self._robot.data.body_pos_w[:, self._body_id].squeeze(1)
        # self._frame_positions[:, 2] = com_pos_w
        self._frame_orientations[:, 0] = self._robot.data.root_quat_w
        self._frame_orientations[:, 1] = self._desired_ori_w
        # self._frame_orientations[:, 2] = self._robot.data.body_quat_w[:, self._body_id].squeeze(1)
        # self._frame_orientations[:, 2] = com_ori_w
        self.frame_visualizer.visualize(self._frame_positions.flatten(0, 1), self._frame_orientations.flatten(0,1))
