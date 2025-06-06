from __future__ import annotations

import torch

# Isaac SDK imports
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab.utils.math as isaac_math_utils
from omni.isaac.lab_assets import CRAZYFLIE_CFG
from omni.isaac.lab.sim.spawners.shapes import SphereCfg, spawn_sphere
from omni.isaac.lab.sim.spawners.materials import VisualMaterialCfg, PreviewSurfaceCfg, spawn_preview_surface
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Usd, UsdShade, Gf


# Python imports
import gymnasium as gym
import numpy as np
import math

# Local imports
from configs.aerial_manip_asset import AERIAL_MANIPULATOR_0DOF_CFG, AERIAL_MANIPULATOR_QUAD_ONLY_CFG, CRAZYFLIE_BRUSHLESS_CFG
import utils.trajectory_utilities as traj_utils
import utils.math_utilities as math_utils
import utils.flatness_utilities as flatness_utils




# Visualization Class
class AerialManipulatorTrajectoryTrackingEnvWindow(BaseEnvWindow):
    """4Window manager for the Quadcopter environment."""

    def __init__(self, env: AerialManipulatorTrajectoryTrackingEnv, window_name: str = "Aerial Manipulator Trajectory Tracking - IsaacLab"):
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


""" 
Base configuration class for the Trajectory Tracking Environment
This class contains the common configuration parameters for the environment.
It is inherited by the specific configuration classes for different types of aerial vehicles.
"""
@configclass
class AerialManipulatorTrajectoryTrackingEnvBaseCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    sim_rate_hz = 100
    policy_rate_hz = 50
    decimation = sim_rate_hz // policy_rate_hz
    ui_window_class_type = AerialManipulatorTrajectoryTrackingEnvWindow
    num_states = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_rate_hz,
        render_interval=1,
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
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.2,
        ),
        debug_vis=False,
    )

    action_space= gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    observation_space= gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,))
    state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)


    # ------- Trajectory --------
    traj_update_dt = 0.02

    trajectory_type = "lissaajous"
    trajectory_horizon = 10

    lissajous_amplitudes = [0, 0, 0, 0]
    lissajous_amplitudes_rand_ranges = [0.0, 0.0, 0.0, 0.0]
    lissajous_frequencies = [0, 0, 0, 0]
    lissajous_frequencies_rand_ranges = [0.0, 0.0, 0.0, 0.0]
    lissajous_phases = [0, 0, 0, 0]
    lissajous_phases_rand_ranges = [0.0, 0.0, 0.0, 0.0]
    lissajous_offsets = [0, 0, 3.0, 0]
    lissajous_offsets_rand_ranges = [0.0, 0.0, 0.0, 0.0]

    # -------- Control --------
    control_mode: str = "CTBM"  # will be overridden by Crazyflie child cfg
    use_motor_dynamics: bool = False  # toggled by child cfg
    pd_loop_rate_hz: int = 100 # Used when CTBR or CTATT modes are selected for control 
    pd_loop_decimation = sim_rate_hz // pd_loop_rate_hz # decimation from sim physics rate
    control_latency_steps = 0 # number of steps to delay the control input (for testing purposes)

    # -------- Scales / gains (defaults – child cfgs override) --------
    attitude_scale_xy: float = 0.3
    attitude_scale_z: float = 3.14159
    body_rate_scale_xy: float = 10.0
    body_rate_scale_z: float = 2.5

    kp_att: float = 1575.0
    kd_att: float = 229.93
    kp_omega: float = 1.0
    kd_omega: float = 0.1

    moment_scale_xy: float = 0.5
    moment_scale_z: float = 0.025
    thrust_to_weight: float = 3.0



    # reward scales
    pos_radius = 0.8
    pos_radius_curriculum = 50000000 # 10e6
    lin_vel_reward_scale = -0.05 # -0.05 
    ang_vel_reward_scale = -0.01 # -0.01
    pos_distance_reward_scale = 15.0 #15.0
    pos_error_reward_scale = 0.0# -1.0
    ori_error_reward_scale = 0.0 # -0.5
    joint_vel_reward_scale = 0.0 # -0.01
    action_norm_reward_scale = 0.0 # -0.01
    previous_action_norm_reward_scale = 0.0 # -0.01
    yaw_error_reward_scale = -2.0 # -0.01
    yaw_distance_reward_scale = 0.0 # -0.01
    yaw_radius = 0.2 
    yaw_smooth_transition_scale = 0.0
    stay_alive_reward = 0.0
    crash_penalty = 0.0

    scale_reward_with_time = False
    square_reward_errors = False
    square_pos_error = True
    square_pos_error = True
    penalize_action = False
    penalize_previous_action = False


    # ------- Initialization --------
    goal_cfg = "rand" # "rand", "fixed", or "initial"
    init_cfg = "rand" # "default" or "rand"
    goal_pos = None
    goal_vel = None
    init_pos_ranges=[0.0, 0.0, 0.0]
    init_lin_vel_ranges=[0.0, 0.0, 0.0]
    init_yaw_ranges=[0.0]
    init_ang_vel_ranges=[0.0, 0.0, 0.0]
    goal_pos_range = 2.0
    goal_yaw_range = 3.14159


    task_body = "root" # "root" or "endeffector" or "vehicle" or "COM"
    goal_body = "root" # "root" or "endeffector" or "vehicle" or "COM"
    reward_task_body = "root"
    reward_goal_body = "root"    
    body_name = "vehicle"
    has_end_effector = False
    use_grav_vector = True
    use_full_ori_matrix = True
    use_yaw_representation = False
    use_previous_actions = False
    use_yaw_representation_for_trajectory=True
    use_ang_vel_from_trajectory=True

    eval_mode = False
    gc_mode = False
    viz_mode = "triad" # or robot
    viz_history_length = 100
    robot_color=[0.0, 0.0, 0.0]
    viz_ref_offset=[0.0,0.0,0.0]

    mass=0.8
    # Domain Randomization
    dr_dict = {
        'thrust_to_weight':  0.0,
        'mass': 0.0,
        'inertia': 0.0,
        'arm_length': 0.0,
        'k_eta': 0.0,
        'k_m': 0.0,
        'tau_m': 0.0,
        'kp_att': 0.0,
        'kd_att': 0.0,
    }

"""
0-DOF Aerial Manipulator Vehicle Trajectory Tracking Environment Configuration

This vehicle represents a 0-DOF aerial manipulator, which is a quadrotor with a fixed end-effector.
The trajectory tracking is of the end-effector pose. 
"""
@configclass
class AerialManipulator0DOFTrajectoryTrackingEnvCfg(AerialManipulatorTrajectoryTrackingEnvBaseCfg):
    # env
    num_actions = 4
    num_joints = 0
    num_observations = 91 
    has_end_effector = True
    
    # Robot-specific Params
    mass = 0.8
    thrust_to_weight = 3.0
    dr_dict = {
        'thrust_to_weight':  0.0,
        'mass': 0.0,
        'inertia': 0.0,
        'k_eta': 0.0,
        'k_m': 0.0,
        'tau_m': 0.0,
        'kp_att': 0.0,
        'kd_att': 0.0,
    }
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_0DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

"""
Quadrotor Trajectory Tracking Environment Configuration

This vehicle represents a quadrotor, capable of agile trajectory tracking. 
This model is the same as the 0-DOF aerial manipulator, but without the end-effector.
The trajectory tracking is of the vehicle pose.
"""
@configclass
class QuadrotorTrajectoryTrackingEnvCfg(AerialManipulatorTrajectoryTrackingEnvBaseCfg):
    # env
    num_actions = 4
    num_joints = 0
    num_observations = 91 
    has_end_effector = False
    
    # Robot-specific Params
    mass = 0.8
    thrust_to_weight = 3.0
    dr_dict = {
        'thrust_to_weight':  0.0,
        'mass': 0.0,
        'inertia': 0.0,
        'k_eta': 0.0,
        'k_m': 0.0,
        'tau_m': 0.0,
        'kp_att': 0.0,
        'kd_att': 0.0,
    }
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_QUAD_ONLY_CFG.replace(prim_path="/World/envs/env_.*/Robot")

"""
Crazyflie Trajectory Tracking Environment Configuration

This vehicle represents a realistic Crazyflie quadrotor model, capable of agile trajectory tracking.
The model specifically uses the Brushless Crazyflie 2.1 configuration, as well as preliminary modeled numbers of the motor dynamics and low-level control.
The trajectory tracking is of the vehicle pose.
"""
@configclass
class BrushlessCrazyflieTrajectoryTrackingEnvCfg(AerialManipulatorTrajectoryTrackingEnvBaseCfg):
    # env
    num_actions = 4
    num_joints = 0
    num_observations = 91 # TODO: Need to update this..

    sim_rate_hz = 1000
    decimation = 10 # 10x decimation from sim physics rate
    pd_loop_rate_hz = 500
    pd_loop_decimation = sim_rate_hz // pd_loop_rate_hz # decimation from sim physics rate

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

    # control rates
    sim_rate_hz = 1000
    policy_rate_hz = 50
    pd_loop_rate_hz = 100
    decimation = sim_rate_hz // policy_rate_hz
    pd_loop_decimation = sim_rate_hz // pd_loop_rate_hz

    # control
    control_mode = "CTBR"
    use_motor_dynamics = True

    # Robot params
    mass = 0.039
    Ixx = 3e-5
    Iyy = 3e-5
    Izz = 3.5e-5

    # Motor dynamics
    arm_length = 0.05
    k_eta = 4.81e-8 # Measured from thrust stand data
    k_m = 7.8e-10 #unchanged
    tau_m = 0.017 # slower motor dynamics
    motor_speed_min = 0.0
    motor_speed_max = 2500.0
    init_motor_speed = 1000.0

    kp_att = 3264.54 # 544
    kd_att = 361.58 # 46.64

    # CTBR Parameters
    kp_omega = 5.27 # measured on static test stand
    kd_omega = 1.0
    body_rate_scale_xy = 10.0
    body_rate_scale_z = 2.5

    # CTATT Parameters
    attitude_scale = torch.pi/6 # 30 degrees

    task_body = "body" 
    goal_body = "body" 
    reward_task_body = "body"
    reward_goal_body = "body"    
    body_name = "body"
    has_end_effector = False

    
    # robot
    robot: ArticulationCfg = CRAZYFLIE_BRUSHLESS_CFG.replace(prim_path="/World/envs/env_.*/Robot")



class AerialManipulatorTrajectoryTrackingEnv(DirectRLEnv):
    cfg: AerialManipulatorTrajectoryTrackingEnvBaseCfg

    def __init__(self, cfg: AerialManipulatorTrajectoryTrackingEnvBaseCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_space= gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.num_actions,))
        self.observation_space= gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.num_observations,))
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,))

        # Actions / Actuation interfaces
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._joint_torques = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._wrench_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust_to_weight = self.cfg.thrust_to_weight * torch.ones(self.num_envs, device=self.device)
        self._motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds_des = torch.zeros(self.num_envs, 4, device=self.device)
        self.min_thrust = torch.zeros(self.num_envs, device=self.device)
        self.max_thrust = torch.ones(self.num_envs, device=self.device) * (self.cfg.thrust_to_weight)
        self._action_queue = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).tile((self.cfg.control_latency_steps+1, self.num_envs, 1)) # for control latency


        # Parameters for potential Domain Randomization
        self._thrust_to_weight = self.cfg.thrust_to_weight * torch.ones(self.num_envs, device=self.device)
        if self.cfg.use_motor_dynamics:
            self._tau_m = self.cfg.tau_m * torch.ones(self.num_envs, device=self.device)
            self._arm_length = self.cfg.arm_length * torch.ones(self.num_envs, device=self.device)
            self._k_m = self.cfg.k_m * torch.ones(self.num_envs, device=self.device)
            self._k_eta = self.cfg.k_eta * torch.ones(self.num_envs, device=self.device)
            self._kp_att = self.cfg.kp_att * torch.ones(self.num_envs, device=self.device)
            self._kd_att = self.cfg.kd_att * torch.ones(self.num_envs, device=self.device)

        # Robot data
        action_history_length = 1 if "action_history_length" not in self.cfg.to_dict().keys() else self.cfg.action_history_length
        state_history_length = 1 if "state_history_length" not in self.cfg.to_dict().keys() else self.cfg.state_history_length
        self._action_history = torch.zeros(self.num_envs, action_history_length, self.cfg.num_actions, device=self.device)
        self._state_history = torch.zeros(self.num_envs, state_history_length, 3, device=self.device)

        # Goal State   
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_ori_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._desired_pos_traj_w = torch.zeros(self.num_envs, 1+self.cfg.trajectory_horizon, 3, device=self.device)
        self._desired_ori_traj_w = torch.zeros(self.num_envs, 1+self.cfg.trajectory_horizon, 4, device=self.device)
        self._pos_traj = torch.zeros(5, self.num_envs, 1+self.cfg.trajectory_horizon, 3, device=self.device)
        self._yaw_traj = torch.zeros(5, self.num_envs, 1+self.cfg.trajectory_horizon, device=self.device)
        
        # Initialization for Lisssajous and Polynomial Trajectories
        self.lissajous_amplitudes = torch.tensor(self.cfg.lissajous_amplitudes, device=self.device).tile((self.num_envs, 1)).float()
        self.lissajous_amplitudes_rand_ranges = torch.tensor(self.cfg.lissajous_amplitudes_rand_ranges, device=self.device).float()
        self.lissajous_frequencies = torch.tensor(self.cfg.lissajous_frequencies, device=self.device).tile((self.num_envs, 1)).float()
        self.lissajous_frequencies_rand_ranges = torch.tensor(self.cfg.lissajous_frequencies_rand_ranges, device=self.device).float()
        self.lissajous_phases = torch.tensor(self.cfg.lissajous_phases, device=self.device).tile((self.num_envs, 1)).float()
        self.lissajous_phases_rand_ranges = torch.tensor(self.cfg.lissajous_phases_rand_ranges, device=self.device).float()
        self.lissajous_offsets = torch.tensor(self.cfg.lissajous_offsets, device=self.device).tile((self.num_envs, 1)).float()
        self.lissajous_offsets_rand_ranges = torch.tensor(self.cfg.lissajous_offsets_rand_ranges, device=self.device).float()

        # max_coefficients = max(len(self.cfg.polynomial_x_coefficients), len(self.cfg.polynomial_y_coefficients), len(self.cfg.polynomial_z_coefficients), len(self.cfg.polynomial_yaw_coefficients))
        # self.polynomial_coefficients = torch.zeros(self.num_envs, 4, max_coefficients, device=self.device)
        # self.polynomial_coefficients[:, 0, :len(self.cfg.polynomial_x_coefficients)] = torch.tensor(self.cfg.polynomial_x_coefficients, device=self.device).tile((self.num_envs, 1))
        # self.polynomial_coefficients[:, 1, :len(self.cfg.polynomial_y_coefficients)] = torch.tensor(self.cfg.polynomial_y_coefficients, device=self.device).tile((self.num_envs, 1))
        # self.polynomial_coefficients[:, 2, :len(self.cfg.polynomial_z_coefficients)] = torch.tensor(self.cfg.polynomial_z_coefficients, device=self.device).tile((self.num_envs, 1))
        # self.polynomial_coefficients[:, 3, :len(self.cfg.polynomial_yaw_coefficients)] = torch.tensor(self.cfg.polynomial_yaw_coefficients, device=self.device).tile((self.num_envs, 1))
        # self.polynomial_yaw_rand_ranges = torch.tensor(self.cfg.polynomial_yaw_rand_ranges, device=self.device).float()

        # Time(needed for trajectory tracking)
        self._time = torch.zeros(self.num_envs, 1, device=self.device)

        # Motor Dynamics initialization
        if self.cfg.use_motor_dynamics:
            r2o2 = math.sqrt(2.0) / 2.0
            self._rotor_positions = torch.cat(
                [
                    (self._arm_length.unsqueeze(1) * torch.tensor([r2o2, r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                    (self._arm_length.unsqueeze(1) * torch.tensor([r2o2, -r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                    (self._arm_length.unsqueeze(1) * torch.tensor([-r2o2, -r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                    (self._arm_length.unsqueeze(1) * torch.tensor([-r2o2, r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                ],
                dim=1, 
            ).to(self.device)
            self._rotor_directions = torch.tensor([1, -1, 1, -1], device=self.device).tile(self.num_envs, 1)
            self.k = self._k_m / self._k_eta
            self.f_to_TM = torch.cat(
                [
                    torch.ones(self.num_envs, 1, 4, device=self.device),
                    torch.linalg.cross(self._rotor_positions, torch.tensor([0.0, 0.0, 1.0], device=self.device).tile(self.num_envs, 1, 1))[:,:, 0:2].transpose(-2,-1),
                    self.k.view(self.num_envs, 1, 1) * self._rotor_directions.view(self.num_envs, 1, 4),
                ],
                dim=1
            )
            self.TM_to_f = torch.linalg.inv(self.f_to_TM)
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "pos_error",
                "pos_distance",
                "ori_error",
                "yaw_error",
                "yaw_distance",
                "action_norm",
                "previous_action_norm",
                "stay_alive",
                "crash_penalty"
            ]
        }

        self._episode_error_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "pos_error",
                "pos_distance",
                "ori_error",
                "yaw_error",
                "yaw_distance",
                "lin_vel",
                "ang_vel",
                "action_norm",
                "previous_action_norm",
                "stay_alive",
                "crash_penalty"
            ]
        }


        # Robot specific data
        self._body_id = self._robot.find_bodies(self.cfg.body_name)[0]
        self._com_id = self._robot.find_bodies("COM")[0]

        assert len(self._body_id) == 1, "There should be only one body with the name \'vehicle\' or \'body\'"

        if self.cfg.has_end_effector:
            self._ee_id = self._robot.find_bodies("endeffector")[0] # also the root of the system

        
        if self.cfg.num_joints > 0:
            self._shoulder_joint_idx = self._robot.find_joints(".*joint1")[0][0]
        if self.cfg.num_joints > 1:
            self._wrist_joint_idx = self._robot.find_joints(".*joint2")[0][0]
        self._total_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self.total_mass = self._total_mass
        self.quad_inertia = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).squeeze()
        self.default_inertia = self.quad_inertia.clone()

        if "Ixx" in self.cfg.to_dict().keys():
            self.inertia_tensor = torch.diag(
                torch.tensor(
                    [
                        self.cfg.Ixx,
                        self.cfg.Iyy,
                        self.cfg.Izz,
                    ],
                    device=self.device,
                )
            ).unsqueeze(0).tile(self.num_envs, 1, 1)
            self.default_inertia = self.inertia_tensor[0].clone().to(self.device)
        else:
            self.inertia_tensor = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).tile(self.num_envs, 1, 1).to(self.device)
            self.default_inertia = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].to(self.device)
        self._robot_inertia = self.inertia_tensor.clone().to(self.device)

        if self.cfg.has_end_effector:
            self.arm_offset = self._robot.root_physx_view.get_link_transforms()[0, self._body_id,:3].squeeze() - \
                                self._robot.root_physx_view.get_link_transforms()[0, self._ee_id,:3].squeeze() 
        else:
            self.arm_offset = torch.zeros(3, device=self.device)
        
        # Compute position and orientation offset between the end effector and the vehicle
        quad_pos = self._robot.data.body_pos_w[0, self._body_id]
        quad_ori = self._robot.data.body_quat_w[0, self._body_id]

        com_pos = self._robot.data.body_pos_w[0, self._com_id]
        com_ori = self._robot.data.body_quat_w[0, self._com_id]

        if self.cfg.has_end_effector:
            ee_pos = self._robot.data.body_pos_w[0, self._ee_id]
            ee_ori = self._robot.data.body_quat_w[0, self._ee_id]
            


        # get center of mass of whole system (vehicle + end effector)
        self.vehicle_mass = self._robot.root_physx_view.get_masses()[0, self._body_id].sum()
        self._nominal_mass = self._robot.root_physx_view.get_masses()[:, self._body_id].clone().to(self.device)
        self._robot_mass = self._robot.root_physx_view.get_masses()[:, self._body_id].clone().to(self.device)
        self.arm_mass = self._total_mass - self.vehicle_mass

        self.com_pos_w = torch.zeros(1, 3, device=self.device)
        for i in range(self._robot.num_bodies):
            self.com_pos_w += self._robot.root_physx_view.get_masses()[0, i] * self._robot.root_physx_view.get_link_transforms()[0, i, :3].squeeze()
        self.com_pos_w /= self._robot.root_physx_view.get_masses()[0].sum()

        if self.cfg.has_end_effector:
            self.com_pos_e, self.com_ori_e = isaac_math_utils.subtract_frame_transforms(ee_pos, ee_ori, com_pos, com_ori)

        self.vehicle_mass = self._robot.root_physx_view.get_masses().clone().to(self.device)[:,self._body_id].squeeze(1)

        
        self.arm_length = torch.linalg.norm(self.arm_offset, dim=-1)



        self.position_offset = quad_pos
        self.orientation_offset = quad_ori


        self._gravity_magnitude = torch.tensor(self.cfg.sim.gravity, device=self.device).norm()
        self._robot_weight = (self._total_mass * self._gravity_magnitude).item()
        self._robot_weight = torch.tensor(self._robot_weight, device=self.device).tile((self.num_envs))
        self._grav_vector_unit = torch.tensor([0.0, 0.0, -1.0], device=self.device).tile((self.num_envs, 1))
        self._grav_vector = torch.tensor(self.cfg.sim.gravity, device=self.device).tile((self.num_envs, 1))

        self._default_masses = self._robot.root_physx_view.get_masses()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._grav_vector_unit = torch.tensor([0.0, 0.0, -1.0], device=self.device).tile((self.num_envs, 1))
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).squeeze()

        # Visualization marker data
        if self.cfg.viz_mode == "triad" or self.cfg.viz_mode == "frame":
            self._frame_positions = torch.zeros(self.num_envs, 2, 3, device=self.device)
            self._frame_orientations = torch.zeros(self.num_envs, 2, 4, device=self.device)
        elif self.cfg.viz_mode == "robot":
            self._robot_positions = torch.zeros(self.num_envs, 3, device=self.device)
            self._robot_orientations = torch.zeros(self.num_envs, 4, device=self.device)
            self._robot_pos_history = torch.zeros(self.num_envs, self.cfg.viz_history_length, 3, device=self.device)
            self._robot_ori_history = torch.zeros(self.num_envs, self.cfg.viz_history_length, 4, device=self.device)
            self._goal_pos_history = torch.zeros(self.num_envs, self.cfg.viz_history_length, 3, device=self.device)
            self._goal_ori_history = torch.zeros(self.num_envs, self.cfg.viz_history_length, 4, device=self.device)
        elif self.cfg.viz_mode == "viz":
            self._robot_positions = torch.zeros(self.num_envs, 3, device=self.device)
            self._robot_orientations = torch.zeros(self.num_envs, 4, device=self.device)
        else:
            raise ValueError("Visualization mode not recognized: ", self.cfg.viz_mode)

        self.local_num_envs = self.num_envs

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _compute_motor_speeds(self, wrench_des):
        """
        Compute the desired motor speeds from the desired wrench.
        """
        f_des = torch.bmm(self.TM_to_f, wrench_des.unsqueeze(2)).squeeze(2) # (n_envs, 4)
        motor_speed_squared = f_des / self._k_eta.unsqueeze(1) # (n_envs, 4)
        motor_speeds_des = torch.sign(motor_speed_squared) * torch.sqrt(torch.abs(motor_speed_squared))
        motor_speeds_des = motor_speeds_des.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)
        return motor_speeds_des

    def _get_moment_from_ctatt(self, actions):
        """
        Compute the desired moment from the actions using the CTATT control method.
        This method uses the flatness-based control approach to compute the desired moment.
        The actions are expected to be in the range [-1, 1] and represent the desired attitude in terms of roll, pitch, and yaw.
        The actions are expected to be in the following format:
        actions[:, 0] = Collective Thrust (not used in this method)
        actions[:, 1] = Roll (x-axis rotation)
        actions[:, 2] = Pitch (y-axis rotation)
        actions[:, 3] = Yaw (z-axis rotation)
        """
        ori_matrix = isaac_math_utils.matrix_from_quat(self._robot.data.root_quat_w)
        
        # Flatness based control
        shape_des = flatness_utils.s2_projection(actions[:, 1]* self.cfg.attitude_scale_xy, actions[:, 2]* self.cfg.attitude_scale_xy)
        psi_des = actions[:,3] * self.cfg.attitude_scale_z
        ori_des_matrix = flatness_utils.getRotationFromShape(shape_des, psi_des)

        euler_des = actions[:, 1:] * self.cfg.attitude_scale
        ori_des_matrix = isaac_math_utils.matrix_from_euler(euler_des, "XYZ") # (n_envs, 3, 3) 

        S_err = 0.5 * (torch.bmm(ori_des_matrix.transpose(-2, -1), ori_matrix) - torch.bmm(ori_matrix.transpose(-2, -1), ori_des_matrix)) # (n_envs, 3, 3)
        att_err = math_utils.vee_map(S_err) # (n_envs, 3)
        omega_des = torch.zeros(self.num_envs, 3, device=self.device)
        omega_err = self._robot.data.root_ang_vel_b - omega_des # (n_envs, 3)

        att_pd = torch.bmm(att_err.unsqueeze(2), -self._kp_att.reshape(-1, 1, 1)).squeeze(2) - torch.bmm(omega_err.unsqueeze(2), self._kd_att.reshape(-1, 1, 1)).squeeze(2) # (n_envs, 3)
        I_omega = torch.bmm(self.inertia_tensor, self._robot.data.root_ang_vel_b.unsqueeze(2)).squeeze(2).to(self.device)
        cmd_moment = torch.bmm(self.inertia_tensor, att_pd.unsqueeze(2)).squeeze(2) + \
                    torch.cross(self._robot.data.root_ang_vel_b, I_omega, dim=1) 
        
        return cmd_moment
    
    def _get_moment_from_ctbr(self, actions):
        """
        Compute the desired moment from the actions using the CTBR control method.
        This method uses a PD controller to compute the desired moment based on the body rates.
        The actions are expected to be in the range [-1, 1] and represent the desired body rates.
        The actions are expected to be in the following format:
        actions[:, 0] = Collective Thrust (not used in this method)
        actions[:, 1] = Body X moment (roll)
        actions[:, 2] = Body Y moment (pitch)
        actions[:, 3] = Body Z moment (yaw)
        """
        omega_des = torch.zeros(self.num_envs, 3, device=self.device)
        omega_des[:, :2] = self.cfg.body_rate_scale_xy * actions[:, 1:3]
        omega_des[:, 2] = self.cfg.body_rate_scale_z * actions[:, 3]
        
        omega_err = self._robot.data.root_ang_vel_b - omega_des
        omega_dot_err = (omega_err - self._previous_omega_err) / self.cfg.pd_loop_rate_hz
        omega_dot = -self.cfg.kp_omega * omega_err - self.cfg.kd_omega * omega_dot_err
        self._previous_omega_err = omega_err

        cmd_moment = torch.bmm(self.inertia_tensor, omega_dot.unsqueeze(2)).squeeze(2)
        return cmd_moment

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Pre-physics step function to apply actions and compute desired wrench.
        This function updates the action queue, computes the desired wrench, and applies the actions to the robot.
        The actions are expected to be in the range [-1, 1] and represent the desired control inputs.
        The actions are expected to be in the following format:
        actions[:, 0] = Collective Thrust (scaled to the range [min_thrust, max_thrust])
        actions[:, 1:] = Moment, Body Rate (roll, pitch, yaw) or attitude angles depending on the control mode.
        """
        # self._action_queue = torch.roll(self._action_queue, shifts=-1, dims=0) # roll the action queue to make room for the new action
        # self._action_queue[-1] = actions.clone().clamp(-1.0, 1.0) # add the new action to the end of the queue
        # self._actions = self._action_queue[0]
        self._actions = actions.clone().clamp(-1.0, 1.0)

        self._action_history = torch.roll(self._action_history, shifts=1, dims=1) # roll the action history to make room for the new action
        self._action_history[:, 0] = self._actions.clone().clamp(-1.0, 1.0) # add the new action to the history


        if self.cfg.use_motor_dynamics:
            self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * (4.0 * self.max_thrust - 4.0 * self.min_thrust) + self.min_thrust # scale thrust to the range [min_thrust, max_thrust]
        else:
            self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * (self._thrust_to_weight * self._robot_weight)

        if self.cfg.control_mode == "CTBM":
            self._wrench_des[:, 1:3] = self.cfg.moment_scale_xy * self._actions[:, 1:3]
            self._wrench_des[:, 3] = self.cfg.moment_scale_z * self._actions[:, 3]
            return
        elif self.cfg.control_mode == "CTATT":
            self._wrench_des[:,1:] = self._get_moment_from_ctatt(self._actions)
        elif self.cfg.control_mode == "CTBR":
            self._wrench_des[:,1:] = self._get_moment_from_ctbr(self._actions)
            
        else:
            raise NotImplementedError(f"Control mode {self.cfg.control_mode} is not implemented.")

        self._motor_speeds_des = self._compute_motor_speeds(self._wrench_des)
        self.pd_loop_counter = 0 # Reset the PD loop counter


    def _apply_action(self):
        """
        Apply the actions to the robot and compute the motor speeds.
        This function updates the motor speeds based on the desired wrench and applies the actions to the robot.
        This function is called cfg.decimation times per physics step, depending on the pd_loop_decimation.
        """
        # Skip low-level motor dynamics
        if not self.cfg.use_motor_dynamics:
            self._thrust[:, 0, 2] = self._wrench_des[:, 0]
            self._moment[:, 0, :] = self._wrench_des[:, 1:]
            self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
            return

        # Update PD loop at the appropriate rate (100Hz or whatever pd_loop_rate_hz is)
        if self.pd_loop_counter % self.cfg.pd_loop_decimation == 0:
            if self.cfg.control_mode == "CTATT":
                self._wrench_des[:,1:] = self._get_moment_from_ctatt(self._actions)
            elif self.cfg.control_mode == "CTBR":
                self._wrench_des[:,1:] = self._get_moment_from_ctbr(self._actions)
                
            # Recompute motor speeds based on fresh wrench calculation
            self._motor_speeds_des = self._compute_motor_speeds(self._wrench_des)

        self.pd_loop_counter += 1

        motor_accel = torch.bmm((1.0/self._tau_m).reshape(self.num_envs, 1, 1), (self._motor_speeds_des - self._motor_speeds).unsqueeze(1)).squeeze(1) # (n_envs, 4)
        self._motor_speeds += motor_accel * self.physics_dt
        self._motor_speeds = self._motor_speeds.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max) # Motor saturation
        motor_forces = self.cfg.k_eta * self._motor_speeds ** 2
   
        wrench = torch.bmm(self.f_to_TM, motor_forces.unsqueeze(2)).squeeze(2) # (n_envs, 4)
        
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
            self.cfg.pos_radius = 0.8 * (0.5 ** (total_timesteps // self.cfg.pos_radius_curriculum))

        
    def update_goal_state(self):
        """
        Update the goal state based on the trajectory type.
        This function computes the desired position and orientation trajectory based on the configured trajectory type.
        It updates the desired position and orientation for the current environment based on the trajectory type.
        The trajectory types supported are:
        - "lissaajous": Lissajous curve trajectory
        - "polynomial": Polynomial trajectory
        - "combined": Combination of Lissajous and Polynomial trajectories
        - "none": No trajectory, just use the current position and orientation as the goal
        If the trajectory type is "none", the desired position and orientation are set to the current position and orientation of the robot.
        If the trajectory type is not recognized, a NotImplementedError is raised.
        This function is called every cfg.traj_update_dt seconds, which is determined by the cfg.policy_rate_hz and cfg.traj_update_dt.
        """
        env_ids = (self.episode_length_buf % int(self.cfg.traj_update_dt*self.cfg.policy_rate_hz)== 0).nonzero(as_tuple=False)
        
        if len(env_ids) == 0 or env_ids.size(0) == 0:
            return
        

        current_time = self.episode_length_buf[env_ids]
        future_timesteps = torch.arange(0, 1+self.cfg.trajectory_horizon, device=self.device)
        time = (current_time + future_timesteps.unsqueeze(0)) * self.cfg.traj_update_dt

        if self.cfg.trajectory_type == "lissaajous":
            pos_traj, yaw_traj = traj_utils.eval_lissajous_curve(time, self.lissajous_amplitudes, self.lissajous_frequencies, self.lissajous_phases, self.lissajous_offsets, derivatives=4)
        elif self.cfg.trajectory_type == "polynomial":
            pos_traj, yaw_traj = traj_utils.eval_polynomial_curve(time, self.polynomial_coefficients, derivatives=4)
        elif self.cfg.trajectory_type == "combined":
            pos_lissajous, yaw_lissajous = traj_utils.eval_lissajous_curve(time, self.lissajous_amplitudes, self.lissajous_frequencies, self.lissajous_phases, self.lissajous_offsets, derivatives=4)
            pos_poly, yaw_poly = traj_utils.eval_polynomial_curve(time, self.polynomial_coefficients, derivatives=4)
            pos_traj = pos_lissajous + pos_poly
            yaw_traj = yaw_lissajous + yaw_poly
        else:
            raise NotImplementedError("Trajectory type not implemented")
    
        self._pos_traj = pos_traj
        self._yaw_traj = yaw_traj
        
        
        self._desired_pos_traj_w[env_ids.squeeze(1)] = (pos_traj[0,env_ids.squeeze(1)]).transpose(1,2)
        self._desired_ori_traj_w[env_ids.squeeze(1)] = math_utils.quat_from_yaw(yaw_traj[0,env_ids.squeeze(1)])

        self._desired_pos_w[env_ids] = self._desired_pos_traj_w[env_ids, 0]
        self._desired_ori_w[env_ids] = self._desired_ori_traj_w[env_ids, 0]
        

    def _get_observations(self) -> torch.Dict[str, torch.Tensor | torch.Dict[str, torch.Tensor]]:
        """
        Returns the observation dictionary. Policy observations are in the key "policy".
        """
        self._apply_curriculum(self.common_step_counter * self.num_envs)
        self.update_goal_state()
        
        
        base_pos_w, base_ori_w, lin_vel_w, ang_vel_w = self.get_frame_state_from_task(self.cfg.task_body)
        goal_pos_w, goal_ori_w = self.get_goal_state_from_task(self.cfg.goal_body)


        # Find the error of the end-effector to the desired position and orientation
        # The root state of the robot is the end-effector frame in this case
        # Batched over number of environments, returns (num_envs, 3) and (num_envs, 4) tensors
        pos_error_b, ori_error_b = isaac_math_utils.subtract_frame_transforms(
            base_pos_w, base_ori_w, 
            goal_pos_w, goal_ori_w
        )

        future_pos_error_b = []
        future_ori_error_b = []
        for i in range(self.cfg.trajectory_horizon):
            goal_pos_traj_w, goal_ori_traj_w = self.convert_ee_goal_from_task(self._desired_pos_traj_w[:, i+1].squeeze(1), self._desired_ori_traj_w[:, i+1].squeeze(1), self.cfg.goal_body)

            waypoint_pos_error_b, waypoint_ori_error_b = isaac_math_utils.subtract_frame_transforms(base_pos_w, base_ori_w, goal_pos_traj_w, goal_ori_traj_w)
            future_pos_error_b.append(waypoint_pos_error_b) # append (n, 3) tensor
            future_ori_error_b.append(waypoint_ori_error_b) # append (n, 4) tensor
        if len(future_pos_error_b) > 0:
            future_pos_error_b = torch.stack(future_pos_error_b, dim=1) # stack to (n, horizon, 3) tensor
            future_ori_error_b = torch.stack(future_ori_error_b, dim=1) # stack to (n, horizon, 4) tensor
 
            if self.cfg.use_yaw_representation_for_trajectory:
                future_ori_error_b = math_utils.yaw_from_quat(future_ori_error_b).reshape(self.num_envs, self.cfg.trajectory_horizon, 1)
        else:
            future_pos_error_b = torch.zeros(self.num_envs, self.cfg.trajectory_horizon, 3, device=self.device)
            future_ori_error_b = torch.zeros(self.num_envs, self.cfg.trajectory_horizon, 4, device=self.device)

        

        # Compute the orientation error as a yaw error in the body frame
        # goal_yaw_w = isaac_math_utils.yaw_quat(self._desired_ori_w)
        goal_yaw_w = isaac_math_utils.yaw_quat(goal_ori_w)
        current_yaw_w = isaac_math_utils.yaw_quat(base_ori_w)
        # yaw_error_w = quat_mul(quat_inv(current_yaw_w), goal_yaw_w)
        yaw_error_w = math_utils.yaw_error_from_quats(current_yaw_w, goal_yaw_w, dof=self.cfg.num_joints).view(self.num_envs, 1)
        
        if self.cfg.use_yaw_representation:
            yaw_representation = yaw_error_w
        else:
            yaw_representation = torch.zeros(self.num_envs, 0, device=self.device)
        

        if self.cfg.use_full_ori_matrix:
            ori_representation_b = isaac_math_utils.matrix_from_quat(ori_error_b).flatten(-2, -1)
        else:
            ori_representation_b = torch.zeros(self.num_envs, 0, device=self.device)

        if self.cfg.use_grav_vector:
            grav_vector_b = isaac_math_utils.quat_rotate_inverse(base_ori_w, self._grav_vector_unit) # projected gravity vector in the cfg frame
        else:
            grav_vector_b = torch.zeros(self.num_envs, 0, device=self.device)
        
        
        if self.cfg.use_ang_vel_from_trajectory and self.cfg.trajectory_horizon > 0:
            ang_vel_des = torch.zeros_like(ang_vel_w)
            ang_vel_des[:,2] = self._yaw_traj[1, :, 0]
            ang_vel_error_w = ang_vel_des - ang_vel_w
        else:
            ang_vel_error_w = ang_vel_w
        
        # Old computation for lin and ang vel:
        lin_vel_error_w = self._pos_traj[1, :, :, 0] - lin_vel_w
        lin_vel_b = isaac_math_utils.quat_rotate_inverse(base_ori_w, lin_vel_error_w)
        ang_vel_b = isaac_math_utils.quat_rotate_inverse(base_ori_w, ang_vel_error_w)


        # Compute the joint states
        shoulder_joint_pos = torch.zeros(self.num_envs, 0, device=self.device)
        shoulder_joint_vel = torch.zeros(self.num_envs, 0, device=self.device)
        wrist_joint_pos = torch.zeros(self.num_envs, 0, device=self.device)
        wrist_joint_vel = torch.zeros(self.num_envs, 0, device=self.device)
        if self.cfg.num_joints > 0:
            shoulder_joint_pos = self._robot.data.joint_pos[:, self._shoulder_joint_idx].unsqueeze(1)
            shoulder_joint_vel = self._robot.data.joint_vel[:, self._shoulder_joint_idx].unsqueeze(1)
        if self.cfg.num_joints > 1:
            wrist_joint_pos = self._robot.data.joint_pos[:, self._wrist_joint_idx].unsqueeze(1)
            wrist_joint_vel = self._robot.data.joint_pos[:, self._wrist_joint_idx].unsqueeze(1)

        # Previous Action
        if self.cfg.use_previous_actions:
            previous_actions = self._previous_actions
        else:
            previous_actions = torch.zeros(self.num_envs, 0, device=self.device)

        obs = torch.cat(
            [
                pos_error_b,                                # (num_envs, 3)
                ori_representation_b,                       # (num_envs, 0) if not using full ori matrix, (num_envs, 9) if using full ori matrix
                yaw_representation,                         # (num_envs, 4) if using yaw representation (quat), 0 otherwise
                grav_vector_b,                              # (num_envs, 3) if using gravity vector, 0 otherwise
                lin_vel_b,                                  # (num_envs, 3)
                ang_vel_b,                                  # (num_envs, 3)
                shoulder_joint_pos,                         # (num_envs, 1)
                wrist_joint_pos,                            # (num_envs, 1)
                shoulder_joint_vel,                         # (num_envs, 1)
                wrist_joint_vel,                            # (num_envs, 1)
                previous_actions,                           # (num_envs, 4)
                future_pos_error_b.flatten(-2, -1),         # (num_envs, horizon * 3)
                future_ori_error_b.flatten(-2, -1)          # (num_envs, horizon * 4) if use_yaw_representation_for_trajectory, else (num_envs, horizon, 1)
            ],
            dim=-1                                          # (num_envs, 22 + 7*horizon)
        )

        
        
        # We also need the state information for other controllers like the decoupled controller.
        # This is the full state of the robot
        quad_pos_w, quad_ori_w, quad_lin_vel_w, quad_ang_vel_w = self.get_frame_state_from_task("COM")
        ee_pos_w, ee_ori_w, ee_lin_vel_w, ee_ang_vel_w = self.get_frame_state_from_task("root")
    
        if self.cfg.gc_mode:
            future_com_pos_w = []
            future_com_ori_w = []
            for i in range(self.cfg.trajectory_horizon):
                des_com_pos_w, des_com_ori_w = self.convert_ee_goal_to_com_goal(self._desired_pos_traj_w[:, i].squeeze(1), self._desired_ori_traj_w[:, i].squeeze(1))
                future_com_pos_w.append(des_com_pos_w)
                future_com_ori_w.append(des_com_ori_w)

            if len(future_com_pos_w) > 0:
                future_com_pos_w = torch.stack(future_com_pos_w, dim=1)
                future_com_ori_w = torch.stack(future_com_ori_w, dim=1)
            else:
                future_com_pos_w = torch.zeros(self.num_envs, self.cfg.trajectory_horizon, 3, device=self.device)
                future_com_ori_w = torch.zeros(self.num_envs, self.cfg.trajectory_horizon, 4, device=self.device)

            goal_pos_w, goal_ori_w = self.get_goal_state_from_task("COM")

            gc_obs = torch.cat(
                [
                    quad_pos_w,                                                 # (num_envs, 3)
                    quad_ori_w,                                                 # (num_envs, 4)
                    quad_lin_vel_w,                                             # (num_envs, 3)
                    quad_ang_vel_w,                                             # (num_envs, 3)
                    goal_pos_w,                                                 # (num_envs, 3)
                    math_utils.yaw_from_quat(goal_ori_w).unsqueeze(1),          # (num_envs, 1)
                    future_com_pos_w.flatten(-2, -1),                           # (num_envs, horizon * 3)
                    future_com_ori_w.flatten(-2, -1)                            # (num_envs, horizon * 4)
                ],
                dim=-1                                                          # (num_envs, 17 + 3*horizon)
            )
        else:
            gc_obs = None

        if self.cfg.eval_mode:
            pos_traj = self._pos_traj[:3,:,:,0].permute(1,0,2).reshape(self.num_envs, -1)
            yaw_traj = self._yaw_traj[:2,:,0].permute(1,0).reshape(self.num_envs, -1)
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
                    shoulder_joint_pos,                         # (num_envs, 1) [26] 
                    wrist_joint_pos,                            # (num_envs, 1) [27]
                    shoulder_joint_vel,                         # (num_envs, 1) [28]
                    wrist_joint_vel,                            # (num_envs, 1) [29]
                    self._desired_pos_w,                        # (num_envs, 3) [30-33] [26-29]
                    self._desired_ori_w,                        # (num_envs, 4) [33-37] [29-33]
                    pos_traj,                                   # (num_envs, 3 * horizon) [37-37 + 3*horizon] [33-33 + 3*horizon]   
                    yaw_traj,                                   # (num_envs, 1 * horizon) [37 + 3*horizon - 37 + 3*horizon + horizon] [33 + 3*horizon - 33 + 3*horizon + horizon]
                ],
                dim=-1                                          # (num_envs, 18)
            )
            self._state = full_state
        else:
            full_state = None

        return {"policy": obs, "gc": gc_obs, "full_state": full_state}

    def _get_rewards(self) -> torch.Tensor:
        """
        Returns the reward tensor.
        """
        base_pos_w, base_ori_w, lin_vel_w, ang_vel_w = self.get_frame_state_from_task(self.cfg.reward_task_body)
        goal_pos_w, goal_ori_w = self.get_goal_state_from_task(self.cfg.reward_goal_body)
        
        # Computes the error from the desired position and orientation
        pos_error = torch.linalg.norm(goal_pos_w - base_pos_w, dim=1)
        # pos_distance = 1.0 - torch.tanh(pos_error / self.cfg.pos_radius)
        if self.cfg.square_pos_error:
            pos_distance = torch.exp(- (pos_error **2) / self.cfg.pos_radius)
        else:
            pos_distance = torch.exp(- (pos_error) / self.cfg.pos_radius)

        ori_error = isaac_math_utils.quat_error_magnitude(goal_ori_w, base_ori_w)

        smooth_transition_func = 1.0 - torch.exp(-1.0 / torch.max(self.cfg.yaw_smooth_transition_scale*pos_error - 10.0, torch.zeros_like(pos_error)))

        yaw_error = math_utils.yaw_error_from_quats(goal_ori_w, base_ori_w, self.cfg.num_joints).unsqueeze(1)
        yaw_error = torch.linalg.norm(yaw_error, dim=1)

        # yaw_distance = (1.0 - torch.tanh(yaw_error / self.cfg.yaw_radius)) * smooth_transition_func
        yaw_distance = torch.exp(- (yaw_error **2) / self.cfg.yaw_radius)
        yaw_error = yaw_error * smooth_transition_func


        # Velocity error components, used for stabliization tuning
        if self.cfg.trajectory_horizon > 0:
            lin_vel_error_w = self._pos_traj[1, :, :, 0] - lin_vel_w
        else:
            lin_vel_error_w = torch.zeros_like(lin_vel_w, device=self.device) - lin_vel_w
        lin_vel_b = isaac_math_utils.quat_rotate_inverse(base_ori_w, lin_vel_error_w)
        if self.cfg.use_ang_vel_from_trajectory and self.cfg.trajectory_horizon > 0:
            ang_vel_des = torch.zeros_like(ang_vel_w)
            ang_vel_des[:,2] = self._yaw_traj[1, :, 0]
            ang_vel_error_w = ang_vel_des - ang_vel_w
        else:
            ang_vel_error_w = torch.zeros_like(ang_vel_w) - ang_vel_w
        ang_vel_b = isaac_math_utils.quat_rotate_inverse(base_ori_w, ang_vel_error_w)


        lin_vel_error = torch.norm(lin_vel_b, dim=1)
        ang_vel_error = torch.norm(ang_vel_b, dim=1)
        joint_vel_error = torch.norm(self._robot.data.joint_vel, dim=1)

        # action_error = torch.sum(torch.square(self._actions), dim=1) 
        action_error = torch.norm(self._actions, dim=1)
        previous_action_error = torch.norm(self._actions - self._previous_actions, dim=1)


        if self.cfg.scale_reward_with_time:
            time_scale = 1.0 / self.cfg.policy_rate_hz
        else:
            time_scale = 1.0

        if self.cfg.square_reward_errors:
            pos_error = pos_error ** 2
            pos_distance = pos_distance ** 2
            ori_error = ori_error ** 2
            yaw_error = yaw_error ** 2
            yaw_distance = yaw_distance ** 2
            lin_vel_error = lin_vel_error ** 2
            ang_vel_error = ang_vel_error ** 2
            joint_vel_error = joint_vel_error ** 2
            action_error = action_error ** 2
            previous_action_error = previous_action_error ** 2

        crash_penalty_time = self.cfg.crash_penalty * (self.max_episode_length - self.episode_length_buf)


        rewards = {
            "pos_error": pos_error * self.cfg.pos_error_reward_scale * time_scale,
            "pos_distance": pos_distance * self.cfg.pos_distance_reward_scale * time_scale,
            "ori_error": ori_error * self.cfg.ori_error_reward_scale * time_scale,
            "yaw_error": yaw_error  * self.cfg.yaw_error_reward_scale * time_scale,
            "yaw_distance": yaw_distance * self.cfg.yaw_distance_reward_scale * time_scale,
            "lin_vel": lin_vel_error * self.cfg.lin_vel_reward_scale * time_scale,
            "ang_vel": ang_vel_error * self.cfg.ang_vel_reward_scale * time_scale,
            "action_norm": action_error * self.cfg.action_norm_reward_scale * time_scale,
            "previous_action_norm": previous_action_error * self.cfg.previous_action_norm_reward_scale * time_scale,
            "stay_alive": torch.ones_like(pos_error) * self.cfg.stay_alive_reward * time_scale,
            "crash_penalty": self.reset_terminated[:].float() * crash_penalty_time * time_scale,
        }
        errors = {
            "pos_error": pos_error,
            "pos_distance": pos_distance,
            "ori_error": ori_error,
            "yaw_error": yaw_error,
            "yaw_distance": yaw_distance,
            "lin_vel": lin_vel_error,
            "ang_vel": ang_vel_error,
            "action_norm": action_error,
            "previous_action_norm": previous_action_error,
            "stay_alive": torch.ones_like(pos_error),
            "crash_penalty": self.reset_terminated[:].float(),
        }

        # 7 x 1024 -> 1024
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        for key, value in errors.items():
            self._episode_error_sums[key] += value
        return reward

    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the tensors corresponding to termination and truncation. 
        """

        # Check if end effector or body has collided with the ground
        if self.cfg.has_end_effector:
            died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.0, self._robot.data.body_state_w[:, self._body_id, 2].squeeze() < 0.0)
        else:
            died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)

        # Check if the robot is too high
        # died = torch.logical_or(died, self._robot.data.root_pos_w[:, 2] > 10.0)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        Resets the environment at the specified indices.
        """

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        base_pos_w, base_ori_w, _, _ = self.get_frame_state_from_task(self.cfg.task_body)

        # Logging the episode sums
        final_distance_to_goal = torch.linalg.norm(self._desired_pos_w[env_ids] - base_pos_w[env_ids], dim=1).mean()
        final_ori_error_to_goal = isaac_math_utils.quat_error_magnitude(self._desired_ori_w[env_ids], base_ori_w[env_ids]).mean()
        final_yaw_error_to_goal = isaac_math_utils.quat_error_magnitude(isaac_math_utils.yaw_quat(self._desired_ori_w[env_ids]), isaac_math_utils.yaw_quat(base_ori_w[env_ids])).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = self._episode_sums[key][env_ids].mean()
            extras["Episode Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        for key in self._episode_error_sums.keys():
            episodic_sum_avg = self._episode_error_sums[key][env_ids].mean()
            extras["Episode Error/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_error_sums[key][env_ids] = 0.0
        extras["Metrics/Final Distance to Goal"] = final_distance_to_goal
        extras["Metrics/Final Orientation Error to Goal"] = final_ori_error_to_goal
        extras["Metrics/Final Yaw Error to Goal"] = final_yaw_error_to_goal
        extras["Episode Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/Position Radius"] = self.cfg.pos_radius
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and not self.cfg.eval_mode:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        elif self.cfg.eval_mode:
            self.episode_length_buf[env_ids] = 0

        # Update the trajectories for the reset environments
        self.initialize_trajectories(env_ids)
        self.update_goal_state()

        # print("Goal State: ", self._desired_pos_w[env_ids[0]], self._desired_ori_w[env_ids[0]])

        # Reset Robot state
        self._robot.reset()
        
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # if self.cfg.num_joints > 0:
        #     # print("Resetting shoulder joint to pi/2")
        #     shoulder_joint_pos = torch.tensor(torch.pi/2, device=self.device, requires_grad=False).float()
        #     shoulder_joint_vel = torch.tensor(0.0, device=self.device, requires_grad=False).float()
        #     self._robot.write_joint_state_to_sim(shoulder_joint_pos, shoulder_joint_vel, joint_ids=self._shoulder_joint_idx, env_ids=env_ids)

        self.domain_randomization(env_ids)

        if self.cfg.init_cfg == "rand":
            default_root_state = self._robot.data.default_root_state[env_ids]
            # Initialize the robot on the trajectory with the correct velocity
            traj_pos_start = self._pos_traj[0, env_ids, :, 0]
            traj_vel_start = self._pos_traj[1, env_ids, :, 0]
            traj_yaw_start = self._yaw_traj[0, env_ids, 0]
            pos_rand = (torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_pos_ranges, device=self.device).float()
            vel_rand = (torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_lin_vel_ranges, device=self.device).float()
            yaw_rand = (torch.rand(len(env_ids), 1, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_yaw_ranges, device=self.device).float()
            ang_vel_rand = (torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_ang_vel_ranges, device=self.device).float()
            init_yaw = math_utils.quat_from_yaw(traj_yaw_start + yaw_rand.squeeze(1))

            default_root_state[:, :3] = traj_pos_start + pos_rand
            default_root_state[:, 3:7] = init_yaw
            default_root_state[:, 7:10] = traj_vel_start + vel_rand
            default_root_state[:, 10:13] = ang_vel_rand
        elif self.cfg.init_cfg == "fixed":
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            default_root_state[:, 2] = 3.0
            # default_root_state[:, 3:7] = self._desired_ori_w[env_ids]
        else:
            default_root_state = self._robot.data.default_root_state[env_ids]
            # Initialize the robot on the trajectory with the correct velocity
            default_root_state[:, :3] = self._desired_pos_w[env_ids]
            default_root_state[:, 3:7] = self._desired_ori_w[env_ids]
            default_root_state[:, 7:10] = self._pos_traj[1, env_ids, :, 0]
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Update viz_histories
        if self.cfg.viz_mode == "robot":
            self._robot_pos_history[env_ids] = default_root_state[:, :3].unsqueeze(1).tile(1, self.cfg.viz_history_length, 1)
            self._robot_ori_history[env_ids] = default_root_state[:, 3:7].unsqueeze(1).tile(1, self.cfg.viz_history_length, 1)
            self._goal_pos_history[env_ids] = self._desired_pos_w[env_ids].unsqueeze(1).tile(1, self.cfg.viz_history_length, 1)
            self._goal_ori_history[env_ids] = self._desired_ori_w[env_ids].unsqueeze(1).tile(1, self.cfg.viz_history_length, 1)
        
        # if self.cfg.num_joints > 0:
        #     default_root_state[:, 3:7] = torch.tensor([0.5, -0.5, -0.5, 0.5], device=self.device, requires_grad=False).float().tile((env_ids.size(0), 1))
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

    def domain_randomization(self, env_ids: torch.Tensor | None):
        if env_ids is None or env_ids.shape[0] == 0:
            return
        
        reinit_motor_dynamics = False

        if self.cfg.dr_dict.get("thrust_to_weight", 0.0) > 0:
            self._thrust_to_weight[env_ids] = torch.zeros_like(self._thrust_to_weight[env_ids]).normal_(mean=0.0, std=0.4) + self.cfg.thrust_to_weight

        if self.cfg.dr_dict.get("mass", 0.0) > 0:
            dr_range = self.cfg.dr_dict["mass"]
            self._robot_mass[env_ids] = torch.zeros_like(self._robot_mass[env_ids]).uniform_(1-dr_range, 1+dr_range) * self.cfg.mass
            self._robot_weight[env_ids] = (self._robot_mass[env_ids] * self._gravity_magnitude).squeeze()
            new_masses = self._robot.root_physx_view.get_masses().clone().to(self.device)
            new_masses[env_ids, self._body_id] = self._robot_mass[env_ids].squeeze(1)
            self._robot.root_physx_view.set_masses(new_masses.cpu(), env_ids.cpu())
            self.vehicle_mass = self._robot.root_physx_view.get_masses().clone().to(self.device)[:,self._body_id].squeeze(1)


        if self.cfg.dr_dict.get("inertia", 0.0) > 0:
            dr_range = self.cfg.dr_dict["inertia"]
            self._robot_inertia[env_ids] = torch.zeros_like(self._robot_inertia[env_ids]).uniform_(1-dr_range, 1+dr_range) * self.default_inertia.view(-1, 3, 3).tile(env_ids.shape[0], 1, 1)
            new_inertia = self._robot.root_physx_view.get_inertias().clone().to(self.device)
            new_inertia[env_ids, self._body_id] = self._robot_inertia[env_ids].view(env_ids.shape[0], 9)
            self._robot.root_physx_view.set_inertias(new_inertia.cpu(), env_ids.cpu())
            self.vehicle_inertia = self._robot.root_physx_view.get_inertias().clone().to(self.device)[:,self._body_id].view(-1, 3, 3).squeeze(1)  

        if self.cfg.dr_dict.get("tau_m", 0.0) > 0:
            dr_range = self.cfg.dr_dict["tau_m"]
            self._tau_m[env_ids] = torch.zeros_like(self._tau_m[env_ids], device=self.device).uniform_(1-dr_range, 1+dr_range) * self.cfg.tau_m

        if self.cfg.dr_dict.get("k_eta", 0.0) > 0:
            dr_range = self.cfg.dr_dict["k_eta"]
            self._k_eta[env_ids] = torch.zeros_like(self._k_eta[env_ids], device=self.device).uniform_(1-dr_range, 1+dr_range) * self.cfg.k_eta
            reinit_motor_dynamics = True

        if self.cfg.dr_dict.get("k_m", 0.0) > 0:
            self._k_m[env_ids] = torch.zeros_like(self._k_m[env_ids], device=self.device).uniform_(-1e-10, 1e-10) + self.cfg.k_m*torch.ones(self._k_m[env_ids].shape, device=self.device)
            reinit_motor_dynamics = True
        
        if self.cfg.dr_dict.get("arm_length", 0.0) > 0:
            self._arm_length[env_ids] = torch.zeros_like(self._arm_length[env_ids], device=self.device).uniform_(-0.01, 0.01) + self.cfg.arm_length*torch.ones(self._arm_length[env_ids].shape, device=self.device)
            reinit_motor_dynamics = True

        if self.cfg.dr_dict.get("kp_att", 0.0) > 0:
            self._kp_att[env_ids] = torch.zeros_like(self._kp_att[env_ids], device=self.device).uniform_(1-self.cfg.dr_dict["kp_att"], 1+self.cfg.dr_dict["kp_att"]) * self.cfg.kp_att
        
        if self.cfg.dr_dict.get("kd_att", 0.0) > 0:
            self._kd_att[env_ids] = torch.zeros_like(self._kd_att[env_ids], device=self.device).uniform_(1-self.cfg.dr_dict["kd_att"], 1+self.cfg.dr_dict["kd_att"]) * self.cfg.kd_att

        if reinit_motor_dynamics:
            self.reinitialize_motor_dynamics(env_ids)

        if self.cfg.use_motor_dynamics:
            self._motor_speeds[env_ids] = torch.sqrt(self._robot_weight[env_ids] / (4 * self._k_eta[env_ids])).unsqueeze(1).tile((1, 4)).to(self.device)
            self.max_thrust[env_ids] = self.cfg.motor_speed_max**2 * self._k_eta[env_ids]

    def reinitialize_motor_dynamics(self, env_ids: torch.Tensor | None = None):
        if env_ids is None or env_ids.shape[0] == 0:
            return
        r2o2 = math.sqrt(2.0) / 2.0
        num_envs = len(env_ids)
        self._rotor_positions[env_ids] = torch.cat(
            [
                (self._arm_length[env_ids].unsqueeze(1) * torch.tensor([r2o2, r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                (self._arm_length[env_ids].unsqueeze(1) * torch.tensor([r2o2, -r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                (self._arm_length[env_ids].unsqueeze(1) * torch.tensor([-r2o2, -r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                (self._arm_length[env_ids].unsqueeze(1) * torch.tensor([-r2o2, r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
            ],
            dim=1, 
        )
        self.k[env_ids] = self._k_m[env_ids] / self._k_eta[env_ids]
        self.f_to_TM[env_ids] = torch.cat(
            [
                torch.ones(num_envs, 1, 4, device=self.device),
                torch.linalg.cross(self._rotor_positions[env_ids], torch.tensor([0.0, 0.0, 1.0], device=self.device).tile(num_envs, 1, 1))[:,:, 0:2].transpose(-2,-1),
                self.k[env_ids].view(num_envs, 1, 1) * self._rotor_directions[env_ids].view(num_envs, 1, 4),
            ],
            dim=1
        )
        self.TM_to_f[env_ids] = torch.linalg.inv(self.f_to_TM[env_ids])


    def initialize_trajectories(self, env_ids):
        """
        Initializes the trajectory for the environment ids.
        """
        num_envs = env_ids.size(0)

        # Randomize Lissajous parameters
        random_amplitudes = ((torch.rand(num_envs, 4, device=self.device)) * 2.0 - 1.0) * self.lissajous_amplitudes_rand_ranges
        random_frequencies = ((torch.rand(num_envs, 4, device=self.device))) * self.lissajous_frequencies_rand_ranges
        random_phases = ((torch.rand(num_envs, 4, device=self.device)) * 2.0 - 1.0) * self.lissajous_phases_rand_ranges
        random_offsets = ((torch.rand(num_envs, 4, device=self.device)) * 2.0 - 1.0) * self.lissajous_offsets_rand_ranges

        terrain_offsets = torch.zeros_like(random_offsets, device=self.device)
        terrain_offsets[:, :2] = self._terrain.env_origins[env_ids, :2]
        
        self.lissajous_amplitudes[env_ids] = torch.tensor(self.cfg.lissajous_amplitudes, device=self.device).tile((num_envs, 1)).float() + random_amplitudes
        self.lissajous_frequencies[env_ids] = torch.tensor(self.cfg.lissajous_frequencies, device=self.device).tile((num_envs, 1)).float() + random_frequencies
        self.lissajous_phases[env_ids] = torch.tensor(self.cfg.lissajous_phases, device=self.device).tile((num_envs, 1)).float() + random_phases
        self.lissajous_offsets[env_ids] = torch.tensor(self.cfg.lissajous_offsets, device=self.device).tile((num_envs, 1)).float() + random_offsets + terrain_offsets
        

    def get_frame_state_from_task(self, task_body:str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if task_body == "root":
            base_pos_w = self._robot.data.root_pos_w
            base_ori_w = self._robot.data.root_quat_w
            lin_vel_w = self._robot.data.root_lin_vel_w
            ang_vel_w = self._robot.data.root_ang_vel_w
        elif task_body == "endeffector":
            base_pos_w = self._robot.data.body_pos_w[:, self._ee_id].squeeze(1)
            base_ori_w = self._robot.data.body_quat_w[:, self._ee_id].squeeze(1)
            lin_vel_w = self._robot.data.root_lin_vel_w
            ang_vel_w = self._robot.data.root_ang_vel_w
        elif task_body == "vehicle" or task_body == "body":
            base_pos_w = self._robot.data.body_pos_w[:, self._body_id].squeeze(1)
            base_ori_w = self._robot.data.body_quat_w[:, self._body_id].squeeze(1)
            lin_vel_w = self._robot.data.body_lin_vel_w[:, self._body_id].squeeze(1)
            ang_vel_w = self._robot.data.body_ang_vel_w[:, self._body_id].squeeze(1)
        elif task_body == "COM":
            frame_id = self._robot.find_bodies("COM")[0]
            base_pos_w = self._robot.data.body_pos_w[:, frame_id].squeeze(1)
            base_ori_w = self._robot.data.body_quat_w[:, frame_id].squeeze(1)
            lin_vel_w = self._robot.data.body_lin_vel_w[:, frame_id].squeeze(1)
            ang_vel_w = self._robot.data.body_ang_vel_w[:, frame_id].squeeze(1)
        else:
            raise ValueError("Invalid task body: ", self.cfg.task_body)

        return base_pos_w, base_ori_w, lin_vel_w, ang_vel_w

    def get_goal_state_from_task(self, goal_body:str) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.has_end_effector:
            goal_pos_w = self._desired_pos_w
            goal_ori_w = self._desired_ori_w
        elif goal_body == "root":
            goal_pos_w = self._desired_pos_w
            goal_ori_w = self._desired_ori_w
        elif goal_body == "endeffector":
            goal_pos_w = self._desired_pos_w
            goal_ori_w = self._desired_ori_w
        elif goal_body == "COM":
            # desired_pos, desired_yaw = self.compute_desired_pose_from_transform(self._desired_pos_w, self._desired_ori_w, self.com_pos_e)
            desired_pos, desired_yaw = math_utils.compute_desired_pose_from_transform(self._desired_pos_w, self._desired_ori_w, self.com_pos_e, 0)
            goal_pos_w = desired_pos
            goal_ori_w = math_utils.quat_from_yaw(desired_yaw)
        else:
            raise ValueError("Invalid goal body: ", goal_body)

        return goal_pos_w, goal_ori_w
    
    def convert_ee_goal_from_task(self, ee_pos_w, ee_ori_w, task_body:str) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.has_end_effector:
            desired_pos, desired_ori = ee_pos_w, ee_ori_w
        elif task_body == "root":
            desired_pos, desired_ori = ee_pos_w, ee_ori_w
        elif task_body == "endeffector":
            desired_pos, desired_ori = ee_pos_w, ee_ori_w
        elif task_body == "vehicle":
            desired_pos, desired_yaw = math_utils.compute_desired_pose_from_transform(ee_pos_w, ee_ori_w, self._robot.data.body_pos_w[:, self._body_id].squeeze(1), 0)
            desired_ori = math_utils.quat_from_yaw(desired_yaw)
        elif task_body == "COM":
            desired_pos, desired_yaw = math_utils.compute_desired_pose_from_transform(ee_pos_w, ee_ori_w, self.com_pos_e, 0)
            desired_ori = math_utils.quat_from_yaw(desired_yaw)
        else:
            raise ValueError("Invalid task body: ", task_body)

        return desired_pos, desired_ori
    
    def convert_ee_goal_to_com_goal(self, ee_pos_w: torch.Tensor, ee_ori_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        desired_pos, desired_yaw = math_utils.compute_desired_pose_from_transform(ee_pos_w, ee_ori_w, self.com_pos_e, 0)
        return desired_pos, math_utils.quat_from_yaw(desired_yaw)

    def _setup_scene(self):
        if sum(self.cfg.robot_color) > 0:
            print("Setting robot color to: ", self.cfg.robot_color)
            print(self.cfg.robot.spawn.visual_material)
            self.cfg.robot.spawn.visual_material=sim_utils.GlassMdlCfg(glass_color=tuple(self.cfg.robot_color))
            print(self.cfg.robot.spawn.visual_material)
            
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "frame_visualizer"):
                if self.cfg.viz_mode == "triad" or self.cfg.viz_mode == "frame": 
                    frame_marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Markers",
                                            markers={
                                            "frame": sim_utils.UsdFileCfg(
                                                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                                                scale=(0.1, 0.1, 0.1),
                                            ),})
                elif self.cfg.viz_mode == "robot":
                    history_color = tuple(self.cfg.robot_color) if sum(self.cfg.robot_color) > 0 else (0.05, 0.05, 0.05)
                    frame_marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Markers",
                                            markers={
                                            "robot_mesh": sim_utils.UsdFileCfg(
                                                usd_path=self.cfg.robot.spawn.usd_path,
                                                scale=(1.0, 1.0, 1.0),
                                                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.0, 0.1, 0.0)),
                                            ),
                                            "robot_history": sim_utils.SphereCfg(
                                                radius=0.01,
                                                visual_material=sim_utils.GlassMdlCfg(glass_color=history_color),
                                            ),
                                            "goal_history": sim_utils.SphereCfg(
                                                radius=0.01,
                                                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.0, 0.1, 0.0)),
                                            ),})
                elif self.cfg.viz_mode == "viz":
                    robot_color = tuple(self.cfg.robot_color) if sum(self.cfg.robot_color) > 0 else (0.05, 0.05, 0.05)
                    frame_marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Markers",
                                            markers={
                                            "goal_mesh": sim_utils.UsdFileCfg(
                                                usd_path=self.cfg.robot.spawn.usd_path,
                                                scale=(1.0, 1.0, 1.0),
                                                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.0, 0.1, 0.0)),
                                            ),
                                            "robot_mesh": sim_utils.UsdFileCfg(
                                                usd_path=self.cfg.robot.spawn.usd_path,
                                                scale=(1.0, 1.0, 1.0),
                                                visual_material=sim_utils.GlassMdlCfg(glass_color=robot_color),
                                            ),
                                            })
                else:
                    raise ValueError("Visualization mode not recognized: ", self.cfg.viz_mode)
    
                self.frame_visualizer = VisualizationMarkers(frame_marker_cfg)
                # set their visibility to true
                self.frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "frame_visualizer"):
                self.frame_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        # update the markers
        # Update frame positions for debug visualization
        if self.cfg.viz_mode == "triad" or self.cfg.viz_mode == "frame":
            self._frame_positions[:, 0] = self._robot.data.root_pos_w
            self._frame_positions[:, 1] = self._desired_pos_w
            
            self._frame_orientations[:, 0] = self._robot.data.root_quat_w
            self._frame_orientations[:, 1] = self._desired_ori_w
            
            self.frame_visualizer.visualize(self._frame_positions.flatten(0, 1), self._frame_orientations.flatten(0,1))
        elif self.cfg.viz_mode == "robot":
            self._robot_positions = self._desired_pos_w + torch.tensor(self.cfg.viz_ref_offset, device=self.device).unsqueeze(0).tile((self.num_envs, 1))
            self._robot_orientations = self._desired_ori_w

            self._goal_pos_history = self._goal_pos_history.roll(1, dims=1)
            self._goal_pos_history[:, 0] = self._desired_pos_w
            self._goal_ori_history = self._goal_ori_history.roll(1, dims=1)
            self._goal_ori_history[:, 0] = self._desired_ori_w

            self._robot_pos_history = self._robot_pos_history.roll(1, dims=1)
            self._robot_pos_history[:, 0] = self._robot.data.root_pos_w
            self._robot_ori_history = self._robot_ori_history.roll(1, dims=1)
            self._robot_ori_history[:, 0] = self._robot.data.root_quat_w

            translation_pos = torch.cat([self._robot_positions, self._robot_pos_history.flatten(0, 1), self._goal_pos_history.flatten(0, 1)], dim=0)
            translation_ori = torch.cat([self._robot_orientations, self._robot_ori_history.flatten(0, 1), self._goal_ori_history.flatten(0, 1)], dim=0)
            marker_indices = [0]*self.num_envs + [1]*self.num_envs*self.cfg.viz_history_length + [2]*self.num_envs*self.cfg.viz_history_length
            self.frame_visualizer.visualize(translation_pos, translation_ori, marker_indices=marker_indices)
        elif self.cfg.viz_mode == "viz":
            self._robot_positions = self._desired_pos_w
            self._robot_orientations = self._desired_ori_w

            goal_pos = self._desired_pos_w.clone()
            goal_ori = self._desired_ori_w.clone()

            robot_pos = self._robot.data.root_pos_w.clone()
            robot_ori = self._robot.data.root_quat_w.clone()

            translation_pos = torch.cat([goal_pos, robot_pos], dim=0)
            translation_ori = torch.cat([goal_ori, robot_ori], dim=0)
            marker_indices = [0]*self.num_envs + [1]*self.num_envs
            self.frame_visualizer.visualize(translation_pos, translation_ori, marker_indices=marker_indices)

