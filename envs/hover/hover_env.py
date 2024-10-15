from __future__ import annotations

import torch

# Isaac SDK imports
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import (subtract_frame_transforms, combine_frame_transforms, matrix_from_quat, quat_error_magnitude, 
                                        random_orientation, quat_inv, quat_rotate_inverse, quat_mul, yaw_quat, quat_conjugate,
                                        quat_rotate, normalize, wrap_to_pi)
from omni.isaac.lab_assets import CRAZYFLIE_CFG

# Local imports
from configs.aerial_manip_asset import AERIAL_MANIPULATOR_2DOF_CFG, AERIAL_MANIPULATOR_1DOF_CFG, AERIAL_MANIPULATOR_1DOF_WRIST_CFG, AERIAL_MANIPULATOR_0DOF_CFG, AERIAL_MANIPULATOR_0DOF_DEBUG_CFG
from utils.math_utilities import yaw_from_quat, yaw_error_from_quats, quat_from_yaw, compute_desired_pose_from_transform

class AerialManipulatorEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: AerialManipulatorHoverEnv, window_name: str = "Aerial Manipulator - IsaacLab"):
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
class AerialManipulatorHoverEnvBaseCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    sim_rate_hz = 100
    policy_rate_hz = 50
    decimation = sim_rate_hz // policy_rate_hz
    ui_window_class_type = AerialManipulatorEnvWindow
    num_states = 0
    debug_vis = True

    seed = 0

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
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # action scaling
    # moment_scale_xy = 1.0
    # moment_scale_z = 0.05
    # thrust_to_weight = 3.0
    moment_scale_xy = 0.5
    moment_scale_z = 0.025 # 0.025 # 0.1
    thrust_to_weight = 3.0

    # reward scales
    pos_radius = 0.8
    pos_radius_curriculum = 0 # 10e6
    lin_vel_reward_scale = -0.05 # -0.05
    ang_vel_reward_scale = -0.01 # -0.01
    pos_distance_reward_scale = 15.0 #15.0
    pos_error_reward_scale = 0.0# -1.0
    ori_error_reward_scale = 0.0 # -0.5
    joint_vel_reward_scale = 0.0 # -0.01
    action_norm_reward_scale = 0.0 # -0.01
    yaw_error_reward_scale = -2.0 # -0.01
    yaw_distance_reward_scale = 0.0 # -0.01
    yaw_radius = 0.2 
    yaw_smooth_transition_scale = 0.0
    stay_alive_reward = 0.0
    crash_penalty = 0.0
    scale_reward_with_time = False
    square_reward_errors = False

    

    # Task condionionals for the environment - modifies the goal
    goal_cfg = "rand" # "rand", "fixed", or "initial"
    # "rand" - Random goal position and orientation
    # "fixed" - Fixed goal position and orientation set apriori
    # "initial" - Goal position and orientation is the initial position and orientation of the robot
    goal_pos = None
    goal_ori = None

    goal_pos_range = 2.0
    goal_yaw_range = 3.14159

    init_cfg = "default" # "default" or "rand"

    task_body = "root" # "root" or "endeffector" or "vehicle" or "COM"
    goal_body = "root" # "root" or "endeffector" or "vehicle" or "COM"
    reward_task_body = "root"
    reward_goal_body = "root"
    body_name = "vehicle"
    has_end_effector = True
    use_grav_vector = True
    use_full_ori_matrix = True
    use_yaw_representation = True

    shoulder_joint_active = True
    wrist_joint_active = True

    eval_mode = False
    gc_mode = False


@configclass
class AerialManipulator2DOFHoverEnvCfg(AerialManipulatorHoverEnvBaseCfg):
    # env
    num_actions = 6
    num_joints = 2
    num_observations = 16
    # 3(vel) + 3(ang vel) + 3(pos) + 3(ori) + 2(joint pos) + 2(joint vel) = 16
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_2DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    
    shoulder_torque_scalar = robot.actuators["shoulder"].effort_limit
    wrist_torque_scalar = robot.actuators["wrist"].effort_limit

@configclass
class AerialManipulator2DOFHoverPoseEnvCfg(AerialManipulatorHoverEnvBaseCfg):
    # env
    num_actions = 6
    num_joints = 2
    num_observations = 22
    # 3(vel) + 3(ang vel) + 3(pos) + 9(ori) + 2(joint pos) + 2(joint vel) = 22
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_2DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    
    shoulder_torque_scalar = robot.actuators["shoulder"].effort_limit
    wrist_torque_scalar = robot.actuators["wrist"].effort_limit
    use_full_ori_matrix = True

@configclass
class AerialManipulator1DOFHoverEnvCfg(AerialManipulatorHoverEnvBaseCfg):
    # env
    num_actions = 5
    num_joints = 1
    num_observations = 14 
    # 3(vel) + 3(ang vel) + 3(pos) + 3(ori) + 1(joint pos) + 1(joint vel) = 14
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_1DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")


    shoulder_torque_scalar = robot.actuators["shoulder"].effort_limit

@configclass
class AerialManipulator1DOFWristHoverEnvCfg(AerialManipulatorHoverEnvBaseCfg):
    # env
    num_actions = 5
    num_joints = 1
    num_observations = 14 
    # 3(vel) + 3(ang vel) + 3(pos) + 3(ori) + 1(joint pos) + 1(joint vel) = 14
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_1DOF_WRIST_CFG.replace(prim_path="/World/envs/env_.*/Robot")


    wrist_torque_scalar = robot.actuators["wrist"].effort_limit

@configclass
class AerialManipulator0DOFHoverEnvCfg(AerialManipulatorHoverEnvBaseCfg):
    # env
    num_actions = 4
    num_joints = 0
    num_observations = 12 # TODO: Need to update this..
    # 3(vel) + 3(ang vel) + 3(pos) + 9(ori) = 18
    # 3(vel) + 3(ang vel) + 3(pos) + 3(grav vector body frame) = 12
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_0DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

@configclass
class AerialManipulator0DOFDebugHoverEnvCfg(AerialManipulatorHoverEnvBaseCfg):
    # env
    num_actions = 4
    num_joints = 0
    num_observations = 12 # TODO: Need to update this..
    # 3(vel) + 3(ang vel) + 3(pos) + 9(ori) = 18
    # 3(vel) + 3(ang vel) + 3(pos) + 3(grav vector body frame) = 12
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_0DOF_DEBUG_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class CrazyflieHoverEnvCfg(AerialManipulatorHoverEnvBaseCfg):
    # env
    num_actions = 4
    num_joints = 0
    # num_observations = 18
    num_observations = 12

    moment_scale_xy = 0.01
    moment_scale_z = 0.01
    thrust_to_weight = 1.9

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    task_body = "vehicle"
    body_name = "body"
    has_end_effector = False
    


class AerialManipulatorHoverEnv(DirectRLEnv):
    cfg: AerialManipulatorHoverEnvBaseCfg

    def __init__(self, cfg: AerialManipulatorHoverEnvBaseCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        torch.manual_seed(self.cfg.seed)

        # Actions / Actuation interfaces
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._joint_torques = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        self._body_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._body_moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Goal State
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_ori_w = torch.zeros(self.num_envs, 4, device=self.device)

        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "endeffector_lin_vel",
                "endeffector_ang_vel",
                "endeffector_pos_error",
                "endeffector_pos_distance",
                "endeffector_ori_error",
                "endeffector_yaw_error",
                "endeffector_yaw_distance",
                "joint_vel",
                "action_norm",
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
                "joint_vel",
                "action_norm",
                "stay_alive",
                "crash_penalty"
            ]
        }

        if self.cfg.goal_cfg == "fixed":
            assert self.cfg.goal_pos is not None and self.cfg.goal_ori is not None, "Goal position and orientation must be set for fixed goal task"

        # Robot specific data
        self._body_id = self._robot.find_bodies(self.cfg.body_name)[0]
        assert len(self._body_id) == 1, "There should be only one body with the name \'vehicle\' or \'body\'"

        if self.cfg.has_end_effector:
            self._ee_id = self._robot.find_bodies("endeffector")[0] # also the root of the system

        self._com_id = self._robot.find_bodies("COM")[0]
        
        if self.cfg.num_joints > 0:
            self._shoulder_joint_idx = self._robot.find_joints(".*joint1")[0][0]
        if self.cfg.num_joints > 1:
            self._wrist_joint_idx = self._robot.find_joints(".*joint2")[0][0]
        self._total_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self.total_mass = self._total_mass
        self.quad_inertia = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).squeeze()
        # self.quad_inertia = self._robot.root_physx_view.get_inertias()[0, self._com_id, :].view(-1, 3, 3).squeeze()
        self.arm_offset = self._robot.root_physx_view.get_link_transforms()[0, self._body_id,:3].squeeze() - \
                            self._robot.root_physx_view.get_link_transforms()[0, self._ee_id,:3].squeeze() 
        
        self.arm_length = torch.linalg.norm(self.arm_offset, dim=-1)
        
        # Compute position and orientation offset between the end effector and the vehicle
        quad_pos = self._robot.data.body_pos_w[0, self._body_id]
        quad_ori = self._robot.data.body_quat_w[0, self._body_id]

        # print("Vehicle Pos: ", self._robot.data.body_pos_w[0, self._robot.find_bodies("vehicle")[0]])
        # print("Virtual Vehicle Pos: ", self._robot.data.body_pos_w[0, self._robot.find_bodies("virtualvehicle")[0]])



        ee_pos = self._robot.data.body_pos_w[0, self._ee_id]
        ee_ori = self._robot.data.body_quat_w[0, self._ee_id]

        com_pos = self._robot.data.body_pos_w[0, self._com_id]
        com_ori = self._robot.data.body_quat_w[0, self._com_id]


        # get center of mass of whole system (vehicle + end effector)
        self.vehicle_mass = self._robot.root_physx_view.get_masses()[0, self._body_id].sum()
        self.arm_mass = self._total_mass - self.vehicle_mass

        self.com_pos_w = torch.zeros(1, 3, device=self.device)
        for i in range(self._robot.num_bodies):
            self.com_pos_w += self._robot.root_physx_view.get_masses()[0, i] * self._robot.root_physx_view.get_link_transforms()[0, i, :3].squeeze()
        self.com_pos_w /= self._robot.root_physx_view.get_masses()[0].sum()

        # self.com_pos_e, self.com_ori_e = subtract_frame_transforms(ee_pos, ee_ori, self.com_pos_w, quad_ori)
        self.com_pos_e, self.com_ori_e = subtract_frame_transforms(ee_pos, ee_ori, com_pos, com_ori)

        # print("[Isaac Init] COM Pos: ", self.com_pos_w)
        # print("[Isaac Init] COM pos EE frame: ", self.com_pos_e)


        self.position_offset = quad_pos
        # self.orientation_offset = quat_mul(quad_ori, quat_conjugate(ee_ori))
        self.orientation_offset = quad_ori


        self._gravity_magnitude = torch.tensor(self.cfg.sim.gravity, device=self.device).norm()
        self._robot_weight = (self._total_mass * self._gravity_magnitude).item()
        self._grav_vector_unit = torch.tensor([0.0, 0.0, -1.0], device=self.device).tile((self.num_envs, 1))

        # Visualization marker data
        self._frame_positions = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._frame_orientations = torch.zeros(self.num_envs, 2, 4, device=self.device)

        self.local_num_envs = self.num_envs

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0) # clamp the actions to [-1, 1]

        # Need to compute joint torques, body forces, and body moments
        # TODO: Implement pre-physics step

        # CTBM + Joint Torques Model
        # Action[0] = Collective Thrust
        # Action[1] = Body X moment
        # Action[2] = Body Y moment
        # Action[3] = Body Z moment
        # Action[4] = Joint 1 Torque if joint exists
        # Action[5] = Joint 2 Torque if joint exists
        self._body_forces[:, 0, 2] = ((self._actions[:, 0] + 1.0) / 2.0) * (self._robot_weight * self.cfg.thrust_to_weight)
        self._body_moment[:, 0, :2] = self._actions[:, 1:3] * self.cfg.moment_scale_xy
        self._body_moment[:, 0, 2] = self._actions[:, 3] * self.cfg.moment_scale_z

        # print("Body Forces: ", self._body_forces)
        # print("Body Moments: ", self._body_moment)
        if self.cfg.num_joints > 0:
            self._joint_torques[:, self._shoulder_joint_idx] = self._actions[:, 4] * self.cfg.shoulder_torque_scalar
        if self.cfg.num_joints > 1:
            self._joint_torques[:, self._wrist_joint_idx] = self._actions[:, 5] * self.cfg.wrist_torque_scalar
            # self._joint_torques[:, self._wrist_joint_idx] = 0.0 # Turn off wrist joint for now


    def _apply_action(self):
        """
        Apply the torques directly to the joints based on the actions.
        Apply the propellor forces/moments to the vehicle body.
        """
        if self.cfg.num_joints > 0:
            self._robot.set_joint_effort_target(self._joint_torques[:,self._shoulder_joint_idx], joint_ids=self._shoulder_joint_idx)
        if self.cfg.num_joints > 1:
            self._robot.set_joint_effort_target(self._joint_torques[:,self._wrist_joint_idx], joint_ids=self._wrist_joint_idx)

        self._robot.set_external_force_and_torque(self._body_forces, self._body_moment, body_ids=self._body_id)

    def _apply_curriculum(self, total_timesteps):
        """
        Apply the curriculum to the environment.
        """
        # print("[Isaac Env: Curriculum] Total Timesteps: ", total_timesteps, " Pos Radius: ", self.cfg.pos_radius)
        if self.cfg.pos_radius_curriculum > 0:
            # half the pos radius every pos_radius_curriculum timesteps
            self.cfg.pos_radius = 0.8 * (0.5 ** (total_timesteps // self.cfg.pos_radius_curriculum))

    def _get_observations(self) -> torch.Dict[str, torch.Tensor | torch.Dict[str, torch.Tensor]]:
        """
        Returns the observation dictionary. Policy observations are in the key "policy".
        """
        self._apply_curriculum(self.common_step_counter * self.num_envs)
        
        
        base_pos_w, base_ori_w, lin_vel_w, ang_vel_w = self.get_frame_state_from_task(self.cfg.task_body)
        goal_pos_w, goal_ori_w = self.get_goal_state_from_task(self.cfg.goal_body)


        # Find the error of the end-effector to the desired position and orientation
        # The root state of the robot is the end-effector frame in this case
        # Batched over number of environments, returns (num_envs, 3) and (num_envs, 4) tensors
        # pos_error_b, ori_error_b = subtract_frame_transforms(self._desired_pos_w, self._desired_ori_w, 
        #                                                      base_pos, base_ori)
        pos_error_b, ori_error_b = subtract_frame_transforms(
            base_pos_w, base_ori_w, 
            # self._desired_pos_w, self._desired_ori_w
            goal_pos_w, goal_ori_w
        )

        # Compute the orientation error as a yaw error in the body frame
        # goal_yaw_w = yaw_quat(self._desired_ori_w)
        goal_yaw_w = yaw_quat(goal_ori_w)
        current_yaw_w = yaw_quat(base_ori_w)
        # yaw_error_w = quat_mul(quat_inv(current_yaw_w), goal_yaw_w)
        yaw_error_w = yaw_error_from_quats(current_yaw_w, goal_yaw_w, dof=self.cfg.num_joints).view(self.num_envs, 1)
        
        if self.cfg.use_yaw_representation:
            yaw_representation = yaw_error_w
        else:
            yaw_representation = torch.zeros(self.num_envs, 0, device=self.device)
        

        if self.cfg.use_full_ori_matrix:
            ori_representation_b = matrix_from_quat(ori_error_b).flatten(-2, -1)
        else:
            ori_representation_b = torch.zeros(self.num_envs, 0, device=self.device)

        if self.cfg.use_grav_vector:
            grav_vector_b = quat_rotate_inverse(base_ori_w, self._grav_vector_unit) # projected gravity vector in the cfg frame
        else:
            grav_vector_b = torch.zeros(self.num_envs, 0, device=self.device)
        
        # Compute the linear and angular velocities of the end-effector in body frame
        lin_vel_b = quat_rotate_inverse(base_ori_w, lin_vel_w)
        ang_vel_b = quat_rotate_inverse(base_ori_w, ang_vel_w)

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
            ],
            dim=-1                                          # (num_envs, 22)
        )

        
        
        # We also need the state information for other controllers like the decoupled controller.
        # This is the full state of the robot
        # print("[Isaac Env: Observations] \"Frame\" Pos: ", base_pos_w)
        # quad_pos_w, quad_ori_w, quad_lin_vel_w, quad_ang_vel_w = self.get_frame_state_from_task("vehicle")
        quad_pos_w, quad_ori_w, quad_lin_vel_w, quad_ang_vel_w = self.get_frame_state_from_task("COM")
        ee_pos_w, ee_ori_w, ee_lin_vel_w, ee_ang_vel_w = self.get_frame_state_from_task("root")
        # print("[Isaac Env: Observations] Quad pos: ", quad_pos_w)
        # print("[Isaac Env: Observations] EE pos: ", ee_pos_w)

        if self.cfg.gc_mode:
            gc_obs = torch.cat(
                [
                    quad_pos_w,
                    quad_ori_w,
                    quad_lin_vel_w,
                    quad_ang_vel_w,
                    goal_pos_w,
                    yaw_from_quat(goal_ori_w).unsqueeze(1),
                ],
                dim=-1
            )
        else:
            gc_obs = None

        if self.cfg.eval_mode:
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
                ],
                dim=-1                                          # (num_envs, 18)
            )
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
        pos_distance = torch.exp(- (pos_error **2) / self.cfg.pos_radius)

        ori_error = quat_error_magnitude(goal_ori_w, base_ori_w)
        
        goal_yaw_w = yaw_quat(goal_ori_w)
        current_yaw_w = yaw_quat(base_ori_w)
        # yaw_error_w = quat_mul(quat_inv(current_yaw_w), goal_yaw_w)
        # yaw_error = quat_error_magnitude(yaw_error_w, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).tile((self.num_envs, 1)))

        smooth_transition_func = 1.0 - torch.exp(-1.0 / torch.max(self.cfg.yaw_smooth_transition_scale*pos_error - 10.0, torch.zeros_like(pos_error)))

        # other_yaw_error = yaw_error_from_quats(goal_yaw_w, current_yaw_w, self.cfg.num_joints).unsqueeze(1)
        yaw_error = yaw_error_from_quats(goal_ori_w, base_ori_w, self.cfg.num_joints).unsqueeze(1)
        # other_yaw_error = torch.sum(torch.square(other_yaw_error), dim=1)
        yaw_error = torch.linalg.norm(yaw_error, dim=1)

        # yaw_distance = (1.0 - torch.tanh(yaw_error / self.cfg.yaw_radius)) * smooth_transition_func
        yaw_distance = torch.exp(- (yaw_error **2) / self.cfg.yaw_radius)
        yaw_error = yaw_error * smooth_transition_func

        combined_error = (pos_error)**2 + (yaw_error * self.arm_length)**2
        combined_distance = torch.exp(- (combined_error) / self.cfg.pos_radius)

        # Velocity error components, used for stabliization tuning
        lin_vel_b = quat_rotate_inverse(base_ori_w, lin_vel_w)
        ang_vel_b = quat_rotate_inverse(base_ori_w, ang_vel_w)
        # lin_vel_error = torch.linalg.norm(lin_vel_b, dim=-1)
        # ang_vel_error = torch.linalg.norm(ang_vel_b, dim=-1)
        # lin_vel_error = torch.sum(torch.square(lin_vel_b), dim=1)
        lin_vel_error = torch.norm(lin_vel_b, dim=1)
        # ang_vel_error = torch.sum(torch.square(ang_vel_b), dim=1)
        ang_vel_error = torch.norm(ang_vel_b, dim=1)
        # if self.cfg.num_joints == 0:
        #     joint_vel_error = torch.zeros(1, device=self.device)
        # elif self.cfg.num_joints > 1:
        #     joint_vel_error = torch.linalg.norm(self._robot.data.joint_vel[self._wrist_joint_idx:self._shoulder_joint_idx], dim=-1)
        # elif self.cfg.num_joints > 0:
        #     joint_vel_error = torch.linalg.norm(self._robot.data.joint_vel[self._shoulder_joint_idx], dim=-1)
        # joint_vel_error = torch.linalg.norm(self._robot.data.joint_vel, dim=-1)
        # joint_vel_error = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        joint_vel_error = torch.norm(self._robot.data.joint_vel, dim=1)

        # action_error = torch.sum(torch.square(self._actions), dim=1) 
        action_error = torch.norm(self._actions, dim=1)

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
            combined_distance = combined_distance ** 2

        crash_penalty_time = self.cfg.crash_penalty * (self.max_episode_length - self.episode_length_buf)


        rewards = {
            "endeffector_pos_error": pos_error * self.cfg.pos_error_reward_scale * time_scale,
            "endeffector_pos_distance": pos_distance * self.cfg.pos_distance_reward_scale * time_scale,
            "endeffector_ori_error": ori_error * self.cfg.ori_error_reward_scale * time_scale,
            "endeffector_yaw_error": yaw_error * self.arm_length * self.cfg.yaw_error_reward_scale * time_scale,
            "endeffector_yaw_distance": yaw_distance * self.cfg.yaw_distance_reward_scale * time_scale,
            "endeffector_lin_vel": lin_vel_error * self.cfg.lin_vel_reward_scale * time_scale,
            "endeffector_ang_vel": ang_vel_error * self.arm_length * self.cfg.ang_vel_reward_scale * time_scale,
            # "joint_vel": joint_vel_error * self.cfg.joint_vel_reward_scale * time_scale,
            "joint_vel": combined_distance * self.cfg.joint_vel_reward_scale * time_scale,
            "action_norm": action_error * self.cfg.action_norm_reward_scale * time_scale,
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
            "joint_vel": joint_vel_error,
            "action_norm": action_error,
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
        final_ori_error_to_goal = quat_error_magnitude(self._desired_ori_w[env_ids], base_ori_w[env_ids]).mean()
        final_yaw_error_to_goal = quat_error_magnitude(yaw_quat(self._desired_ori_w[env_ids]), yaw_quat(base_ori_w[env_ids])).mean()
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
        
        # Sample new goal position and orientation
        if self.cfg.goal_cfg == "rand":
            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-self.cfg.goal_pos_range, self.cfg.goal_pos_range)
            # self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2])
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            # self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(3 - self.cfg.goal_pos_range, 3 + self.cfg.goal_pos_range)

            # print("[DEBUG] Desired Pos: ", self._desired_pos_w[env_ids])

            if self.cfg.num_joints > 0:
                self._desired_ori_w[env_ids] = random_orientation(env_ids.size(0), device=self.device)
            else:
                random_yaw = torch.zeros_like(self._desired_ori_w[env_ids, 0]).uniform_(-self.cfg.goal_yaw_range, self.cfg.goal_yaw_range)
                self._desired_ori_w[env_ids] = quat_from_yaw(random_yaw)
        elif self.cfg.goal_cfg == "fixed":
            self._desired_pos_w[env_ids] = torch.tensor(self.cfg.goal_pos, device=self.device).tile((env_ids.size(0), 1))
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_ori_w[env_ids] = torch.tensor(self.cfg.goal_ori, device=self.device).tile((env_ids.size(0), 1))
        elif self.cfg.goal_cfg == "initial":
            default_root_state = self._robot.data.default_root_state[env_ids]
            self._desired_pos_w[env_ids] = default_root_state[:, :3]
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_ori_w[env_ids] = default_root_state[:, 3:7]
        else:
            raise ValueError("Invalid goal task: ", self.cfg.goal_cfg)

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

        if self.cfg.init_cfg == "rand":
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :2] = torch.zeros_like(default_root_state[:, :2]).uniform_(-2.0, 2.0)
            default_root_state[:, 2] = torch.zeros_like(default_root_state[:, 2]).uniform_(0.5, 1.5)
        elif self.cfg.init_cfg == "fixed":
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] = torch.tensor([0.0, 0.0, 0.5], device=self.device, requires_grad=False).float().tile((env_ids.size(0), 1))
            default_root_state[:, 3:7] = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], device=self.device, requires_grad=False).float().tile((env_ids.size(0), 1))
        else:
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:,2] = 3.0 * torch.ones_like(default_root_state[:, 2])
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        # if self.cfg.num_joints > 0:
        #     default_root_state[:, 3:7] = torch.tensor([0.5, -0.5, -0.5, 0.5], device=self.device, requires_grad=False).float().tile((env_ids.size(0), 1))
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    
    def get_frame_state_w(self, frame_name:str):
        frame_id = self._robot.find_bodies(frame_name)[0]
        return self._robot.data.body_state_w[:, frame_id]

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
        elif task_body == "vehicle":
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
        else:
            raise ValueError("Invalid goal body: ", self.cfg.goal_body)

        return goal_pos_w, goal_ori_w

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

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "frame_visualizer"):
                marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Markers",
                                        markers={
                                        "frame": sim_utils.UsdFileCfg(
                                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                                            scale=(0.1, 0.1, 0.1),
                                        ),})
                self.frame_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "frame_visualizer"):
                self.frame_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        com_pos_w, com_ori_w = combine_frame_transforms(self._robot.data.root_pos_w, self._robot.data.root_quat_w, self.com_pos_e.tile(self.local_num_envs, 1), self.com_ori_e.tile(self.local_num_envs, 1))

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