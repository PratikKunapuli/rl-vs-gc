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
from omni.isaac.lab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_error_magnitude, random_orientation, quat_inv, quat_rotate_inverse, quat_mul, yaw_quat
from omni.isaac.lab_assets import CRAZYFLIE_CFG

# Local imports
from configs.aerial_manip_asset import AERIAL_MANIPULATOR_2DOF_CFG, AERIAL_MANIPULATOR_1DOF_CFG, AERIAL_MANIPULATOR_1DOF_WRIST_CFG, AERIAL_MANIPULATOR_0DOF_CFG


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
    sim_rate_hz = 500
    policy_rate_hz = 100
    decimation = sim_rate_hz // policy_rate_hz
    ui_window_class_type = AerialManipulatorEnvWindow
    num_states = 0
    debug_vis = True

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

    # action scaling
    # moment_scale_xy = 1.0
    # moment_scale_z = 0.05
    # thrust_to_weight = 3.0
    moment_scale_xy = 0.5
    moment_scale_z = 0.025
    thrust_to_weight = 3.0

    # reward scales
    pos_radius = 0.8
    lin_vel_reward_scale = -0.05 # -0.05
    ang_vel_reward_scale = -0.01 # -0.01
    pos_distance_reward_scale = 15.0 #15.0
    pos_error_reward_scale = 0.0# -1.0
    ori_error_reward_scale = 0.0 # -0.5
    joint_vel_reward_scale = 0.0 # -0.01
    action_norm_reward_scale = 0.0 # -0.01

    # Task condionionals for the environment - modifies the goal
    goal_cfg = "rand" # "rand", "fixed", or "initial"
    # "rand" - Random goal position and orientation
    # "fixed" - Fixed goal position and orientation set apriori
    # "initial" - Goal position and orientation is the initial position and orientation of the robot
    goal_pos = None
    goal_ori = None

    init_cfg = "default" # "default" or "rand"

    task_body = "root" # "root" or "endeffector" or "vehicle"
    body_name = "vehicle"
    has_end_effector = True
    use_full_ori_matrix = False
    use_yaw_representation = False

    shoulder_joint_active = True
    wrist_joint_active = True


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
                "joint_vel",
                "action_norm"
            ]
        }

        self._episode_error_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "pos_error",
                "pos_distance",
                "ori_error",
                "lin_vel",
                "ang_vel",
                "joint_vel",
                "action_norm"
            ]
        }

        if self.cfg.goal_cfg == "fixed":
            assert self.cfg.goal_pos is not None and self.cfg.goal_ori is not None, "Goal position and orientation must be set for fixed goal task"

        # Robot specific data
        self._body_id = self._robot.find_bodies(self.cfg.body_name)[0]
        assert len(self._body_id) == 1, "There should be only one body with the name \'vehicle\' or \'body\'"

        if self.cfg.has_end_effector:
            self._ee_id = self._robot.find_bodies("endeffector")[0] # also the root of the system

        
        if self.cfg.num_joints > 0:
            self._shoulder_joint_idx = self._robot.find_joints(".*joint1")[0][0]
        if self.cfg.num_joints > 1:
            self._wrist_joint_idx = self._robot.find_joints(".*joint2")[0][0]
        self._total_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.cfg.sim.gravity, device=self.device).norm()
        self._robot_weight = (self._total_mass * self._gravity_magnitude).item()
        self._grav_vector_unit = torch.tensor([0.0, 0.0, -1.0], device=self.device).tile((self.num_envs, 1))

        # Visualization marker data
        self._frame_positions = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._frame_orientations = torch.zeros(self.num_envs, 2, 4, device=self.device)

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

    def _get_observations(self) -> torch.Dict[str, torch.Tensor | torch.Dict[str, torch.Tensor]]:
        """
        Returns the observation dictionary. Policy observations are in the key "policy".
        """
        if self.cfg.task_body == "root":
            base_pos = self._robot.data.root_pos_w
            base_ori = self._robot.data.root_quat_w
            lin_vel_w = self._robot.data.root_lin_vel_w
            ang_vel_w = self._robot.data.root_ang_vel_w
        elif self.cfg.task_body == "endeffector":
            base_pos = self._robot.data.body_pos_w[:, self._ee_id].squeeze(1)
            base_ori = self._robot.data.body_quat_w[:, self._ee_id].squeeze(1)
            lin_vel_w = self._robot.data.root_lin_vel_w
            ang_vel_w = self._robot.data.root_ang_vel_w
        elif self.cfg.task_body == "vehicle":
            base_pos = self._robot.data.body_pos_w[:, self._body_id].squeeze(1)
            base_ori = self._robot.data.body_quat_w[:, self._body_id].squeeze(1)
            lin_vel_w = self._robot.data.body_lin_vel_w[:, self._body_id].squeeze(1)
            ang_vel_w = self._robot.data.body_ang_vel_w[:, self._body_id].squeeze(1)
        else:
            raise ValueError("Invalid task body: ", self.cfg.task_body)


        # Find the error of the end-effector to the desired position and orientation
        # The root state of the robot is the end-effector frame in this case
        # Batched over number of environments, returns (num_envs, 3) and (num_envs, 4) tensors
        # pos_error_b, ori_error_b = subtract_frame_transforms(self._desired_pos_w, self._desired_ori_w, 
        #                                                      base_pos, base_ori)
        pos_error_b, ori_error_b = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], 
            self._desired_pos_w, self._desired_ori_w
        )

        # Compute the orientation error as a yaw error in the body frame
        goal_yaw_w = yaw_quat(self._desired_ori_w)
        current_yaw_w = yaw_quat(base_ori)
        yaw_error_w = quat_mul(quat_inv(current_yaw_w), goal_yaw_w)

        if self.cfg.use_yaw_representation:
            yaw_representation = yaw_error_w
        else:
            yaw_representation = torch.zeros(self.num_envs, 0, device=self.device)
        

        if self.cfg.use_full_ori_matrix:
            ori_representation_b = matrix_from_quat(ori_error_b).flatten(-2, -1)
        else:
            ori_representation_b = quat_rotate_inverse(base_ori, self._grav_vector_unit) # projected gravity vector in the cfg frame
        # Compute the linear and angular velocities of the end-effector in body frame
        lin_vel_b = quat_rotate_inverse(base_ori, lin_vel_w)
        ang_vel_b = quat_rotate_inverse(base_ori, ang_vel_w)

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
                ori_representation_b,                       # (num_envs, 3) if not using full ori matrix, (num_envs, 9) if using full ori matrix
                yaw_representation,                         # (num_envs, 4) if using yaw representation (quat), 0 otherwise
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
        full_state = torch.cat(
            [
                base_pos,                                   # (num_envs, 3)
                base_ori,                                   # (num_envs, 4)
                lin_vel_w,                                  # (num_envs, 3)
                ang_vel_w,                                  # (num_envs, 3)
                shoulder_joint_pos,                         # (num_envs, 1)
                wrist_joint_pos,                            # (num_envs, 1)
                shoulder_joint_vel,                         # (num_envs, 1)
                wrist_joint_vel,                            # (num_envs, 1)
            ],
            dim=-1                                          # (num_envs, 18)
        )

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """
        Returns the reward tensor.
        """
        if self.cfg.task_body == "root":
            base_pos = self._robot.data.root_pos_w
            base_ori = self._robot.data.root_quat_w
            lin_vel_w = self._robot.data.root_lin_vel_w
            ang_vel_w = self._robot.data.root_ang_vel_w
        elif self.cfg.task_body == "endeffector":
            base_pos = self._robot.data.body_pos_w[:, self._ee_id].squeeze(1)
            base_ori = self._robot.data.body_quat_w[:, self._ee_id].squeeze(1)
            lin_vel_w = self._robot.data.root_lin_vel_w
            ang_vel_w = self._robot.data.root_ang_vel_w
        elif self.cfg.task_body == "vehicle":
            base_pos = self._robot.data.body_pos_w[:, self._body_id].squeeze(1)
            base_ori = self._robot.data.body_quat_w[:, self._body_id].squeeze(1)
            lin_vel_w = self._robot.data.body_lin_vel_w[:, self._body_id].squeeze(1)
            ang_vel_w = self._robot.data.body_ang_vel_w[:, self._body_id].squeeze(1)
        else:
            raise ValueError("Invalid task body: ", self.cfg.task_body)
        
        # Computes the error from the desired position and orientation
        pos_error = torch.linalg.norm(self._desired_pos_w - base_pos, dim=1)
        ori_error = quat_error_magnitude(self._desired_ori_w, base_ori)
        pos_distance = 1.0 - torch.tanh(pos_error / self.cfg.pos_radius)

        # Velocity error components, used for stabliization tuning
        lin_vel_b = quat_rotate_inverse(base_ori, lin_vel_w)
        ang_vel_b = quat_rotate_inverse(base_ori, ang_vel_w)
        # lin_vel_error = torch.linalg.norm(lin_vel_b, dim=-1)
        # ang_vel_error = torch.linalg.norm(ang_vel_b, dim=-1)
        lin_vel_error = torch.sum(torch.square(lin_vel_b), dim=1)
        ang_vel_error = torch.sum(torch.square(ang_vel_b), dim=1)
        # if self.cfg.num_joints == 0:
        #     joint_vel_error = torch.zeros(1, device=self.device)
        # elif self.cfg.num_joints > 1:
        #     joint_vel_error = torch.linalg.norm(self._robot.data.joint_vel[self._wrist_joint_idx:self._shoulder_joint_idx], dim=-1)
        # elif self.cfg.num_joints > 0:
        #     joint_vel_error = torch.linalg.norm(self._robot.data.joint_vel[self._shoulder_joint_idx], dim=-1)
        # joint_vel_error = torch.linalg.norm(self._robot.data.joint_vel, dim=-1)
        joint_vel_error = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)

        action_error = torch.sum(torch.square(self._actions), dim=1) 

        # rewards = {
        #     "endeffector_pos_error": pos_error * self.cfg.pos_error_reward_scale * self.step_dt,
        #     "endeffector_pos_distance": pos_distance * self.cfg.pos_distance_reward_scale * self.step_dt,
        #     "endeffector_ori_error": ori_error * self.cfg.ori_error_reward_scale * self.step_dt,
        #     "endeffector_lin_vel": lin_vel_error * self.cfg.lin_vel_reward_scale * self.step_dt,
        #     "endeffector_ang_vel": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
        #     "joint_vel": joint_vel_error * self.cfg.joint_vel_reward_scale * self.step_dt,
        # }
        rewards = {
            "endeffector_pos_error": pos_error * self.cfg.pos_error_reward_scale,
            "endeffector_pos_distance": pos_distance * self.cfg.pos_distance_reward_scale,
            "endeffector_ori_error": ori_error * self.cfg.ori_error_reward_scale,
            "endeffector_lin_vel": lin_vel_error * self.cfg.lin_vel_reward_scale,
            "endeffector_ang_vel": ang_vel_error * self.cfg.ang_vel_reward_scale,
            "joint_vel": joint_vel_error * self.cfg.joint_vel_reward_scale,
            "action_norm": action_error * self.cfg.action_norm_reward_scale,
        }
        errors = {
            "pos_error": pos_error,
            "pos_distance": pos_distance,
            "ori_error": ori_error,
            "lin_vel": lin_vel_error,
            "ang_vel": ang_vel_error,
            "joint_vel": joint_vel_error,
            "action_norm": action_error,
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

        # Logging the episode sums
        final_distance_to_goal = torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1).mean()
        final_ori_error_to_goal = quat_error_magnitude(self._desired_ori_w[env_ids], self._robot.data.root_quat_w[env_ids]).mean()
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
        extras["Episode Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        # Sample new goal position and orientation
        if self.cfg.goal_cfg == "rand":
            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
            self._desired_ori_w[env_ids] = random_orientation(env_ids.size(0), device=self.device)
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
        if self.cfg.num_joints > 0:
            # print("Resetting shoulder joint to pi/2")
            shoulder_joint_pos = torch.tensor(torch.pi/2, device=self.device, requires_grad=False).float()
            shoulder_joint_vel = torch.tensor(0.0, device=self.device, requires_grad=False).float()
            self._robot.write_joint_state_to_sim(shoulder_joint_pos, shoulder_joint_vel, joint_ids=self._shoulder_joint_idx, env_ids=env_ids)

        if self.cfg.init_cfg == "rand":
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :2] = torch.zeros_like(default_root_state[:, :2]).uniform_(-2.0, 2.0)
            default_root_state[:, 2] = torch.zeros_like(default_root_state[:, 2]).uniform_(0.5, 1.5)
        else:
            default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        if self.cfg.num_joints > 0:
            default_root_state[:, 3:7] = torch.tensor([0.5, -0.5, -0.5, 0.5], device=self.device, requires_grad=False).float().tile((env_ids.size(0), 1))
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    

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
        # update the markers
        # Update frame positions for debug visualization
        self._frame_positions[:, 0] = self._robot.data.root_pos_w
        self._frame_positions[:, 1] = self._desired_pos_w
        self._frame_orientations[:, 0] = self._robot.data.root_quat_w
        self._frame_orientations[:, 1] = self._desired_ori_w
        self.frame_visualizer.visualize(self._frame_positions.flatten(0, 1), self._frame_orientations.flatten(0,1))