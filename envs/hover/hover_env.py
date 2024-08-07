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
from omni.isaac.lab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_error_magnitude, random_orientation

# Local imports
from configs.aerial_manip_asset import AERIAL_MANIPULATOR_CFG

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
class AerialManipulatorHoverEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 1
    num_actions = 6
    num_joints = 2
    num_observations = 22 # TODO: Need to update this..
    num_states = 0
    debug_vis = True

    ui_window_class_type = AerialManipulatorEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
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
    robot: ArticulationCfg = AERIAL_MANIPULATOR_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # action scaling
    moment_scale = 0.01
    thrust_to_weight = 3.0
    shoulder_torque_scalar = 0.9
    wrist_torque_scalar = 0.1

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.05 # -0.01
    pos_distance_reward_scale = 15.0
    pos_error_reward_scale = -1.0
    ori_error_reward_scale = -0.5
    joint_vel_reward_scale = -0.01

class AerialManipulatorHoverEnv(DirectRLEnv):
    cfg: AerialManipulatorHoverEnvCfg

    def __init__(self, cfg: AerialManipulatorHoverEnvCfg, render_mode: str | None = None, initial_state_dict=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Actions / Actuation interfaces
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._joint_torques = torch.zeros(self.num_envs, self.cfg.num_joints, device=self.device)
        self._body_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._body_moment = torch.zeros(self.num_envs, 3, device=self.device)

        # Goal State
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_ori_w = torch.zeros(self.num_envs, 4, device=self.device)

        # State
        self.initial_state_dict = initial_state_dict

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
            ]
        }

        # Robot specific data
        self._body_id = self._robot.find_bodies("vehicle")[0]
        self._ee_id = self._robot.find_bodies("endeffector")[0] # also the root of the system
        self._joint_ids = self._robot.find_joints(".*joint.*")[0]
        self._total_mass = self._robot.root_physx_view.get_masses().sum()
        self._gravity_magnitude = torch.tensor(self.cfg.sim.gravity, device=self.device).norm()

        # Visualization marker data
        self._frame_positions = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._frame_orientations = torch.zeros(self.num_envs, 2, 4, device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0).to(self.device) # clamp the actions to [-1, 1]

        # Need to compute joint torques, body forces, and body moments
        # TODO: Implement pre-physics step

        # CTBM + Joint Torques Model
        # Action[0] = Collective Thrust
        # Action[1] = Body X moment
        # Action[2] = Body Y moment
        # Action[3] = Body Z moment
        # Action[4] = Joint 1 Torque
        # Action[5] = Joint 2 Torque
        self._body_forces[:, 2] = (self._actions[:, 0] + 1.0) / 2.0 * (self._total_mass * self._gravity_magnitude * self.cfg.thrust_to_weight)
        self._body_moment[:, :] = self._actions[:, 1:4] * self.cfg.moment_scale
        self._joint_torques[:, 0] = self._actions[:, 4] * self.cfg.shoulder_torque_scalar
        self._joint_torques[:, 1] = self._actions[:, 5] * self.cfg.wrist_torque_scalar

    def _apply_action(self):
        """
        Apply the torques directly to the joints based on the actions.
        Apply the propellor forces/moments to the body.
        """
        self._robot.set_joint_effort_target(self._joint_torques, joint_ids=self._joint_ids)
        self._robot.set_external_force_and_torque(self._body_forces, self._body_moment, body_ids=self._body_id)

    def _get_observations(self) -> torch.Dict[str, torch.Tensor | torch.Dict[str, torch.Tensor]]:
        """
        Returns the observation dictionary. Policy observations are in the key "policy".
        """

        # Find the error of the end-effector to the desired position and orientation
        # The root state of the robot is the end-effector frame in this case
        # Batched over number of environments, returns (num_envs, 3) and (num_envs, 4) tensors
        pos_error_b, ori_error_b = subtract_frame_transforms(self._desired_pos_w, self._desired_ori_w, 
                                                             self._robot.data.root_pos_w, self._robot.data.root_quat_w)
        
        # Compute the linear and angular velocities of the end-effector in body frame
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b

        # Compute the joint states
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        obs = torch.cat(
            [
                pos_error_b,                                # (num_envs, 3)
                matrix_from_quat(ori_error_b).flatten(-2),  # (num_envs, 9)
                lin_vel_b,                                  # (num_envs, 3)
                ang_vel_b,                                  # (num_envs, 3)
                joint_pos,                                  # (num_envs, num_joints) = (num_envs, 2)
                joint_vel                                   # (num_envs, num_joints) = (num_envs, 2)
            ],
            dim=-1                                          # (num_envs, 22)
        )
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """
        Returns the reward tensor.
        """
        
        # Computes the error from the desired position and orientation
        pos_error = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=-1)
        ori_error = quat_error_magnitude(self._desired_ori_w, self._robot.data.root_quat_w)
        pos_distance = 1.0 - torch.tanh(pos_error / 0.5)

        # Velocity error components, used for stabliization tuning
        lin_vel_error = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=-1)
        ang_vel_error = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=-1)
        joint_vel_error = torch.linalg.norm(self._robot.data.joint_vel, dim=-1)

        rewards = {
            "endeffector_pos_error": pos_error * self.cfg.pos_error_reward_scale * self.step_dt,
            "endeffector_pos_distance": pos_distance * self.cfg.pos_distance_reward_scale * self.step_dt,
            "endeffector_ori_error": ori_error * self.cfg.ori_error_reward_scale * self.step_dt,
            "endeffector_lin_vel": lin_vel_error * self.cfg.lin_vel_reward_scale * self.step_dt,
            "endeffector_ang_vel": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "joint_vel": joint_vel_error * self.cfg.joint_vel_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the tensors corresponding to termination and truncation. 
        """

        # Check if end effector or body has collided with the ground
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.0, self._robot.data.body_state_w[:, self._body_id, 2] < 0.0)

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
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = self._episode_sums[key][env_ids].mean()
            extras["Episode Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Metrics/Final Distance to Goal"] = final_distance_to_goal
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
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 2.0)
        self._desired_ori_w[env_ids] = random_orientation(env_ids.size(0), device=self.device)

        # Reset Robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7])
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:])

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