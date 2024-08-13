# These imports need to go first
from omni.isaac.lab.app import AppLauncher
import argparse # Used for Isaac
parser = argparse.ArgumentParser(description="Run demo with Isaac Sim")
parser.add_argument("--time_step", type=float, default=1.0 / 60.0, help="Time step for the simulation")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless=False
# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# Local imports
from configs.aerial_manip_asset import AERIAL_MANIPULATOR_CFG, AERIAL_MANIPULATOR_2DOF_CFG, AERIAL_MANIPULATOR_1DOF_CFG, AERIAL_MANIPULATOR_0DOF_CFG

# Isaac Lab/Isaac Sim imports
# import isaacsim
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_inv

from omni.isaac.lab_assets import CRAZYFLIE_CFG


import tyro # used for everything else
import torch
import numpy as np


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create the robots
    aerial_manipulator_cfg = AERIAL_MANIPULATOR_1DOF_CFG
    print("Soft Limits: ", aerial_manipulator_cfg.soft_joint_pos_limit_factor)
    aerial_manipulator_cfg.spawn.rigid_props.disable_gravity = True
    aerial_manipulator_cfg.spawn.func("/World/AerialManipulator/Robot_1", aerial_manipulator_cfg.spawn, translation=(0.0, 0.0, 0.0))

    # aerial_manipulator_cfg.actuators["shoulder"].stiffness = 0.01
    # aerial_manipulator_cfg.actuators["shoulder"].damping = 0.01
    # aerial_manipulator_cfg.actuators["wrist"].stiffness = 0.01
    # aerial_manipulator_cfg.actuators["wrist"].damping = 0.01

    # create handles for the robots
    origins = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    aerial_manipulator = Articulation(aerial_manipulator_cfg.replace(prim_path="/World/AerialManipulator/Robot.*"))

    crazyflie_cfg = CRAZYFLIE_CFG
    crazyflie_cfg.spawn.func("/World/Crazyflie/Robot_1", crazyflie_cfg.spawn, translation=(0.0, 0.0, 0.0))
    crazyflie = Articulation(CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie/Robot.*"))

    # Create Marker for visualization
    marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Markers",
                                         markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),})
    marker = VisualizationMarkers(marker_cfg)

    # scene_entities = {"aerial_manipulator": aerial_manipulator, "marker": marker,}
    scene_entities = {"aerial_manipulator": aerial_manipulator, "marker": marker, "crazyflie": crazyflie}
    return scene_entities, origins

def main():
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 1.0, 4.5], [0.0, 0.0, 5.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    aerial_manipulator = scene_entities["aerial_manipulator"]
    marker_frame = scene_entities["marker"]
    crazyflie = scene_entities["crazyflie"]

    marker_location = torch.tensor([[0.0, 0.0, 0.5]], device=sim.device)
    marker_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    joint1_fixed_pos = torch.pi/2.0
    joint2_fixed_pos = -torch.pi/4.0


    aerial_manipulator_body_id = aerial_manipulator.find_bodies("vehicle.*")[0]
    aerial_manipulator_ee_id = aerial_manipulator.find_bodies("endeffector")[0]
    desired_joint1_pos = None
    desired_joint2_pos = None
    desired_joint_vel = None
    if "joint1" in aerial_manipulator.joint_names:
        aerial_manipulator_joint1_id = aerial_manipulator.find_joints(".*joint1.*")[0]
        desired_joint1_pos = torch.tensor([joint1_fixed_pos], device=sim.device).float()
        desired_joint_vel = torch.tensor([0.0], device=sim.device).float()
        
        aerial_manipulator.write_joint_state_to_sim(desired_joint1_pos, desired_joint_vel, joint_ids=[aerial_manipulator_joint1_id])
    
    if "joint2" in aerial_manipulator.joint_names:
        aerial_manipulator_joint2_id = aerial_manipulator.find_joints(".*joint2.*")[0]
        desired_joint2_pos = torch.tensor([joint2_fixed_pos], device=sim.device).float()
        desired_joint_vel = torch.tensor([0.0], device=sim.device).float()
        
        aerial_manipulator.write_joint_state_to_sim(desired_joint2_pos, desired_joint_vel, joint_ids=[aerial_manipulator_joint2_id])

    aerial_manipulator_mass = aerial_manipulator.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    print("Aerial Manipulator Mass: ", aerial_manipulator_mass)

    

    joint1_pos_desired = torch.tensor([joint1_fixed_pos], device=sim.device).float()
    joint2_pos_desired = torch.tensor([joint2_fixed_pos], device=sim.device).float()

    print(aerial_manipulator.joint_names)



    import code; code.interact(local=locals())

    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            sim_time = 0.0
            aerial_manipulator.reset()
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            # reset dof state
            joint_pos, joint_vel = aerial_manipulator.data.default_joint_pos, aerial_manipulator.data.default_joint_vel
            # aerial_manipulator.write_joint_state_to_sim(joint_pos, joint_vel)

            if "joint1" in aerial_manipulator.joint_names:
                aerial_manipulator.write_joint_state_to_sim(desired_joint1_pos, desired_joint_vel, joint_ids=[aerial_manipulator_joint1_id])
            
            if "joint2" in aerial_manipulator.joint_names:
                aerial_manipulator.write_joint_state_to_sim(desired_joint2_pos, desired_joint_vel, joint_ids=[aerial_manipulator_joint2_id])

            # Set body to be hover orientation
            aerial_manipulator_body_state = aerial_manipulator.data.body_state_w[0, aerial_manipulator_body_id].squeeze()
            print("Body State: ", aerial_manipulator_body_state)
            default_root_state = aerial_manipulator.data.default_root_state
            default_root_state[:, 3:7] = quat_inv(aerial_manipulator_body_state[3:7])
            new_root_state = default_root_state[:, :7]
            print("New Root State: ", new_root_state)

            aerial_manipulator.write_root_pose_to_sim(default_root_state[:, :7])
            aerial_manipulator.write_root_velocity_to_sim(aerial_manipulator.data.default_root_state[:, 7:])
            
            print("[INFO]: Resetting Aerial manipulator: ", aerial_manipulator.data.default_root_state)

            # desired_joint_pos = torch.tensor([np.random.uniform(-3, 3), np.random.uniform(-3, 3)], device=sim.device).float()
            # desired_joint_vel = torch.tensor([0.0, 0.0], device=sim.device).float()
            # desired_joint_effort = torch.tensor([0.0, 0.0], device=sim.device).float()

            if "joint1" in aerial_manipulator.joint_names:
                aerial_manipulator.set_joint_position_target(joint1_pos_desired, joint_ids=[aerial_manipulator_joint1_id])
            
            if "joint2" in aerial_manipulator.joint_names:
                aerial_manipulator.set_joint_position_target(joint2_pos_desired, joint_ids=[aerial_manipulator_joint2_id])
            # aerial_manipulator.set_joint_position_target(desired_joint_pos, joint_ids=aerial_manipulator_joints)

            # aerial_manipulator.set_joint_velocity_target(desired_joint_vel, joint_ids=aerial_manipulator_joints)
            # aerial_manipulator.set_joint_effort_target(desired_joint_effort, joint_ids=aerial_manipulator_joints)
            # print("Desired Joint Pos: ", desired_joint_pos)

        # Update marker
        root_pos = aerial_manipulator.data.root_pos_w
        root_ori = aerial_manipulator.data.root_quat_w
        body_pos = aerial_manipulator.data.body_state_w[0, aerial_manipulator_body_id, :3]
        body_ori = aerial_manipulator.data.body_state_w[0, aerial_manipulator_body_id, 3:7]

        marker_pos = torch.stack([root_pos, body_pos], dim=0).squeeze()
        marker_ori = torch.stack([root_ori, body_ori], dim=0).squeeze()
        marker_frame.visualize(marker_pos, marker_ori)
        
        # aerial_manipulator.write_root_pose_to_sim(aerial_manipulator.data.default_root_state[:, :7])
        # aerial_manipulator.write_root_velocity_to_sim(aerial_manipulator.data.default_root_state[:, 7:])

        
        print("\nDesired Joint Pos: ", desired_joint1_pos, " ", desired_joint2_pos, " Current Joint Pos: ", aerial_manipulator.data.joint_pos)
        print("Desired Joint Vel: ", desired_joint_vel, " Current Joint Vel: ", aerial_manipulator.data.joint_vel)
        # aerial_manipulator.set_joint_position_target(desired_joint_pos, joint_ids=aerial_manipulator_joints)
        # aerial_manipulator.set_joint_velocity_target(desired_joint_vel, joint_ids=aerial_manipulator_joints)
        # aerial_manipulator.set_joint_effort_target(desired_joint_effort, joint_ids=aerial_manipulator_joints)
        
        print("Joint Torques Computed: ", aerial_manipulator.data.computed_torque)
        print("Joint Torques Applied: ", aerial_manipulator.data.applied_torque)
        # aerial_manipulator.write_data_to_sim()
        # aerial_manipulator.update(sim_dt)

        # Apply External Forces
        forces = torch.zeros(1, 1, 3, device=sim.device)
        torques = torch.zeros_like(forces)
        # forces[..., 2] = aerial_manipulator_mass * gravity
        aerial_manipulator.set_external_force_and_torque(forces, torques, body_ids=aerial_manipulator_body_id)
        aerial_manipulator.write_data_to_sim()

        # Perform step
        sim.step()
        # Increment counter
        count += 1
        sim_time += sim_dt
        # Update buffers
        aerial_manipulator.update(sim_dt)


if __name__ == "__main__":
    main()
    