from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from utils.assets import MODELS_PATH


AERIAL_MANIPULATOR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/aerialmanipulator.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # Default joint positions and velocities is 0.0
    ),
    actuators={
        # "base": ImplicitActuatorCfg(
        #     joint_names_expr=[".*"],
        #     stiffness=0.0,
        #     damping=0.0,
        # ),
        "shoulder": IdealPDActuatorCfg( # Stiffness, damping, armature, friction need to be set. 
            joint_names_expr=["joint1"],
            stiffness=0.01,
            damping=0.01,
            armature=0.0,
            friction=0.0,
        ),
        "wrist": IdealPDActuatorCfg( # Stiffness, damping, armature, friction need to be set. 
            joint_names_expr=["joint2"],
            stiffness=0.01,
            damping=0.01,
            armature=0.0,
            friction=0.0,
        ),
    },
)
"""Configuration for the Aerial Manipulator."""
