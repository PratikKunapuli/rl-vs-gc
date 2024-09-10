from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from utils.assets import MODELS_PATH

from omni.isaac.lab.sim.spawners.shapes import SphereCfg, spawn_sphere
from omni.isaac.lab.sim.spawners.materials import VisualMaterialCfg, PreviewSurfaceCfg


AERIAL_MANIPULATOR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/aerial_manipulator.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
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

    # Available joints: joint1, joint2
    actuators={ 
        "shoulder": IdealPDActuatorCfg( # Stiffness, damping, armature, friction need to be set. 
            joint_names_expr=["joint1"],
            effort_limit=0.6,
            velocity_limit=float(1e5),
            stiffness=0.0,
            damping=0.0,
            armature=0.0,
            friction=0.0,
        ),
        "wrist": IdealPDActuatorCfg( # Stiffness, damping, armature, friction need to be set. 
            joint_names_expr=["joint2"],
            effort_limit=0.3,
            velocity_limit=float(1e5),
            stiffness=0.0,
            damping=0.0,
            armature=0.0,
            friction=0.0,
        ),
    },
)
"""Configuration for the Aerial Manipulator 2DOF."""


AERIAL_MANIPULATOR_2DOF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/aerial_manipulator_2dof.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
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
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "prop1": 0.0,
            "prop2": -0.0,
            "prop3": 0.0,
            "prop4": -0.0,
            "joint1": 0.0,
            "joint2": 0.0,
        },
    ),

    # Available joints: joint1, joint2
    actuators={ 
        "shoulder": IdealPDActuatorCfg( # Stiffness, damping, armature, friction need to be set. 
            joint_names_expr=["joint1"],
            effort_limit=0.6,
            velocity_limit=float(1e5),
            stiffness=0.0,
            damping=0.0,
            armature=0.0,
            friction=0.0,
        ),
        "wrist": IdealPDActuatorCfg( # Stiffness, damping, armature, friction need to be set. 
            joint_names_expr=["joint2"],
            effort_limit=0.3,
            velocity_limit=float(1e5),
            stiffness=0.0,
            damping=0.0,
            armature=0.0,
            friction=0.0,
        ),
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*prop.*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Aerial Manipulator 2DOF."""

AERIAL_MANIPULATOR_1DOF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/aerial_manipulator_1dof.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
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
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "prop1": 0.0,
            "prop2": -0.0,
            "prop3": 0.0,
            "prop4": -0.0,
            "joint1": 0.0,
        },
    ),

    # Available joints: joint1
    actuators={ 
        "shoulder": IdealPDActuatorCfg( # Stiffness, damping, armature, friction need to be set. 
            joint_names_expr=["joint1"],
            effort_limit=0.6,
            velocity_limit=float(1e5),
            stiffness=0.0,
            damping=0.0,
            armature=0.0,
            friction=0.0,
        ),
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*prop.*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Aerial Manipulator 1DOF."""


AERIAL_MANIPULATOR_1DOF_WRIST_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/aerial_manipulator_1dof_wrist.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
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
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "prop1": 0.0,
            "prop2": -0.0,
            "prop3": 0.0,
            "prop4": -0.0,
            "joint2": 0.0,
        },
    ),

    # Available joints: joint1
    actuators={ 
        "wrist": IdealPDActuatorCfg( # Stiffness, damping, armature, friction need to be set. 
            joint_names_expr=["joint2"],
            effort_limit=0.3,
            velocity_limit=float(1e5),
            stiffness=0.0,
            damping=0.0,
            armature=0.0,
            friction=0.0,
        ),
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*prop.*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Aerial Manipulator 1DOF."""

AERIAL_MANIPULATOR_0DOF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/aerial_manipulator_0dof.usd",
        # usd_path=f"{MODELS_PATH}/aerial_manipulator_0dof_debug.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
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
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "prop1": 0.0,
            "prop2": -0.0,
            "prop3": 0.0,
            "prop4": -0.0,
        },
    ),

    # Available joints: 
    actuators={ 
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Aerial Manipulator 0DOF."""

AERIAL_MANIPULATOR_0DOF_DEBUG_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/aerial_manipulator_0dof_debug.usd",
        # usd_path=f"{MODELS_PATH}/aerial_manipulator_debug.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
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
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "prop1": 0.0,
            "prop2": -0.0,
            "prop3": 0.0,
            "prop4": -0.0,
        },
    ),

    # Available joints: 
    actuators={ 
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


BALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn = SphereCfg(
        radius=0.04,
        visual_material=PreviewSurfaceCfg(diffuse_color = (1.0, 0.0, 0.0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.05,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.001,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            restitution=0.9,
            static_friction=0.5,
            dynamic_friction=0.5,
        ),
    ),
    collision_group=0,
)