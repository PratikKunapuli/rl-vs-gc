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
        # usd_path=f"{MODELS_PATH}/aerial_manipulator_2dof.usd",
        usd_path=f"{MODELS_PATH}/uam_2dof.usd",
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
            # "joint1": 0.0,
            # "joint2": 0.0,
            "joint_wrist": 0.0,
            "joint_shoulder": 0.0,
        },
    ),

    # Available joints: joint1, joint2
    actuators={ 
        "shoulder": IdealPDActuatorCfg( # Stiffness, damping, armature, friction need to be set. 
            # joint_names_expr=["joint1"],
            joint_names_expr=["joint_shoulder"],
            effort_limit=0.6,
            velocity_limit=float(1e5),
            stiffness=0.0,
            damping=0.0,
            armature=0.0,
            friction=0.0,
        ),
        "wrist": IdealPDActuatorCfg( # Stiffness, damping, armature, friction need to be set. 
            # joint_names_expr=["joint2"],
            joint_names_expr=["joint_wrist"],
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
"""Configuration for the Aerial Manipulator 0DOF."""

AERIAL_MANIPULATOR_0DOF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{MODELS_PATH}/aerial_manipulator_0dof.usd",
        usd_path=f"{MODELS_PATH}/uam_0dof_com_middle.usd",
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
        usd_path=f"{MODELS_PATH}/uam_0dof_com_middle.usd",
        # usd_path=f"{MODELS_PATH}/aerial_manipulator_0dof_debug.usd",
        # usd_path=f"{MODELS_PATH}/aerial_manipulator_0dof.usd",
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


"""
Configurations for the 0-DOF Aerial Manipulator with shorter arm, varied COM locations along the arm
"""
AERIAL_MANIPULATOR_0DOF_SMALL_ARM_COM_V_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/uam_0dof_small_arm_com_v.usd",
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

AERIAL_MANIPULATOR_0DOF_SMALL_ARM_COM_MIDDLE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/uam_0dof_small_arm_com_middle.usd",
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

AERIAL_MANIPULATOR_0DOF_SMALL_ARM_COM_EE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/uam_0dof_small_arm_com_ee.usd",
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

"""
Aerial Manipulator 0DOF Long Arm
"""
AERIAL_MANIPULATOR_0DOF_LONG_ARM_COM_MIDDLE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/uam_0dof_long_arm_com_middle.usd",
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





""" Configuration for the Aerial Manipulator quad base only"""
AERIAL_MANIPULATOR_QUAD_ONLY_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/uam_quadrotor.usd",
        # usd_path=f"{MODELS_PATH}/uam_quadrotor_new.usd",
        # usd_path=f"{MODELS_PATH}/aerial_manipulator_0dof.usd",
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

"""

Ball Catching Assets Below

"""

AERIAL_MANIPULATOR_0DOF_BALL_CATCHING_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{MODELS_PATH}/aerial_manipulator_0dof_catching.usd",
        usd_path=f"{MODELS_PATH}/uam_0dof_com_middle_catching.usd",
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

AERIAL_MANIPULATOR_0DOF_DEBUG_BALL_CATCHING_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/aerial_manipulator_0dof_debug_catching.usd",
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
            static_friction=20.0,
            dynamic_friction=20.0,
        ),
    ),
    collision_group=0,
)

# Crazyflie Manipulator
CRAZYFLIE_MANIPULATOR_0DOF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/Crazyflie_manipulator_v2.usd",
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
        pos=(0.0, -0.05, 0.5),
        # Default joint positions and velocities is 0.0
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
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

CRAZYFLIE_MANIPULATOR_0DOF_LONG_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/Crazyflie_manipulator_v2_long.usd",
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
        pos=(0.0, -0.1, 0.5),
        # Default joint positions and velocities is 0.0
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
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

# Brushless Crazyflie
CRAZYFLIE_BRUSHLESS_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODELS_PATH}/crazyflie_brushless.usd",
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
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
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