# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Aerial Manipulator environment for hovering.
"""

import gymnasium as gym

from . import hover_env
from .hover_env import AerialManipulatorHoverEnv
from .hover_env import AerialManipulator2DOFHoverEnvCfg, AerialManipulator2DOFHoverPoseEnvCfg, AerialManipulator1DOFHoverEnvCfg, AerialManipulator0DOFHoverEnvCfg, AerialManipulator0DOFDebugHoverEnvCfg
from .hover_env import CrazyflieHoverEnvCfg
from .hover_env import AerialManipulator0DOFSmallArmCOMVehicleHoverEnvCfg, AerialManipulator0DOFSmallArmCOMMiddleHoverEnvCfg, AerialManipulator0DOFSmallArmCOMEndEffectorHoverEnvCfg
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-AerialManipulator-2DOF-Hover-v0",
    entry_point = "envs.hover.hover_env:AerialManipulatorHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator2DOFHoverEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-2DOF-HoverPose-v0",
    entry_point = "envs.hover.hover_env:AerialManipulatorHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator2DOFHoverPoseEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-1DOF-Hover-v0",
    entry_point = "envs.hover.hover_env:AerialManipulatorHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator1DOFHoverEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-1DOF-Wrist-Hover-v0",
    entry_point = "envs.hover.hover_env:AerialManipulatorHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator1DOFHoverEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-0DOF-Hover-v0",
    entry_point = "envs.hover.hover_env:AerialManipulatorHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFHoverEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-0DOF-Debug-Hover-v0",
    entry_point = "envs.hover.hover_env:AerialManipulatorHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFDebugHoverEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-0DOF-SmallArmCOM-V-Hover-v0",
    entry_point = "envs.hover.hover_env:AerialManipulatorHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFSmallArmCOMVehicleHoverEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-0DOF-SmallArmCOM-Middle-Hover-v0",
    entry_point = "envs.hover.hover_env:AerialManipulatorHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFSmallArmCOMMiddleHoverEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-0DOF-SmallArmCOM-EndEffector-Hover-v0",
    entry_point = "envs.hover.hover_env:AerialManipulatorHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFSmallArmCOMEndEffectorHoverEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)



gym.register(
    id="Isaac-Crazyflie-0DOF-Hover-v0",
    entry_point = "envs.hover.hover_env:AerialManipulatorHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrazyflieHoverEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)






# gym.register(
#     id="Isaac-AerialManipulator-Hover-Vehicle-v0",
#     entry_point = "envs.hover.hover_env_vehicle:AerialManipulatorHoverEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": AerialManipulatorHoverEnvCfgVehicle,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml"
#     },
# )