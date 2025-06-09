# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Aerial Manipulator environment for hovering.
"""

import gymnasium as gym

from . import ball_catching_env
from .ball_catching_env import AerialManipulator0DOFBallCatchingEnvCfg
from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-AerialManipulator-0DOF-BallCatch-v0",
    entry_point = "envs.ball_catching.ball_catching_env:AerialManipulatorBallCatchingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFBallCatchingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)