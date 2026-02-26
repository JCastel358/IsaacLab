# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg


@configclass
class UnitreeGo2RobustPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    max_iterations = 1500
    experiment_name = "unitree_go2_robust"


@configclass
class UnitreeGo2RobustFlatFinetunePPORunnerCfg(UnitreeGo2RobustPPORunnerCfg):
    max_iterations = 600
