# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2RobustRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    """Robust sim-to-real oriented Go2 rough-terrain velocity-tracking environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # commands: direct yaw-rate commands and conservative deployment envelope
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = None

        # domain randomization: broaden contact and inertial uncertainty
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 4.0)
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)},
            },
        )
        self.events.base_external_force_torque.params["force_range"] = (-20.0, 20.0)
        self.events.base_external_force_torque.params["torque_range"] = (-3.0, 3.0)

        # reset randomization: avoid overfitting to near-static resets
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.4, 0.4),
                "z": (-0.2, 0.2),
                "roll": (-0.35, 0.35),
                "pitch": (-0.35, 0.35),
                "yaw": (-0.8, 0.8),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.reset_robot_joints.params["velocity_range"] = (-1.0, 1.0)

        # disturbances: restore pushes with bounded yaw impulse matching controller envelope
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 10.0),
            params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8), "yaw": (-1.0, 1.0)}},
        )

        # actuator / joint dynamics randomization
        self.events.robot_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
                ),
                "stiffness_distribution_params": (0.85, 1.15),
                "damping_distribution_params": (0.7, 1.3),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.robot_joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
                ),
                "friction_distribution_params": (0.0, 0.05),
                "operation": "add",
                "distribution": "uniform",
            },
        )
        self.events.robot_joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
                ),
                "armature_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # rewards: emphasize yaw tracking and stance robustness
        self.rewards.track_lin_vel_xy_exp.weight = 1.1
        self.rewards.track_ang_vel_z_exp.weight = 1.4
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.dof_torques_l2.weight = -3.0e-4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.03
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-0.5,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
        )
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.05,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )
        self.rewards.stand_still_joint_deviation_l1 = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.1,
            params={"command_name": "base_velocity", "command_threshold": 0.1},
        )


@configclass
class UnitreeGo2RobustFlatEnvCfg(UnitreeGo2RobustRoughEnvCfg):
    """Robust sim-to-real oriented Go2 flat-terrain velocity-tracking environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # stronger orientation tracking on flat terrain
        self.rewards.flat_orientation_l2.weight = -5.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class UnitreeGo2RobustRoughEnvCfg_PLAY(UnitreeGo2RobustRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random disturbances
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class UnitreeGo2RobustFlatEnvCfg_PLAY(UnitreeGo2RobustFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random disturbances
        self.events.base_external_force_torque = None
        self.events.push_robot = None
