"""
Spider Standing Environment for Isaac Lab
Phase 1: Standing and Balancing with 8-legged spider robot
"""

import math
import numpy as np
import torch
from typing import Dict, Tuple
import os

# Isaac Lab imports
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_to_euler_xyz, quat_conjugate, quat_mul
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg


@configclass
class SpiderStandingEnvCfg(DirectRLEnvCfg):
    """Configuration for Spider Standing environment"""

    # Environment settings
    decimation = 2  # Control frequency = physics_frequency / decimation
    episode_length_s = 10.0
    num_envs = 4096
    num_observations = 99  # 24*2*3 (joint pos/vel × 3 history) + 3 (ang_vel) + 3 (lin_acc) + 3 (orientation error)
    num_actions = 24  # 24 joint position targets

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,  # 100 Hz physics
        render_interval=decimation,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=3.0, replicate_physics=True)

    # Task-specific parameters
    target_height = 0.18  # Target torso height (meters)

    # Reward weights
    reward_weights = {
        "orientation": 1.0,      # exp(-3 * tilt^2)
        "height": 5.0,           # -5 * |height_error|
        "alive": 1.0,            # Alive bonus
        "action_rate": 0.001,    # Action rate penalty
        "joint_vel": 0.0001,     # Joint velocity penalty
    }

    # Termination conditions
    max_tilt_deg = 60.0         # Maximum tilt before termination
    min_height = 0.10           # Minimum height before termination

    # Domain randomization
    mass_randomization = 0.4     # ±40%
    friction_range = (0.5, 1.5)  # Friction randomization
    damping_randomization = 0.5  # ±50%

    # External push settings
    push_interval_range = (3.0, 5.0)  # Seconds between pushes
    push_force_range = (10.0, 80.0)   # Newtons

    # Actuator delay
    actuator_delay_range = (0.01, 0.03)  # 10-30 ms


class SpiderStandingEnv(DirectRLEnv):
    """Spider robot standing and balancing environment"""

    cfg: SpiderStandingEnvCfg

    def __init__(self, cfg: SpiderStandingEnvCfg, render_mode: str = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize robot
        self._init_robot()

        # History buffers for observations (3-step history)
        self.history_length = 3
        self.joint_pos_history = torch.zeros(
            self.num_envs, self.history_length, self.num_actions, device=self.device
        )
        self.joint_vel_history = torch.zeros(
            self.num_envs, self.history_length, self.num_actions, device=self.device
        )

        # Action history for action rate penalty
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        # Push tracking
        self.push_timer = torch.zeros(self.num_envs, device=self.device)
        self.next_push_time = torch.zeros(self.num_envs, device=self.device)

        # Actuator delay simulation
        self.actuator_delays = torch.zeros(self.num_envs, device=self.device)
        self.delayed_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        # Domain randomization parameters
        self._randomize_domain()

        # Default joint positions (standing pose)
        self.default_joint_pos = torch.tensor([
            # Leg 0 (FR): coxa, femur, tibia
            0.0, 0.5, -1.0,
            # Leg 1 (FL): coxa, femur, tibia
            0.0, 0.5, -1.0,
            # Leg 2 (MFR): coxa, femur, tibia
            0.0, 0.5, -1.0,
            # Leg 3 (MFL): coxa, femur, tibia
            0.0, 0.5, -1.0,
            # Leg 4 (MBR): coxa, femur, tibia
            0.0, 0.5, -1.0,
            # Leg 5 (MBL): coxa, femur, tibia
            0.0, 0.5, -1.0,
            # Leg 6 (BR): coxa, femur, tibia
            0.0, 0.5, -1.0,
            # Leg 7 (BL): coxa, femur, tibia
            0.0, 0.5, -1.0,
        ], device=self.device).repeat(self.num_envs, 1)

        # Success tracking for logging
        self.standing_success = torch.zeros(self.num_envs, device=self.device)

    def _init_robot(self):
        """Initialize the spider robot articulation"""
        # Get path to URDF
        urdf_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/urdf/spider.urdf"
        )

        # Create articulation configuration
        robot_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=urdf_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=10.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.25),  # Start slightly above target height
                rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
                joint_pos={
                    ".*": 0.0,  # Default all joints to 0
                },
            ),
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=["coxa_.*", "femur_.*", "tibia_.*"],
                    effort_limit=10.0,
                    velocity_limit=5.0,
                    stiffness=40.0,
                    damping=2.0,
                ),
            },
        )

        self.robot: Articulation = Articulation(robot_cfg)

    def _setup_scene(self):
        """Setup the scene with ground plane and robot"""
        # Ground plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)

        # Clone robot environments
        self.scene.clone_environments(copy_from_source=False)

        # Add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # Add lighting
        cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step"""
        # Simulate actuator delay
        if torch.rand(1).item() < 0.01:  # Periodically update delays
            self.actuator_delays = torch.rand(self.num_envs, device=self.device) * \
                                   (self.cfg.actuator_delay_range[1] - self.cfg.actuator_delay_range[0]) + \
                                   self.cfg.actuator_delay_range[0]

        # Apply delayed actions (simple 1-step delay for efficiency)
        actual_actions = self.delayed_actions.clone()
        self.delayed_actions = actions.clone()

        # Set joint position targets
        self.robot.set_joint_position_target(actual_actions)

    def _apply_action(self):
        """Apply buffered actions (called by framework)"""
        pass  # Actions applied in _pre_physics_step

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations"""
        # Get current joint states
        joint_pos = self.robot.data.joint_pos[:, :self.num_actions]
        joint_vel = self.robot.data.joint_vel[:, :self.num_actions]

        # Update history buffers (FIFO)
        self.joint_pos_history = torch.roll(self.joint_pos_history, shifts=-1, dims=1)
        self.joint_pos_history[:, -1, :] = joint_pos

        self.joint_vel_history = torch.roll(self.joint_vel_history, shifts=-1, dims=1)
        self.joint_vel_history[:, -1, :] = joint_vel

        # Get torso state
        torso_quat = self.robot.data.root_quat_w  # (num_envs, 4) - quaternion (w, x, y, z)
        torso_ang_vel = self.robot.data.root_ang_vel_w  # (num_envs, 3)
        torso_lin_acc = self.robot.data.root_lin_acc_w  # (num_envs, 3)

        # Compute orientation error (deviation from upright)
        # Target orientation is identity quaternion (0, 0, 0, 1) in scalar-last convention
        # But Isaac Lab uses scalar-first (w, x, y, z)
        target_quat = torch.zeros_like(torso_quat)
        target_quat[:, 0] = 1.0  # w = 1 for identity

        # Compute orientation error as euler angles
        euler = quat_to_euler_xyz(torso_quat)
        orientation_error = euler  # Roll, pitch, yaw

        # Flatten history buffers
        joint_pos_flat = self.joint_pos_history.reshape(self.num_envs, -1)  # (num_envs, 72)
        joint_vel_flat = self.joint_vel_history.reshape(self.num_envs, -1)  # (num_envs, 72)

        # Concatenate all observations
        obs = torch.cat([
            joint_pos_flat,      # 72 dims
            joint_vel_flat,      # 72 dims
            torso_ang_vel,       # 3 dims
            torso_lin_acc,       # 3 dims
            orientation_error,   # 3 dims (roll, pitch, yaw)
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards"""
        # Get current state
        torso_pos = self.robot.data.root_pos_w
        torso_quat = self.robot.data.root_quat_w
        joint_vel = self.robot.data.joint_vel[:, :self.num_actions]

        # Current height
        height = torso_pos[:, 2]
        height_error = torch.abs(height - self.cfg.target_height)

        # Orientation error (tilt)
        euler = quat_to_euler_xyz(torso_quat)
        roll, pitch = euler[:, 0], euler[:, 1]
        tilt = torch.sqrt(roll**2 + pitch**2)  # Total tilt magnitude

        # Reward components
        orientation_reward = torch.exp(-3.0 * tilt**2)
        height_reward = -height_error
        alive_reward = torch.ones_like(height)

        # Action rate penalty
        action_rate = torch.sum(torch.abs(self.delayed_actions - self.last_actions), dim=-1)
        action_rate_penalty = -action_rate

        # Joint velocity penalty
        joint_vel_penalty = -torch.sum(joint_vel**2, dim=-1)

        # Total reward
        reward = (
            self.cfg.reward_weights["orientation"] * orientation_reward +
            self.cfg.reward_weights["height"] * height_reward +
            self.cfg.reward_weights["alive"] * alive_reward +
            self.cfg.reward_weights["action_rate"] * action_rate_penalty +
            self.cfg.reward_weights["joint_vel"] * joint_vel_penalty
        )

        # Update last actions
        self.last_actions = self.delayed_actions.clone()

        # Track standing success (height within 10% and tilt < 10 degrees)
        success_mask = (torch.abs(height_error) < 0.018) & (tilt < 0.174)  # 10 deg = 0.174 rad
        self.standing_success = success_mask.float()

        return reward

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation conditions"""
        # Get current state
        torso_pos = self.robot.data.root_pos_w
        torso_quat = self.robot.data.root_quat_w

        # Height check
        height = torso_pos[:, 2]
        height_fail = height < self.cfg.min_height

        # Tilt check
        euler = quat_to_euler_xyz(torso_quat)
        roll, pitch = euler[:, 0], euler[:, 1]
        tilt = torch.sqrt(roll**2 + pitch**2)
        tilt_fail = tilt > (self.cfg.max_tilt_deg * math.pi / 180.0)

        # Termination: height or tilt failure
        terminated = height_fail | tilt_fail

        # Truncation: time limit reached
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments"""
        super()._reset_idx(env_ids)

        # Reset robot to initial state with small randomization
        joint_pos = self.default_joint_pos[env_ids].clone()
        joint_pos += torch.randn_like(joint_pos) * 0.1  # Add noise

        joint_vel = torch.zeros((len(env_ids), self.num_actions), device=self.device)

        # Reset root state
        root_pos = torch.zeros((len(env_ids), 3), device=self.device)
        root_pos[:, 2] = self.cfg.target_height + torch.randn(len(env_ids), device=self.device) * 0.05

        root_quat = torch.zeros((len(env_ids), 4), device=self.device)
        root_quat[:, 0] = 1.0  # w = 1
        # Add small random orientation
        small_euler = torch.randn((len(env_ids), 3), device=self.device) * 0.1

        root_vel = torch.zeros((len(env_ids), 6), device=self.device)

        # Apply reset
        self.robot.write_root_pose_to_sim(
            torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids
        )
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset history buffers
        self.joint_pos_history[env_ids] = 0.0
        self.joint_vel_history[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.delayed_actions[env_ids] = 0.0

        # Reset push timers
        self.push_timer[env_ids] = 0.0
        self.next_push_time[env_ids] = torch.rand(len(env_ids), device=self.device) * \
                                       (self.cfg.push_interval_range[1] - self.cfg.push_interval_range[0]) + \
                                       self.cfg.push_interval_range[0]

        # Re-randomize domain parameters
        self._randomize_domain(env_ids)

    def _randomize_domain(self, env_ids: torch.Tensor = None):
        """Apply domain randomization"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Mass randomization (±40%)
        mass_scale = 1.0 + (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * self.cfg.mass_randomization
        # Note: Actual mass setting requires USD property access, simplified here

        # Friction randomization (0.5-1.5)
        friction = torch.rand(len(env_ids), device=self.device) * \
                  (self.cfg.friction_range[1] - self.cfg.friction_range[0]) + \
                  self.cfg.friction_range[0]
        # Note: Actual friction setting requires material property access

        # Joint damping randomization (±50%)
        damping_scale = 1.0 + (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * self.cfg.damping_randomization
        # Note: Actual damping setting requires articulation property access

    def _apply_random_pushes(self):
        """Apply random external pushes"""
        self.push_timer += self.cfg.sim.dt

        # Check which environments need a push
        push_mask = self.push_timer >= self.next_push_time

        if push_mask.any():
            push_envs = torch.where(push_mask)[0]

            # Generate random push forces
            num_pushes = len(push_envs)
            push_magnitude = torch.rand(num_pushes, device=self.device) * \
                           (self.cfg.push_force_range[1] - self.cfg.push_force_range[0]) + \
                           self.cfg.push_force_range[0]

            # Random direction (horizontal plane)
            push_angle = torch.rand(num_pushes, device=self.device) * 2.0 * math.pi
            push_force = torch.zeros((num_pushes, 3), device=self.device)
            push_force[:, 0] = push_magnitude * torch.cos(push_angle)
            push_force[:, 1] = push_magnitude * torch.sin(push_angle)

            # Apply forces to torso
            # Note: Force application requires physics API access
            # self.robot.apply_external_force(push_force, env_ids=push_envs)

            # Reset timers for pushed environments
            self.push_timer[push_envs] = 0.0
            self.next_push_time[push_envs] = torch.rand(num_pushes, device=self.device) * \
                                             (self.cfg.push_interval_range[1] - self.cfg.push_interval_range[0]) + \
                                             self.cfg.push_interval_range[0]

    def step(self, actions: torch.Tensor):
        """Environment step"""
        # Apply random pushes
        self._apply_random_pushes()

        # Continue with normal step
        return super().step(actions)

    def get_metrics(self) -> Dict[str, float]:
        """Get additional metrics for logging"""
        metrics = {}

        # Standing success rate
        metrics["standing_success_rate"] = self.standing_success.mean().item()

        # Average torso height
        height = self.robot.data.root_pos_w[:, 2]
        metrics["avg_torso_height"] = height.mean().item()

        # Average tilt
        euler = quat_to_euler_xyz(self.robot.data.root_quat_w)
        roll, pitch = euler[:, 0], euler[:, 1]
        tilt = torch.sqrt(roll**2 + pitch**2)
        metrics["avg_tilt_deg"] = (tilt * 180.0 / math.pi).mean().item()

        return metrics
