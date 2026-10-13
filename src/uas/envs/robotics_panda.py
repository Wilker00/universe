"""Franka Panda 3D robotic manipulation environment and wrapper for UAS.

Provides continuous 3D manipulation tasks (Reach, Push, Pick-and-Place) with
Cartesian end-effector and parallel-jaw gripper control.
Supports both native panda-gym execution and a built-in deterministic kinematic
simulation harness for headless CI and lightweight benchmarking.
"""

from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


PANDA_TASKS = {
    "panda_reach": "Move end-effector to 3D target coordinates.",
    "panda_push": "Push an object block to target 3D goal coordinates.",
    "panda_pick_and_place": "Grasp object block and lift it to target coordinates.",
}


class PandaManipulationEnv(gym.Env):
    """3D Robotic Manipulation Environment for Franka Emika Panda.

    Action Space:
        Box(-1.0, 1.0, shape=(4,), dtype=float32)
        - [0:3]: End-effector Cartesian velocity [dx, dy, dz]
        - [3]: Gripper action [-1.0 = open, 1.0 = close]

    Observation Space:
        Box(-inf, inf, shape=(13,), dtype=float32)
        - [0:3]: End-effector position [x, y, z]
        - [3:6]: End-effector velocity [vx, vy, vz]
        - [6]: Gripper width [0.0 = fully closed, 0.08 = fully open]
        - [7:10]: Object position [ox, oy, oz]
        - [10:13]: Goal position [gx, gy, gz]
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_id: str = "panda_reach",
        max_steps: int = 50,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if task_id not in PANDA_TASKS:
            raise ValueError(
                f"Unknown panda task: {task_id}. Available: {list(PANDA_TASKS.keys())}"
            )
        self.task_id = task_id
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 4D action space: [dx, dy, dz, gripper]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 13D observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32,
        )

        self._step_count = 0
        self._ee_pos = np.array([0.0, 0.0, 0.2], dtype=np.float32)
        self._ee_vel = np.zeros(3, dtype=np.float32)
        self._gripper_width = 0.08
        self._object_pos = np.array([0.1, 0.0, 0.02], dtype=np.float32)
        self._goal_pos = np.array([0.2, 0.2, 0.1], dtype=np.float32)
        self._rng = np.random.default_rng(0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._ee_pos = np.array([0.0, 0.0, 0.2], dtype=np.float32) + self._rng.uniform(
            -0.02, 0.02, size=3
        ).astype(np.float32)
        self._ee_vel = np.zeros(3, dtype=np.float32)
        self._gripper_width = 0.08

        if self.task_id == "panda_reach":
            self._object_pos = np.zeros(3, dtype=np.float32)
            self._goal_pos = np.array([0.15, 0.15, 0.25], dtype=np.float32) + self._rng.uniform(
                -0.05, 0.05, size=3
            ).astype(np.float32)
        elif self.task_id == "panda_push":
            self._object_pos = np.array([0.1, 0.0, 0.02], dtype=np.float32)
            self._goal_pos = np.array([0.25, 0.15, 0.02], dtype=np.float32)
        else:  # pick and place
            self._object_pos = np.array([0.1, 0.0, 0.02], dtype=np.float32)
            self._goal_pos = np.array([0.15, 0.15, 0.20], dtype=np.float32)

        obs = self._get_obs()
        return obs, {"is_success": False, "task_id": self.task_id}

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self._ee_pos,
                self._ee_vel,
                np.array([self._gripper_width], dtype=np.float32),
                self._object_pos,
                self._goal_pos,
            ],
            axis=0,
        ).astype(np.float32)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._step_count += 1

        dt = 0.05
        max_vel = 0.2  # m/s
        self._ee_vel = action[:3] * max_vel
        self._ee_pos = self._ee_pos + self._ee_vel * dt
        self._ee_pos = np.clip(self._ee_pos, [-0.5, -0.5, 0.0], [0.5, 0.5, 0.6])

        # Gripper actuation
        gripper_cmd = float(action[3])
        if gripper_cmd > 0:
            self._gripper_width = max(0.0, self._gripper_width - 0.02)
        else:
            self._gripper_width = min(0.08, self._gripper_width + 0.02)

        # Object physics
        if self.task_id in ("panda_push", "panda_pick_and_place"):
            dist_to_obj = float(np.linalg.norm(self._ee_pos - self._object_pos))
            if dist_to_obj < 0.05:
                # Gripper contact
                if self.task_id == "panda_pick_and_place" and self._gripper_width < 0.03:
                    # Object grasped and follows gripper
                    self._object_pos = self._ee_pos.copy()
                else:
                    # Push along velocity vector
                    push_dir = self._ee_vel[:2]
                    self._object_pos[:2] += push_dir * dt

        # Reward & success calculation
        if self.task_id == "panda_reach":
            dist = float(np.linalg.norm(self._ee_pos - self._goal_pos))
            reward = -dist
            is_success = dist < 0.05
        else:
            dist = float(np.linalg.norm(self._object_pos - self._goal_pos))
            ee_to_obj = float(np.linalg.norm(self._ee_pos - self._object_pos))
            reward = -dist - 0.1 * ee_to_obj
            is_success = dist < 0.05

        terminated = bool(is_success)
        truncated = bool(self._step_count >= self.max_steps)
        if is_success:
            reward += 10.0

        obs = self._get_obs()
        info = {
            "is_success": is_success,
            "distance": dist,
            "step_count": self._step_count,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            # Return synthetic 64x64 frame representing 3D workspace
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            x_pixel = int(np.clip((self._ee_pos[0] + 0.5) * 64, 0, 63))
            y_pixel = int(np.clip((self._ee_pos[1] + 0.5) * 64, 0, 63))
            frame[max(0, y_pixel - 2) : min(64, y_pixel + 2), max(0, x_pixel - 2) : min(64, x_pixel + 2)] = [255, 0, 0]
            return frame
        return None

    def close(self) -> None:
        pass


def make_panda_env(
    task_id: str = "panda_reach", **kwargs: Any
) -> PandaManipulationEnv:
    """Create a Franka Panda manipulation environment."""
    return PandaManipulationEnv(task_id=task_id, **kwargs)

