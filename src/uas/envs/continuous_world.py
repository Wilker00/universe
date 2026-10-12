"""A small continuous 2D benchmark for testing UAS memory transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ENVIRONMENT_ID = "uas-continuous-world-v0"
TASK_FAMILY = "continuous-navigation-v1"
OBSERVATION_SCHEMA = "agent-goal-obstacle-step-v1"
ACTION_SCHEMA = "continuous-delta-xy-v1"


@dataclass(frozen=True)
class TaskSpec:
    """Configuration shared by training and held-out navigation tasks."""

    task_id: str
    description: str
    start: tuple[float, float]
    goal: tuple[float, float]
    obstacle: tuple[float, float, float, float] | None = None
    start_jitter: float = 0.035
    goal_jitter: float = 0.02

    def vector(self) -> np.ndarray:
        """Return normalized task features used by the memory retriever."""
        obstacle = self.obstacle or (0.0, 0.0, 0.0, 0.0)
        return np.asarray((*self.start, *self.goal, *obstacle), dtype=np.float32)


TASKS: dict[str, TaskSpec] = {
    "open": TaskSpec(
        task_id="open",
        description="Reach the target in an open continuous room.",
        start=(0.12, 0.15),
        goal=(0.86, 0.84),
    ),
    "upper_gap": TaskSpec(
        task_id="upper_gap",
        description="Reach the target by navigating above a vertical obstacle.",
        start=(0.12, 0.16),
        goal=(0.87, 0.82),
        obstacle=(0.43, 0.00, 0.57, 0.66),
    ),
    "lower_gap": TaskSpec(
        task_id="lower_gap",
        description="Reach the target by navigating below a vertical obstacle.",
        start=(0.12, 0.84),
        goal=(0.87, 0.18),
        obstacle=(0.43, 0.34, 0.57, 1.00),
    ),
    "held_out": TaskSpec(
        task_id="held_out",
        description="Reach a shifted target by navigating above an unfamiliar vertical obstacle.",
        start=(0.10, 0.12),
        goal=(0.89, 0.78),
        obstacle=(0.39, 0.00, 0.55, 0.59),
    ),
}


class ContinuousWorldEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous navigation with fixed schemas across related task layouts.

    The environment deliberately stays small. It exists to test whether
    governed retrieval improves adaptation, not to act as a realistic robot
    simulator.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        task_id: str = "open",
        *,
        max_steps: int = 150,
        max_action: float = 0.055,
        goal_radius: float = 0.065,
        agent_radius: float = 0.022,
        render_mode: str | None = None,
        render_size: int = 512,
    ) -> None:
        super().__init__()
        if task_id not in TASKS:
            available = ", ".join(sorted(TASKS))
            raise ValueError(f"Unknown task '{task_id}'. Available tasks: {available}")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")

        self.task = TASKS[task_id]
        self.task_id = task_id
        self.max_steps = int(max_steps)
        self.max_action = float(max_action)
        self.goal_radius = float(goal_radius)
        self.agent_radius = float(agent_radius)
        self.render_mode = render_mode
        self.render_size = int(render_size)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        observation_low = np.asarray(
            [0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32
        )
        observation_high = np.asarray(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32,
        )

        self.agent_position = np.zeros(2, dtype=np.float32)
        self.goal_position = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.collision_count = 0

    @property
    def task_vector(self) -> np.ndarray:
        return self.task.vector()

    def _jittered_point(
        self, point: tuple[float, float], jitter: float
    ) -> np.ndarray:
        value = np.asarray(point, dtype=np.float32)
        if jitter:
            value = value + self.np_random.uniform(-jitter, jitter, size=2)
        return np.clip(value, self.agent_radius, 1.0 - self.agent_radius).astype(
            np.float32
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.agent_position = self._jittered_point(
            self.task.start, self.task.start_jitter
        )
        self.goal_position = self._jittered_point(
            self.task.goal, self.task.goal_jitter
        )
        self.step_count = 0
        self.collision_count = 0
        return self._get_observation(), self._get_info(success=False, collision=False)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action_array.shape}")
        action_array = np.clip(action_array, -1.0, 1.0)
        magnitude = float(np.linalg.norm(action_array))
        if magnitude > 1.0:
            action_array = action_array / magnitude

        previous_navigation_distance = self._navigation_distance(self.agent_position)
        candidate = np.clip(
            self.agent_position + action_array * self.max_action,
            self.agent_radius,
            1.0 - self.agent_radius,
        ).astype(np.float32)
        collision = self._path_hits_obstacle(self.agent_position, candidate)
        if collision:
            candidate = self.agent_position.copy()
            self.collision_count += 1

        self.agent_position = candidate
        self.step_count += 1
        distance = self._distance_to_goal()
        navigation_distance = self._navigation_distance(self.agent_position)
        success = distance <= self.goal_radius
        terminated = bool(success)
        truncated = bool(self.step_count >= self.max_steps and not terminated)

        progress = previous_navigation_distance - navigation_distance
        reward = float(5.0 * progress - 0.01)
        if collision:
            reward -= 0.20
        if success:
            reward += 2.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(success=success, collision=collision),
        )

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal_position - self.agent_position))

    def _navigation_distance(self, point: np.ndarray) -> float:
        """Approximate shortest-path distance for a useful shaping signal."""
        direct_distance = float(np.linalg.norm(self.goal_position - point))
        if self.task.obstacle is None or not self._path_hits_obstacle(
            point, self.goal_position
        ):
            return direct_distance

        x_min, y_min, x_max, y_max = self.task.obstacle
        clearance = self.agent_radius + 0.025
        candidates: list[float] = []
        upper_y = y_max + clearance
        if upper_y < 1.0 - self.agent_radius:
            upper_left = np.asarray([x_min - clearance, upper_y], dtype=np.float32)
            upper_right = np.asarray([x_max + clearance, upper_y], dtype=np.float32)
            candidates.append(
                float(
                    np.linalg.norm(point - upper_left)
                    + np.linalg.norm(upper_left - upper_right)
                    + np.linalg.norm(upper_right - self.goal_position)
                )
            )
        lower_y = y_min - clearance
        if lower_y > self.agent_radius:
            lower_left = np.asarray([x_min - clearance, lower_y], dtype=np.float32)
            lower_right = np.asarray([x_max + clearance, lower_y], dtype=np.float32)
            candidates.append(
                float(
                    np.linalg.norm(point - lower_left)
                    + np.linalg.norm(lower_left - lower_right)
                    + np.linalg.norm(lower_right - self.goal_position)
                )
            )
        return min(candidates, default=direct_distance)

    def _obstacle_values(self) -> tuple[float, float, float, float]:
        return self.task.obstacle or (0.0, 0.0, 0.0, 0.0)

    def _get_observation(self) -> np.ndarray:
        obstacle = self._obstacle_values()
        delta = self.goal_position - self.agent_position
        return np.asarray(
            [
                *self.agent_position,
                *self.goal_position,
                *delta,
                *obstacle,
                float(self.task.obstacle is not None),
                self.step_count / self.max_steps,
            ],
            dtype=np.float32,
        )

    def _get_info(self, *, success: bool, collision: bool) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_family": TASK_FAMILY,
            "distance": self._distance_to_goal(),
            "success": bool(success),
            "collision": bool(collision),
            "collision_count": self.collision_count,
            "step_count": self.step_count,
            "observation_schema": OBSERVATION_SCHEMA,
            "action_schema": ACTION_SCHEMA,
        }

    def _path_hits_obstacle(self, start: np.ndarray, end: np.ndarray) -> bool:
        if self.task.obstacle is None:
            return False
        x_min, y_min, x_max, y_max = self.task.obstacle
        x_min -= self.agent_radius
        y_min -= self.agent_radius
        x_max += self.agent_radius
        y_max += self.agent_radius
        samples = np.linspace(start, end, num=16, dtype=np.float32)
        inside_x = (samples[:, 0] >= x_min) & (samples[:, 0] <= x_max)
        inside_y = (samples[:, 1] >= y_min) & (samples[:, 1] <= y_max)
        return bool(np.any(inside_x & inside_y))

    def render(self) -> np.ndarray:
        canvas = np.full(
            (self.render_size, self.render_size, 3), 248, dtype=np.uint8
        )

        def to_pixel(point: np.ndarray | tuple[float, float]) -> tuple[int, int]:
            x = int(float(point[0]) * (self.render_size - 1))
            y = int((1.0 - float(point[1])) * (self.render_size - 1))
            return x, y

        if self.task.obstacle is not None:
            x_min, y_min, x_max, y_max = self.task.obstacle
            left = int(x_min * self.render_size)
            right = int(x_max * self.render_size)
            top = int((1.0 - y_max) * self.render_size)
            bottom = int((1.0 - y_min) * self.render_size)
            canvas[top:bottom, left:right] = np.asarray([55, 65, 81], dtype=np.uint8)

        self._draw_disc(canvas, to_pixel(self.goal_position), self.goal_radius, (42, 176, 96))
        self._draw_disc(canvas, to_pixel(self.agent_position), self.agent_radius, (44, 112, 214))
        return canvas

    def _draw_disc(
        self,
        canvas: np.ndarray,
        center: tuple[int, int],
        radius_world: float,
        color: tuple[int, int, int],
    ) -> None:
        radius = max(2, int(radius_world * self.render_size))
        x_center, y_center = center
        y, x = np.ogrid[: self.render_size, : self.render_size]
        mask = (x - x_center) ** 2 + (y - y_center) ** 2 <= radius**2
        canvas[mask] = np.asarray(color, dtype=np.uint8)
