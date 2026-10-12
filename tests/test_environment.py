import numpy as np
import pytest

from uas.envs import ContinuousWorldEnv


def test_reset_is_reproducible_and_in_observation_space() -> None:
    environment = ContinuousWorldEnv(task_id="held_out")
    first, first_info = environment.reset(seed=42)
    second, second_info = environment.reset(seed=42)

    np.testing.assert_allclose(first, second)
    assert environment.observation_space.contains(first)
    assert first_info["task_id"] == "held_out"
    assert second_info["task_id"] == "held_out"


def test_agent_can_complete_open_task() -> None:
    environment = ContinuousWorldEnv(task_id="open")
    observation, _ = environment.reset(seed=3)
    terminated = False

    for _ in range(environment.max_steps):
        direction = observation[2:4] - observation[0:2]
        norm = np.linalg.norm(direction)
        action = direction / max(norm, 1e-6)
        observation, _, terminated, truncated, info = environment.step(action)
        if terminated or truncated:
            break

    assert terminated
    assert info["success"]


def test_obstacle_collision_blocks_motion() -> None:
    environment = ContinuousWorldEnv(task_id="upper_gap")
    environment.reset(seed=0)
    environment.agent_position = np.asarray([0.39, 0.30], dtype=np.float32)
    previous = environment.agent_position.copy()

    _, reward, _, _, info = environment.step(np.asarray([1.0, 0.0], dtype=np.float32))

    np.testing.assert_allclose(environment.agent_position, previous)
    assert info["collision"]
    assert info["collision_count"] == 1
    assert reward < -0.1


def test_render_returns_rgb_frame() -> None:
    environment = ContinuousWorldEnv(task_id="held_out", render_mode="rgb_array")
    environment.reset(seed=0)
    frame = environment.render()

    assert frame.shape == (512, 512, 3)
    assert frame.dtype == np.uint8


@pytest.mark.parametrize(
    ("task_id", "waypoints"),
    [
        ("hard_upper", ((0.25, 0.86), (0.75, 0.86))),
        ("hard_lower", ((0.25, 0.14), (0.75, 0.14))),
    ],
)
def test_hard_tasks_have_a_feasible_narrow_passage(
    task_id: str, waypoints: tuple[tuple[float, float], ...]
) -> None:
    environment = ContinuousWorldEnv(task_id=task_id)
    observation, _ = environment.reset(seed=0)
    targets = [np.asarray(point, dtype=np.float32) for point in waypoints]
    targets.append(observation[2:4].copy())
    target_index = 0

    for _ in range(environment.max_steps):
        direction = targets[target_index] - observation[0:2]
        if np.linalg.norm(direction) < 0.045 and target_index < len(targets) - 1:
            target_index += 1
            direction = targets[target_index] - observation[0:2]
        action = direction / max(float(np.linalg.norm(direction)), 1e-6)
        observation, _, terminated, truncated, info = environment.step(action)
        if terminated or truncated:
            break

    assert terminated
    assert info["success"]
