"""Tests for the framework-neutral MemoryBridge API."""

from pathlib import Path

import numpy as np
import pytest

from uas import Episode, MemoryBridge


def _make_dummy_episode(
    task: str = "reach",
    embodiment: str = "franka_panda",
    framework: str = "isaac_lab",
    success: bool = True,
    reward: float = 8.5,
    steps: int = 10,
    tags: list[str] | None = None,
) -> Episode:
    return Episode(
        task=task,
        embodiment=embodiment,
        framework=framework,
        success=success,
        total_reward=reward,
        observations=np.random.randn(steps + 1, 9).astype(np.float32),
        actions=np.random.randn(steps, 7).astype(np.float32),
        rewards=np.ones(steps, dtype=np.float32) * (reward / steps),
        terminated=np.array([False] * (steps - 1) + [True]),
        truncated=np.zeros(steps, dtype=bool),
        metadata={"tags": tags or ["arm", "test"], "collisions": 0},
    )


def test_bridge_record_and_get(tmp_path: Path) -> None:
    bridge = MemoryBridge(tmp_path / "bridge")
    ep = _make_dummy_episode(task="pick_cube", reward=10.0)

    ep_id = bridge.record(ep)
    assert bridge.count() == 1
    assert ep_id == ep.episode_id

    loaded = bridge.get(ep_id)
    assert loaded is not None
    assert loaded.task == "pick_cube"
    assert loaded.embodiment == "franka_panda"
    assert loaded.framework == "isaac_lab"
    assert loaded.success is True
    assert loaded.total_reward == 10.0
    assert loaded.steps == 10
    assert loaded.observations.shape == (11, 9)
    assert loaded.actions.shape == (10, 7)
    assert np.allclose(loaded.actions, ep.actions)


def test_bridge_search_filters(tmp_path: Path) -> None:
    bridge = MemoryBridge(tmp_path / "bridge")

    # Log 3 different episodes from different frameworks & embodiments
    bridge.record(
        _make_dummy_episode(
            task="reach",
            embodiment="franka_panda",
            framework="isaac_lab",
            success=True,
            reward=9.0,
            tags=["grasping"],
        )
    )
    bridge.record(
        _make_dummy_episode(
            task="push",
            embodiment="franka_panda",
            framework="isaac_lab",
            success=False,
            reward=2.0,
            tags=["sliding"],
        )
    )
    bridge.record(
        _make_dummy_episode(
            task="reach",
            embodiment="continuous_2d",
            framework="gymnasium",
            success=True,
            reward=5.0,
            tags=["navigation"],
        )
    )

    assert bridge.count() == 3
    assert set(bridge.embodiments()) == {"franka_panda", "continuous_2d"}
    assert set(bridge.tasks()) == {"reach", "push"}

    # Search by task
    results = bridge.search(task="reach")
    assert len(results) == 2

    # Search by embodiment
    results = bridge.search(embodiment="franka_panda")
    assert len(results) == 2

    # Search success only
    results = bridge.search(success_only=True)
    assert len(results) == 2

    # Search min_reward
    results = bridge.search(min_reward=6.0)
    assert len(results) == 1
    assert results[0].task == "reach"
    assert results[0].framework == "isaac_lab"

    # Search by tags
    results = bridge.search(tags=["navigation"])
    assert len(results) == 1
    assert results[0].embodiment == "continuous_2d"


def test_episode_shape_validation() -> None:
    with pytest.raises(ValueError, match="Reward count"):
        Episode(
            task="test",
            embodiment="test",
            framework="test",
            success=True,
            total_reward=1.0,
            observations=np.zeros((6, 2)),
            actions=np.zeros((5, 2)),
            rewards=np.zeros((4,)),  # Mismatch!
            terminated=np.zeros((5,), dtype=bool),
            truncated=np.zeros((5,), dtype=bool),
        )
