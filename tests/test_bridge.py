"""Tests for the framework-neutral MemoryBridge API."""

import os
from pathlib import Path

import numpy as np
import pytest

from uas.bridge import CompatibilityReport, Episode, MemoryBridge


def _make_dummy_episode(
    task: str = "reach",
    embodiment: str = "franka_panda",
    framework: str = "isaac_lab",
    success: bool = True,
    reward: float = 8.5,
    steps: int = 10,
    tags: list[str] | None = None,
    discrete: bool = False,
    obs_dim: int = 9,
    act_dim: int = 7,
) -> Episode:
    obs = np.random.randn(steps + 1, obs_dim).astype(np.float32)
    if discrete:
        acts = np.random.randint(0, 4, size=(steps,)).astype(np.int64)
    else:
        acts = np.random.randn(steps, act_dim).astype(np.float32)
    return Episode(
        task=task,
        embodiment=embodiment,
        framework=framework,
        success=success,
        total_reward=reward,
        observations=obs,
        actions=acts,
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


def test_bridge_search_filters_and_tag_limit(tmp_path: Path) -> None:
    bridge = MemoryBridge(tmp_path / "bridge")

    # Insert 10 high-reward episodes WITHOUT the tag "rare_tag" (rewards 20.0 to 11.0)
    for i in range(10):
        bridge.record(
            _make_dummy_episode(
                task="generic_task",
                reward=20.0 - i,
                tags=["common_tag"],
            )
        )

    # Insert 5 lower-reward episodes WITH the tag "rare_tag" (rewards 5.0 to 1.0)
    for i in range(5):
        bridge.record(
            _make_dummy_episode(
                task="specific_task",
                reward=5.0 - i,
                tags=["rare_tag"],
            )
        )

    assert bridge.count() == 15

    # Tag search with limit=5 must NOT be choked by the 10 top-reward episodes
    results = bridge.search(tags=["rare_tag"], limit=5)
    assert len(results) == 5
    assert all("rare_tag" in r.metadata.get("tags", []) for r in results)
    assert results[0].total_reward == 5.0


def test_bridge_sqlite_connection_closed_on_cleanup(tmp_path: Path) -> None:
    bridge_dir = tmp_path / "bridge_conn_test"
    bridge = MemoryBridge(bridge_dir)
    ep = _make_dummy_episode()
    bridge.record(ep)
    _ = bridge.search(limit=1)

    # Windows file lock test: If connection was not closed, unlinking will raise PermissionError
    db_file = bridge_dir / "bridge.sqlite3"
    assert db_file.exists()
    db_file.unlink()  # Must succeed cleanly on Windows without PermissionError!


def test_discrete_action_semantics_preserved() -> None:
    ep = _make_dummy_episode(discrete=True)
    assert ep.action_type == "discrete"
    assert np.issubdtype(ep.actions.dtype, np.integer)
    assert ep.actions.dtype == np.int64


def test_compatibility_report() -> None:
    ep = _make_dummy_episode(obs_dim=9, act_dim=7)

    # Exact match
    match_report = ep.check_compatibility(
        other_obs_schema="continuous",
        other_action_schema="continuous",
        other_obs_dim=(9,),
        other_act_dim=(7,),
    )
    assert match_report.compatible is True

    # Dimension mismatch
    mismatch_report = ep.check_compatibility(
        other_obs_schema="continuous",
        other_action_schema="continuous",
        other_obs_dim=(12,),
        other_act_dim=(7,),
    )
    assert mismatch_report.compatible is False
    assert mismatch_report.observation_compatible is False
    assert mismatch_report.action_compatible is True


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
