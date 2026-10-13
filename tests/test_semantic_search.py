from pathlib import Path

import numpy as np
import pytest

from uas.bridge.client import MemoryBridge
from uas.bridge.models import Episode


def _create_test_episode(
    task: str,
    task_description: str,
    reward: float = 10.0,
    tags: list[str] | None = None,
) -> Episode:
    obs = np.zeros((6, 4), dtype=np.float32)
    acts = np.zeros((5, 2), dtype=np.float32)
    rews = np.ones(5, dtype=np.float32) * (reward / 5.0)
    term = np.zeros(5, dtype=bool)
    term[-1] = True
    trunc = np.zeros(5, dtype=bool)

    return Episode(
        task=task,
        task_description=task_description,
        embodiment="robot",
        framework="uas",
        success=True,
        total_reward=reward,
        observations=obs,
        actions=acts,
        rewards=rews,
        terminated=term,
        truncated=trunc,
        metadata={"tags": tags or []},
    )


def test_semantic_search_ranking(tmp_path: Path) -> None:
    bridge = MemoryBridge(tmp_path / "bridge")

    ep_reach = _create_test_episode(
        task="panda_reach",
        task_description="Reach target in 3D workspace with robotic arm",
        tags=["manipulation", "reaching"],
    )
    ep_grasp = _create_test_episode(
        task="panda_pick_and_place",
        task_description="Grasp red block with parallel jaw gripper and lift to goal height",
        tags=["grasping", "block"],
    )
    ep_nav = _create_test_episode(
        task="continuous_navigation",
        task_description="Navigate continuous mobile agent around obstacles in a 2D room",
        tags=["navigation", "mobile"],
    )

    bridge.record(ep_reach)
    bridge.record(ep_grasp)
    bridge.record(ep_nav)

    # 1. Query for grasping/lifting
    grasp_results = bridge.search_semantic("grasp block with gripper", limit=2)
    assert len(grasp_results) > 0
    top_ep, score = grasp_results[0]
    assert top_ep.task == "panda_pick_and_place"
    assert score > 0.2

    # 2. Query for navigation
    nav_results = bridge.search_semantic("navigate mobile agent around obstacles", limit=2)
    assert len(nav_results) > 0
    top_nav, nav_score = nav_results[0]
    assert top_nav.task == "continuous_navigation"
    assert nav_score > 0.2

    # 3. Query for arm reaching
    reach_results = bridge.search_semantic("robotic arm reach target", limit=2)
    assert len(reach_results) > 0
    top_reach, reach_score = reach_results[0]
    assert top_reach.task == "panda_reach"
    assert reach_score > 0.2


def test_semantic_search_empty_query(tmp_path: Path) -> None:
    bridge = MemoryBridge(tmp_path / "bridge")
    ep = _create_test_episode(task="test", task_description="some task")
    bridge.record(ep)

    assert bridge.search_semantic("") == []
    assert bridge.search_semantic("   ") == []
