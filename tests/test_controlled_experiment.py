from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from uas.controlled_experiment import summarize_controlled_results
from uas.controlled_replay import (
    sample_balanced_transitions,
    select_controlled_episodes,
)
from uas.envs import ACTION_SCHEMA, ENVIRONMENT_ID, OBSERVATION_SCHEMA, TASK_FAMILY
from uas.memory import EpisodeRecord, MemoryStore, Trajectory
from uas.training import query_for_task


def _trajectory(length: int = 20) -> Trajectory:
    return Trajectory(
        observations=np.zeros((length + 1, 12), dtype=np.float32),
        actions=np.zeros((length, 2), dtype=np.float32),
        rewards=np.zeros(length, dtype=np.float32),
        terminated=np.asarray([False] * (length - 1) + [True]),
        truncated=np.zeros(length, dtype=np.bool_),
    )


def _add_episode(
    store: MemoryStore,
    *,
    index: int,
    success: bool,
    random_policy: bool,
    length: int = 20,
) -> None:
    record = EpisodeRecord.create(
        environment_id=ENVIRONMENT_ID,
        task_id=f"source-{index}",
        task_family=TASK_FAMILY,
        task_description=f"source {index}",
        observation_schema=OBSERVATION_SCHEMA,
        action_schema=ACTION_SCHEMA,
        agent_id="random-explorer" if random_policy else "trained-agent",
        policy_version="random-v1" if random_policy else "sac-v1",
        policy_reference=None,
        seed=index,
        success=success,
        total_reward=float(success),
        normalized_score=1.0 if success else 0.1,
        steps=length,
        termination_reason="success" if success else "time_limit",
        collision_count=0,
        task_vector=(float(index) / 20.0,) * 8,
    )
    store.add_episode(record, _trajectory(length))


def test_controlled_selection_and_sampling_are_exact_and_reproducible(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    for index in range(8):
        _add_episode(
            store,
            index=index,
            success=index >= 4,
            random_policy=index < 4,
        )
    query = query_for_task("hard_upper")

    random_episodes = select_controlled_episodes(
        store, query, mode="random", episode_budget=3, seed=11
    )
    governed = select_controlled_episodes(
        store, query, mode="governed", episode_budget=3, seed=11
    )
    assert len(random_episodes) == 3
    assert all(item.record.policy_version == "random-v1" for item in random_episodes)
    assert len(governed) == 3
    assert all(item.record.success for item in governed)

    first = sample_balanced_transitions(
        store, governed, transition_budget=8, seed=42
    )
    second = sample_balanced_transitions(
        store, governed, transition_budget=8, seed=42
    )
    first_refs = [
        (item.episode.record.episode_id, item.transition_index) for item in first
    ]
    second_refs = [
        (item.episode.record.episode_id, item.transition_index) for item in second
    ]
    assert first_refs == second_refs
    assert len(first_refs) == 8
    assert len(set(first_refs)) == 8
    counts = Counter(episode_id for episode_id, _ in first_refs)
    assert sorted(counts.values()) == [2, 3, 3]


def test_controlled_sampling_rejects_an_unfair_budget(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    _add_episode(
        store, index=0, success=True, random_policy=False, length=2
    )
    selected = select_controlled_episodes(
        store,
        query_for_task("hard_upper"),
        mode="governed",
        episode_budget=1,
        seed=0,
    )
    with pytest.raises(ValueError, match="balanced quota"):
        sample_balanced_transitions(
            store, selected, transition_budget=3, seed=0
        )


def test_controlled_summary_reports_auc_and_negative_transfer() -> None:
    rows = []
    for mode, success_rates in (
        ("no_memory", (0.0, 0.5, 1.0)),
        ("governed", (0.0, 1.0, 1.0)),
    ):
        for timestep, success_rate in zip((0, 100, 200), success_rates, strict=True):
            rows.append(
                {
                    "task_id": "hard_upper",
                    "mode": mode,
                    "seed": 0,
                    "timestep": timestep,
                    "success_rate": success_rate,
                    "mean_reward": success_rate,
                    "mean_collisions": 0.0,
                }
            )

    summary, comparisons = summarize_controlled_results(rows, train_steps=200)

    governed = summary["tasks"]["hard_upper"]["governed"]
    assert governed["normalized_success_auc"]["mean"] == 0.75
    assert governed["first_threshold"]["mean_steps"] == 100.0
    assert governed["sustained_threshold"]["mean_steps"] == 100.0
    comparison = comparisons[0]
    assert comparison["mode"] == "governed"
    assert comparison["mean_auc_delta_vs_no_memory"] == 0.25
    assert comparison["negative_transfer_rate"] == 0.0
    assert comparison["mean_sustained_threshold_step_delta"] == -100.0
