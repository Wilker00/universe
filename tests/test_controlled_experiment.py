from collections import Counter
import csv
from pathlib import Path

import numpy as np
import pytest

from uas.controlled_experiment import (
    run_controlled_experiment,
    summarize_controlled_results,
)
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


def test_controlled_experiment_resumes_completed_runs(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    _add_episode(store, index=0, success=True, random_policy=False, length=10)

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_rows = [
        {
            "task_id": "hard_upper",
            "mode": "no_memory",
            "seed": 0,
            "timestep": 0,
            "success_rate": 0.0,
            "mean_reward": 0.0,
            "mean_steps": 10.0,
            "mean_collisions": 0.0,
            "selected_episodes": 0,
            "selected_successful_episodes": 0,
            "prefilled_transitions": 0,
        },
        {
            "task_id": "hard_upper",
            "mode": "no_memory",
            "seed": 0,
            "timestep": 10,
            "success_rate": 0.8,
            "mean_reward": 0.8,
            "mean_steps": 10.0,
            "mean_collisions": 0.0,
            "selected_episodes": 0,
            "selected_successful_episodes": 0,
            "prefilled_transitions": 0,
        },
    ]
    csv_path = output_dir / "learning_curves.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(existing_rows[0].keys()))
        writer.writeheader()
        writer.writerows(existing_rows)

    run_controlled_experiment(
        memory_directory=tmp_path / "memory",
        output_directory=output_dir,
        task_ids=["hard_upper"],
        modes=["no_memory"],
        seeds=[0],
        train_steps=10,
        evaluation_interval=10,
        evaluation_episodes=1,
        episode_budget=1,
        replay_transition_budget=1,
    )

    with csv_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 2
    assert reader[1]["timestep"] == "10"
    summary_path = output_dir / "summary.json"
    assert summary_path.is_file()


def test_governed_diverse_selection(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")

    # Add 4 episodes with identical vector A (score 1.0)
    for i in range(4):
        record = EpisodeRecord.create(
            environment_id=ENVIRONMENT_ID,
            task_id=f"cluster_a_{i}",
            task_family=TASK_FAMILY,
            task_description="cluster a",
            observation_schema=OBSERVATION_SCHEMA,
            action_schema=ACTION_SCHEMA,
            agent_id="agent",
            policy_version="sac-v1",
            policy_reference=None,
            seed=i,
            success=True,
            total_reward=10.0,
            normalized_score=1.0,
            steps=10,
            termination_reason="success",
            collision_count=0,
            task_vector=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        store.add_episode(record, _trajectory(10))

    # Add 2 episodes with distinct vector B (score 0.9)
    for i in range(2):
        record = EpisodeRecord.create(
            environment_id=ENVIRONMENT_ID,
            task_id=f"cluster_b_{i}",
            task_family=TASK_FAMILY,
            task_description="cluster b",
            observation_schema=OBSERVATION_SCHEMA,
            action_schema=ACTION_SCHEMA,
            agent_id="agent",
            policy_version="sac-v1",
            policy_reference=None,
            seed=10 + i,
            success=True,
            total_reward=9.0,
            normalized_score=0.9,
            steps=10,
            termination_reason="success",
            collision_count=0,
            task_vector=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
        store.add_episode(record, _trajectory(10))

    query = query_for_task("hard_upper")

    # Greedy governed takes only from cluster A because scores are highest (1.0)
    greedy = select_controlled_episodes(
        store, query, mode="governed", episode_budget=2, seed=0
    )
    assert all("cluster_a" in ep.record.task_id for ep in greedy)

    # Governed diverse balances score with diversity and selects from both cluster A and cluster B
    diverse = select_controlled_episodes(
        store, query, mode="governed_diverse", episode_budget=2, seed=0
    )
    task_ids = [ep.record.task_id for ep in diverse]
    assert any("cluster_a" in tid for tid in task_ids)
    assert any("cluster_b" in tid for tid in task_ids)


def test_random_uniform_selection(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    for i in range(15):
        _add_episode(store, index=i, success=i % 2 == 0, random_policy=i % 3 == 0, length=20)

    query = query_for_task("hard_upper")
    selected_1 = select_controlled_episodes(
        store, query, mode="random_uniform", episode_budget=6, seed=42
    )
    selected_2 = select_controlled_episodes(
        store, query, mode="random_uniform", episode_budget=6, seed=42
    )
    assert len(selected_1) == 6
    assert [e.record.episode_id for e in selected_1] == [e.record.episode_id for e in selected_2]

    # Verify loud failure when budget exceeds available candidates
    with pytest.raises(ValueError, match="eligible episodes"):
        select_controlled_episodes(
            store, query, mode="random_uniform", episode_budget=20, seed=42
        )


def test_stratified_success_selection(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    idx = 0
    for layout in ("open", "upper_gap", "lower_gap"):
        for _ in range(5):
            record = EpisodeRecord.create(
                environment_id=ENVIRONMENT_ID,
                task_id=layout,
                task_family=TASK_FAMILY,
                task_description=layout,
                observation_schema=OBSERVATION_SCHEMA,
                action_schema=ACTION_SCHEMA,
                agent_id="agent",
                policy_version="sac-v1",
                policy_reference=None,
                seed=idx,
                success=True,
                total_reward=10.0,
                normalized_score=1.0,
                steps=20,
                termination_reason="success",
                collision_count=0,
                task_vector=(0.1 * idx,) * 8,
            )
            store.add_episode(record, _trajectory(20))
            idx += 1

    query = query_for_task("hard_upper")
    selected = select_controlled_episodes(
        store, query, mode="stratified_success", episode_budget=12, seed=123
    )
    assert len(selected) == 12
    counts = Counter(ep.record.task_id for ep in selected)
    assert counts["open"] == 4
    assert counts["upper_gap"] == 4
    assert counts["lower_gap"] == 4
    assert all(ep.record.success for ep in selected)

    # Test balanced transitions sampling
    transitions = sample_balanced_transitions(
        store, selected, transition_budget=200, seed=456
    )
    assert len(transitions) == 200
    t_counts = Counter(t.episode.record.episode_id for t in transitions)
    assert sorted(t_counts.values()) == [16] * 4 + [17] * 8

    with pytest.raises(ValueError, match="evenly divisible"):
        select_controlled_episodes(
            store, query, mode="stratified_success", episode_budget=10, seed=123
        )

    with pytest.raises(ValueError, match="requires 6 successful episodes"):
        select_controlled_episodes(
            store, query, mode="stratified_success", episode_budget=18, seed=123
        )


def test_random_concentrated_selection(tmp_path: Path) -> None:
    store_real = MemoryStore(tmp_path / "memory_real")
    from uas.envs.continuous_world import ContinuousWorldEnv
    envs = {l: ContinuousWorldEnv(task_id=l) for l in ("open", "upper_gap", "lower_gap")}
    idx = 0
    for layout, env in envs.items():
        tvec = env.task_vector
        for _ in range(10):
            record = EpisodeRecord.create(
                environment_id=ENVIRONMENT_ID,
                task_id=layout,
                task_family=TASK_FAMILY,
                task_description=layout,
                observation_schema=OBSERVATION_SCHEMA,
                action_schema=ACTION_SCHEMA,
                agent_id="random-explorer",
                policy_version="random-v1",
                policy_reference=None,
                seed=idx,
                success=False,
                total_reward=0.0,
                normalized_score=0.0,
                steps=20,
                termination_reason="time_limit",
                collision_count=0,
                task_vector=tuple(tvec),
            )
            store_real.add_episode(record, _trajectory(20))
            idx += 1

    query_upper = query_for_task("hard_upper")
    selected_upper = select_controlled_episodes(
        store_real, query_upper, mode="random_concentrated", episode_budget=12, seed=99
    )
    assert len(selected_upper) == 12
    counts_u = Counter(ep.record.task_id for ep in selected_upper)
    assert counts_u["upper_gap"] == 8
    assert counts_u["open"] == 4
    assert counts_u["lower_gap"] == 0

    query_lower = query_for_task("hard_lower")
    selected_lower = select_controlled_episodes(
        store_real, query_lower, mode="random_concentrated", episode_budget=12, seed=99
    )
    assert len(selected_lower) == 12
    counts_l = Counter(ep.record.task_id for ep in selected_lower)
    assert counts_l["lower_gap"] == 8
    assert counts_l["upper_gap"] == 4
    assert counts_l["open"] == 0

    with pytest.raises(ValueError, match="requires 16 random episodes"):
        select_controlled_episodes(
            store_real, query_upper, mode="random_concentrated", episode_budget=24, seed=99
        )


