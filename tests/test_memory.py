from pathlib import Path

import numpy as np

from uas.envs import ACTION_SCHEMA, ENVIRONMENT_ID, OBSERVATION_SCHEMA, TASK_FAMILY
from uas.memory import EpisodeRecord, MemoryStore, TaskQuery, Trajectory


def make_trajectory() -> Trajectory:
    return Trajectory(
        observations=np.zeros((2, 12), dtype=np.float32),
        actions=np.zeros((1, 2), dtype=np.float32),
        rewards=np.asarray([1.0], dtype=np.float32),
        terminated=np.asarray([True]),
        truncated=np.asarray([False]),
    )


def make_record(
    *,
    task_id: str,
    vector: tuple[float, ...],
    success: bool,
    score: float,
    task_family: str = TASK_FAMILY,
) -> EpisodeRecord:
    return EpisodeRecord.create(
        environment_id=ENVIRONMENT_ID,
        task_id=task_id,
        task_family=task_family,
        task_description=task_id,
        observation_schema=OBSERVATION_SCHEMA,
        action_schema=ACTION_SCHEMA,
        agent_id="test-agent",
        policy_version="test-v1",
        policy_reference=None,
        seed=1,
        success=success,
        total_reward=1.0,
        normalized_score=score,
        steps=1,
        termination_reason="success" if success else "time_limit",
        collision_count=0,
        task_vector=vector,
    )


def test_memory_round_trip_and_retrieval_modes(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    successful = make_record(
        task_id="successful-source",
        vector=(0.0,) * 8,
        success=True,
        score=1.0,
    )
    close_failure = make_record(
        task_id="close-failure",
        vector=(0.9,) * 8,
        success=False,
        score=0.4,
    )
    incompatible = make_record(
        task_id="other-family",
        vector=(1.0,) * 8,
        success=True,
        score=1.0,
        task_family="unrelated-family",
    )
    for record in (successful, close_failure, incompatible):
        store.add_episode(record, make_trajectory())

    query = TaskQuery(
        task_id="held-out",
        task_family=TASK_FAMILY,
        task_description="held out",
        observation_schema=OBSERVATION_SCHEMA,
        action_schema=ACTION_SCHEMA,
        task_vector=(1.0,) * 8,
    )

    assert store.count() == 3
    assert len(store.retrieve(query, mode="naive")) == 2
    assert store.retrieve(query, mode="similarity")[0].record == close_failure
    governed = store.retrieve(query, mode="governed")
    assert [item.record for item in governed] == [successful]
    assert store.load_trajectory(successful).observations.shape == (2, 12)


def test_rejects_empty_trajectory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    record = make_record(
        task_id="source", vector=(0.0,) * 8, success=True, score=1.0
    )
    trajectory = Trajectory(
        observations=np.zeros((1, 12), dtype=np.float32),
        actions=np.zeros((0, 2), dtype=np.float32),
        rewards=np.zeros(0, dtype=np.float32),
        terminated=np.zeros(0, dtype=np.bool_),
        truncated=np.zeros(0, dtype=np.bool_),
    )

    try:
        store.add_episode(record, trajectory)
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("Empty trajectory should be rejected")
