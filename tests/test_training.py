from pathlib import Path

import numpy as np

from uas.envs import ContinuousWorldEnv
from uas.memory import MemoryStore
from uas.rollouts import run_episode, store_episode
from uas.training import make_model, prefill_replay_buffer, query_for_task


def test_retrieved_episode_prefills_sac_replay_buffer(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    source_environment = ContinuousWorldEnv(task_id="upper_gap")

    def move_up(_observation: np.ndarray) -> np.ndarray:
        return np.asarray([0.0, 1.0], dtype=np.float32)

    trajectory, summary = run_episode(source_environment, move_up, seed=0)
    summary["success"] = True
    summary["termination_reason"] = "success"
    record = store_episode(
        store,
        source_environment,
        trajectory,
        summary,
        seed=0,
        agent_id="test-agent",
        policy_version="test-v1",
    )
    assert record.success

    target_environment = ContinuousWorldEnv(task_id="held_out")
    model = make_model(target_environment, seed=0, learning_starts=0)
    retrieved = store.retrieve(query_for_task("held_out"), mode="governed")
    added = prefill_replay_buffer(model, store, retrieved, max_transitions=10)

    assert added == 10
    assert model.replay_buffer.size() == 10
