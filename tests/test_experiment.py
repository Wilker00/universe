import json
from pathlib import Path

from uas.experiment import _checkpoint_results, summarize_results


def test_summary_uses_latest_evaluation_per_seed() -> None:
    rows = [
        {
            "mode": "no_memory",
            "seed": 0,
            "timestep": 0,
            "success_rate": 0.0,
            "mean_reward": -1.0,
            "mean_collisions": 2.0,
        },
        {
            "mode": "no_memory",
            "seed": 0,
            "timestep": 100,
            "success_rate": 1.0,
            "mean_reward": 2.0,
            "mean_collisions": 0.0,
        },
    ]

    summary = summarize_results(rows)

    assert summary["no_memory"]["final_timestep"] == 100
    assert summary["no_memory"]["mean_success_rate"] == 1.0
    assert summary["no_memory"]["mean_best_success_rate"] == 1.0
    assert summary["no_memory"]["seeds_reaching_threshold"] == 1
    assert summary["no_memory"]["mean_steps_to_threshold"] == 100.0


def test_checkpoint_results_preserves_partial_progress(tmp_path: Path) -> None:
    rows = [
        {
            "task_id": "held_out",
            "mode": "governed",
            "seed": 0,
            "timestep": 0,
            "success_rate": 0.0,
            "mean_reward": -1.0,
            "mean_steps": 150.0,
            "mean_collisions": 1.0,
            "retrieved_episodes": 3,
            "prefilled_transitions": 100,
        }
    ]
    _checkpoint_results(tmp_path, rows, expected_runs=1, train_steps=100)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert (tmp_path / "learning_curves.csv").is_file()
    assert summary["_checkpoint"]["evaluation_rows"] == 1
    assert summary["_checkpoint"]["completed_runs"] == 0
    assert not summary["_checkpoint"]["complete"]
    assert not (tmp_path / "success_rate.png").exists()

    rows.append({**rows[0], "timestep": 100, "success_rate": 1.0})
    _checkpoint_results(
        tmp_path,
        rows,
        expected_runs=1,
        train_steps=100,
        include_plot=True,
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["_checkpoint"]["completed_runs"] == 1
    assert summary["_checkpoint"]["complete"]
    assert (tmp_path / "success_rate.png").is_file()
