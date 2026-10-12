from uas.experiment import summarize_results


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
