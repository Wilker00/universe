# UAS v0.2: Controlled Memory-Transfer Experiment

UAS v0.2 tests a narrower and more defensible question than v0.1:

> Under equal episode and replay-transition budgets, does memory selection improve sample efficiency or reduce negative transfer on harder held-out tasks?

The v0.1 benchmark verified the complete memory-to-training pipeline, but it mixed retrieval strategy with replay volume. Naive and similarity retrieval each supplied 4,062 transitions while governed retrieval supplied 462. Every condition also reached 100% final success, making the task too easy to separate the methods reliably. V0.2 keeps the published v0.1 result unchanged and adds a new controlled protocol.

## Harder held-out tasks

The source memory still comes from `open`, `upper_gap`, and `lower_gap`. V0.2 evaluates two new layouts that use the same observation and action schemas but require longer paths through narrower passages:

| Task | Layout |
| --- | --- |
| `hard_upper` | A wide obstacle blocks the room up to `y=0.80`, leaving a narrow upper route. |
| `hard_lower` | A wide obstacle blocks the room down to `y=0.20`, leaving a narrow lower route. |

The target tasks are never added to source memory by the experiment.

## Conditions

| Mode | Episode-selection rule |
| --- | --- |
| `no_memory` | Train without replay prefill. |
| `random` | Use episodes produced by the random-policy collector. This is a known-poor-memory control. |
| `naive` | Seed-shuffle all schema-compatible episodes without filtering outcomes. |
| `similarity` | Select the most similar schema-compatible episodes, regardless of outcome. |
| `governed` | Require successful, schema-compatible episodes and rank them by retrieval score. |

## Fairness controls

Every memory condition receives:

- exactly 12 source episodes;
- exactly 200 replay transitions;
- 16 or 17 transitions from each selected episode;
- sampling without replacement; and
- deterministic selection and transition-sampling seeds recorded in the manifest.

The run fails before training if any condition cannot satisfy the configured budgets. It never silently duplicates transitions or gives a condition less memory.

All five conditions use the same SAC architecture, hyperparameters, training budget, evaluation seeds, and `learning_starts=100`. V0.1 started memory conditions immediately but delayed the scratch condition; v0.2 removes that confound.

## Default experiment

First create the 48-episode source memory if it does not already exist:

```powershell
uas build-memory --reset
```

Then run the controlled benchmark:

```powershell
uas experiment-v2
```

The defaults are:

- 2 held-out tasks;
- 5 conditions;
- 10 seeds per task and condition;
- 3,000 online environment steps per run;
- evaluation every 250 steps using 25 episodes; and
- 100 total training runs and 1,300 evaluation rows.

This is substantially larger than v0.1 and may take more than an hour on a CPU. A structural smoke run is:

```powershell
uas experiment-v2 `
  --tasks hard_upper `
  --modes no_memory governed `
  --seeds 0 `
  --train-steps 10 `
  --eval-interval 5 `
  --eval-episodes 1
```

The smoke run only validates the pipeline. It is not research evidence.

## Metrics

V0.2 reports, by task and condition:

- final success, reward, and collisions;
- best observed success;
- normalized area under the success curve (AUC);
- first evaluation reaching 80% success;
- earliest evaluation after which success remains at or above 80%;
- approximate 95% confidence intervals for means; and
- matched AUC differences against `no_memory` for the same task and seed.

A run is counted as negative transfer when its normalized success AUC is lower than the matching no-memory run. Negative threshold-step differences are improvements because fewer steps are better.

The confidence intervals are mean plus or minus 1.96 standard errors. With only ten seeds they are descriptive, not a substitute for a larger statistical study.

## Outputs

The default output directory is `artifacts/v0.2-results/`:

- `learning_curves.csv` contains every evaluation;
- `comparisons.csv` contains matched transfer effects against no memory;
- `summary.json` contains task-level and overall aggregates;
- `replay_manifest.json` records selected episodes and exact transition indices;
- `run_config.json` records the complete protocol and package versions; and
- `success_rate.png` plots mean curves with one-standard-deviation shading.

CSV and JSON results are atomically replaced after every evaluation. The chart is refreshed after every completed task/mode/seed run. The summary checkpoint states how many runs are complete.

Policy checkpoints are disabled by default because the report artifacts are sufficient to review the experiment. Pass `--save-models` only when local policy inspection is required. Generated files remain under the ignored `artifacts/` directory until a completed result has been reviewed and deliberately published.

## Publication criteria

Do not claim that governed memory helps merely because one curve crosses 80% first. A useful v0.2 result should show a positive AUC difference with uncertainty that is reasonably separated from zero, a lower negative-transfer rate than unfiltered memory, and similar behavior across both held-out tasks.

Publish the result even if those criteria are not met. A controlled negative result still identifies which part of the UAS hypothesis needs redesign.

## Remaining limitations

- All tasks still belong to one small continuous-navigation environment.
- The task descriptor is a hand-designed vector rather than a learned representation.
- Replay prefill tests episodic experience reuse, not reusable policies or robot skills.
- The approximate confidence interval assumes independent seed-level measurements.
- Hyperparameters are fixed rather than tuned separately for each condition.
- The benchmark does not provide robotics, cross-simulator, or simulation-to-real evidence.
