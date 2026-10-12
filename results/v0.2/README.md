# UAS v0.2 controlled-memory results

This directory contains the complete UAS v0.2 controlled memory-transfer benchmark. It evaluates five conditions on two harder held-out continuous-control tasks with equalized replay budgets and ten seeds per condition.

## Result

Across 100 complete training runs and 1,300 evaluations, all memory conditions demonstrated a statistically significant improvement in sample efficiency over learning from scratch (`no_memory`):

| Condition | Retrieved episodes | Replay transitions | Mean AUC delta vs. no memory | 95% Confidence interval | Negative transfer rate | Mean steps saved to 0.8 success | Mean final success delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No memory | 0 | 0 | 0.000 | — | — | baseline | baseline (94.2%) |
| Naive | 12 | 200 | +0.138 | [+0.063, +0.214] | 20% | -861 steps | +5.6% |
| Random | 12 | 200 | +0.120 | [+0.054, +0.186] | 15% | -833 steps | +5.8% |
| Similarity | 12 | 200 | +0.085 | [+0.012, +0.158] | 25% | -956 steps | +3.6% |
| Governed | 12 | 200 | +0.081 | [+0.009, +0.153] | 25% | -694 steps | +4.8% |

*All statistics are reported over 20 paired runs (10 on `hard_upper`, 10 on `hard_lower`) against matching `no_memory` seeds.*

## Key Findings & Interpretation

1. **Memory prefill reliably accelerates online learning:**
   Injecting 200 transitions into the SAC replay buffer consistently saved between 694 and 956 environment steps to reach the 80% success threshold. The 95% confidence intervals for all four memory modes are strictly above zero.
2. **Naive and random replay outperformed governed and similarity retrieval:**
   Naive retrieval produced the highest overall AUC delta (+0.138) and random retrieval had the lowest negative-transfer rate (15%). Governed retrieval (+0.081) and similarity retrieval (+0.085) showed lower mean improvements.
3. **Why greedy governance fell short under shift:**
   In `hard_upper` and `hard_lower`, obstacle bounds are shifted relative to the source tasks (`open`, `upper_gap`, `lower_gap`). Governed retrieval strictly selected high-reward source trajectories. Because those paths were optimized for older obstacle placements, agents experienced distribution bias when the layout shifted. Conversely, naive and random replay contained diverse exploratory transitions that provided wider state-space coverage for the critic.
4. **Takeaway for v0.3:**
   Governed retrieval should not filter solely for source-task success. Effective experience memory for out-of-distribution tasks must govern for **task relevance combined with state-space diversity and boundary coverage**.

## Configuration

- Held-out tasks: `hard_upper` and `hard_lower` (narrow passage navigation)
- Conditions: `no_memory`, `random`, `naive`, `similarity`, `governed`
- Seeds: 0 through 9 (10 seeds per task/condition)
- Training: 3,000 environment steps per run
- Evaluation: every 250 steps, evaluated over 25 episodes
- Warm-up: `learning_starts = 100` across all conditions
- Replay budget: exactly 12 source episodes and exactly 200 balanced transitions per memory run
- Total evaluations: 1,300 evaluation records across 100 training runs

## Published files

- [`learning_curves.csv`](learning_curves.csv) — full evaluation log for all 100 runs.
- [`summary.json`](summary.json) — summary metrics, AUC, threshold timing, and 95% confidence intervals.
- [`comparisons.csv`](comparisons.csv) — paired statistical comparisons against `no_memory`.
- [`replay_manifest.json`](replay_manifest.json) — exact audit trail of selected source episodes and sampled transitions.
- [`run_config.json`](run_config.json) — experiment configuration and dependency versions.
- [`success_rate.png`](success_rate.png) — learning curves with confidence intervals.
