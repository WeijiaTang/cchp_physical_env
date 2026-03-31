# Multi-Seed Yearly Eval Snapshot

Timestamp: `2026-03-30 Asia/Shanghai`

## Data Sources

Independent yearly `2025` evals were completed for:

- `DDPG+rule_residual` seeds `1025/1026/1027`
- `rbDQN` seeds `1025/1026/1027`
- `PPO+rule_residual` seeds `1025/1026/1027`
- `SAC+rule_residual` seeds `1025/1026/1027`
- `TD3+rule_residual` seeds `1025/1026/1027`

Primary numeric export:

- `results/tables/paper/drl_multi_seed_yearly_eval_2026-03-30.csv`
- `results/tables/paper/drl_multi_seed_yearly_eval_2026-03-30.md`

## Executive Summary

- The seed-sensitivity story is now materially cleaner because the comparison has been moved from training-pool indicators to independent yearly evals.
- All five algorithms now satisfy `heat reliability = 1.0` and `violation_rate = 0.0` across all three seeds.
- The remaining discriminators are annual cost, cooling reliability, yearly cooling unmet, and switching burden.

## Table 1. Mean ± Std Across Seeds

| Model | total_cost (M) | rel_cool | unmet_c (MWh) | starts_gt | starts_ech | Reading |
|---|---:|---:|---:|---:|---:|---|
| `DDPG+rule_residual` | `14.690 ± 0.276` | `0.992749 ± 0.004006` | `43.860 ± 24.230` | `391.0 ± 14.9` | `336.7 ± 20.4` | Best balanced same-info seed behavior |
| `rbDQN` | `14.911 ± 0.386` | `0.994671 ± 0.006810` | `32.237 ± 41.195` | `468.0 ± 56.6` | `529.0 ± 0.0` | Best upside, but one weak cooling seed remains |
| `TD3+rule_residual` | `15.477 ± 0.307` | `0.990314 ± 0.007839` | `58.588 ± 47.416` | `387.7 ± 15.7` | `356.7 ± 28.6` | Moderately stable in cost, but cooling tail risk is visible |
| `SAC+rule_residual` | `15.656 ± 1.277` | `0.990599 ± 0.003966` | `56.864 ± 23.990` | `385.7 ± 1.2` | `393.3 ± 42.3` | Largest cost variance across seeds |
| `PPO+rule_residual` | `15.703 ± 0.596` | `0.995831 ± 0.002622` | `25.220 ± 15.861` | `278.3 ± 90.1` | `666.3 ± 75.5` | Cooling is acceptable, but annual cost and ECH dependence are both high |

## Table 2. Per-Seed Highlights

| Model | Best seed reading | Weak seed reading |
|---|---|---|
| `DDPG+rule_residual` | `seed=1027`: `14.302 M`, `rel_cool=0.995465` | `seed=1026`: `rel_cool=0.987086`, `unmet_c=78.117 MWh` |
| `rbDQN` | `seed=1026`: `14.448 M`, `rel_cool=0.999894` | `seed=1027`: `15.394 M`, `rel_cool=0.985052` |
| `PPO+rule_residual` | `seed=1025`: `rel_cool=0.998893`, `unmet_c=6.699 MWh` | `seed=1025` also has the highest cost: `16.360 M` |
| `SAC+rule_residual` | `seed=1027`: `14.536 M`, `rel_cool=0.993862` | `seed=1025`: cost spikes to `17.443 M`; `seed=1026` drops to `rel_cool=0.985017` |
| `TD3+rule_residual` | `seed=1026`: `15.474 M`, `rel_cool=0.996776` | `seed=1027`: `rel_cool=0.979283`, `unmet_c=125.319 MWh` |

## Manuscript Reading

- `DDPG+rule_residual` is currently the strongest candidate if the paper prioritizes a stable continuous-control narrative.
- `rbDQN` still represents the strongest discrete-control line and remains competitive in cost, but it has a more visible bad-seed tail.
- `PPO+rule_residual` is no longer an immediate failure line after independent yearly eval; however, its cost and ECH burden remain too large for it to become the main paper champion.
- `SAC+rule_residual` and `TD3+rule_residual` still look more like secondary baselines than main-result candidates.

## Bottom Line

- The previous concern that seed sensitivity had only been studied on the training pool is now resolved for the five main DRL algorithms.
- The current same-info multi-seed shortlist should be:
  - `DDPG+rule_residual` as the main stable continuous-control line
  - `rbDQN` as the main discrete-control line
  - `PPO+rule_residual` as a useful but secondary comparison line
