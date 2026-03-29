# Latest Oracle + Kaggle DRL Master Snapshot

Timestamp: `2026-03-29 09:40:08 +08:00`

## Data Sources

Latest planner-side Oracle snapshot:

- `runs/oracle_milp_full/eval/summary.json`
- `runs/oracle_milp_full/eval/cost_breakdown.csv`
- `runs/oracle_milp_full/eval/step_log_light.csv`

Latest Kaggle DRL cohort:

- `kaggle/CCHP-SB3-ddpg+mlp03290830/runs/20260328_112434_292954_train_sb3_ddpg_mlp_k32`
- `kaggle/CCHP-SB3-dqn+mlp03290830/runs/20260328_112435_662485_train_sb3_dqn_mlp_k32`
- `kaggle/CCHP-SB3-ppo+mlp03290830/runs/20260328_112455_949521_train_sb3_ppo_mlp_k32`
- `kaggle/CCHP-SB3-sac+mlp03290830/runs/20260328_112454_348432_train_sb3_sac_mlp_k32`
- `kaggle/CCHP-SB3-td3+mlp03290830/runs/20260328_112450_593601_train_sb3_td3_mlp_k32`

Primary files read from each DRL run:

- `train/summary.json`
- `eval/summary.json`

## Executive Summary

- The latest Kaggle DRL batch is now strong enough for main-result drafting: all five runs reached `electric reliability = 1.0`, `heat reliability = 1.0`, `yearly unmet heat = 0`, and `violation_rate = 0`.
- Within the same-info DRL group, `PPO+rule_residual` is the current cost leader, `rbDQN` is the strongest cooling-reliability line, and `DDPG+rule_residual` is the strongest balanced continuous-control candidate.
- The repaired Oracle is now also usable as the planner-side reference: it reaches `total_cost = 8.766 M`, `heat reliability = 1.0`, `cooling reliability = 0.999883`, with only `0.707 MWh` yearly cooling unmet and essentially negligible yearly violation.

## Table 1. Master Result Table

Paper gate for the same-info DRL line:

- electric reliability `= 1.0`
- heat reliability `>= 0.99`
- cooling reliability `>= 0.99`

| Group | Rank | Model | total_cost (M) | rel_e | rel_h | rel_c | unmet_h (MWh) | unmet_c (MWh) | viol_rate | export_penalty (M) | starts_gt | starts_ech | Reading |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Same-info DRL | 1 | `PPO+rule_residual` | 14.306 | 1.000000 | 1.000000 | 0.997348 | 0.000 | 16.045 | 0.000000000 | 0.620 | 366 | 810 | Lowest total cost in the latest DRL batch |
| Same-info DRL | 2 | `DDPG+rule_residual` | 14.893 | 1.000000 | 1.000000 | 0.998077 | 0.000 | 11.635 | 0.000000000 | 1.367 | 371 | 272 | Best balanced continuous-control result |
| Same-info DRL | 3 | `rbDQN` | 14.995 | 1.000000 | 1.000000 | 0.999716 | 0.000 | 1.718 | 0.000000000 | 1.294 | 522 | 529 | Best cooling reliability among DRL, but higher switching burden |
| Same-info DRL | 4 | `TD3+rule_residual` | 15.104 | 1.000000 | 1.000000 | 0.995213 | 0.000 | 28.958 | 0.000000000 | 1.313 | 389 | 353 | Usable, but clearly behind the top three |
| Same-info DRL | 5 | `SAC+rule_residual` | 15.257 | 1.000000 | 1.000000 | 0.989801 | 0.000 | 61.695 | 0.000000000 | 1.891 | 414 | 437 | Weakest cooling-quality result in the latest batch |
| Planner-side reference | 1 | `MILP-MPC Oracle` | 8.766 | 1.000000 | 1.000000 | 0.999883 | 0.000 | 0.707 | 0.000028539 | 0.018 | 365 | 725 | Current best planner-side reference after planner repair |

## Table 2. Oracle Health Check

The latest Oracle result should be read as a repaired planner-side benchmark rather than as a stale or fallback-driven artifact.

| Item | Value | Interpretation |
|---|---:|---|
| `fallback_dispatch_count` | 0 | No rule-style emergency fallback was needed |
| `online_repair_steps` | 0 | No online repair was needed in the final yearly replay |
| `optimizer_partial_solution_count` | 355 | Partial MILP solutions were accepted when available |
| `optimizer_failure_count` | 0 | No hard optimizer failure remained |
| `cooling unmet` | 0.707 MWh | Small enough for manuscript-level reference use |
| `violation_step_count` | 1 | Only one yearly step produced a counted violation |

## Table 3. Manuscript Reading

| Topic | Current reading |
|---|---|
| Main same-info winner | `PPO+rule_residual` is the current lowest-cost DRL line under the latest Kaggle batch. |
| Main continuous-control line | `DDPG+rule_residual` is the strongest continuous-control candidate because it remains close to the DRL cost frontier while keeping stronger cooling quality than `PPO`. |
| Main reliability-heavy DRL line | `rbDQN` delivers the best yearly cooling reliability among DRL and remains highly credible for the paper's discrete-control discussion. |
| Oracle status | The repaired `MILP-MPC Oracle` is now strong enough to serve as the planner-side upper-bound style reference in the manuscript. |
| Result-section readiness | The latest DRL cohort plus the repaired Oracle are now good enough for structured result-section drafting, with Oracle and same-info DRL kept explicitly separated in the narrative. |

## Recommended Use in the Paper

Recommended manuscript framing at the current stage:

1. Use the latest five-model Kaggle batch as the main same-info DRL result table.
2. Use the repaired Oracle as the planner-side reference table.
3. Highlight three different strengths instead of forcing one single DRL narrative:
   - `PPO+rule_residual` for lowest same-info cost
   - `DDPG+rule_residual` for the strongest continuous-control trade-off
   - `rbDQN` for the strongest reliability-oriented DRL line

## Bottom Line

- The newest Kaggle DRL results are materially better than the earlier unstable cohorts and are now suitable for paper drafting.
- The latest Oracle rerun is no longer dominated by fallback or repair artifacts and can now be treated as a credible planner-side benchmark.
- The combined evidence supports a cleaner paper story: a reliable same-info DRL leaderboard on one side, and a repaired high-performance Oracle reference on the other.
