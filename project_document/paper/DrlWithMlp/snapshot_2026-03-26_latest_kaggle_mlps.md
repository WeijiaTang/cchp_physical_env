# Latest Kaggle MLP Benchmark Snapshot

Timestamp: `2026-03-26 21:30 Asia/Shanghai`

## Data Sources

Latest DRL cohort used in this snapshot:

- `kaggle/CCHP-SB3-dqn+mlp03262100/runs/20260326_034951_233565_train_sb3_dqn_mlp_k32`
- `kaggle/CCHP-SB3-ddpg+mlp03262100/runs/20260326_035328_996847_train_sb3_ddpg_mlp_k32`
- `kaggle/CCHP-SB3-ppo+mlp03262100/runs/20260326_035238_271631_train_sb3_ppo_mlp_k32`
- `kaggle/CCHP-SB3-sac+mlp03262100/runs/20260326_035055_197126_train_sb3_sac_mlp_k32`
- `kaggle/CCHP-SB3-td3+mlp03262100/runs/20260326_035143_375863_train_sb3_td3_mlp_k32`

Reference baselines used here:

- `tmp_verify/rule_dynamic_om_regress/eval/summary.json`
- `tmp_verify/easy_rule_dynamic_om_regress/eval/summary.json`
- `tmp_verify/oracle_milp_full_v3/eval/summary.json`
- `tmp_verify/oracle_ga_full_v3/eval/summary.json`

Comparison cohort for change tracking:

- `03260630`

## Executive Summary

- This batch is materially better than the previous `03260630` batch on the heat side.
- The most important structural change is not a small cost tweak:
  - all five latest DRL runs achieved `heat reliability = 1.0`
  - all five latest DRL runs achieved `unmet_h = 0`
  - all five latest DRL runs reduced `idle_heat_backup` to `0`
- This strongly suggests that the new heat-side observation set and boiler lower-bound shield are working as intended.
- Current main same-info DRL candidates:
  - `rbDQN`: best overall KPI
  - `DDPG+rule_residual`: best continuous-control trade-off

## Table 1. Same-Info DRL Leaderboard

| Rank | Model | total_cost (M) | rel_heat | rel_cool | unmet_h (MWh) | unmet_c (MWh) | viol_rate | export_penalty (M) | starts_gt | starts_ech | Note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `rbDQN` | 14.387 | 1.000000 | 0.999179 | 0.000 | 4.964 | 0.000000 | 1.308 | 531 | 529 | Best aggregate KPI; discrete control still strongest |
| 2 | `DDPG+rule_residual` | 14.389 | 1.000000 | 0.996078 | 0.000 | 23.725 | 0.004366 | 1.231 | 374 | 318 | Best continuous-control result; smoother than DQN |
| 3 | `SAC+rule_residual` | 14.518 | 1.000000 | 0.991880 | 0.000 | 49.121 | 0.008219 | 1.230 | 420 | 426 | Much improved vs prior cohort, but still less clean than DDPG |
| 4 | `PPO+rule_residual` | 15.067 | 1.000000 | 0.982844 | 0.000 | 103.776 | 0.000114 | 0.713 | 354 | 1045 | Heat fixed, cooling still weak |
| 5 | `TD3+rule_residual` | 15.442 | 1.000000 | 0.998634 | 0.000 | 8.261 | 0.012785 | 1.841 | 427 | 253 | Cooling strong, but violations/export penalty still heavy |

## Table 2. Heat-Side Repair Effect vs Previous Cohort (`03260630`)

| Model | cost delta (M) | heat rel delta | cool rel delta | unmet_h delta (MWh) | unmet_c delta (MWh) | viol_rate delta | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `rbDQN` | -9.885 | +0.000009 | -0.000821 | -0.137 | +4.964 | +0.000000 | Massive cost drop; heat unchanged because it was already near-perfect |
| `DDPG+rule_residual` | -1.536 | +0.013480 | -0.000332 | -202.144 | +2.008 | -0.008533 | Strongest evidence that heat repair worked |
| `PPO+rule_residual` | -0.258 | +0.002532 | -0.014605 | -37.966 | +88.345 | +0.000057 | Heat improved, but cooling degraded materially |
| `SAC+rule_residual` | -3.777 | +0.001613 | -0.007177 | -24.193 | +43.411 | +0.008219 | No longer frozen, but still not as stable as desired |
| `TD3+rule_residual` | -0.027 | +0.010647 | +0.004202 | -159.665 | -25.420 | -0.001884 | Heat and cooling both improved, but cost gain is small |

## Table 3. Engineering Baselines and Oracle Side Leaderboard

| Group | Policy | total_cost (M) | rel_heat | rel_cool | unmet_h (MWh) | unmet_c (MWh) | viol_rate | starts_gt | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Same-info baseline | `rule` | 18.650 | 0.999110 | 1.000000 | 13.340 | 0.000 | 0.000000 | 391 | Strong engineering baseline; now clearly beaten on cost by latest DRL |
| Same-info baseline | `easy_rule` | 20.269 | 1.000000 | 1.000000 | 0.000 | 0.000 | 0.000000 | 80 | Reliability-perfect but economically weak |
| Oracle / planner | `milp_mpc` | 9.098 | 0.986219 | 1.000000 | 206.662 | 0.000 | 0.000000 | 366 | Lower raw cost, but heat reliability still below main paper gate |
| Oracle / planner | `ga_mpc` | 9.136 | 0.985736 | 1.000000 | 213.897 | 0.000 | 0.000000 | 366 | Similar to MILP; still not same-info comparable |

## Table 4. Why the Heat Repair Looks Real

| Model | heat_unmet_steps | idle_heat_backup_steps | Key diagnostic signal |
|---|---:|---:|---|
| `rbDQN` | 0 | 0 | `heat_backup_shield_applied = 19681` |
| `DDPG+rule_residual` | 0 | 0 | `heat_backup_shield_applied = 12373` |
| `PPO+rule_residual` | 0 | 0 | `heat_backup_shield_applied = 7761` |
| `SAC+rule_residual` | 0 | 0 | `heat_backup_shield_applied = 11160` |
| `TD3+rule_residual` | 0 | 0 | `heat_backup_shield_applied = 12214` |

Interpretation:

- The latest cohort does not merely “look better” in total cost.
- The specific failure mode identified earlier, “heat deficit + backup idle”, is effectively removed in this batch.
- Therefore, the current gains are consistent with the intended code change, not just random checkpoint luck.

## Current Paper-Narrative Candidate

Recommended main narrative:

1. The heat-side observability and same-info boiler shield repaired the dominant failure mode in the previous DRL runs.
2. After that repair, same-info DRL clearly outperformed `rule` and `easy_rule` on economic cost while maintaining acceptable reliability.
3. `rbDQN` remained the strongest overall method, which is consistent with the reference paper’s observation that discrete action design is advantageous when on/off structure is important.
4. Among continuous methods, `DDPG+rule_residual` is the strongest engineering compromise because it combines low cost with lower switching burden than `rbDQN`.

## What Is Still Not Fully Final

- The latest Kaggle training summaries still show training best-gate thresholds at `0.999`, not the intended final default `0.99`.
- The `rule / easy_rule` baselines used here are from local `tmp_verify`, not a final “same-commit / same-export style” rerun.
- Oracle is still not eligible for the same main leaderboard because its raw low cost is achieved under lower heat reliability than the paper gate.

## Provisional Use in the Paper

Recommended status: `near-final, but not final-lock`.

Safe to use now for:

- internal paper drafting
- figure/table prototyping
- main result narrative drafting

Recommended before final submission:

1. rerun `rule / easy_rule` with the same final code snapshot and export style
2. rerun one confirmation DRL batch with the intended training best-gate default propagated as `0.99`
3. freeze the final same-info leaderboard and Oracle side leaderboard separately
