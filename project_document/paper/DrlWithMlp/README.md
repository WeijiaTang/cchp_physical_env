# DRL With MLP Result Notes

Timestamp: `2026-03-26 21:30 Asia/Shanghai`

## Purpose

This folder stores paper-oriented benchmark summaries for the `MLP`-backbone DRL line in this repository.
The goal is to convert raw experiment outputs into dense, review-friendly markdown records.

## Scope

- Same-info DRL leaderboard:
  - `rbDQN`
  - `DDPG+rule_residual`
  - `PPO+rule_residual`
  - `SAC+rule_residual`
  - `TD3+rule_residual`
- Engineering baselines:
  - `rule`
  - `easy_rule`
- Oracle / planner side leaderboard:
  - `milp_mpc`
  - `ga_mpc`

## Source Convention

Primary data sources come from:

- `kaggle/CCHP-SB3-<algo>+mlp<stamp>/runs/*/train/summary.json`
- `kaggle/CCHP-SB3-<algo>+mlp<stamp>/runs/*/train/convergence_summary.json`
- `kaggle/CCHP-SB3-<algo>+mlp<stamp>/runs/*/eval/summary.json`
- `kaggle/CCHP-SB3-<algo>+mlp<stamp>/runs/*/eval/behavior_metrics.json`
- `tmp_verify/*/eval/summary.json` for local baselines and Oracle snapshots

## File Layout

- `README.md`
  - folder convention and interpretation rules
- `snapshot_2026-03-26_latest_kaggle_mlps.md`
  - latest paper-style benchmark snapshot for `03262100`
- `snapshot_2026-03-27_latest_kaggle_mlps.md`
  - latest paper-style benchmark snapshot for `03272030/03282030`

## Interpretation Rules

- Same-info DRL and Oracle/planner must not be mixed into one main leaderboard.
- `total_cost` alone is not sufficient.
  - Heat/cooling reliability
  - unmet energy
  - violation rate
  - export penalty
  - starts / toggle burden
  must be read together.
- Current main paper gate:
  - electric reliability `= 1.0`
  - heat reliability `>= 0.99`
  - cooling reliability `>= 0.99`
- Oracle remains an upper-bound reference, not the main teacher for same-info DRL.

## Current Status

As of the latest Kaggle MLP batch (`03272030/03282030`):

- The recent shared-physics and training-control changes are clearly present in Kaggle artifacts:
  - `best gate heat/cool = 0.99`
  - `plateau_control.enabled = true`
  - final fine-tune learning rate reached `5e-5`
  - off-policy runs used `rule_replay_prefill_v1`
- All five latest DRL runs reached:
  - `electric reliability = 1.0`
  - `heat reliability = 1.0`
  - `unmet_h = 0`
  - `violation_rate = 0`
- All five latest DRL runs show nonzero physical ABS participation, which is strong evidence that the updated shared physics is actually being used during Kaggle training/evaluation.
- Main same-info candidates are now:
  - `rbDQN` for strongest aggregate KPI
  - `DDPG+rule_residual` for strongest continuous-control trade-off
- Current caution points before final paper lock-in:
  - `SAC+rule_residual` still misses the cooling gate on final-year eval
  - `PPO+rule_residual` passes the final-year gate but did not pass the training selection gate on the 2024 fixed-window pool
  - `rule / easy_rule / Oracle` should still be rerun once under the exact same final code snapshot for the final manuscript tables
