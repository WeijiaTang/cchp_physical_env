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

As of the latest Kaggle MLP batch:

- The heat-side repair appears to be effective.
- All five latest DRL runs reached `heat reliability = 1.0` and `unmet_h = 0`.
- Main same-info candidates are now:
  - `rbDQN` for strongest aggregate KPI
  - `DDPG+rule_residual` for strongest continuous-control trade-off
- Before final paper lock-in, one more confirmation run is still recommended:
  - rerun `rule / easy_rule` with the same final code snapshot
  - confirm the training best-gate default is truly propagated as `0.99` in Kaggle
