# Snapshot 2026-03-31 All-Available Multi-Seed Yearly Eval

Timestamp: 2026-03-31

Primary table:
- [drl_all_available_multi_seed_yearly_eval_2026-03-31.md](D:/EnergyStorage/CCHP/results/tables/paper/drl_all_available_multi_seed_yearly_eval_2026-03-31.md)

Aggregate CSV:
- [drl_all_available_multi_seed_yearly_eval_aggregate_2026-03-31.csv](D:/EnergyStorage/CCHP/results/tables/paper/drl_all_available_multi_seed_yearly_eval_aggregate_2026-03-31.csv)

## Scope and boundary

This snapshot summarizes the latest independent yearly `2025` evaluation retained for each available training seed of:
- `DDPG`
- `SAC`
- `TD3`
- `rbDQN`
- `PPO`

The statistical unit is the independently trained run, not the duplicated `seed_*` eval subdirectory inside a single training run.

The currently retained publication-facing training-seed pool is:
- `0, 1, 42, 51, 61, 810, 1025, 1026, 1027, 1919, 114514`

Coverage is algorithm-dependent:
- `DDPG+rule_residual`: 11 seeds
- `SAC+rule_residual`: 11 seeds
- `TD3+rule_residual`: 11 seeds
- `rbDQN`: 11 seeds
- `PPO+rule_residual`: 11 seeds

## Mean reading across all available seeds

| Model | n seeds | total_cost (M) | rel_cool | unmet_c (MWh) | starts_gt | starts_ech |
|---|---:|---:|---:|---:|---:|---:|
| `DDPG+rule_residual` | 11 | `14.827 ± 0.241` | `0.993084 ± 0.004623` | `41.834 ± 27.962` | `385.3 ± 13.8` | `331.7 ± 57.8` |
| `SAC+rule_residual` | 11 | `15.118 ± 0.873` | `0.991727 ± 0.004128` | `50.046 ± 24.970` | `388.4 ± 15.0` | `418.2 ± 35.0` |
| `TD3+rule_residual` | 11 | `15.127 ± 0.357` | `0.991175 ± 0.007667` | `53.380 ± 46.375` | `399.1 ± 27.0` | `431.0 ± 162.9` |
| `rbDQN` | 11 | `14.870 ± 1.420` | `0.997463 ± 0.003013` | `15.349 ± 18.228` | `462.5 ± 48.5` | `529.0 ± 0.0` |
| `PPO+rule_residual` | 11 | `15.822 ± 1.418` | `0.991818 ± 0.012944` | `49.491 ± 78.296` | `260.8 ± 143.6` | `702.9 ± 310.5` |
