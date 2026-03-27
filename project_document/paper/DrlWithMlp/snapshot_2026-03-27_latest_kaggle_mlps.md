# Latest Kaggle MLP Benchmark Snapshot

Timestamp: `2026-03-27 20:50:36 +08:00`

## Data Sources

Latest DRL cohort used in this snapshot:

- `kaggle/CCHP-SB3-dqn+mlp03272030/runs/20260327_032502_853296_train_sb3_dqn_mlp_k32`
- `kaggle/CCHP-SB3-ddpg+mlp03272030/runs/20260327_032418_930697_train_sb3_ddpg_mlp_k32`
- `kaggle/CCHP-SB3-ppo+mlp03272030/runs/20260327_032544_641642_train_sb3_ppo_mlp_k32`
- `kaggle/CCHP-SB3-sac+mlp03272030/runs/20260327_032508_703855_train_sb3_sac_mlp_k32`
- `kaggle/CCHP-SB3-td3+mlp03272030/runs/20260327_032436_184749_train_sb3_td3_mlp_k32`

Comparison cohort for change tracking:

- `kaggle/CCHP-SB3-dqn+mlp03270750/runs/20260326_142945_286021_train_sb3_dqn_mlp_k32`
- `kaggle/CCHP-SB3-ddpg+mlp03270750/runs/20260326_142848_668259_train_sb3_ddpg_mlp_k32`
- `kaggle/CCHP-SB3-ppo+mlp03270750/runs/20260326_142904_135913_train_sb3_ppo_mlp_k32`
- `kaggle/CCHP-SB3-sac+mlp03270750/runs/20260326_142943_662413_train_sb3_sac_mlp_k32`
- `kaggle/CCHP-SB3-td3+mlp03270750/runs/20260326_142919_344702_train_sb3_td3_mlp_k32`

Primary files read from each run:

- `train/summary.json`
- `eval/summary.json`
- `eval/behavior_metrics.json`
- `eval/step_log_light.csv`

## Executive Summary

- This batch is the clearest evidence so far that the recent shared-physics repair is really propagating into Kaggle DRL runs.
- All five latest DRL runs achieved:
  - `electric reliability = 1.0`
  - `heat reliability = 1.0`
  - `unmet_h = 0`
  - `violation_rate = 0`
  - `invalid_abs_request = 0`
- The main competition is now no longer “who avoids heat collapse”; it is “who best manages cooling quality and economic cost”.
- Current same-info ranking:
  - `rbDQN` remains the strongest overall result
  - `DDPG+rule_residual` is the strongest continuous-control result
  - `TD3+rule_residual` is now close enough to remain a serious secondary candidate

## Table 1. Same-Info DRL Leaderboard

Paper gate used for this internal snapshot:

- electric reliability `= 1.0`
- heat reliability `>= 0.99`
- cooling reliability `>= 0.99`

| Rank | Model | total_cost (M) | rel_heat | rel_cool | unmet_h (MWh) | unmet_c (MWh) | viol_rate | export_penalty (M) | starts_gt | starts_ech | Final-year gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `rbDQN` | 13.120 | 1.000000 | 0.998365 | 0.000 | 9.891 | 0.000000 | 0.599 | 413 | 529 | Pass |
| 2 | `DDPG+rule_residual` | 14.868 | 1.000000 | 0.998331 | 0.000 | 10.095 | 0.000000 | 1.505 | 383 | 350 | Pass |
| 3 | `TD3+rule_residual` | 15.017 | 1.000000 | 0.997372 | 0.000 | 15.899 | 0.000000 | 1.407 | 384 | 315 | Pass |
| 4 | `PPO+rule_residual` | 15.041 | 1.000000 | 0.994244 | 0.000 | 34.818 | 0.000000 | 0.753 | 348 | 1091 | Pass |
| 5 | `SAC+rule_residual` | 15.372 | 1.000000 | 0.986607 | 0.000 | 81.016 | 0.000000 | 1.365 | 399 | 458 | Fail |

## Table 2. Training-Protocol Verification

This table is included to verify that the latest Kaggle outputs really reflect the intended final training protocol.

| Model | Residual / warm-start / prefill evidence | Train best-gate passed | Selected-best timestep | Final LR | Plateau fine-tune |
|---|---|---|---:|---:|---|
| `rbDQN` | `rule_replay_prefill_v1`, no residual | Pass | 750000 | 5e-5 | Yes |
| `DDPG+rule_residual` | residual enabled + `rule_replay_prefill_v1` | Pass | 950000 | 5e-5 | Yes |
| `TD3+rule_residual` | residual enabled + `rule_replay_prefill_v1` | Pass | 700000 | 5e-5 | Yes |
| `PPO+rule_residual` | residual enabled + PPO warm-start applied | Fail | 1200000 | 5e-5 | Yes |
| `SAC+rule_residual` | residual enabled + `rule_replay_prefill_v1` | Fail | 450000 | 5e-5 | Yes |

Interpretation:

- The latest Kaggle runs do use the intended `0.99 / 0.99` train-time gate.
- The low-LR fine-tune mechanism is clearly active in all five runs.
- Off-policy algorithms were no longer trained “cold”; they were launched with rule-based replay prefill.

## Table 3. Physics-Uptake Evidence

This table is the most direct evidence that the recent shared physical-environment change is actually reflected in Kaggle DRL behavior.

| Model | abs_blocked_rate | u_abs_phys_nonzero_rate | q_abs_cool mean (MW) | cool_unmet_steps | Interpretation |
|---|---:|---:|---:|---:|---|
| `rbDQN` | 0.8374 | 0.1626 | 0.2265 | 90 | Conservative ABS use, but still clearly nonzero |
| `DDPG+rule_residual` | 0.2025 | 0.7975 | 0.7129 | 112 | Strong continuous ABS participation |
| `TD3+rule_residual` | 0.2132 | 0.7868 | 0.7085 | 160 | Similar to DDPG; more aggressive than DQN |
| `PPO+rule_residual` | 0.3434 | 0.6566 | 0.8458 | 368 | ABS is active, but cooling quality still lags |
| `SAC+rule_residual` | 0.2923 | 0.7077 | 0.7503 | 760 | ABS is active, but control quality remains weak |

Interpretation:

- In the earlier “ABS trapped behind deadzone” failure mode, we often saw near-zero useful ABS participation or a flood of invalid requests.
- That is not what the latest Kaggle runs show.
- All five latest runs have `invalid_abs_request = 0`, and all five have nonzero physical ABS participation.
- Therefore, the new main bottleneck is no longer “ABS cannot enter the control loop”; it is “how effectively each algorithm uses the now-available cooling pathway”.

## Table 4. Change vs Previous Kaggle Batch (`03270750`)

| Model | cost delta (M) | cool rel delta | unmet_c delta (MWh) | viol_rate delta | Reading |
|---|---:|---:|---:|---:|---|
| `rbDQN` | -1.544 | -0.001339 | +8.097 | +0.000000 | Much cheaper, slightly worse cooling |
| `DDPG+rule_residual` | +0.479 | +0.002253 | -13.629 | -0.004366 | More expensive, but cleaner and more reliable |
| `TD3+rule_residual` | -0.425 | -0.001263 | +7.638 | -0.012785 | Cost improved, but cooling softened slightly |
| `PPO+rule_residual` | -0.026 | +0.011400 | -68.958 | -0.000114 | Largest cooling-quality improvement |
| `SAC+rule_residual` | +0.854 | -0.005273 | +31.895 | -0.008219 | Cleaner constraint-wise, but still weaker overall |

Interpretation:

- The latest batch is not a uniform “all metrics improved” story.
- It is a more meaningful engineering story:
  - several algorithms became cleaner and more physically plausible
  - `PPO` improved substantially on cooling reliability
  - `DDPG` improved in cooling quality and constraint cleanliness
  - `DQN` remained the cheapest result, even though its cooling margin softened slightly

## Current Paper-Narrative Candidate

Recommended main narrative for the DRL-MLP line:

1. The recent environment-side repair successfully removed the previous dominant failure mode on the heat side.
2. After that repair, all five latest DRL algorithms achieved perfect electricity and heat reliability with zero yearly heat unmet.
3. The competition among algorithms moved to cooling quality and economy, which is a much more credible stage for paper discussion.
4. `rbDQN` remains the strongest same-info method in total cost, consistent with the reference-paper intuition that structured/discrete decisions remain advantageous when unit switching matters.
5. `DDPG+rule_residual` is now the strongest continuous-control candidate because it nearly matches `rbDQN` on yearly cooling quality while keeping a smoother and more compact control pattern than the more unstable continuous baselines from earlier cohorts.

## Recommended Use in the Paper

Recommended status: `good enough for main-result drafting`.

Safe to use now for:

- main same-info DRL result tables
- result-section draft writing
- preliminary comparison figures
- algorithm discussion around discrete vs continuous control

Still recommended before final manuscript lock:

1. rerun `rule / easy_rule` with the exact same final code snapshot and export style
2. rerun the Oracle side table under the same final code snapshot
3. add one final multi-seed aggregation round for the shortlisted DRL methods

## Bottom Line

- The latest Kaggle batch is materially better than the older unstable DRL cohorts.
- The recent modifications are not “missing from Kaggle”; they are clearly visible in both protocol metadata and yearly behavior.
- The current same-info shortlist is:
  - `rbDQN` as the main best-result line
  - `DDPG+rule_residual` as the main continuous-control line
  - `TD3+rule_residual` as a useful secondary continuous baseline
