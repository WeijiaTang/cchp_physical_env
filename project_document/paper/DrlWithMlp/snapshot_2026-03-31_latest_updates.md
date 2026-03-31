# Snapshot 2026-03-31 Latest Updates

Timestamp: 2026-03-31

Data sources:
- `kaggle/CCHP-SB3-dqn+mlp03310740`
- `kaggle/CCHP-SB3-ppo+mlp03310740`
- local DDPG runs under `runs/seed_sens_20260329`
- SAC and TD3 train batches under `runs/funhpc/0331`

## Main message

This update adds fourteen new final yearly evaluations and confirms that the `funhpc/0331` SAC and TD3 batches had already completed training before evaluation.

The strongest final result in this batch is still the latest rbDQN run:
- total cost `15.127 M`
- heat reliability `1.0000`
- cooling reliability `0.999978`
- unmet cooling `0.133 MWh`
- zero violation

The latest PPO run is slightly cheaper:
- total cost `14.986 M`

but the quality trade-off is clear:
- cooling reliability drops to `0.990912`
- unmet cooling rises to `54.977 MWh`
- ECH starts increase to `789`

Two local DDPG runs have now also been evaluated yearly:
- `seed=1919`: total cost `14.685 M`, cooling reliability `0.986439`
- `seed=810`: total cost `14.820 M`, cooling reliability `0.995529`

## Interpretation for the paper

At this stage, the new DQN result is easier to defend as a primary same-information DRL result because it remains very close to full yearly heat and cooling satisfaction while keeping the policy behavior clean.

The latest PPO run is interesting because it narrows the cost gap, but it does so with visibly thinner cooling reliability and stronger dependence on electric chilling. It is therefore better treated as a competitive but less robust variant, not yet as the flagship result.

The `seed_1919` and `seed_810` eval subdirectories inside these two Kaggle runs reproduce the same yearly metrics as the main eval result. They are therefore useful for packaging consistency checks, but not as independent training-seed sensitivity evidence.

## DDPG reading

With the two new yearly evals, DDPG now has a five-seed yearly set (`1025/1026/1027/1919/810`). Across these five seeds:
- mean total cost is `14.715 ± 0.220 M`
- mean cooling reliability is `0.992043 ± 0.004317`
- mean unmet cooling is `48.132 ± 26.115 MWh`

This means DDPG is still economically strong, but its cooling tail is weaker than the latest DQN result and less clean than a paper flagship should ideally be.

## funhpc status

The `funhpc/0331` SAC and TD3 batches were not copied halfway through training. All ten runs had already reached their configured `1.5M` timesteps with `training_complete = true`, and the yearly evaluations have now been completed.

Across the five yearly seeds:
- `SAC+rule_residual`: `14.668 ± 0.378 M`, `rel_cool = 0.994887 ± 0.001458`
- `TD3+rule_residual`: `14.922 ± 0.280 M`, `rel_cool = 0.991898 ± 0.006589`

This changes the earlier reading in an important way: SAC is no longer a collapsed or unusable line under the current residual-assisted setting. TD3 improves as well, but it still shows a more obvious cooling-tail failure risk.
