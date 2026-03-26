[README-en.md](README.md) | [README-zhcn.md](README-zhcn.md) |
# CCHP (Combined Cooling, Heating, and Power) + Renewables Scheduling Research

A Python research codebase for a CCHP physical environment, reproducible training/evaluation pipelines, and benchmark-style experiment collection.

This project supports:
- rule-based baselines
- deep sequence policy training (`mlp` / `transformer` / `mamba`)
- SB3 multi-algorithm training (`ppo` / `sac` / `td3` / `ddpg`) with sequence backbones
- unified CLI for `summary`, `train`, `eval`, `sb3-train`, `sb3-eval`, `ablation`, `calibrate`, and `collect`

- Notebook entry: `main.ipynb`
- CLI entry: `python -m cchp_physical_env`

## What This Repo Is

Core modules:
- `src/cchp_physical_env/__main__.py`: unified CLI entry
- `src/cchp_physical_env/core/data.py`: frozen-schema loading, validation, repair, sampling
- `src/cchp_physical_env/core/config_loader.py`: config loading and CLI override merge
- `src/cchp_physical_env/pipeline/runner.py`: baseline train/eval runner
- `src/cchp_physical_env/pipeline/sequence.py`: sequence policy utilities and adapters
- `src/cchp_physical_env/policy/trainer.py`: PPO-style deep sequence trainer
- `src/cchp_physical_env/policy/sb3.py`: SB3 training/evaluation integration
- `src/cchp_physical_env/config/config.yaml`: environment + training config

## Data Convention

Expected processed files under repo root:
- train: `data/processed/cchp_main_15min_2024.csv`
- eval: `data/processed/cchp_main_15min_2025.csv`

Frozen convention:
- timezone: `Asia/Shanghai`
- resolution: `15min`
- yearly steps: `35040`
- leap-day handling: drop `02/29`

Frozen schema includes:
- load / renewables: `p_dem_mw`, `qh_dem_mw`, `qc_dem_mw`, `pv_mw`, `wt_mw`
- weather: `t_amb_k`, `sp_pa`, `rh_pct`, `wind_speed`, `wind_direction`, `ghi_wm2`, `dni_wm2`, `dhi_wm2`
- price / tax: `price_e`, `price_gas`, `carbon_tax`

Validation / repair rules:
- timestamps must be unique and monotonic increasing
- data must be single-year and aligned to the frozen 15min index
- load-like columns: time interpolation + boundary fill
- weather columns: forward fill
- price/tax columns: hard fail on missing values

## Installation

Python 3.11+ is recommended.

Using `uv`:

```bash
uv sync
```

Using plain `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional dependencies:
- SB3: `uv pip install -e '.[sb3]'`

PyTorch is intentionally not pinned in `pyproject.toml`. Install the correct CPU/CUDA build for your machine before long deep-learning runs.

The `mamba` backbone used by both `sequence_rule` and SB3 now comes from `transformers`, so there is no separate `mamba-ssm` package or extra in this repo.

## CLI Quick Start

Windows / PowerShell (`pwsh`):

```powershell
uv run python -m cchp_physical_env summary
uv run python -m cchp_physical_env --help
```

Linux / Debian / bash:

```bash
python -m cchp_physical_env summary
python -m cchp_physical_env --help
```

## Shell Conventions

PowerShell (`pwsh`) line continuation uses the backtick:

```powershell
uv run python -m cchp_physical_env train `
  --policy rule `
  --episodes 800 `
  --episode-days 14 `
  --seed 40
```

Linux / Debian / bash line continuation uses `\`:

```bash
python -m cchp_physical_env train \
  --policy rule \
  --episodes 800 \
  --episode-days 14 \
  --seed 40
```

Below, Windows examples use `uv run python -m ...`, while Linux examples use direct `python -m ...`.

## Training Parameter Guide

### Deep sequence training arguments

Use these arguments when `train` runs the built-in PPO-style sequence trainer, i.e.:
- `--policy sequence_rule`
- `--sequence-adapter mlp|transformer|mamba`

Recommended meaning of major arguments:
- `--history-steps`: sequence window length fed into the backbone; larger values give longer temporal context but increase memory and compute.
- `--episode-days`: number of days sampled per training episode; controls horizon length and rollout diversity.
- `--train-steps`: total rollout steps collected for training updates.
- `--batch-size`: minibatch size per gradient update.
- `--update-epochs`: how many passes are made over each rollout batch.
- `--lr`: optimizer learning rate.
- `--device`: `auto`, `cpu`, `cuda`, or `cuda:<index>`.
- `--seed`: random seed for reproducibility.

Practical interpretation:
- if GPU memory is tight, reduce `--history-steps` or `--batch-size`
- if training is unstable, first lower `--lr`
- if policy underfits, increase `--train-steps`
- if rollout horizon is too short for storage dynamics, increase `--episode-days`

### Baseline / heuristic arguments

Use these arguments for lightweight baselines:
- `--policy rule`
- `--policy random`
- `--policy sequence_rule --sequence-adapter rule`

Main arguments:
- `--episodes`: number of sampled episodes for baseline training.
- `--episode-days`: length of each sampled episode.
- `--history-steps`: only meaningful for `sequence_rule` variants.
- `--seed`: reproducibility control.

### SB3 training arguments

Use these arguments with either explicit `sb3-train` or unified `train --sb3-enabled`.

Algorithm / agent arguments:
- `--algo` or `--sb3-algo`: RL algorithm, one of `ppo`, `sac`, `td3`, `ddpg`.
- `--backbone` or `--sb3-backbone`: feature extractor backbone, one of `mlp`, `transformer`, `mamba`.
- `--history-steps` or `--sb3-history-steps`: sequence window length for the SB3 observation tensor.
- `--total-timesteps` or `--sb3-total-timesteps`: total environment interaction steps.
- `--n-envs` or `--sb3-n-envs`: number of parallel environments.
- `--learning-rate` or `--sb3-learning-rate`: optimizer learning rate.
- `--batch-size` or `--sb3-batch-size`: batch size used by the algorithm.
- `--gamma` or `--sb3-gamma`: reward discount factor.
- `--vec-norm-obs` / `--vec-norm-reward` (or `--sb3-vec-norm-obs` / `--sb3-vec-norm-reward`): enable VecNormalize for observations / rewards.
- `--eval-freq` / `--eval-episode-days` (or `--sb3-eval-freq` / `--sb3-eval-episode-days`): evaluation frequency and horizon during training for best-checkpoint selection.
- PPO-only (`ppo`): `--ppo-n-steps`, `--ppo-gae-lambda`, `--ppo-ent-coef`, `--ppo-clip-range` (or `--sb3-ppo-*`).
- Off-policy-only (`sac`/`td3`/`ddpg`): `--learning-starts`, `--train-freq`, `--gradient-steps`, `--tau`, `--action-noise-std`, `--buffer-size`, `--optimize-memory-usage` (or `--sb3-*`).
- `--device`: `auto`, `cpu`, `cuda`, or `cuda:<index>`.
- `--seed`: reproducibility control.

Practical interpretation:
- `ppo` is usually the safest starting point
- `sac` is often a strong continuous-control baseline
- `transformer` / `mamba` backbones are heavier and slower than `mlp`
- `--n-envs` speeds up collection but increases CPU / RAM load
- `--total-timesteps` is the first knob to increase for longer runs

## Recommended Benchmark Recipes

Global options such as `--train-path`, `--eval-path`, and `--env-config` must appear **before** the subcommand. The CLI accepts comma-separated seeds (e.g., `--seed 0,42,123`) and fans out runs sequentially; aggregate the outputs with `scripts/debug/_diag_run.py` if you need consolidated tables.

### Rule baseline (long horizon)

```bash
python -m cchp_physical_env \
  --train-path data/processed/cchp_main_15min_2024.csv \
  --eval-path data/processed/cchp_main_15min_2025.csv \
  train \
  --policy rule \
  --episodes 800 \
  --episode-days 14 \
  --seed 42 \
  --run-root runs/baseline_rule
```

### Sequence trainer (transformer backbone)

```bash
python -m cchp_physical_env \
  --train-path data/processed/cchp_main_15min_2024.csv \
  --eval-path data/processed/cchp_main_15min_2025.csv \
  train \
  --policy sequence_rule \
  --sequence-adapter transformer \
  --history-steps 32 \
  --episode-days 14 \
  --train-steps 409600 \
  --batch-size 256 \
  --update-epochs 8 \
  --lr 1e-4 \
  --device cuda \
  --seed 42 \
  --run-root runs/seq_transformer_long
```

Replicate with `--sequence-adapter mlp` and `mamba` for the full benchmark.

### SB3 (SAC + transformer backbone)

```bash
python -m cchp_physical_env \
  --train-path data/processed/cchp_main_15min_2024.csv \
  --eval-path data/processed/cchp_main_15min_2025.csv \
  sb3-train \
  --algo sac \
  --backbone transformer \
  --history-steps 32 \
  --total-timesteps 2000000 \
  --episode-days 14 \
  --n-envs 4 \
  --learning-rate 3e-4 \
  --batch-size 512 \
  --gamma 0.99 \
  --vec-norm-obs \
  --vec-norm-reward \
  --eval-freq 50000 \
  --eval-episode-days 14 \
  --learning-starts 5000 \
  --train-freq 1 \
  --gradient-steps 1 \
  --tau 0.005 \
  --action-noise-std 0.1 \
  --buffer-size 50000 \
  --optimize-memory-usage \
  --device cuda \
  --seed 42 \
  --run-root runs/sb3_sac_transformer_long
```

### Constraint ablation

Toggle constraint modeling by re-running any command with `--constraint-mode reward_only`. Compare with the default `physics_in_loop` run to quantify constraint fidelity.

### Cross-year evaluation

After training, evaluate on the frozen 2025 dataset:

```bash
python -m cchp_physical_env \
  --train-path data/processed/cchp_main_15min_2024.csv \
  --eval-path data/processed/cchp_main_15min_2025.csv \
  eval \
  --checkpoint runs/seq_transformer_long/<timestamp>_train_sequence_transformer/checkpoints/baseline_policy.json \
  --run-dir runs/seq_transformer_long_eval \
  --seed 42 \
  --device auto
```

### Training artifacts

Each `train` invocation creates:

- `runs/<timestamp>_.../train/train_statistics.json` – frozen dataset stats.
- `runs/<timestamp>_.../train/episodes.csv` – per-episode KPIs (sufficient for tables/figures).
- `runs/<timestamp>_.../train/summary.json` – aggregated means.
- `runs/<timestamp>_.../checkpoints/baseline_policy.json` – metadata + checkpoint pointer.
- `runs/<timestamp>_.../eval/*.csv` – written by `eval` / `sb3-eval`.

These artifacts, together with the diagnostics under `scripts/debug/`, cover the quantitative material typically required in a paper (KPI tables, constraint ablations, and per-step traces).

## Full CLI Command Coverage

This section lists every supported top-level CLI subcommand in `python -m cchp_physical_env`.

### 1) `summary`

Validate train/eval data and print frozen-schema summary.

Windows / `pwsh`:

```powershell
uv run python -m cchp_physical_env summary
```

Linux / Debian / bash:

```bash
python -m cchp_physical_env summary
```

Override environment config or constraint mode if needed:

```bash
python -m cchp_physical_env summary \
  --env-config src/cchp_physical_env/config/config.yaml \
  --constraint-mode physics_in_loop
```

### 2) `train`

The unified training entry has three routing modes:
- baseline route: `rule`, `random`, `sequence_rule + rule`
- built-in deep sequence route: `sequence_rule + mlp|transformer|mamba`
- SB3 route: `--sb3-enabled`

### Notebook override mapping for `main.ipynb`

The parameter block in [`main.ipynb`](main.ipynb) is not obsolete. Those keys still map to the current `train` / `eval` pipeline.

Environment overrides:
- `env_overrides.constraint_mode` -> `--constraint-mode`
- `env_overrides.physics_backend` -> environment backend override in config loading

Training overrides:
- `policy` -> `--policy`
- `sequence_adapter` -> `--sequence-adapter`
- `history_steps` -> `--history-steps`
- `episode_days` -> `--episode-days`
- `episodes` -> `--episodes`
- `train_steps` -> `--train-steps`
- `batch_size` -> `--batch-size`
- `update_epochs` -> `--update-epochs`
- `lr` -> `--lr`
- `device` -> `--device`
- `seed` -> `--seed`
- `sb3_enabled` -> `--sb3-enabled`
- `sb3_algo` -> `--sb3-algo`
- `sb3_backbone` -> `--sb3-backbone`
- `sb3_history_steps` -> `--sb3-history-steps`
- `sb3_total_timesteps` -> `--sb3-total-timesteps`
- `sb3_n_envs` -> `--sb3-n-envs`
- `sb3_learning_rate` -> `--sb3-learning-rate`
- `sb3_batch_size` -> `--sb3-batch-size`
- `sb3_gamma` -> `--sb3-gamma`
- `sb3_buffer_size` -> `--sb3-buffer-size`
- `sb3_optimize_memory_usage` -> `--sb3-optimize-memory-usage` / `--no-sb3-optimize-memory-usage`

How to enable each experiment path in the notebook parameter block:
- random baseline: set `policy='random'`, keep `sb3_enabled=false`
- rule baseline: set `policy='rule'`, keep `sb3_enabled=false`
- sequence rule baseline: set `policy='sequence_rule'`, `sequence_adapter='rule'`, keep `sb3_enabled=false`
- deep sequence training: set `policy='sequence_rule'`, `sequence_adapter='mlp'|'transformer'|'mamba'`, keep `sb3_enabled=false`
- SB3 training: set `sb3_enabled=true`, then choose `sb3_algo` and `sb3_backbone`

Important routing note:
- if `sb3_enabled=true`, `train` will prioritize the SB3 route even if `policy='sequence_rule'`
- therefore when SB3 is enabled, `policy` / `sequence_adapter` should be treated as metadata only (to avoid ambiguity, you may set `policy='sb3'` as a record value).

Notebook default:
- `main.ipynb` defaults are the **paper profile**, aligned with `src/cchp_physical_env/config/config.yaml` (deep sequence: `policy=sequence_rule`, `sequence_adapter=transformer`, `sb3_enabled=false`).

### 3) `eval`

The unified evaluation entry:
- evaluates baseline checkpoints
- auto-detects SB3 checkpoints when `artifact_type=sb3_policy`
- can infer `run_dir` from checkpoint if not provided (baseline + SB3)

### 4) `sb3-train`

Explicit Stable-Baselines3 training entry.

### 5) `sb3-eval`

Explicit SB3 evaluation entry. Requires both `--run-dir` and `--checkpoint`.

Notes:
- `--run-dir` should point to the *run root directory* (the folder that contains `train/`, `eval/`, `checkpoints/`), not `.../eval`.
- Add `--stochastic` to sample actions stochastically (default is deterministic).

### 6) `calibrate`

Run parameter calibration search on train/eval data.

### 7) `ablation`

Run constraint-mode ablation across `physics_in_loop` and/or `reward_only`.

### 8) `collect`

Scan `runs/` and aggregate evaluation outputs into benchmark CSV tables.

## Training + Evaluation Recipes

This section is organized by actual experiment type. Every training example is followed by an evaluation command.

Paper defaults (no extra flags):

```bash
python -m cchp_physical_env train \
  --env-config src/cchp_physical_env/config/config.yaml
```

### 1) Rule Baseline

Train (`pwsh`):

```powershell
uv run python -m cchp_physical_env train `
  --policy rule `
  --episodes 800 `
  --episode-days 14 `
  --seed 40
```

Train (Linux / Debian / bash):

```bash
python -m cchp_physical_env train \
  --policy rule \
  --episodes 800 \
  --episode-days 14 \
  --seed 40
```

Eval:

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 2) Random Baseline

Train (`pwsh`):

```powershell
uv run python -m cchp_physical_env train `
  --policy random `
  --episodes 800 `
  --episode-days 14 `
  --seed 40
```

Train (Linux / Debian / bash):

```bash
python -m cchp_physical_env train \
  --policy random \
  --episodes 800 \
  --episode-days 14 \
  --seed 40
```

Eval:

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 3) Heuristic Sequence Baseline

This uses `policy=sequence_rule` with `sequence_adapter=rule`.

Train (`pwsh`):

```powershell
uv run python -m cchp_physical_env train `
  --policy sequence_rule `
  --sequence-adapter rule `
  --history-steps 16 `
  --episodes 800 `
  --episode-days 14 `
  --seed 40
```

Train (Linux / Debian / bash):

```bash
python -m cchp_physical_env train \
  --policy sequence_rule \
  --sequence-adapter rule \
  --history-steps 16 \
  --episodes 800 \
  --episode-days 14 \
  --seed 40
```

Eval:

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 4) Deep Sequence Policy: MLP

This goes through the built-in PPO-style sequence trainer.

Train (`pwsh`):

```powershell
uv run python -m cchp_physical_env train `
  --policy sequence_rule `
  --sequence-adapter mlp `
  --history-steps 32 `
  --episode-days 14 `
  --train-steps 409600 `
  --batch-size 256 `
  --update-epochs 8 `
  --lr 0.0001 `
  --device auto `
  --seed 40
```

Train (Linux / Debian / bash):

```bash
python -m cchp_physical_env train \
  --policy sequence_rule \
  --sequence-adapter mlp \
  --history-steps 32 \
  --episode-days 14 \
  --train-steps 409600 \
  --batch-size 256 \
  --update-epochs 8 \
  --lr 0.0001 \
  --device auto \
  --seed 40
```

Eval:

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 5) Deep Sequence Policy: Transformer

Train (`pwsh`):

```powershell
uv run python -m cchp_physical_env train `
  --policy sequence_rule `
  --sequence-adapter transformer `
  --history-steps 32 `
  --episode-days 14 `
  --train-steps 409600 `
  --batch-size 256 `
  --update-epochs 8 `
  --lr 0.0001 `
  --device auto `
  --seed 40
```

Train (Linux / Debian / bash):

```bash
python -m cchp_physical_env train \
  --policy sequence_rule \
  --sequence-adapter transformer \
  --history-steps 32 \
  --episode-days 14 \
  --train-steps 409600 \
  --batch-size 256 \
  --update-epochs 8 \
  --lr 0.0001 \
  --device auto \
  --seed 40
```

Eval:

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 6) Deep Sequence Policy: Mamba

Train (`pwsh`):

```powershell
uv run python -m cchp_physical_env train `
  --policy sequence_rule `
  --sequence-adapter mamba `
  --history-steps 32 `
  --episode-days 14 `
  --train-steps 409600 `
  --batch-size 256 `
  --update-epochs 8 `
  --lr 0.0001 `
  --device auto `
  --seed 40
```

Train (Linux / Debian / bash):

```bash
python -m cchp_physical_env train \
  --policy sequence_rule \
  --sequence-adapter mamba \
  --history-steps 32 \
  --episode-days 14 \
  --train-steps 409600 \
  --batch-size 256 \
  --update-epochs 8 \
  --lr 0.0001 \
  --device auto \
  --seed 40
```

Eval:

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 7) SB3: SAC + Transformer

You can either use `train --sb3-enabled ...` or the explicit `sb3-train` command. README uses the explicit command here.

Train (`pwsh`):

```powershell
uv run python -m cchp_physical_env sb3-train `
  --algo sac `
  --backbone transformer `
  --history-steps 32 `
  --total-timesteps 2000000 `
  --episode-days 14 `
  --n-envs 4 `
  --learning-rate 0.0003 `
  --batch-size 512 `
  --gamma 0.99 `
  --vec-norm-obs `
  --vec-norm-reward `
  --eval-freq 50000 `
  --eval-episode-days 14 `
  --learning-starts 5000 `
  --train-freq 1 `
  --gradient-steps 1 `
  --tau 0.005 `
  --action-noise-std 0.1 `
  --buffer-size 50000 `
  --optimize-memory-usage `
  --device auto `
  --seed 40
```

Train (Linux / Debian / bash):

```bash
python -m cchp_physical_env sb3-train \
  --algo sac \
  --backbone transformer \
  --history-steps 32 \
  --total-timesteps 2000000 \
  --episode-days 14 \
  --n-envs 4 \
  --learning-rate 0.0003 \
  --batch-size 512 \
  --gamma 0.99 \
  --vec-norm-obs \
  --vec-norm-reward \
  --eval-freq 50000 \
  --eval-episode-days 14 \
  --learning-starts 5000 \
  --train-freq 1 \
  --gradient-steps 1 \
  --tau 0.005 \
  --action-noise-std 0.1 \
  --buffer-size 50000 \
  --optimize-memory-usage \
  --device auto \
  --seed 40
```

Optional: add `--eval-after-train` to run the full 2025 evaluation immediately after training, and write outputs into the same `run_dir/eval/`.

Eval (`sb3-eval`):

```bash
python -m cchp_physical_env sb3-eval \
  --run-dir runs/<sb3_train_run> \
  --checkpoint runs/<sb3_train_run>/checkpoints/baseline_policy.json \
  --device auto \
  --seed 40
```

You can also use the unified `eval` entry for the same checkpoint (by default, it writes back into the training `run_dir` inferred from the checkpoint):

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<sb3_train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 8) SB3: PPO + Mamba

Train (`pwsh`):

```powershell
uv run python -m cchp_physical_env sb3-train `
  --algo ppo `
  --backbone mamba `
  --history-steps 32 `
  --total-timesteps 2000000 `
  --episode-days 14 `
  --n-envs 4 `
  --learning-rate 0.0003 `
  --batch-size 512 `
  --gamma 0.99 `
  --vec-norm-obs `
  --vec-norm-reward `
  --eval-freq 50000 `
  --eval-episode-days 14 `
  --learning-starts 5000 `
  --train-freq 1 `
  --gradient-steps 1 `
  --tau 0.005 `
  --action-noise-std 0.1 `
  --buffer-size 50000 `
  --optimize-memory-usage `
  --device auto `
  --seed 40
```

Train (Linux / Debian / bash):

```bash
python -m cchp_physical_env sb3-train \
  --algo ppo \
  --backbone mamba \
  --history-steps 32 \
  --total-timesteps 2000000 \
  --episode-days 14 \
  --n-envs 4 \
  --learning-rate 0.0003 \
  --batch-size 512 \
  --gamma 0.99 \
  --vec-norm-obs \
  --vec-norm-reward \
  --eval-freq 50000 \
  --eval-episode-days 14 \
  --learning-starts 5000 \
  --train-freq 1 \
  --gradient-steps 1 \
  --tau 0.005 \
  --action-noise-std 0.1 \
  --buffer-size 50000 \
  --optimize-memory-usage \
  --device auto \
  --seed 40
```

Eval:

```bash
python -m cchp_physical_env sb3-eval \
  --run-dir runs/<sb3_train_run> \
  --checkpoint runs/<sb3_train_run>/checkpoints/baseline_policy.json \
  --device auto \
  --seed 40
```

## Unified `train` Entry for SB3

If you prefer using one single training entry, the `train` command can also route into SB3.

`pwsh`:

```powershell
uv run python -m cchp_physical_env train `
  --sb3-enabled `
  --sb3-algo sac `
  --sb3-backbone transformer `
  --sb3-history-steps 32 `
  --sb3-total-timesteps 2000000 `
  --episode-days 14 `
  --sb3-learning-rate 0.0003 `
  --sb3-batch-size 512 `
  --sb3-gamma 0.99 `
  --sb3-vec-norm-obs `
  --sb3-vec-norm-reward `
  --sb3-eval-freq 50000 `
  --sb3-eval-episode-days 14 `
  --sb3-learning-starts 5000 `
  --sb3-train-freq 1 `
  --sb3-gradient-steps 1 `
  --sb3-tau 0.005 `
  --sb3-action-noise-std 0.1 `
  --sb3-buffer-size 50000 `
  --sb3-optimize-memory-usage `
  --device auto `
  --seed 40
```

Linux / Debian / bash:

```bash
python -m cchp_physical_env train \
  --sb3-enabled \
  --sb3-algo sac \
  --sb3-backbone transformer \
  --sb3-history-steps 32 \
  --sb3-total-timesteps 2000000 \
  --episode-days 14 \
  --sb3-learning-rate 0.0003 \
  --sb3-batch-size 512 \
  --sb3-gamma 0.99 \
  --sb3-vec-norm-obs \
  --sb3-vec-norm-reward \
  --sb3-eval-freq 50000 \
  --sb3-eval-episode-days 14 \
  --sb3-learning-starts 5000 \
  --sb3-train-freq 1 \
  --sb3-gradient-steps 1 \
  --sb3-tau 0.005 \
  --sb3-action-noise-std 0.1 \
  --sb3-buffer-size 50000 \
  --sb3-optimize-memory-usage \
  --device auto \
  --seed 40
```

Eval:

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<sb3_train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

Oracle
```bash
uv run python -m cchp_physical_env eval `
  --policy milp_mpc `
  --history-steps 32 `
  --env-config src\cchp_physical_env\config\config.yaml `
  --run-dir runs\oracle_milp_full
```
```bash
python -m cchp_physical_env eval `
  --policy ga_mpc `
  --history-steps 32 `
  --env-config src\cchp_physical_env\config\config.yaml `
  --run-dir runs\oracle_ga_full
```
## Experiment Utilities

### Constraint Ablation

`pwsh`:

```powershell
uv run python -m cchp_physical_env ablation `
  --policy rule `
  --modes physics_in_loop,reward_only `
  --seed 40
```

Linux / Debian / bash:

```bash
python -m cchp_physical_env ablation \
  --policy rule \
  --modes physics_in_loop,reward_only \
  --seed 40
```

Sequence variant example:

```bash
python -m cchp_physical_env ablation \
  --policy sequence_rule \
  --sequence-adapter rule \
  --history-steps 16 \
  --modes physics_in_loop,reward_only \
  --seed 40
```

### Calibration

`pwsh`:

```powershell
uv run python -m cchp_physical_env calibrate `
  --config docs/spec/calibration_config.json `
  --n-samples 6 `
  --run-root runs
```

Linux / Debian / bash:

```bash
python -m cchp_physical_env calibrate \
  --config docs/spec/calibration_config.json \
  --n-samples 6 \
  --run-root runs
```

With sequence-related overrides:

```bash
python -m cchp_physical_env calibrate \
  --config docs/spec/calibration_config.json \
  --n-samples 6 \
  --history-steps 16 \
  --sequence-adapter rule \
  --seed 40 \
  --run-root runs
```

### Collect benchmark tables

`pwsh`:

```powershell
uv run python -m cchp_physical_env collect `
  --runs-root runs `
  --output runs/paper/benchmark_summary.csv `
  --full-output runs/paper/benchmark_summary_full.csv
```

Linux / Debian / bash:

```bash
python -m cchp_physical_env collect \
  --runs-root runs \
  --output runs/paper/benchmark_summary.csv \
  --full-output runs/paper/benchmark_summary_full.csv
```

## Important Notes

- `train` always uses 2024 processed data.
- `eval` and `sb3-eval` always use 2025 processed data.
- `summary` checks frozen-schema consistency between train and eval files.
- `eval` automatically dispatches to SB3 evaluation when the checkpoint has `artifact_type=sb3_policy`.
- `sb3-eval` requires `--run-dir`; unified `eval` can auto-create or infer it.
- `--constraint-mode` is supported by most subcommands as an override to `env.constraint_mode`.
- `collect` accepts `--constraint-mode` only for compatibility; it is not used by the aggregation logic.
