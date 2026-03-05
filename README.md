[README-en.md](README.md) | [README-zhcn.md](README-zhcn.md)|
## CCHP (Combined Cooling, Heating, and Power) + Renewables Scheduling Research

This repository provides a **Python** physical environment for a CCHP energy system (electricity-heat-cooling coupling) and a minimal CLI pipeline for reproducible experiments.

The project started from an upstream baseline (`reference/repo/joint_bes_gt_dispatch`) and is being extended towards a CCHP / multi-energy system research codebase.

- Notebook entry (Kaggle-ready): `main.ipynb`

## Repository structure (core modules)

- `src/cchp_physical_env/__main__.py`: CLI entrypoint (`summary/train/eval/calibrate/ablation/sb3-train/sb3-eval`)
- `src/cchp_physical_env/core/data.py`: frozen-schema data loading, validation, missing-value repair, episode sampling
- `src/cchp_physical_env/core/config_loader.py`: load/merge config and CLI overrides
- `src/cchp_physical_env/pipeline/runner.py`: baseline train/eval runner (`rule`, `random`, `sequence_rule`)
- `src/cchp_physical_env/pipeline/sequence.py`: sequence policy wrapper and adapters
- `src/cchp_physical_env/policy/trainer.py`: PPO-style deep sequence trainer (`transformer`/`mamba`/`mlp`)
- `src/cchp_physical_env/policy/sb3.py`: optional Stable-Baselines3 train/eval integration (`ppo`/`sac`/`td3`/`ddpg`)
- `src/cchp_physical_env/config/config.yaml`: unified environment + training config

## Data (frozen paths)

The CLI expects the following files under the repo root (do not rename):

- Train: `data/processed/cchp_main_15min_2024.csv`
- Eval: `data/processed/cchp_main_15min_2025.csv`

Time convention:

- Timezone: `Asia/Shanghai`
- Resolution: `15min`
- Yearly steps: `35040`
- Leap day handling: drop `02/29`

Frozen schema (`core/data.py`) includes:

- Load/renewables: `p_dem_mw`, `qh_dem_mw`, `qc_dem_mw`, `pv_mw`, `wt_mw`
- Weather: `t_amb_k`, `sp_pa`, `rh_pct`, `wind_speed`, `wind_direction`, `ghi_wm2`, `dni_wm2`, `dhi_wm2`
- Price/tax: `price_e`, `price_gas`, `carbon_tax`

Validation/repair behavior:

- Timestamp must be monotonic increasing and unique
- Data must be single-year and aligned to expected 15min index
- Load-like columns: time interpolation + boundary fill
- Weather columns: forward fill (ZOH)
- Price/tax missing values: hard fail (no implicit fill)

## Quickstart (recommended: `uv`)

This project uses Python 3.11+. We recommend using `uv` for dependency management.

### 1) Install dependencies

```bash
uv sync
```

### 2) Run CLI

By default the CLI reads the environment config:

`src/cchp_physical_env/config/config.yaml`

```bash
uv run python -m cchp_physical_env summary
uv run python -m cchp_physical_env train --episodes 1 --episode-days 7 --policy rule --seed 2
```

Then evaluate the latest run:

```powershell
$runDir = Get-ChildItem runs -Directory | Sort-Object Name | Select-Object -Last 1
$ckpt = Join-Path $runDir.FullName 'checkpoints/baseline_policy.json'
uv run python -m cchp_physical_env eval --checkpoint $ckpt --seed 2
```

### 3) Run CLI (without `uv`)

If you prefer plain `pip` + `python`:

```bash
python -m venv .venv
```

```bash
# PowerShell
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m cchp_physical_env --help
```

If you want to explicitly specify the config path:

```bash
uv run python -m cchp_physical_env summary --env-config src/cchp_physical_env/config/config.yaml
```

## CLI commands

- `summary`: validate 2024/2025 datasets (schema consistency + basic stats)
- `train`: run baseline training skeleton (2024)
- `eval`: evaluate on 2025 (supports loading a checkpoint)
- `calibrate`: run calibration search (minimal sampling)
- `ablation`: compare constraint modes (e.g. `physics_in_loop` vs `reward_only`)
- `sb3-train`: run optional Stable-Baselines3 training (`ppo`/`sac`/`td3`/`ddpg`)
- `sb3-eval`: evaluate a Stable-Baselines3 checkpoint run

Run `uv run python -m cchp_physical_env --help` to see all options.

## Policy / algorithm selection

The training/evaluation entrypoints share a unified `policy` option:

- **`policy=random`**
  Baseline random policy (no learning).
- **`policy=rule`**
  Baseline rule-based policy (no deep learning).
- **`policy=sequence_rule`**
  Sequence policy. If `sequence_adapter=transformer|mamba|mlp`, training uses PPO-style update (`SequencePolicyTrainer._ppo_update`).

For `policy=sequence_rule`, the following backbones are supported:

- **`sequence_adapter=transformer`**
- **`sequence_adapter=mamba`**
- **`sequence_adapter=mlp`** (structure ablation baseline under the same PPO-style sequence trainer)

Typical config knobs are under `training:` in `src/cchp_physical_env/config/config.yaml`:

- **Common**: `seed`, `episode_days` (constrained to `[7, 30]`)
- **Baseline (rule/random)**: `episodes`
- **Sequence (PPO-style)**: `history_steps`, `train_steps`, `batch_size`, `update_epochs`, `lr`, `device`

Example (sequence training, `mlp` ablation):

```bash
uv run python -m cchp_physical_env train `
  --policy sequence_rule `
  --sequence-adapter mlp `
  --history-steps 16 `
  --episode-days 7 `
  --train-steps 4096 `
  --batch-size 256 `
  --device cpu
```

## SB3 multi-algorithm comparison (optional dependency)

Install optional dependency first:

```bash
uv pip install -e '.[sb3]'
```

If your environment does not have `torch`, install it according to your platform (CPU/CUDA) before running `sb3-train`.

Recommended usage (route via `train`):

- Set `training.sb3_enabled: true` and `training.sb3_algo: sac` in `src/cchp_physical_env/config/config.yaml`
- Then run:

```bash
uv run python -m cchp_physical_env train
```

CLI override example (force `td3` and override config):

```bash
uv run python -m cchp_physical_env train `
  --sb3-enabled `
  --sb3-algo td3 `
  --sb3-total-timesteps 200000
```

Explicit SB3 entrypoint (equivalent, optional):

```bash
uv run python -m cchp_physical_env sb3-train `
  --algo sac `
  --total-timesteps 200000 `
  --episode-days 14 `
  --n-envs 1 `
  --device auto
```

Evaluation:

```bash
uv run python -m cchp_physical_env eval `
  --checkpoint runs/<sb3_train_run>/checkpoints/baseline_policy.json
```

Optional explicit SB3 evaluation (fallback):

```bash
uv run python -m cchp_physical_env sb3-eval `
  --run-dir runs/<new_eval_dir> `
  --checkpoint runs/<sb3_train_run>/checkpoints/baseline_policy.json
```

Supported SB3 algorithms: `ppo`, `sac`, `td3`, `ddpg`.

## Training & evaluation flow

- `summary`: load two yearly CSV files, run frozen-schema checks, print JSON summary
- `train`:
  - baseline path (`rule/random`): use `pipeline/runner.py` and save train artifacts
  - sequence deep path (`sequence_rule + transformer/mamba/mlp`): use `policy/trainer.py`
- `eval`: run 2025 evaluation and export evaluation summary
- `sb3-train/sb3-eval`: optional SB3 train/eval path via `policy/sb3.py`

Year constraints in code:

- Training year must be `2024`
- Evaluation year must be `2025`

## Run artifacts

All runs are saved under `runs/<timestamp>_<mode>_<policy>/` (or sequence/SB3 variant), typically including:

- `train/train_statistics.json`
- `train/episodes.csv` (baseline training episodes)
- `eval/` outputs (evaluation logs/summary)
- `checkpoints/` (e.g. `baseline_policy.json` or deep model artifacts)

`eval --checkpoint .../checkpoints/...` will infer `run_dir` from checkpoint parent path.

Paper-friendly eval exports (generated automatically on `eval`):

- `eval/summary_flat.csv` (1-row flattened KPI table)
- `eval/cost_breakdown.csv`, `eval/violation_counts.csv`, `eval/diagnostic_counts.csv`
- `eval/step_log_light.csv` (drops per-step JSON fields, easier for plotting)
- `eval/daily_agg.csv` (daily aggregation for curves/figures)

Collect benchmark tables across runs:

```bash
uv run python -m cchp_physical_env collect
```

## Config notes

`src/cchp_physical_env/config/config.yaml` has two top-level blocks:

- `env`: physical/environment and penalty parameters (Option-C: fully configurable)
- `training`: policy/training hyperparameters

CLI argument precedence:

- Explicit CLI args override `training:` defaults from YAML
- Unspecified options fall back to YAML

## Kaggle

Use `main.ipynb`.

In Kaggle, add two datasets:

- A code dataset that contains `pyproject.toml` and `src/`
- A data dataset that contains `data/processed/*.csv`

The notebook will rsync both into a writable working directory and then run `summary/train/eval`.
