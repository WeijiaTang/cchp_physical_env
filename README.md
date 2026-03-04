English[README-en.md](README.md)|中文[README-zhcn.md](README-zhcn.md)|
## CCHP (Combined Cooling, Heating, and Power) + Renewables Scheduling Research

This repository provides a **Python** physical environment for a CCHP energy system (electricity-heat-cooling coupling) and a minimal CLI pipeline for reproducible experiments.

The project started from an upstream baseline (`reference/repo/joint_bes_gt_dispatch`) and is being extended towards a CCHP / multi-energy system research codebase.

- Notebook entry (Kaggle-ready): `main.ipynb`

## Data (frozen paths)

The CLI expects the following files under the repo root (do not rename):

- Train: `data/processed/cchp_main_15min_2024.csv`
- Eval: `data/processed/cchp_main_15min_2025.csv`

Time convention:

- Timezone: `Asia/Beijing`
- Resolution: `15min`
- Yearly steps: `35040`
- Leap day handling: drop `02/29`

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
uv run python -m cchp_physical_env eval --checkpoint runs/<run_id>/checkpoints/baseline_policy.json --seed 2
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

Run `uv run python -m cchp_physical_env --help` to see all options.

## Kaggle

Use `main.ipynb`.

In Kaggle, add two datasets:

- A code dataset that contains `pyproject.toml` and `src/`
- A data dataset that contains `data/processed/*.csv`

The notebook will rsync both into a writable working directory and then run `summary/train/eval`.

## Upstream baseline (optional)

The upstream baseline lives in `reference/repo/joint_bes_gt_dispatch/` and comes with its own run scripts.

Example:

```powershell
cd reference/repo/joint_bes_gt_dispatch
python main/rule_based.py
python main/run_dqn.py
```
