[README-en.md](README.md) | [README-zhcn.md](README-zhcn.md)|
# CCHP（冷热电三联供）+ 新能源调度研究仓库

本仓库提供一个 **Python** CCHP 物理环境（电-热-冷耦合）以及最小化的 CLI 训练/评估流程，用于可复现实验与研究对比。

英文版说明见：`README.md`。

- Kaggle/Notebook 入口：`main.ipynb`

## 仓库结构（核心模块）

- `src/cchp_physical_env/__main__.py`：CLI 入口（`summary/train/eval/calibrate/ablation/sb3-train/sb3-eval`）
- `src/cchp_physical_env/core/data.py`：冻结 schema 数据读取、校验、缺失值修复、episode 采样
- `src/cchp_physical_env/core/config_loader.py`：配置读取与 CLI 覆盖合并
- `src/cchp_physical_env/pipeline/runner.py`：基线训练/评估执行器（`rule`、`random`、`sequence_rule`）
- `src/cchp_physical_env/pipeline/sequence.py`：序列策略封装与 adapter
- `src/cchp_physical_env/policy/trainer.py`：PPO-style 深度序列训练器（`transformer`/`mamba`/`mlp`）
- `src/cchp_physical_env/policy/sb3.py`：可选 Stable-Baselines3 训练/评估集成（`ppo`/`sac`/`td3`/`ddpg`）
- `src/cchp_physical_env/config/config.yaml`：统一环境参数与训练参数配置

## 数据（冻结路径）

CLI 默认要求仓库根目录下存在如下文件（不要改文件名）：

- 训练：`data/processed/cchp_main_15min_2024.csv`
- 评价：`data/processed/cchp_main_15min_2025.csv`

时间口径冻结：

- 时区：`Asia/Shanghai`
- 分辨率：`15min`
- 全年步数：`35040`
- 闰日处理：删除 `2/29`

冻结 schema（`core/data.py`）包含：

- 负荷/新能源：`p_dem_mw`、`qh_dem_mw`、`qc_dem_mw`、`pv_mw`、`wt_mw`
- 天气：`t_amb_k`、`sp_pa`、`rh_pct`、`wind_speed`、`wind_direction`、`ghi_wm2`、`dni_wm2`、`dhi_wm2`
- 价格/碳税：`price_e`、`price_gas`、`carbon_tax`

校验与修复逻辑：

- 时间戳必须严格递增且不可重复
- 数据必须是单年，并对齐到预期 15min 索引
- 负荷类列：时间插值 + 边界前后向填充
- 天气列：前向填充（ZOH）
- 价格/税列缺失：直接报错（不做隐式补齐）

## 快速开始（强烈推荐使用 `uv`）

本项目要求 Python 3.11+。推荐使用 `uv` 管理依赖与运行命令。

### 1) 安装依赖

```bash
uv sync
```

### 2) 运行 CLI

默认读取环境配置：

`src/cchp_physical_env/config/config.yaml`

```bash
uv run python -m cchp_physical_env summary
uv run python -m cchp_physical_env train --episodes 1 --episode-days 7 --policy rule --seed 2
```

然后评估最新一次运行（自动定位 `runs/` 下最新的 checkpoint）：

```powershell
$runDir = Get-ChildItem runs -Directory | Sort-Object Name | Select-Object -Last 1
$ckpt = Join-Path $runDir.FullName 'checkpoints/baseline_policy.json'
uv run python -m cchp_physical_env eval --checkpoint $ckpt --seed 2
```

### 3) 不使用 `uv` 的本地运行方式

如果你希望用 `pip` + `python` 直接运行：

```bash
python -m venv .venv
```

```bash
# PowerShell
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m cchp_physical_env --help
```

如需显式指定配置文件路径：

```bash
uv run python -m cchp_physical_env summary --env-config src/cchp_physical_env/config/config.yaml
```

## CLI 子命令说明

- `summary`：校验 2024/2025 数据集（schema 一致性 + 基础统计）
- `train`：运行基线训练骨架（2024）
- `eval`：在 2025 上评估（支持加载 checkpoint）
- `calibrate`：运行标定搜索（最小采样）
- `ablation`：对比约束模式（如 `physics_in_loop` vs `reward_only`）
- `sb3-train`：运行可选 Stable-Baselines3 训练（`ppo`/`sac`/`td3`/`ddpg`）
- `sb3-eval`：评估 Stable-Baselines3 训练产出的 checkpoint

可用参数请查看：

```bash
uv run python -m cchp_physical_env --help
```

## 策略 / 算法选择

训练与评估使用统一 `policy` 参数：

- **`policy=random`**
  随机策略基线（不学习）。
- **`policy=rule`**
  规则策略基线（不训练深度模型）。
- **`policy=sequence_rule`**
  序列策略；当 `sequence_adapter=transformer|mamba|mlp` 时，训练走 PPO-style 更新（`SequencePolicyTrainer._ppo_update`）。

当 `policy=sequence_rule` 时，支持的 backbone 为：

- **`sequence_adapter=transformer`**
- **`sequence_adapter=mamba`**
- **`sequence_adapter=mlp`**（同一 PPO-style 序列训练器下的结构消融基线）

常用训练参数位于 `src/cchp_physical_env/config/config.yaml` 的 `training:` 段：

- **通用**：`seed`、`episode_days`（强约束 `[7, 30]`）
- **基线策略（rule/random）**：`episodes`
- **序列策略（PPO-style）**：`history_steps`、`train_steps`、`batch_size`、`update_epochs`、`lr`、`device`

示例（序列策略训练，`mlp` 消融）：

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

## SB3 多算法对比（可选依赖）

先安装可选依赖：

```bash
uv pip install -e '.[sb3]'
```

如果环境中尚未安装 `torch`，请按平台（CPU/CUDA）先安装，再运行 `sb3-train`。

推荐用法（通过 `train` 自动路由）：

- 在 `src/cchp_physical_env/config/config.yaml` 设置：
  - `training.sb3_enabled: true`
  - `training.sb3_algo: sac`
- 然后直接运行：

```bash
uv run python -m cchp_physical_env train
```

CLI 覆盖示例（强制启用 `td3`，覆盖 config）：

```bash
uv run python -m cchp_physical_env train `
  --sb3-enabled `
  --sb3-algo td3 `
  --sb3-total-timesteps 200000
```

可选：显式使用 SB3 子命令（等价入口）：

```bash
uv run python -m cchp_physical_env sb3-train `
  --algo sac `
  --total-timesteps 200000 `
  --episode-days 14 `
  --n-envs 1 `
  --device auto
```

评估示例：

```bash
uv run python -m cchp_physical_env eval `
  --checkpoint runs/<sb3_train_run>/checkpoints/baseline_policy.json
```

可选：显式调用 `sb3-eval`（备用入口）：

```bash
uv run python -m cchp_physical_env sb3-eval `
  --run-dir runs/<new_eval_dir> `
  --checkpoint runs/<sb3_train_run>/checkpoints/baseline_policy.json
```

当前支持的 SB3 算法：`ppo`、`sac`、`td3`、`ddpg`。

## 训练与评估流程

- `summary`：读取两份年度 CSV，执行冻结 schema 校验，输出 JSON 摘要
- `train`：
  - 基线路径（`rule/random`）：调用 `pipeline/runner.py` 并保存训练产物
  - 序列深度路径（`sequence_rule + transformer/mamba/mlp`）：调用 `policy/trainer.py`
- `eval`：运行 2025 评估并输出评估摘要
- `sb3-train/sb3-eval`：通过 `policy/sb3.py` 走可选的 SB3 训练/评估路径

代码中的年份硬约束：

- 训练年必须为 `2024`
- 评估年必须为 `2025`

## 运行产物

运行目录默认在 `runs/<timestamp>_<mode>_<policy>/`（序列训练/SB3 训练会带对应后缀），常见文件包括：

- `train/train_statistics.json`
- `train/episodes.csv`（基线训练的 episode 记录）
- `eval/` 输出（评估日志/汇总）
- `checkpoints/`（如 `baseline_policy.json` 或深度模型产物）

`eval --checkpoint .../checkpoints/...` 会自动从 checkpoint 路径反推 `run_dir`。

论文作图/写作友好的 eval 额外导出（`eval` 自动生成）：

- `eval/summary_flat.csv`（一行扁平化指标表，便于跨 run 汇总）
- `eval/cost_breakdown.csv`、`eval/violation_counts.csv`、`eval/diagnostic_counts.csv`
- `eval/step_log_light.csv`（去掉逐步 JSON 字段，更适合画曲线）
- `eval/daily_agg.csv`（按日聚合，更适合论文图表）

跨 runs 汇总论文表格：

```bash
uv run python -m cchp_physical_env collect
```

## 配置说明

`src/cchp_physical_env/config/config.yaml` 顶层分两段：

- `env`：物理环境与惩罚项参数（Option-C：全量可配）
- `training`：策略选择与训练超参数

参数优先级：

- CLI 显式参数覆盖 YAML 的 `training` 默认值
- 未显式传参时使用 YAML 默认值

## Kaggle

请使用 `main.ipynb`。

在 Kaggle Notebook 右侧 Add data：

- 一个“代码数据集”，顶层包含 `pyproject.toml` 与 `src/`
- 一个“数据数据集”，顶层包含 `data/processed/*.csv`

Notebook 会将两者 rsync 到可写目录，并依次执行 `summary/train/eval`。
