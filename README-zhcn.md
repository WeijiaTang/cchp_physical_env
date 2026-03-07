[README-en.md](README.md) | [README-zhcn.md](README-zhcn.md) |
# CCHP（冷热电三联供）+ 新能源调度研究仓库

这是一个面向研究的 Python CCHP 物理环境与可复现实验仓库，提供统一的训练、评估和结果汇总 CLI。

本项目支持：
- 规则基线策略
- 深度序列策略训练（`mlp` / `transformer` / `mamba`）
- 基于 SB3 的多算法训练（`ppo` / `sac` / `td3` / `ddpg`）以及序列骨干
- 统一 CLI：`summary`、`train`、`eval`、`sb3-train`、`sb3-eval`、`ablation`、`calibrate`、`collect`

- Notebook 入口：`main.ipynb`
- CLI 入口：`python -m cchp_physical_env`

## 这个仓库是什么

核心模块：
- `src/cchp_physical_env/__main__.py`：统一 CLI 入口
- `src/cchp_physical_env/core/data.py`：冻结 schema 数据读取、校验、修复、采样
- `src/cchp_physical_env/core/config_loader.py`：配置读取与 CLI 覆盖合并
- `src/cchp_physical_env/pipeline/runner.py`：基线训练/评估执行器
- `src/cchp_physical_env/pipeline/sequence.py`：序列策略工具与 adapter
- `src/cchp_physical_env/policy/trainer.py`：PPO-style 深度序列训练器
- `src/cchp_physical_env/policy/sb3.py`：SB3 训练/评估集成
- `src/cchp_physical_env/config/config.yaml`：环境参数与训练参数配置

## 数据约定

仓库根目录下默认使用以下 processed 数据文件：
- 训练集：`data/processed/cchp_main_15min_2024.csv`
- 评估集：`data/processed/cchp_main_15min_2025.csv`

冻结口径：
- 时区：`Asia/Shanghai`
- 分辨率：`15min`
- 全年步数：`35040`
- 闰日处理：删除 `02/29`

冻结 schema 包含：
- 负荷 / 新能源：`p_dem_mw`、`qh_dem_mw`、`qc_dem_mw`、`pv_mw`、`wt_mw`
- 天气：`t_amb_k`、`sp_pa`、`rh_pct`、`wind_speed`、`wind_direction`、`ghi_wm2`、`dni_wm2`、`dhi_wm2`
- 价格 / 税：`price_e`、`price_gas`、`carbon_tax`

校验 / 修复规则：
- 时间戳必须唯一且严格递增
- 数据必须是单年并对齐到冻结的 15min 索引
- 负荷类列：时间插值 + 边界填充
- 天气列：前向填充
- 价格/税列：出现缺失时直接报错

## 安装

推荐 Python 3.11+。

Windows 如果使用 `uv`：

```powershell
uv sync
```

Linux / Debian 如果不使用 `uv`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

可选依赖：
- SB3：`uv pip install -e '.[sb3]'`

`pyproject.toml` 中没有写死 PyTorch，请根据本机 CPU/CUDA 环境安装正确版本，再进行长时间深度学习训练。

`sequence_rule` 与 SB3 使用的 `mamba` 骨干现在均由 `transformers` 提供，因此本仓库不再单独提供 `mamba-ssm` 包或 extra。

## CLI 快速开始

Windows / PowerShell（`pwsh`）：

```powershell
uv run python -m cchp_physical_env summary
uv run python -m cchp_physical_env --help
```

Linux / Debian / bash：

```bash
python -m cchp_physical_env summary
python -m cchp_physical_env --help
```

## Shell 换行约定

PowerShell（`pwsh`）使用反引号 `` ` `` 进行换行续写：

```powershell
uv run python -m cchp_physical_env train `
  --policy rule `
  --episodes 800 `
  --episode-days 14 `
  --seed 40
```

Linux / Debian / bash 使用反斜杠 `\` 进行换行续写：

```bash
python -m cchp_physical_env train \
  --policy rule \
  --episodes 800 \
  --episode-days 14 \
  --seed 40
```

下面的命令示例中，Windows 统一使用 `uv run python -m ...`，Linux 统一使用直接 `python -m ...`。

## 训练参数说明

### 深度序列训练参数

当 `train` 进入内置 PPO-style 序列训练器时，使用下面这组参数：
- `--policy sequence_rule`
- `--sequence-adapter mlp|transformer|mamba`

主要参数含义：
- `--history-steps`：送入序列骨干网络的时间窗口长度；越大表示时间上下文越长，但显存和计算量也越高。
- `--episode-days`：每个训练 episode 采样的天数，控制单次 rollout 的时间跨度。
- `--train-steps`：训练阶段累计采样的 rollout 步数。
- `--batch-size`：每次梯度更新的 minibatch 大小。
- `--update-epochs`：每个 rollout batch 被重复优化的轮数。
- `--lr`：优化器学习率。
- `--device`：`auto`、`cpu`、`cuda` 或 `cuda:<index>`。
- `--seed`：随机种子，用于复现实验。

实践理解：
- 如果显存紧张，优先减小 `--history-steps` 或 `--batch-size`
- 如果训练不稳定，优先降低 `--lr`
- 如果策略明显欠拟合，优先增加 `--train-steps`
- 如果储能动态感知不充分，可以增大 `--episode-days`

### 基线 / 启发式参数

下面这些参数用于轻量级基线：
- `--policy rule`
- `--policy random`
- `--policy sequence_rule --sequence-adapter rule`

主要参数：
- `--episodes`：采样训练 episode 的数量。
- `--episode-days`：每个 episode 的长度。
- `--history-steps`：只对 `sequence_rule` 变体有意义。
- `--seed`：复现控制。

### SB3 训练参数

下面这些参数用于显式 `sb3-train`，或者统一入口 `train --sb3-enabled`。

算法 / agent 参数：
- `--algo` 或 `--sb3-algo`：强化学习算法，可选 `ppo`、`sac`、`td3`、`ddpg`。
- `--backbone` 或 `--sb3-backbone`：特征提取骨干，可选 `mlp`、`transformer`、`mamba`。
- `--history-steps` 或 `--sb3-history-steps`：SB3 观测张量使用的时间窗口长度。
- `--total-timesteps` 或 `--sb3-total-timesteps`：总环境交互步数。
- `--n-envs` 或 `--sb3-n-envs`：并行环境数量。
- `--learning-rate` 或 `--sb3-learning-rate`：优化器学习率。
- `--batch-size` 或 `--sb3-batch-size`：算法使用的 batch size。
- `--gamma` 或 `--sb3-gamma`：奖励折扣因子。
- `--device`：`auto`、`cpu`、`cuda` 或 `cuda:<index>`。
- `--seed`：复现控制。

实践理解：
- `ppo` 通常适合作为最稳妥的起点
- `sac` 通常是连续控制中很强的对比基线
- `transformer` / `mamba` 比 `mlp` 更重、更慢
- `--n-envs` 会提升采样吞吐，但也增加 CPU / 内存压力
- `--total-timesteps` 是长时间训练最先应该增加的主旋钮

## CLI 全部命令覆盖

本节按 `python -m cchp_physical_env` 实际支持的顶层子命令完整列出。

### 1）`summary`

校验训练 / 评估数据并打印冻结 schema 摘要。

Windows / `pwsh`：

```powershell
uv run python -m cchp_physical_env summary
```

Linux / Debian / bash：

```bash
python -m cchp_physical_env summary
```

如需覆盖环境配置或约束模式：

```bash
python -m cchp_physical_env summary \
  --env-config src/cchp_physical_env/config/config.yaml \
  --constraint-mode physics_in_loop
```

### 2）`train`

统一训练入口有三种路由：
- 基线路由：`rule`、`random`、`sequence_rule + rule`
- 内置深度序列路由：`sequence_rule + mlp|transformer|mamba`
- SB3 路由：`--sb3-enabled`

### 3）`eval`

统一评估入口：
- 可评估 baseline checkpoint
- 遇到 `artifact_type=sb3_policy` 时会自动识别为 SB3 checkpoint
- 未显式给出 `run_dir` 时可自动推断或自动创建

### 4）`sb3-train`

显式 Stable-Baselines3 训练入口。

### 5）`sb3-eval`

显式 SB3 评估入口，要求同时提供 `--run-dir` 与 `--checkpoint`。

### 6）`calibrate`

在训练 / 评估数据上运行物理参数标定搜索。

### 7）`ablation`

在 `physics_in_loop` 与 / 或 `reward_only` 上运行约束模式消融。

### 8）`collect`

扫描 `runs/`，将评估结果汇总为 benchmark CSV 表。

## 训练 + 评估示例

本节按实际实验类型组织。每个训练命令后面都接一个对应的评估命令。

### 1）规则基线（Rule Baseline）

训练（`pwsh`）：

```powershell
uv run python -m cchp_physical_env train `
  --policy rule `
  --episodes 800 `
  --episode-days 14 `
  --seed 40
```

训练（Linux / Debian / bash）：

```bash
python -m cchp_physical_env train \
  --policy rule \
  --episodes 800 \
  --episode-days 14 \
  --seed 40
```

评估：

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 2）随机基线（Random Baseline）

训练（`pwsh`）：

```powershell
uv run python -m cchp_physical_env train `
  --policy random `
  --episodes 800 `
  --episode-days 14 `
  --seed 40
```

训练（Linux / Debian / bash）：

```bash
python -m cchp_physical_env train \
  --policy random \
  --episodes 800 \
  --episode-days 14 \
  --seed 40
```

评估：

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 3）启发式序列基线（Heuristic Sequence Baseline）

即 `policy=sequence_rule` 且 `sequence_adapter=rule`。

训练（`pwsh`）：

```powershell
uv run python -m cchp_physical_env train `
  --policy sequence_rule `
  --sequence-adapter rule `
  --history-steps 16 `
  --episodes 800 `
  --episode-days 14 `
  --seed 40
```

训练（Linux / Debian / bash）：

```bash
python -m cchp_physical_env train \
  --policy sequence_rule \
  --sequence-adapter rule \
  --history-steps 16 \
  --episodes 800 \
  --episode-days 14 \
  --seed 40
```

评估：

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 4）深度序列策略：MLP

该模式走内置的 PPO-style 序列训练器。

训练（`pwsh`）：

```powershell
uv run python -m cchp_physical_env train `
  --policy sequence_rule `
  --sequence-adapter mlp `
  --history-steps 16 `
  --episode-days 28 `
  --train-steps 409600 `
  --batch-size 1024 `
  --update-epochs 4 `
  --lr 0.0003 `
  --device auto `
  --seed 40
```

训练（Linux / Debian / bash）：

```bash
python -m cchp_physical_env train \
  --policy sequence_rule \
  --sequence-adapter mlp \
  --history-steps 16 \
  --episode-days 28 \
  --train-steps 409600 \
  --batch-size 1024 \
  --update-epochs 4 \
  --lr 0.0003 \
  --device auto \
  --seed 40
```

评估：

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 5）深度序列策略：Transformer

训练（`pwsh`）：

```powershell
uv run python -m cchp_physical_env train `
  --policy sequence_rule `
  --sequence-adapter transformer `
  --history-steps 16 `
  --episode-days 28 `
  --train-steps 409600 `
  --batch-size 1024 `
  --update-epochs 4 `
  --lr 0.0003 `
  --device auto `
  --seed 40
```

训练（Linux / Debian / bash）：

```bash
python -m cchp_physical_env train \
  --policy sequence_rule \
  --sequence-adapter transformer \
  --history-steps 16 \
  --episode-days 28 \
  --train-steps 409600 \
  --batch-size 1024 \
  --update-epochs 4 \
  --lr 0.0003 \
  --device auto \
  --seed 40
```

评估：

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 6）深度序列策略：Mamba

训练（`pwsh`）：

```powershell
uv run python -m cchp_physical_env train `
  --policy sequence_rule `
  --sequence-adapter mamba `
  --history-steps 16 `
  --episode-days 28 `
  --train-steps 409600 `
  --batch-size 1024 `
  --update-epochs 4 `
  --lr 0.0003 `
  --device auto `
  --seed 40
```

训练（Linux / Debian / bash）：

```bash
python -m cchp_physical_env train \
  --policy sequence_rule \
  --sequence-adapter mamba \
  --history-steps 16 \
  --episode-days 28 \
  --train-steps 409600 \
  --batch-size 1024 \
  --update-epochs 4 \
  --lr 0.0003 \
  --device auto \
  --seed 40
```

评估：

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 7）SB3：SAC + Transformer

你既可以使用 `train --sb3-enabled ...`，也可以显式使用 `sb3-train`。README 这里优先展示显式命令。

训练（`pwsh`）：

```powershell
uv run python -m cchp_physical_env sb3-train `
  --algo sac `
  --backbone transformer `
  --history-steps 16 `
  --total-timesteps 2000000 `
  --episode-days 14 `
  --n-envs 1 `
  --learning-rate 0.0003 `
  --batch-size 256 `
  --gamma 0.99 `
  --device auto `
  --seed 40
```

训练（Linux / Debian / bash）：

```bash
python -m cchp_physical_env sb3-train \
  --algo sac \
  --backbone transformer \
  --history-steps 16 \
  --total-timesteps 2000000 \
  --episode-days 14 \
  --n-envs 1 \
  --learning-rate 0.0003 \
  --batch-size 256 \
  --gamma 0.99 \
  --device auto \
  --seed 40
```

评估（`sb3-eval`）：

```bash
python -m cchp_physical_env sb3-eval \
  --run-dir runs/<new_eval_run> \
  --checkpoint runs/<sb3_train_run>/checkpoints/baseline_policy.json \
  --device auto \
  --seed 40
```

同一个 checkpoint 也可以直接用统一入口 `eval`：

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<sb3_train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

### 8）SB3：PPO + Mamba

训练（`pwsh`）：

```powershell
uv run python -m cchp_physical_env sb3-train `
  --algo ppo `
  --backbone mamba `
  --history-steps 16 `
  --total-timesteps 2000000 `
  --episode-days 14 `
  --n-envs 1 `
  --learning-rate 0.0003 `
  --batch-size 256 `
  --gamma 0.99 `
  --device auto `
  --seed 40
```

训练（Linux / Debian / bash）：

```bash
python -m cchp_physical_env sb3-train \
  --algo ppo \
  --backbone mamba \
  --history-steps 16 \
  --total-timesteps 2000000 \
  --episode-days 14 \
  --n-envs 1 \
  --learning-rate 0.0003 \
  --batch-size 256 \
  --gamma 0.99 \
  --device auto \
  --seed 40
```

评估：

```bash
python -m cchp_physical_env sb3-eval \
  --run-dir runs/<new_eval_run> \
  --checkpoint runs/<sb3_train_run>/checkpoints/baseline_policy.json \
  --device auto \
  --seed 40
```

## 使用统一 `train` 入口触发 SB3

如果你希望所有训练都走同一个入口，那么 `train` 也可以路由到 SB3。

`pwsh`：

```powershell
uv run python -m cchp_physical_env train `
  --sb3-enabled `
  --sb3-algo sac `
  --sb3-backbone transformer `
  --sb3-history-steps 16 `
  --sb3-total-timesteps 2000000 `
  --episode-days 14 `
  --sb3-learning-rate 0.0003 `
  --sb3-batch-size 256 `
  --sb3-gamma 0.99 `
  --device auto `
  --seed 40
```

Linux / Debian / bash：

```bash
python -m cchp_physical_env train \
  --sb3-enabled \
  --sb3-algo sac \
  --sb3-backbone transformer \
  --sb3-history-steps 16 \
  --sb3-total-timesteps 2000000 \
  --episode-days 14 \
  --sb3-learning-rate 0.0003 \
  --sb3-batch-size 256 \
  --sb3-gamma 0.99 \
  --device auto \
  --seed 40
```

评估：

```bash
python -m cchp_physical_env eval \
  --checkpoint runs/<sb3_train_run>/checkpoints/baseline_policy.json \
  --seed 40
```

## 实验辅助命令

### 约束模式消融

`pwsh`：

```powershell
uv run python -m cchp_physical_env ablation `
  --policy rule `
  --modes physics_in_loop,reward_only `
  --seed 40
```

Linux / Debian / bash：

```bash
python -m cchp_physical_env ablation \
  --policy rule \
  --modes physics_in_loop,reward_only \
  --seed 40
```

序列变体示例：

```bash
python -m cchp_physical_env ablation \
  --policy sequence_rule \
  --sequence-adapter rule \
  --history-steps 16 \
  --modes physics_in_loop,reward_only \
  --seed 40
```

### 参数标定

`pwsh`：

```powershell
uv run python -m cchp_physical_env calibrate `
  --config docs/spec/calibration_config.json `
  --n-samples 6 `
  --run-root runs
```

Linux / Debian / bash：

```bash
python -m cchp_physical_env calibrate \
  --config docs/spec/calibration_config.json \
  --n-samples 6 \
  --run-root runs
```

带序列相关覆盖的例子：

```bash
python -m cchp_physical_env calibrate \
  --config docs/spec/calibration_config.json \
  --n-samples 6 \
  --history-steps 16 \
  --sequence-adapter rule \
  --seed 40 \
  --run-root runs
```

### 汇总 benchmark 表

`pwsh`：

```powershell
uv run python -m cchp_physical_env collect `
  --runs-root runs `
  --output runs/paper/benchmark_summary.csv `
  --full-output runs/paper/benchmark_summary_full.csv
```

Linux / Debian / bash：

```bash
python -m cchp_physical_env collect \
  --runs-root runs \
  --output runs/paper/benchmark_summary.csv \
  --full-output runs/paper/benchmark_summary_full.csv
```

## 重要说明

- `train` 固定使用 2024 年 processed 数据。
- `eval` 与 `sb3-eval` 固定使用 2025 年 processed 数据。
- `summary` 会检查训练 / 评估文件之间的冻结 schema 一致性。
- `eval` 遇到 `artifact_type=sb3_policy` 的 checkpoint 时会自动路由到 SB3 评估。
- `sb3-eval` 强制要求 `--run-dir`；统一入口 `eval` 可以自动创建或推断。
- 大多数子命令都支持 `--constraint-mode`，用于覆盖 `env.constraint_mode`。
- `collect` 虽然接受 `--constraint-mode`，但只是兼容保留字段，汇总逻辑并不会使用它。
