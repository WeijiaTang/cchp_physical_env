English[README-en.md](README-en.md)|中文[README-zhcn.md](README-zhcn.md)|
# CCHP（冷热电三联供）+ 新能源调度研究仓库

本仓库提供一个 **Python** CCHP 物理环境（电-热-冷耦合）以及最小化的 CLI 训练/评估流程，用于可复现实验与研究对比。

英文版说明见：`README.md`。

- Kaggle/Notebook 入口：`main.ipynb`

## 数据（冻结路径）

CLI 默认要求仓库根目录下存在如下文件（不要改文件名）：

- 训练：`data/processed/cchp_main_15min_2024.csv`
- 评价：`data/processed/cchp_main_15min_2025.csv`

时间口径冻结：

- 时区：`Asia/Beijing`
- 分辨率：`15min`
- 全年步数：`35040`
- 闰年处理：删除 `2/29`

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
uv run python -m cchp_physical_env eval --checkpoint runs/<run_id>/checkpoints/baseline_policy.json --seed 2
```

如需显式指定配置文件路径：

```bash
uv run python -m cchp_physical_env summary --env-config src/cchp_physical_env/config/config.yaml
```

## CLI 子命令说明

- `summary`：读取 2024/2025 数据并做冻结 schema 一致性校验，输出摘要
- `train`：运行 baseline 训练骨架（2024）
- `eval`：运行 baseline 评估（2025，支持加载 checkpoint）
- `calibrate`：运行标定搜索（最小采样）
- `ablation`：运行约束方式消融（例如 `physics_in_loop` vs `reward_only`）

可用参数请查看：

```bash
uv run python -m cchp_physical_env --help
```

## Kaggle

请使用 `main.ipynb`。

在 Kaggle Notebook 右侧 Add data：

- 一个“代码数据集”，顶层包含 `pyproject.toml` 与 `src/`
- 一个“数据数据集”，顶层包含 `data/processed/*.csv`

`main.ipynb` 会将它们 rsync 到可写目录，并依次执行 `summary/train/eval`。

## 上游基线（可选）

上游基线位于：`reference/repo/joint_bes_gt_dispatch/`。

示例：

```powershell
cd reference/repo/joint_bes_gt_dispatch
python main/rule_based.py
python main/run_dqn.py
```
