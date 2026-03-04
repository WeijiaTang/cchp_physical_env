## CCHP（冷热电三联供）+ 新能源（光/风）调度研究仓库

本仓库以 `reference/repo/joint_bes_gt_dispatch`（Energy & AI 2025 的 GT+BES+新能源 联合调度 DRL 基线）为起点，逐步扩展到 **CCHP/多能系统（电-热-冷耦合）**，用于论文建模、算法对比与可复现实验。

## 你应该先看哪里

- 路线图（工程仿真导向）：`docs/roadmap/roadmap.md`
- 基线代码的“文件级地图/扩展点”：`docs/roadmap/code_map.md`
- 接口与方程清单（冻结版）：`docs/roadmap/interface_equations_checklist.md`
- 输入数据规范与可用数据源：`docs/roadmap/data_inputs_and_sources.md`
- 主输入 CSV（环境读取入口，冻结）：
  - 训练：`data/processed/cchp_main_15min_2024.csv`
  - 评价：`data/processed/cchp_main_15min_2025.csv`
- Notebook 入口（教学 + 快速自检）：`main.ipynb`
- 进入编码前的规格（Spec→Code）：`docs/spec/README.md`
- 联网检索沉淀（开源仓库/数据/论文线索）：`docs/chat/search_result-03031017.md`

## 仓库结构（当前）

- `reference/repo/joint_bes_gt_dispatch/`：上游基线代码（GT + 电池 + 风光 + 需求 + DRL 训练管线）
- `docs/`：路线图与检索记录
- `src/`：预留给你后续“CCHP 扩展模块/新环境/数据管线”的主开发区（当前未落代码）

## 运行基线（建议先跑通再扩展）

基线仓库自带 `Pipfile`（Python 3.11），并提供多个可直接运行的入口脚本（DQN/PPO/SAC/TD3/规则基线等）。

示例（从基线目录运行）：

```powershell
cd reference/repo/joint_bes_gt_dispatch
python main/rule_based.py
python main/run_dqn.py
```

说明：
- `main/run_*.py` 里通常会提示你输入一个保存目录名（用于写入 log/）。
- 更具体的入口文件与“扩展到 CCHP 时该改哪里”，见 `docs/roadmap/code_map.md`。
