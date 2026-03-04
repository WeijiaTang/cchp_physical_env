# process-data

目标：把 raw 数据源按 `docs/roadmap/data_inputs_and_sources.md` 的冻结口径，生成环境唯一入口外生数据表：
- `data/processed/cchp_main_15min_2024.csv`（15min × 365 天 = 35,040 行；2024 删除 2/29）

## 生成命令

```powershell
python scripts/process-data/build_cchp_main_15min_2024.py
```

训练/评估两套数据（推荐）：

```powershell
python scripts/process-data/build_cchp_main_15min_2024.py --config scripts/process-data/cchp_main_15min_2024_config.json
python scripts/process-data/build_cchp_main_15min_2024.py --config scripts/process-data/cchp_main_15min_2025_config.json
```

## 配置

默认配置文件：
- `scripts/process-data/cchp_main_15min_2024_config.json`
- `scripts/process-data/cchp_main_15min_2025_config.json`

你可以修改其中：
- 工业园电负荷输入路径（2020，5min CSV 日文件）
- CityLearn 数据集路径（vt_chittenden_county_neighborhood，47×resstock）
- 热/冷负荷缩放比例（相对电负荷峰值）
- PV/WT 装机容量与简化功率曲线参数
- 电价基准与 TOU 计价口径（输出为 `CNY/MWh`）
- `price_gas`、`carbon_tax` 的常数取值

## 产物

- 主表：`data/processed/cchp_main_15min_2024.csv`
- 清单：`data/processed/cchp_main_15min_2024.manifest.json`（记录 config、缩放系数、输入文件数量等，便于论文复现）
