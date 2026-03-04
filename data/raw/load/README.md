# 负荷数据（raw/load）获取与用途（按文档冻结口径）

本项目论文主线（方案 B）需要 15min × 1 年的外生负荷输入，最终会被合并成单一文件：
- `data/processed/cchp_main_15min_2024.csv`

`data/raw/load/` 只负责保存可溯源的原始负荷数据（不要求立即可读、也不要求格式统一）。

## 1) 电负荷（主线：真实工业园）

来源：Scientific Data 2023 工业园真实电负荷数据（Suzhou, China）
- 论文 DOI：`10.1038/s41597-023-02786-9`
- OSF 项目页：`https://osf.io/agk8s/`

落盘位置（建议保持原始压缩包 + 解压目录）：
- `data/raw/load/industrial_park_scidata2023/osf/`

下载清单（已生成）：
- `data/raw/load/industrial_park_scidata2023/osf/download_manifest.md`
- `data/raw/load/industrial_park_scidata2023/osf/download_manifest.csv`

用途说明：
- 冻结：选择其中 **2020** 年的 **5min** 分辨率数据，按“功率序列”聚合到 15min（mean）
- 再缩放到我们冻结的 **10 MW** 峰值，生成 `p_dem_mw`
- 闰年处理：先删除 `2020-02-29`，并在最终主时间索引中删除 `2024-02-29`，保证全年 `35,040` 行

> 注：该数据包内部是大量按天拆分的 `.xlsx`，短期不需要读取；后续 ETL 阶段再把它拼接成全年序列。

## 2) 热/冷负荷（主线：CityLearn）

来源：CityLearn 数据集（MIT License，公开可复现）
- GitHub：`https://github.com/intelligent-environments-lab/CityLearn`

落盘位置（建议只下载某一个 dataset 文件夹，避免整库过大）：
- `data/raw/load/citylearn/`

用途说明：
- 选定一个 dataset + building 子集
- 聚合成园区级 `qh_dem_mw / qc_dem_mw`（再缩放到与电负荷量级匹配）

## 3) 风电（可选）

如果需要 `wt_mw`：
- 可用 OPSD（time_series）或 Fingrid 15min 风电数据
- 建议落盘到 `data/raw/load/wind/`（后续再补下载清单）
