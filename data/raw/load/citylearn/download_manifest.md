# CityLearn 热/冷负荷数据 — 手动下载清单

目录：`data/raw/load/citylearn/`

我们论文主线会把 CityLearn 的“建筑热/冷负荷”聚合成园区级 `qh_dem_mw / qc_dem_mw`，并最终写入：
- `data/processed/cchp_main_15min_2024.csv`

## 推荐数据集（默认，用于热/冷负荷）

优先选择 **ResStock neighborhood** 类数据集（文件名通常是 `resstock-*.csv`），因为它们包含非零的 `heating_demand` 与 `cooling_demand`，更适合 CCHP 的“供热 + 吸收式制冷”叙事。

推荐（默认）：
- CityLearn dataset：`vt_chittenden_county_neighborhood`
- 原因：同时包含较明显的供热与制冷需求（冷/热都非零），规模适中（几十个 building csv），便于聚合成园区级 `qh_dem_mw / qc_dem_mw`。

不推荐作为主线热/冷负荷来源（但可用于电侧 benchmark/消融）：
- `citylearn_challenge_2022_phase_*`：在你当前 clone 的版本里，`Building_*.csv` 的 `cooling_demand/heating_demand/dhw_demand` 基本为 0，更像“电侧/储能”任务，不适合作为 CCHP 热/冷负荷。

## 获取方式 A（你已 clone：直接从本地拷贝该 dataset 目录）

你已 clone 到：
- `reference/repo/CityLearn/data/datasets/`

把下面目录拷贝到（或软链接到）本目录：
- `reference/repo/CityLearn/data/datasets/vt_chittenden_county_neighborhood/`
→ `data/raw/load/citylearn/vt_chittenden_county_neighborhood/`

后续 ETL 只需要读取其中若干个 `resstock-*.csv` 并做聚合即可。

## 获取方式 B（没 clone 时：git sparse-checkout，只拉该 dataset）

如果你想用 git 下载且只拉这个数据集（避免整库过大），可以用 sparse-checkout（可选）：

```powershell
git clone --filter=blob:none --no-checkout https://github.com/intelligent-environments-lab/CityLearn.git CityLearn
cd CityLearn
git sparse-checkout init --cone
git sparse-checkout set data/datasets/vt_chittenden_county_neighborhood
git checkout
```

然后把 `CityLearn/data/datasets/vt_chittenden_county_neighborhood/` 拷贝到：
- `data/raw/load/citylearn/vt_chittenden_county_neighborhood/`

## 注意

- 许可：CityLearn 仓库为 MIT License（适合论文复现/开源发布）。
- 我们主线天气使用北京 ERA5；因此 CityLearn 的天气文件更多用于 baseline/ablation 或字段对照，不强制使用。
- 你本地若使用的是 fork（例如 `nagyzoltan/CityLearn`），建议在论文/附录里同时记录：fork 仓库 URL + `git rev-parse HEAD` 的 commit hash，确保可复现。
