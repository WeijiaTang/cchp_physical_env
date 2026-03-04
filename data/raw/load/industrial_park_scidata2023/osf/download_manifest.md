# 工业园负荷数据（Scientific Data 2023）— 手动下载清单（OSF）

目录：`data/raw/load/industrial_park_scidata2023/osf/`

本清单用于你手动把“论文公开数据包”下载到本目录，并在后续 ETL 阶段用于溯源与校验。

## 1) 必下（主线用：工业园电负荷）

- 文件名：`Electric power load data.zip`
- OSF 节点：`agk8s`
- OSF 文件 ID：`64cb190a4f0a442835994659`
- 下载链接（直链）：`https://files.osf.io/v1/resources/agk8s/providers/osfstorage/64cb190a4f0a442835994659`
- 大小：`193,429,960` bytes（≈ 184.5 MiB）
- 校验：
  - `md5`：`3323cf7983b66368ce58efca00f2be33`
  - `sha256`：`743402af5103b2368d1d0a00af67fa80ff05d24580b388ac7d95cead215fc443`
- 用途（对应我们文档的方案 B）：
  - 取其中 **5min** 分辨率的某一年（建议先用 `2019` 或 `2020`，全年更稳定），按“功率序列”聚合到 **15min mean**
  - 再缩放到 **10 MW 峰值**，得到 `p_dem_mw`
  - 冻结（当前项目口径）：使用 `2020` 年，并先删除 `2020-02-29`（主时间索引也删除 `2024-02-29`），保证全年 `35,040` 行

数据包结构要点（你解压后会看到）：
- 2016–2021 每年一套目录
- 每年包含 `5 min / 30 min / 1 hour` 三种分辨率
- 数据是大量按天拆分的 `.xlsx`（后续 ETL 再处理即可）

## 2) 可选（数据集自带天气）

- 文件名：`Weather data.zip`
- OSF 节点：`agk8s`
- OSF 文件 ID：`64cbdc2b9cbf0333221e48ff`
- 下载链接（直链）：`https://files.osf.io/v1/resources/agk8s/providers/osfstorage/64cbdc2b9cbf0333221e48ff`
- 大小：`10,955,758` bytes（≈ 10.4 MiB）
- 校验：
  - `md5`：`11a1962f14c0bf3a46278bb438682d8f`
  - `sha256`：`fcf6c8e855e7e4ab7d8dedf8cb3f7e198c309f70b2f0bf68fcba1b1cf62e6500`
- 用途：
  - 主线我们用你已下载的 **北京 ERA5**；该天气包可作为“与工业园负荷同源”的备份对照（不强制用）

## 3) 建议的断点续传下载命令（Windows）

用 `curl.exe`（断点续传）：

```powershell
curl.exe -L -C - -o "Electric power load data.zip" "https://files.osf.io/v1/resources/agk8s/providers/osfstorage/64cb190a4f0a442835994659"
curl.exe -L -C - -o "Weather data.zip" "https://files.osf.io/v1/resources/agk8s/providers/osfstorage/64cbdc2b9cbf0333221e48ff"
```

下载完成后校验（推荐 sha256）：

```powershell
Get-FileHash -Algorithm SHA256 "Electric power load data.zip"
Get-FileHash -Algorithm SHA256 "Weather data.zip"
```

## 4) 许可/引用提醒

- 该数据集对应论文 DOI：`10.1038/s41597-023-02786-9`
- OSF 入口页：`https://osf.io/agk8s/`
- 论文发布的数据一般按 **CC BY 4.0** 共享；正式写论文时请以论文/OSF 页面标注为准并按推荐格式引用。
