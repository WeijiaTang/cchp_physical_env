# Ref: docs/spec/task.md
"""
数据处理与校验：冻结 schema、时间口径、缺失值修复、episode 采样。

本模块定义了 processed 数据的"冻结口径"，确保实验可复现：
- 冻结 schema：FROZEN_COLUMNS 定义了最小充分的外生变量集合
- 冻结时间口径：15min 分辨率、Asia/Shanghai 时区、删除闰日 2/29
- 冻结缺失值修复规则：负荷插值、天气前向填充、价格硬失败
- 训练/评估年份硬约束：TRAIN_YEAR=2024、EVAL_YEAR=2025

常见坑：
- 不要在 processed 数据里加入"未来信息"（如未来 24h 价格）
- 训练统计只能从训练集拟合，不能混入评估集
- episode 采样仅允许训练年数据
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# 时间分辨率常量
STEP_MINUTES = 15
STEP_FREQ = "15min"
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES
EXPECTED_STEPS_PER_YEAR = 365 * STEPS_PER_DAY  # 删除闰日后为 35040
TIMEZONE = "Asia/Shanghai"

# 训练/评估年份硬约束（不要在其他地方改）
TRAIN_YEAR = 2024
EVAL_YEAR = 2025

# 冻结 schema：最小充分的外生变量集合
FROZEN_COLUMNS = [
    "timestamp",
    "p_dem_mw",  # 电负荷
    "qh_dem_mw",  # 热负荷
    "qc_dem_mw",  # 冷负荷
    "pv_mw",  # 光伏出力
    "wt_mw",  # 风电出力
    "t_amb_k",  # 环境温度（K）
    "sp_pa",  # 气压（Pa）
    "rh_pct",  # 相对湿度（%）
    "wind_speed",  # 风速
    "wind_direction",  # 风向
    "ghi_wm2",  # 总辐照度
    "dni_wm2",  # 直射辐照度
    "dhi_wm2",  # 散射辐照度
    "price_e",  # 电价
    "price_gas",  # 气价
    "carbon_tax",  # 碳税
]

# 缺失值修复策略分类
LOAD_LIKE_COLUMNS = ["p_dem_mw", "qh_dem_mw", "qc_dem_mw", "pv_mw", "wt_mw"]  # 时间插值
WEATHER_COLUMNS = [
    "t_amb_k",
    "sp_pa",
    "rh_pct",
    "wind_speed",
    "wind_direction",
    "ghi_wm2",
    "dni_wm2",
    "dhi_wm2",
]  # 前向填充
PRICE_COLUMNS = ["price_e", "price_gas", "carbon_tax"]  # 硬失败（不允许缺失）


class DataValidationError(ValueError):
    """数据校验失败。"""


@dataclass(frozen=True, slots=True)
class EpisodeWindow:
    """训练切片的元信息。"""

    start_idx: int
    end_idx: int
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp


def _build_expected_index(year: int, tz_name: str = TIMEZONE) -> pd.DatetimeIndex:
    """
    构建全年 15min 时间索引，并删除闰日 2/29。

    返回：长度为 EXPECTED_STEPS_PER_YEAR（35040）的 DatetimeIndex
    """
    full_index = pd.date_range(
        start=f"{year}-01-01 00:00:00",
        end=f"{year}-12-31 23:45:00",
        freq=STEP_FREQ,
        tz=tz_name,
    )
    without_feb29 = full_index[~((full_index.month == 2) & (full_index.day == 29))]
    if len(without_feb29) != EXPECTED_STEPS_PER_YEAR:
        raise DataValidationError(
            f"期望步数应为 {EXPECTED_STEPS_PER_YEAR}，当前为 {len(without_feb29)}。"
        )
    return without_feb29


def _coerce_timestamp_series(series: pd.Series) -> pd.Series:
    """
    解析时间戳序列，统一转换到 Asia/Shanghai 时区。

    若无时区信息则 localize，有时区则 convert。
    解析失败会抛出 DataValidationError。
    """
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        invalid_count = int(parsed.isna().sum())
        raise DataValidationError(f"`timestamp` 解析失败，存在 {invalid_count} 个非法时间戳。")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(TIMEZONE)
    else:
        parsed = parsed.dt.tz_convert(TIMEZONE)
    return parsed


def _assert_required_columns(df: pd.DataFrame) -> None:
    """校验 DataFrame 是否包含所有冻结列。"""
    missing = [column for column in FROZEN_COLUMNS if column not in df.columns]
    if missing:
        raise DataValidationError(f"缺少冻结列: {missing}")


def _assert_single_year(index: pd.DatetimeIndex) -> int:
    """校验时间索引是否为单年数据，返回年份。"""
    years = sorted({int(value.year) for value in index})
    if len(years) != 1:
        raise DataValidationError(f"文件必须是单年数据，实际年份: {years}")
    return years[0]


def _assert_strictly_monotonic(index: pd.DatetimeIndex) -> None:
    """校验时间索引是否严格递增且无重复。"""
    if not index.is_monotonic_increasing:
        raise DataValidationError("`timestamp` 不是严格递增。")
    if index.has_duplicates:
        duplicated = int(index.duplicated().sum())
        raise DataValidationError(f"`timestamp` 存在重复，重复数量: {duplicated}")


def _drop_feb29_rows(df: pd.DataFrame) -> pd.DataFrame:
    """删除闰日 2/29 的行，保证全年步数为 35040。"""
    index = df.index
    mask = (index.month == 2) & (index.day == 29)
    if mask.any():
        return df.loc[~mask].copy()
    return df


def _assert_step_resolution(index: pd.DatetimeIndex) -> None:
    """
    校验时间索引是否为 15min 分辨率。

    特殊处理：允许 2/28 23:45 -> 3/1 00:00 的跳跃（闰日删除后的边界）
    """
    if len(index) < 2:
        raise DataValidationError("数据行数不足，无法校验 15min 分辨率。")
    step = pd.Timedelta(minutes=STEP_MINUTES)
    index_series = index.to_series()
    diffs = index_series.diff().dropna()

    prev_series = index_series.shift(1)
    leap_skip_mask = (
        (diffs == pd.Timedelta(days=1, minutes=STEP_MINUTES))
        & (index_series.dt.month == 3)
        & (index_series.dt.day == 1)
        & (index_series.dt.hour == 0)
        & (index_series.dt.minute == 0)
        & (prev_series.dt.month == 2)
        & (prev_series.dt.day == 28)
        & (prev_series.dt.hour == 23)
        & (prev_series.dt.minute == 45)
    )
    invalid = diffs[(diffs != step) & (~leap_skip_mask.reindex(diffs.index, fill_value=False))]
    if not invalid.empty:
        sample = invalid.index[0]
        raise DataValidationError(
            f"存在非 15min 间隔，示例位置: {sample}, diff={invalid.iloc[0]}"
        )


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in FROZEN_COLUMNS:
        if column == "timestamp":
            continue
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _repair_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    按冻结规则修复缺失值。

    策略：
    - 负荷/新能源列：时间插值 + 边界前后向填充
    - 天气列：前向填充（ZOH）
    - 价格/税列：硬失败（不允许缺失，避免隐式规则污染实验口径）
    """
    repaired = df.copy()

    # 负荷/新能源列：时间插值 + 边界前后向填充。
    repaired[LOAD_LIKE_COLUMNS] = repaired[LOAD_LIKE_COLUMNS].interpolate(
        method="time", limit_direction="both"
    )
    repaired[LOAD_LIKE_COLUMNS] = repaired[LOAD_LIKE_COLUMNS].ffill().bfill()

    # 天气列：按冻结口径采用前向填充（ZOH）。
    repaired[WEATHER_COLUMNS] = repaired[WEATHER_COLUMNS].ffill()

    # 价格/碳税列：缺失时硬失败，避免隐式规则污染实验口径。
    price_missing = repaired[PRICE_COLUMNS].isna().sum()
    if int(price_missing.sum()) > 0:
        missing_map = {key: int(value) for key, value in price_missing.items() if value > 0}
        raise DataValidationError(
            f"价格/税列存在缺失，当前 spec 未冻结自动补齐规则: {missing_map}"
        )

    still_missing = repaired[FROZEN_COLUMNS[1:]].isna().sum()
    unresolved = {key: int(value) for key, value in still_missing.items() if value > 0}
    if unresolved:
        raise DataValidationError(f"缺失值补齐后仍存在空值: {unresolved}")

    return repaired


def _canonicalize_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    将原始 DataFrame 规范化为冻结口径格式。

    处理流程：
    1. 校验冻结列是否存在
    2. 解析时间戳并设置索引
    3. 校验单年、严格递增、无重复
    4. 删除闰日并对齐到全年主索引
    5. 数值化并修复缺失值
    6. 校验 15min 分辨率和行数
    7. 保证冻结列顺序在前
    """
    _assert_required_columns(raw_df)

    working = raw_df.copy()
    working["timestamp"] = _coerce_timestamp_series(working["timestamp"])
    working = working.set_index("timestamp", drop=True)
    _assert_strictly_monotonic(working.index)

    year = _assert_single_year(working.index)
    working = _drop_feb29_rows(working)
    expected_index = _build_expected_index(year=year)

    # 以 timestamp 主索引对齐，其它列按左连接语义对齐到主索引。
    working = working.reindex(expected_index)
    working = _coerce_numeric_columns(working)
    working = _repair_missing_values(working)

    _assert_step_resolution(working.index)
    if len(working) != EXPECTED_STEPS_PER_YEAR:
        raise DataValidationError(
            f"行数错误，期望 {EXPECTED_STEPS_PER_YEAR}，实际 {len(working)}"
        )
    if ((working.index.month == 2) & (working.index.day == 29)).any():
        raise DataValidationError("数据包含 2/29，违反冻结口径。")

    output = working.reset_index().rename(columns={"index": "timestamp"})
    # 保证冻结列顺序在前，额外列保留在后。
    extra_columns = [column for column in output.columns if column not in FROZEN_COLUMNS]
    ordered_columns = [*FROZEN_COLUMNS, *extra_columns]
    return output.loc[:, ordered_columns]


def load_exogenous_data(path: str | Path) -> pd.DataFrame:
    """
    读取 processed CSV，并执行冻结规则校验与缺失值处理。

    返回：规范化后的 DataFrame，包含冻结列且顺序固定
    抛出：FileNotFoundError / DataValidationError
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    return _canonicalize_frame(raw_df)


def summarize_exogenous_data(df: pd.DataFrame) -> dict:
    """输出可复现实验所需的数据摘要信息。"""

    _assert_required_columns(df)

    timestamp_series = pd.to_datetime(df["timestamp"])
    if timestamp_series.dt.tz is None:
        timestamp_series = timestamp_series.dt.tz_localize(TIMEZONE)
    else:
        timestamp_series = timestamp_series.dt.tz_convert(TIMEZONE)

    summary = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "timestamp_start": timestamp_series.iloc[0].isoformat(),
        "timestamp_end": timestamp_series.iloc[-1].isoformat(),
        "missing_rate_by_col": {
            column: float(df[column].isna().mean()) for column in df.columns
        },
        "basic_stats": {},
    }

    key_columns = [
        "p_dem_mw",
        "qh_dem_mw",
        "qc_dem_mw",
        "price_e",
        "price_gas",
        "carbon_tax",
        "t_amb_k",
        "sp_pa",
        "rh_pct",
        "wind_speed",
        "wind_direction",
        "ghi_wm2",
        "dni_wm2",
        "dhi_wm2",
    ]
    for column in key_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        summary["basic_stats"][column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }

    return summary


def ensure_frozen_schema_consistency(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    """校验 train/eval 至少共享同一冻结列集合。"""

    train_missing = [column for column in FROZEN_COLUMNS if column not in train_df.columns]
    eval_missing = [column for column in FROZEN_COLUMNS if column not in eval_df.columns]
    if train_missing or eval_missing:
        raise DataValidationError(
            f"冻结列不一致，train_missing={train_missing}, eval_missing={eval_missing}"
        )


def _extract_year(df: pd.DataFrame) -> int:
    timestamp_series = pd.to_datetime(df["timestamp"])
    if timestamp_series.dt.tz is None:
        timestamp_series = timestamp_series.dt.tz_localize(TIMEZONE)
    years = sorted({int(item.year) for item in timestamp_series})
    if len(years) != 1:
        raise DataValidationError(f"DataFrame 不是单年数据，年份集合: {years}")
    return years[0]


def make_episode_sampler(
    df: pd.DataFrame, episode_days: int, seed: int
) -> Iterator[tuple[EpisodeWindow, pd.DataFrame]]:
    """
    在训练年数据上按固定随机种子采样 7-30 天 episode。

    参数：
    - df: 训练年数据（必须为 2024）
    - episode_days: episode 长度（7-30 天）
    - seed: 随机种子（保证可复现）

    返回：无限迭代器，每次产出 (EpisodeWindow, episode_df)

    注意：
    - 仅允许训练年数据（TRAIN_YEAR=2024）
    - episode_days 必须在 [7, 30] 范围内
    """
    if episode_days < 7 or episode_days > 30:
        raise ValueError("`episode_days` 必须在 [7, 30] 范围内。")

    year = _extract_year(df)
    if year != TRAIN_YEAR:
        raise DataValidationError(
            f"episode 采样仅允许训练年 {TRAIN_YEAR}，当前数据年份为 {year}"
        )

    total_steps = len(df)
    episode_steps = episode_days * STEPS_PER_DAY
    if episode_steps > total_steps:
        raise DataValidationError(
            f"episode 步数 {episode_steps} 超过全年步数 {total_steps}"
        )

    max_start = total_steps - episode_steps
    rng = np.random.default_rng(seed)

    while True:
        start_idx = int(rng.integers(0, max_start + 1))
        end_idx = start_idx + episode_steps
        episode_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        start_ts = pd.to_datetime(episode_df["timestamp"].iloc[0])
        end_ts = pd.to_datetime(episode_df["timestamp"].iloc[-1])
        window = EpisodeWindow(
            start_idx=start_idx,
            end_idx=end_idx,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
        )
        yield window, episode_df


def compute_training_statistics(
    train_df: pd.DataFrame, columns: list[str] | None = None
) -> dict:
    """
    仅在训练集上拟合统计量，供后续归一化/策略阈值使用。

    参数：
    - train_df: 训练年数据（必须为 2024）
    - columns: 需要统计的列（默认为所有冻结列，除 timestamp）

    返回：包含 mean/std/p05/p50/p95 的统计字典

    注意：
    - 仅允许训练年数据（TRAIN_YEAR=2024）
    - 这些统计量不能用于评估集（防止数据泄漏）
    """
    year = _extract_year(train_df)
    if year != TRAIN_YEAR:
        raise DataValidationError(
            f"训练统计只能来自 {TRAIN_YEAR}，当前年份为 {year}"
        )

    selected_columns = columns or [column for column in FROZEN_COLUMNS if column != "timestamp"]
    stats: dict[str, dict[str, float]] = {}
    for column in selected_columns:
        values = pd.to_numeric(train_df[column], errors="coerce")
        stats[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "p05": float(values.quantile(0.05)),
            "p50": float(values.quantile(0.50)),
            "p95": float(values.quantile(0.95)),
        }
    return {
        "year": TRAIN_YEAR,
        "n_rows": int(len(train_df)),
        "columns": selected_columns,
        "stats": stats,
    }


def dump_statistics_json(statistics: dict, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(statistics, indent=2, ensure_ascii=False), encoding="utf-8")
