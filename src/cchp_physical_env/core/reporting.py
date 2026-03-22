# Ref: docs/spec/task.md (Task-ID: 011)
# Ref: docs/spec/architecture.md (Module: core/)
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def flatten_mapping(mapping: dict[str, Any], *, sep: str = "__") -> dict[str, Any]:
    """
    把嵌套 dict 扁平化成一层（用于论文表格列）。

    - dict 递归：a:{b:1} -> a__b=1
    - list/tuple：转为 JSON 字符串（避免 CSV 单元格结构丢失）
    """

    def _flatten(value: Any, prefix: str, out: dict[str, Any]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                name = f"{prefix}{sep}{key}" if prefix else str(key)
                _flatten(item, name, out)
            return
        if isinstance(value, (list, tuple)):
            out[prefix] = json.dumps(value, ensure_ascii=False)
            return
        out[prefix] = value

    result: dict[str, Any] = {}
    _flatten(mapping, "", result)
    return result


def write_one_row_csv(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(target, index=False)


def write_kv_csv(
    path: str | Path,
    mapping: dict[str, Any],
    *,
    key_col: str = "name",
    value_col: str = "value",
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [{key_col: str(k), value_col: float(v)} for k, v in mapping.items()]
    pd.DataFrame(rows).to_csv(target, index=False)


def build_daily_aggregation(step_log: pd.DataFrame, *, dt_h: float = 0.25) -> pd.DataFrame:
    """
    将 step_log 聚合成日尺度（更适合论文画曲线/柱状图）。

    约定：
    - 成本/能量按 sum
    - 功率按 energy = sum(p * dt_h)
    - 每日可靠性按 1 - unmet/demand（按日）
    """
    if "timestamp" not in step_log.columns:
        raise ValueError("step_log 缺少 timestamp 列，无法做日尺度聚合。")
    ts = pd.to_datetime(step_log["timestamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError("timestamp 存在无法解析的值。")
    df = step_log.copy()
    df["_date"] = ts.dt.date

    sum_cols = [c for c in df.columns if c.startswith("cost_") or c.startswith("energy_")]
    daily = df.groupby("_date", sort=True)[sum_cols].sum(numeric_only=True)
    daily.index = daily.index.astype(str)
    daily.index.name = "date"

    # grid import/export 能量（MWh）
    if "p_grid_import_mw" in df.columns:
        daily["energy_grid_import_mwh"] = (
            df.groupby("_date")["p_grid_import_mw"].sum(numeric_only=True) * float(dt_h)
        )
    if "p_grid_export_mw" in df.columns:
        daily["energy_grid_export_mwh"] = (
            df.groupby("_date")["p_grid_export_mw"].sum(numeric_only=True) * float(dt_h)
        )

    # 日可靠性（按日 unmet/demand）
    demand_map = {
        "electric": "energy_demand_e_mwh",
        "heat": "energy_demand_h_mwh",
        "cooling": "energy_demand_c_mwh",
    }
    unmet_map = {
        "electric": "energy_unmet_e_mwh",
        "heat": "energy_unmet_h_mwh",
        "cooling": "energy_unmet_c_mwh",
    }
    for name in ("electric", "heat", "cooling"):
        d_col = demand_map[name]
        u_col = unmet_map[name]
        if d_col in daily.columns and u_col in daily.columns:
            denom = daily[d_col].clip(lower=1e-9)
            daily[f"reliability_{name}"] = (1.0 - daily[u_col] / denom).clip(lower=0.0, upper=1.0)

    return daily.reset_index()


def write_paper_eval_artifacts(
    eval_dir: str | Path, *, summary: dict[str, Any], step_log: pd.DataFrame, dt_h: float = 0.25
) -> dict[str, str]:
    """
    将 eval 输出补齐为论文友好的格式：
    - summary_flat.csv：一行扁平化表格（便于多 run 汇总）
    - cost_breakdown.csv / violation_counts.csv / diagnostic_counts.csv / state_diagnostic_counts.csv
    - step_log_light.csv：去掉 per-step JSON，适合画曲线
    - daily_agg.csv：按日聚合
    """
    out_dir = Path(eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flat = flatten_mapping(summary)
    write_one_row_csv(out_dir / "summary_flat.csv", flat)

    cost_breakdown = summary.get("cost_breakdown", {})
    if isinstance(cost_breakdown, dict) and cost_breakdown:
        write_kv_csv(out_dir / "cost_breakdown.csv", cost_breakdown, key_col="cost", value_col="value")

    violation_counts = summary.get("violation_counts", {})
    if isinstance(violation_counts, dict) and violation_counts:
        write_kv_csv(out_dir / "violation_counts.csv", violation_counts, key_col="flag", value_col="count")

    diagnostic_counts = summary.get("diagnostic_counts", {})
    if isinstance(diagnostic_counts, dict) and diagnostic_counts:
        write_kv_csv(out_dir / "diagnostic_counts.csv", diagnostic_counts, key_col="flag", value_col="count")

    state_diagnostic_counts = summary.get("state_diagnostic_counts", {})
    if isinstance(state_diagnostic_counts, dict) and state_diagnostic_counts:
        write_kv_csv(
            out_dir / "state_diagnostic_counts.csv",
            state_diagnostic_counts,
            key_col="flag",
            value_col="count",
        )

    light = step_log.copy()
    drop_cols = [
        c
        for c in (
            "violation_flags_json",
            "diagnostic_flags_json",
            "state_diagnostic_flags_json",
            "episode_summary",
        )
        if c in light.columns
    ]
    if drop_cols:
        light = light.drop(columns=drop_cols)
    light.to_csv(out_dir / "step_log_light.csv", index=False)

    daily = build_daily_aggregation(light, dt_h=float(dt_h))
    daily.to_csv(out_dir / "daily_agg.csv", index=False)

    meta = {
        "dt_h": float(dt_h),
        "paper_files": {
            "summary_flat_csv": "summary_flat.csv",
            "cost_breakdown_csv": "cost_breakdown.csv" if isinstance(cost_breakdown, dict) and cost_breakdown else None,
            "violation_counts_csv": "violation_counts.csv" if isinstance(violation_counts, dict) and violation_counts else None,
            "diagnostic_counts_csv": "diagnostic_counts.csv" if isinstance(diagnostic_counts, dict) and diagnostic_counts else None,
            "state_diagnostic_counts_csv": (
                "state_diagnostic_counts.csv"
                if isinstance(state_diagnostic_counts, dict) and state_diagnostic_counts
                else None
            ),
            "step_log_light_csv": "step_log_light.csv",
            "daily_agg_csv": "daily_agg.csv",
        },
    }
    (out_dir / "paper_manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {k: str(v) for k, v in meta["paper_files"].items() if v is not None}
