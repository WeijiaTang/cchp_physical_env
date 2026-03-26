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


def write_learning_curve_artifacts(
    train_dir: str | Path,
    *,
    eval_history_rows: list[dict[str, Any]],
    progress_df: pd.DataFrame | None = None,
    selected_snapshot: dict[str, Any] | None = None,
    reward_leader_snapshot: dict[str, Any] | None = None,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    """
    将训练收敛相关数据导出为论文/可视化友好的结构化文件。

    输出：
    - learning_curve_eval.csv / .json：训练中评估点
    - learning_curve_train.csv：SB3 progress.csv 的可视化副本（若存在）
    - convergence_summary.json：first / selected_best / reward_best / last / last5 统计
    """
    out_dir = Path(train_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    summary: dict[str, Any] = {
        "eval_points": int(len(eval_history_rows)),
        "total_timesteps": None if total_timesteps is None else int(total_timesteps),
    }

    if eval_history_rows:
        eval_csv_path = out_dir / "learning_curve_eval.csv"
        eval_json_path = out_dir / "learning_curve_eval.json"
        flat_rows: list[dict[str, Any]] = []
        for eval_index, row in enumerate(eval_history_rows):
            flat = flatten_mapping(row)
            gate = row.get("gate") or {}
            reliability_mean = row.get("reliability_mean") or {}
            reliability_min = row.get("reliability_min") or {}
            plateau = row.get("plateau") or {}
            flat["eval_index"] = int(eval_index)
            flat["reliability_mean_heat"] = float(reliability_mean.get("heat", 0.0))
            flat["reliability_mean_cooling"] = float(reliability_mean.get("cooling", 0.0))
            flat["reliability_mean_electric"] = float(reliability_mean.get("electric", 0.0))
            flat["reliability_min_heat"] = float(reliability_min.get("heat", 0.0))
            flat["reliability_min_cooling"] = float(reliability_min.get("cooling", 0.0))
            flat["reliability_min_electric"] = float(reliability_min.get("electric", 0.0))
            flat["best_gate_passed"] = bool(gate.get("passed", False))
            flat["best_gate_shortfall_total"] = float((gate.get("shortfall") or {}).get("total", 0.0))
            flat["best_gate_shortfall_max"] = float((gate.get("shortfall") or {}).get("max", 0.0))
            flat["current_learning_rate"] = float(row.get("learning_rate", 0.0))
            flat["plateau_no_improve_evals"] = int(plateau.get("no_improve_evals", 0))
            flat["plateau_stop_requested"] = bool(plateau.get("stop_requested", False))
            flat_rows.append(flat)
        pd.DataFrame(flat_rows).to_csv(eval_csv_path, index=False)
        eval_json_path.write_text(
            json.dumps(eval_history_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        files["learning_curve_eval_csv"] = str(eval_csv_path)
        files["learning_curve_eval_json"] = str(eval_json_path)

        def _shortfall_total(item: dict[str, Any]) -> float:
            gate = item.get("gate") or {}
            shortfall = gate.get("shortfall") or {}
            return float(shortfall.get("total", 0.0))

        def _mean_reward(item: dict[str, Any]) -> float:
            return float(item.get("mean_reward", 0.0))

        def _snapshot_brief(item: dict[str, Any] | None) -> dict[str, Any] | None:
            if not item:
                return None
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else item
            gate = item.get("gate") or {}
            return {
                "timesteps": int(item.get("timesteps", metrics.get("timesteps", 0) or 0)),
                "mean_reward": float(metrics.get("mean_reward", 0.0)),
                "mean_total_cost": float(metrics.get("mean_total_cost", 0.0)),
                "reliability_min": dict(metrics.get("reliability_min", {})),
                "gate_passed": bool(gate.get("passed", False)),
                "gate_shortfall": dict(gate.get("shortfall", {})),
            }

        last5 = eval_history_rows[-5:]
        shortfall_last5 = [_shortfall_total(row) for row in last5]
        reward_last5 = [_mean_reward(row) for row in last5]
        best_selected = _snapshot_brief(selected_snapshot)
        best_reward = _snapshot_brief(reward_leader_snapshot)
        summary.update(
            {
                "first_eval": _snapshot_brief(eval_history_rows[0]),
                "last_eval": _snapshot_brief(eval_history_rows[-1]),
                "selected_best": best_selected,
                "reward_best": best_reward,
                "last5_shortfall_total_mean": (
                    float(sum(shortfall_last5) / len(shortfall_last5)) if shortfall_last5 else None
                ),
                "last5_shortfall_total_std": (
                    float(pd.Series(shortfall_last5, dtype="float64").std(ddof=0))
                    if len(shortfall_last5) >= 1
                    else None
                ),
                "last5_mean_reward_mean": (
                    float(sum(reward_last5) / len(reward_last5)) if reward_last5 else None
                ),
                "last5_mean_reward_std": (
                    float(pd.Series(reward_last5, dtype="float64").std(ddof=0))
                    if len(reward_last5) >= 1
                    else None
                ),
                "selected_before_halfway": (
                    None
                    if best_selected is None or total_timesteps is None or int(total_timesteps) <= 0
                    else bool(int(best_selected["timesteps"]) <= int(total_timesteps) // 2)
                ),
            }
        )

    if progress_df is not None and not progress_df.empty:
        progress_path = out_dir / "learning_curve_train.csv"
        progress_df.to_csv(progress_path, index=False)
        files["learning_curve_train_csv"] = str(progress_path)

    summary_path = out_dir / "convergence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files["convergence_summary_json"] = str(summary_path)

    manifest = {"files": files, "summary": summary}
    manifest_path = out_dir / "learning_curve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    files["learning_curve_manifest_json"] = str(manifest_path)

    return {"files": files, "summary": summary}
