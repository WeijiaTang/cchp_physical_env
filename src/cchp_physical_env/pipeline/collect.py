# Ref: docs/spec/task.md (Task-ID: 011)
# Ref: docs/spec/architecture.md (Module: pipeline/)
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.reporting import flatten_mapping


DEFAULT_PAPER_COLUMNS: tuple[str, ...] = (
    "run_name",
    "run_dir",
    "eval_dir",
    "year",
    "policy",
    "sequence_adapter",
    "algo",
    "backbone",
    "history_steps",
    "seed",
    "total_cost",
    "violation_rate",
    "unmet_energy_mwh__electric",
    "unmet_energy_mwh__heat",
    "unmet_energy_mwh__cooling",
    "reliability__electric",
    "reliability__heat",
    "reliability__cooling",
    "starts__gt",
    "starts__boiler",
    "starts__ech",
    "cost_breakdown__grid",
    "cost_breakdown__gt_fuel",
    "cost_breakdown__gt_om",
    "cost_breakdown__carbon",
    "cost_breakdown__unmet_h",
    "cost_breakdown__unmet_c",
    "cost_breakdown__viol",
    "emissions_ton__total",
)


def _find_eval_summaries(runs_root: str | Path) -> list[Path]:
    root = Path(runs_root)
    if not root.exists():
        raise FileNotFoundError(f"runs_root 不存在: {root}")
    return sorted(root.rglob("eval/summary.json"))


def collect_run_summaries(*, runs_root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in _find_eval_summaries(runs_root):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        eval_dir = summary_path.parent
        run_dir = eval_dir.parent
        row = flatten_mapping(summary)
        row["run_name"] = str(run_dir.name)
        row["run_dir"] = str(run_dir).replace("\\", "/")
        row["eval_dir"] = str(eval_dir).replace("\\", "/")
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "total_cost" in df.columns:
        df = df.sort_values(by=["total_cost"], ascending=[True])
    return df


def write_benchmark_tables(
    *,
    runs_root: str | Path,
    output_csv: str | Path,
    full_output_csv: str | Path | None = None,
) -> dict[str, str]:
    df = collect_run_summaries(runs_root=runs_root)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        out_path.write_text("", encoding="utf-8")
        return {"output_csv": str(out_path)}

    if full_output_csv is not None:
        full_path = Path(full_output_csv)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(full_path, index=False)

    cols = [c for c in DEFAULT_PAPER_COLUMNS if c in df.columns]
    paper_df = df[cols].copy() if cols else df.copy()
    paper_df.to_csv(out_path, index=False)
    result = {"output_csv": str(out_path)}
    if full_output_csv is not None:
        result["full_output_csv"] = str(Path(full_output_csv))
    result["n_runs"] = str(int(len(df)))
    return result
