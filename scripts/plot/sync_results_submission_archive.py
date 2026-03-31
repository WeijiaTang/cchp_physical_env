from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results"
DEFAULT_ARCHIVE_TAG = "paper_ready_2026-03-31"
FULL_TABLE = RESULTS_ROOT / "tables" / "paper" / "drl_all_available_multi_seed_yearly_eval_2026-03-31.csv"
AGG_TABLE = RESULTS_ROOT / "tables" / "paper" / "drl_all_available_multi_seed_yearly_eval_aggregate_2026-03-31.csv"

DRL_COMPACT_FILES: tuple[tuple[str, str], ...] = (
    ("train", "summary.json"),
    ("train", "learning_curve_eval.csv"),
    ("train", "convergence_summary.json"),
    ("eval", "summary.json"),
    ("eval", "summary_flat.csv"),
    ("eval", "behavior_metrics.json"),
    ("eval", "cost_breakdown.csv"),
    ("eval", "diagnostic_counts.csv"),
    ("eval", "state_diagnostic_counts.csv"),
    ("eval", "daily_agg.csv"),
    ("eval", "kpi_by_month.csv"),
    ("eval", "kpi_by_hour.csv"),
)
DRL_REPRESENTATIVE_FILES: tuple[tuple[str, str], ...] = (("eval", "step_log_light.csv"),)
BASELINE_FILES: tuple[tuple[str, str], ...] = (
    ("eval", "summary.json"),
    ("eval", "summary_flat.csv"),
    ("eval", "cost_breakdown.csv"),
    ("eval", "diagnostic_counts.csv"),
    ("eval", "state_diagnostic_counts.csv"),
    ("eval", "daily_agg.csv"),
    ("eval", "step_log_light.csv"),
)
BASELINES: tuple[dict[str, str], ...] = (
    {
        "slug": "oracle_milp_full",
        "label": "Oracle MILP (strict, h32)",
        "source": "runs/oracle_milp_full",
    },
    {
        "slug": "rule_reference_h16",
        "label": "Rule baseline (h16)",
        "source": "runs/rule_h16_full_v1",
    },
)


def _safe_slug(value: str) -> str:
    chars: list[str] = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path)


def _find_run_dir(row: pd.Series) -> Path:
    run_id = str(row["run_id"])
    root_group = str(row["root_group"]).strip().lower()
    search_root = ROOT / root_group
    if not search_root.exists():
        raise FileNotFoundError(f"Search root does not exist for run {run_id}: {search_root}")

    matches: list[Path] = []
    for current_root, dirnames, _ in os.walk(search_root, topdown=True, onerror=lambda _: None):
        if run_id in dirnames:
            matches.append(Path(current_root) / run_id)
    if not matches:
        raise FileNotFoundError(f"Unable to locate run directory for {run_id} under {search_root}")

    scored: list[tuple[int, Path]] = []
    for candidate in matches:
        score = 0
        if (candidate / "train").exists():
            score += 1
        if (candidate / "eval").exists():
            score += 1
        scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], len(str(item[1]))))
    return scored[0][1]


def _copy_selected_files(src_root: Path, dst_root: Path, file_pairs: tuple[tuple[str, str], ...]) -> list[str]:
    copied: list[str] = []
    for section, filename in file_pairs:
        src = src_root / section / filename
        if not src.exists():
            continue
        dst = dst_root / section / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst.relative_to(dst_root)).replace("\\", "/"))
    return copied


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_submission_archive(*, archive_tag: str = DEFAULT_ARCHIVE_TAG) -> dict[str, Any]:
    full_df = _load_table(FULL_TABLE)
    agg_df = _load_table(AGG_TABLE)

    archive_root = RESULTS_ROOT / "archive" / archive_tag
    if archive_root.exists():
        shutil.rmtree(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "catalogs").mkdir(parents=True, exist_ok=True)

    best_idx = full_df.groupby("algo")["total_cost_m"].idxmin()
    representative_keys = {
        (str(full_df.loc[idx, "algo"]), int(full_df.loc[idx, "train_seed"])) for idx in best_idx.tolist()
    }

    catalog_rows: list[dict[str, Any]] = []
    for _, row in full_df.iterrows():
        run_dir = _find_run_dir(row)
        algo = str(row["algo"])
        train_seed = int(row["train_seed"])
        normalized_root = archive_root / "drl" / algo / f"seed_{train_seed}"
        copied_files = _copy_selected_files(run_dir, normalized_root, DRL_COMPACT_FILES)
        is_representative = (algo, train_seed) in representative_keys
        if is_representative:
            copied_files.extend(_copy_selected_files(run_dir, normalized_root, DRL_REPRESENTATIVE_FILES))

        manifest = {
            "algo": algo,
            "model": str(row["model"]),
            "train_seed": train_seed,
            "eval_seed": int(row["eval_seed"]),
            "run_id": str(row["run_id"]),
            "root_group": str(row["root_group"]),
            "source_run_dir": str(run_dir),
            "representative_run": is_representative,
            "copied_files": copied_files,
            "metrics": {
                "total_cost_m": float(row["total_cost_m"]),
                "rel_heat": float(row["rel_heat"]),
                "rel_cool": float(row["rel_cool"]),
                "unmet_c_mwh": float(row["unmet_c_mwh"]),
                "viol_rate": float(row["viol_rate"]),
                "starts_gt": int(row["starts_gt"]),
                "starts_ech": int(row["starts_ech"]),
            },
        }
        _write_json(normalized_root / "source_manifest.json", manifest)

        catalog_row = row.to_dict()
        catalog_row["source_run_dir"] = str(run_dir)
        catalog_row["archive_dir"] = str(normalized_root)
        catalog_row["representative_run"] = is_representative
        catalog_rows.append(catalog_row)

    baseline_rows: list[dict[str, Any]] = []
    for baseline in BASELINES:
        src_root = ROOT / baseline["source"]
        dst_root = archive_root / "baselines" / baseline["slug"]
        copied_files = _copy_selected_files(src_root, dst_root, BASELINE_FILES)
        summary_path = src_root / "eval" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        payload = {
            "slug": baseline["slug"],
            "label": baseline["label"],
            "source_run_dir": str(src_root),
            "copied_files": copied_files,
            "summary_excerpt": {
                "total_cost": summary.get("total_cost"),
                "reliability": summary.get("reliability"),
                "violation_rate": summary.get("violation_rate"),
                "starts": summary.get("starts"),
            },
        }
        _write_json(dst_root / "source_manifest.json", payload)
        baseline_rows.append(
            {
                "slug": baseline["slug"],
                "label": baseline["label"],
                "source_run_dir": str(src_root),
                "archive_dir": str(dst_root),
                "total_cost": summary.get("total_cost"),
                "rel_heat": (summary.get("reliability") or {}).get("heat"),
                "rel_cool": (summary.get("reliability") or {}).get("cooling"),
                "violation_rate": summary.get("violation_rate"),
            }
        )

    catalog_df = pd.DataFrame(catalog_rows)
    catalog_df.to_csv(archive_root / "catalogs" / "drl_eval_catalog.csv", index=False, encoding="utf-8")
    agg_df.to_csv(archive_root / "catalogs" / "drl_eval_aggregate.csv", index=False, encoding="utf-8")
    pd.DataFrame(baseline_rows).to_csv(archive_root / "catalogs" / "baseline_catalog.csv", index=False, encoding="utf-8")

    readme_lines = [
        "# Paper-Ready Submission Archive",
        "",
        f"Archive tag: `{archive_tag}`",
        "",
        "This folder is the normalized publication-facing archive copied from `runs/` and `kaggle/`.",
        "",
        "## Contents",
        "",
        "- `drl/`: per-algorithm, per-seed compact yearly evaluation artifacts",
        "- `baselines/`: rule and Oracle reference runs",
        "- `catalogs/`: normalized CSV catalogs and aggregate tables used by plotting",
        "",
        "## Copy policy",
        "",
        "- All DRL runs keep compact train/eval artifacts needed for paper tables and convergence plots.",
        "- Only representative runs keep `step_log_light.csv` to avoid bloating the archive.",
        "- Original source directories remain untouched; this archive is a submission-facing copy.",
        "",
        "## Source tables",
        "",
        f"- `{FULL_TABLE.relative_to(ROOT).as_posix()}`",
        f"- `{AGG_TABLE.relative_to(ROOT).as_posix()}`",
    ]
    (archive_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    top_manifest = {
        "archive_tag": archive_tag,
        "archive_root": str(archive_root),
        "drl_run_count": int(len(catalog_df)),
        "baseline_count": int(len(baseline_rows)),
        "representative_runs": [
            {
                "algo": str(row["algo"]),
                "train_seed": int(row["train_seed"]),
                "archive_dir": str(row["archive_dir"]),
            }
            for _, row in catalog_df[catalog_df["representative_run"] == True].iterrows()
        ],
        "full_catalog_csv": str(archive_root / "catalogs" / "drl_eval_catalog.csv"),
        "aggregate_csv": str(archive_root / "catalogs" / "drl_eval_aggregate.csv"),
        "baseline_catalog_csv": str(archive_root / "catalogs" / "baseline_catalog.csv"),
    }
    _write_json(archive_root / "manifest.json", top_manifest)
    return top_manifest


def main() -> None:
    payload = build_submission_archive()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
