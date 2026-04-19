from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_AGGREGATE_TABLE = Path("results/tables/paper/drl_all_available_multi_seed_yearly_eval_aggregate_2026-03-31.csv")
DEFAULT_PER_SEED_TABLE = Path("results/tables/paper/drl_all_available_multi_seed_yearly_eval_2026-03-31.csv")
DEFAULT_BASELINE_TABLE = Path("results/tables/paper/multi_seed_nature/baseline_reference_data.csv")
DEFAULT_OUTPUT_DIR = Path("results/tables/paper")

MAIN_GATE_ELECTRIC = 1.0
MAIN_GATE_HEAT = 0.99
MAIN_GATE_COOL = 0.99


@dataclass(frozen=True)
class PAFCSummaryRow:
    algo: str
    model: str
    total_cost_m: float
    rel_electric: float
    rel_heat: float
    rel_cool: float
    unmet_c_mwh: float
    starts_gt: int
    starts_ech: int
    grid_m: float
    boiler_m: float
    gt_fuel_m: float
    gt_om_m: float
    carbon_m: float
    grid_export_penalty_m: float


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _load_pafc_summary(path: Path, *, algo: str, model: str) -> PAFCSummaryRow:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cost_breakdown = dict(payload.get("cost_breakdown", {}) or {})
    reliability = dict(payload.get("reliability", {}) or {})
    unmet_energy = dict(payload.get("unmet_energy_mwh", {}) or {})
    starts = dict(payload.get("starts", {}) or {})
    return PAFCSummaryRow(
        algo=algo,
        model=model,
        total_cost_m=float(payload.get("total_cost", 0.0)) / 1e6,
        rel_electric=float(reliability.get("electric", 0.0)),
        rel_heat=float(reliability.get("heat", 0.0)),
        rel_cool=float(reliability.get("cooling", 0.0)),
        unmet_c_mwh=float(unmet_energy.get("cooling", 0.0)),
        starts_gt=int(starts.get("gt", 0)),
        starts_ech=int(starts.get("ech", 0)),
        grid_m=float(cost_breakdown.get("grid", 0.0)) / 1e6,
        boiler_m=float(cost_breakdown.get("boiler", 0.0)) / 1e6,
        gt_fuel_m=float(cost_breakdown.get("gt_fuel", 0.0)) / 1e6,
        gt_om_m=float(cost_breakdown.get("gt_om", 0.0)) / 1e6,
        carbon_m=float(cost_breakdown.get("carbon", 0.0)) / 1e6,
        grid_export_penalty_m=float(cost_breakdown.get("grid_export_penalty", 0.0)) / 1e6,
    )


def _build_aggregate_rows(
    *,
    baseline_rows: list[dict[str, str]],
    pafc_row: PAFCSummaryRow,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in baseline_rows:
        rows.append(
            {
                "algo": row["algo"],
                "model": row["model"],
                "n_seeds": int(row["n_seeds"]),
                "total_cost_m": float(row["total_cost_mean"]),
                "rel_cool": float(row["rel_cool_mean"]),
                "unmet_c_mwh": float(row["unmet_c_mean"]),
                "starts_gt": float(row["starts_gt_mean"]),
                "starts_ech": float(row["starts_ech_mean"]),
                "source": "paper_aggregate",
            }
        )
    rows.append(
        {
            "algo": pafc_row.algo,
            "model": pafc_row.model,
            "n_seeds": 1,
            "total_cost_m": pafc_row.total_cost_m,
            "rel_cool": pafc_row.rel_cool,
            "unmet_c_mwh": pafc_row.unmet_c_mwh,
            "starts_gt": pafc_row.starts_gt,
            "starts_ech": pafc_row.starts_ech,
            "source": "pafc_eval_summary",
        }
    )
    return sorted(rows, key=lambda item: float(item["total_cost_m"]))


def _passes_main_gate(row: dict[str, str]) -> bool:
    rel_heat = float(row["rel_heat"])
    rel_cool = float(row["rel_cool"])
    viol_rate = float(row["viol_rate"])
    return (
        float(row.get("rel_heat", 0.0)) >= MAIN_GATE_HEAT
        and float(row.get("rel_cool", 0.0)) >= MAIN_GATE_COOL
        and abs(float(row.get("rel_heat", 0.0)) - rel_heat) < 1e-12
        and viol_rate <= 1e-12
    )


def _build_passed_run_rows(
    *,
    per_seed_rows: list[dict[str, str]],
    pafc_row: PAFCSummaryRow,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in per_seed_rows:
        if not _passes_main_gate(row):
            continue
        rows.append(
            {
                "algo": row["algo"],
                "model": row["model"],
                "train_seed": row["train_seed"],
                "eval_seed": row["eval_seed"],
                "total_cost_m": float(row["total_cost_m"]),
                "rel_cool": float(row["rel_cool"]),
                "unmet_c_mwh": float(row["unmet_c_mwh"]),
                "starts_gt": int(row["starts_gt"]),
                "starts_ech": int(row["starts_ech"]),
                "run_id": row["run_id"],
                "source": "paper_per_seed",
            }
        )
    rows = sorted(rows, key=lambda item: float(item["total_cost_m"]))[: max(1, int(top_k))]
    rows.append(
        {
            "algo": pafc_row.algo,
            "model": pafc_row.model,
            "train_seed": "42",
            "eval_seed": "42",
            "total_cost_m": pafc_row.total_cost_m,
            "rel_cool": pafc_row.rel_cool,
            "unmet_c_mwh": pafc_row.unmet_c_mwh,
            "starts_gt": pafc_row.starts_gt,
            "starts_ech": pafc_row.starts_ech,
            "run_id": "pafc_eval_summary",
            "source": "pafc_eval_summary",
        }
    )
    return sorted(rows, key=lambda item: float(item["total_cost_m"]))


def _find_baseline_row(baseline_rows: list[dict[str, str]], *, slug: str) -> dict[str, str]:
    for row in baseline_rows:
        if row["slug"] == slug:
            return row
    raise KeyError(f"baseline slug not found: {slug}")


def _build_markdown(
    *,
    pafc_row: PAFCSummaryRow,
    aggregate_rows: list[dict[str, Any]],
    passed_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, str]],
) -> str:
    best_aggregate = aggregate_rows[0]
    best_passed = passed_rows[0]
    rule_row = _find_baseline_row(baseline_rows, slug="rule_reference_h16")
    oracle_row = _find_baseline_row(baseline_rows, slug="oracle_milp_full")

    lines: list[str] = []
    lines.append("# PAFC vs Existing DRL")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- PAFC total cost: `{_format_float(pafc_row.total_cost_m, 3)} M`, cooling reliability: `{_format_float(pafc_row.rel_cool, 6)}`, unmet cooling: `{_format_float(pafc_row.unmet_c_mwh, 3)} MWh`."
    )
    lines.append(
        f"- Gap vs best DRL aggregate mean ({best_aggregate['model']}): `+{_format_float(pafc_row.total_cost_m - float(best_aggregate['total_cost_m']), 3)} M`."
    )
    lines.append(
        f"- Gap vs best passed single DRL run ({best_passed['model']} seed {best_passed['train_seed']}): `+{_format_float(pafc_row.total_cost_m - float(best_passed['total_cost_m']), 3)} M`."
    )
    lines.append(
        f"- Gap vs rule baseline: `+{_format_float(pafc_row.total_cost_m - float(rule_row['total_cost_m']), 3)} M`; gap vs Oracle MILP: `+{_format_float(pafc_row.total_cost_m - float(oracle_row['total_cost_m']), 3)} M`."
    )
    lines.append(
        f"- Control burden is dramatically lower: GT starts `{pafc_row.starts_gt}`, ECH starts `{pafc_row.starts_ech}`."
    )
    lines.append("")
    lines.append("## PAFC Cost Breakdown")
    lines.append("")
    lines.append("| Metric | Value (M) |")
    lines.append("|---|---:|")
    lines.append(f"| Grid | {_format_float(pafc_row.grid_m, 3)} |")
    lines.append(f"| Boiler | {_format_float(pafc_row.boiler_m, 3)} |")
    lines.append(f"| GT fuel | {_format_float(pafc_row.gt_fuel_m, 3)} |")
    lines.append(f"| GT O&M | {_format_float(pafc_row.gt_om_m, 3)} |")
    lines.append(f"| Carbon | {_format_float(pafc_row.carbon_m, 3)} |")
    lines.append(f"| Grid export penalty | {_format_float(pafc_row.grid_export_penalty_m, 3)} |")
    lines.append("")
    lines.append("## Aggregate Ranking")
    lines.append("")
    lines.append("| Rank | Model | Total cost (M) | Rel. cool | Unmet cool (MWh) | GT starts | ECH starts |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(aggregate_rows, start=1):
        lines.append(
            f"| {idx} | {row['model']} | {_format_float(float(row['total_cost_m']), 3)} | {_format_float(float(row['rel_cool']), 6)} | {_format_float(float(row['unmet_c_mwh']), 3)} | {_format_float(float(row['starts_gt']), 1)} | {_format_float(float(row['starts_ech']), 1)} |"
        )
    lines.append("")
    lines.append("## Passed Single-Run Ranking")
    lines.append("")
    lines.append("| Rank | Model | Train seed | Total cost (M) | Rel. cool | Unmet cool (MWh) | GT starts | ECH starts |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(passed_rows, start=1):
        lines.append(
            f"| {idx} | {row['model']} | {row['train_seed']} | {_format_float(float(row['total_cost_m']), 3)} | {_format_float(float(row['rel_cool']), 6)} | {_format_float(float(row['unmet_c_mwh']), 3)} | {int(row['starts_gt'])} | {int(row['starts_ech'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PAFC-vs-existing-DRL comparison tables.")
    parser.add_argument("--pafc-summary", type=Path, required=True, help="Path to PAFC eval summary.json")
    parser.add_argument("--label", default="PAFC-TD3", help="Display label for the PAFC row")
    parser.add_argument("--algo", default="pafc_td3", help="Algo slug for the PAFC row")
    parser.add_argument("--aggregate-table", type=Path, default=DEFAULT_AGGREGATE_TABLE)
    parser.add_argument("--per-seed-table", type=Path, default=DEFAULT_PER_SEED_TABLE)
    parser.add_argument("--baseline-table", type=Path, default=DEFAULT_BASELINE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--date-tag", default=str(date.today()))
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    pafc_row = _load_pafc_summary(args.pafc_summary, algo=str(args.algo), model=str(args.label))
    aggregate_rows = _build_aggregate_rows(
        baseline_rows=_read_csv_rows(args.aggregate_table),
        pafc_row=pafc_row,
    )
    passed_rows = _build_passed_run_rows(
        per_seed_rows=_read_csv_rows(args.per_seed_table),
        pafc_row=pafc_row,
        top_k=int(args.top_k),
    )
    baseline_rows = _read_csv_rows(args.baseline_table)

    output_dir = args.output_dir
    aggregate_csv = output_dir / f"pafc_vs_existing_drl_aggregate_{args.date_tag}.csv"
    passed_csv = output_dir / f"pafc_vs_existing_drl_passed_runs_{args.date_tag}.csv"
    summary_md = output_dir / f"pafc_vs_existing_drl_{args.date_tag}.md"
    summary_json = output_dir / f"pafc_vs_existing_drl_{args.date_tag}.json"

    _write_csv(
        aggregate_csv,
        aggregate_rows,
        fieldnames=["algo", "model", "n_seeds", "total_cost_m", "rel_cool", "unmet_c_mwh", "starts_gt", "starts_ech", "source"],
    )
    _write_csv(
        passed_csv,
        passed_rows,
        fieldnames=["algo", "model", "train_seed", "eval_seed", "total_cost_m", "rel_cool", "unmet_c_mwh", "starts_gt", "starts_ech", "run_id", "source"],
    )
    summary_md.write_text(
        _build_markdown(
            pafc_row=pafc_row,
            aggregate_rows=aggregate_rows,
            passed_rows=passed_rows,
            baseline_rows=baseline_rows,
        ),
        encoding="utf-8",
    )
    summary_json.write_text(
        json.dumps(
            {
                "pafc": pafc_row.__dict__,
                "aggregate_rows": aggregate_rows,
                "passed_rows": passed_rows,
                "baseline_rows": baseline_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "aggregate_csv": str(aggregate_csv),
                "passed_csv": str(passed_csv),
                "summary_md": str(summary_md),
                "summary_json": str(summary_json),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
