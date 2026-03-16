"""Root-cause analyzer for heat deficits and TES temperature."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import collect_episode, load_context


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify dominant drivers behind unmet heat/cool demand.")
    parser.add_argument("--env-config", type=Path, default=Path("src/cchp_physical_env/config/config.yaml"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/cchp_main_15min_2024.csv"))
    parser.add_argument("--eval", type=Path, default=Path("data/processed/cchp_main_15min_2025.csv"))
    parser.add_argument("--dataset", choices=("train", "eval"), default="eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--export-dir", type=Path, help="Optional directory to dump CSVs for unmet segments.")
    return parser.parse_args()


def _tag_states(df: pd.DataFrame) -> pd.DataFrame:
    tagged = df.copy()
    tagged["heat_supply_mw"] = (
        tagged["q_hrsg_rec_mw"] + tagged["q_boiler_mw"] + tagged["q_tes_discharge_mw"]
    )
    tagged["cool_supply_mw"] = tagged["q_abs_cool_mw"] + tagged["q_ech_cool_mw"]
    tagged["gt_state"] = pd.Series(
        pd.cut(tagged["p_gt_mw"], bins=[-0.1, 0.05, 12.1], labels=["gt_off", "gt_on"])
    ).astype(str)
    tagged["boiler_state"] = pd.cut(
        tagged["q_boiler_mw"], bins=[-0.1, 0.1, 10.0], labels=["boiler_idle", "boiler_on"]
    ).astype(str)
    tagged["tes_state"] = pd.cut(
        tagged["e_tes_mwh"], bins=[-0.1, 6.0, 14.0, 25.0], labels=["tes_low", "tes_mid", "tes_high"]
    ).astype(str)
    tagged["abs_ready"] = tagged["t_tes_hot_k"] >= 358.15
    return tagged


def _summarize_unmet(unmet: pd.DataFrame, label: str) -> None:
    if unmet.empty:
        print(f"No {label} unmet events.")
        return
    demand_col = "qh_dem_mw" if label == "heat" else "qc_dem_mw"
    supply_col = "heat_supply_mw" if label == "heat" else "cool_supply_mw"
    grouped = (
        unmet.groupby(["gt_state", "boiler_state", "tes_state"])
        .agg(
            steps=("timestamp", "count"),
            avg_demand=(demand_col, "mean"),
            avg_supply=(supply_col, "mean"),
        )
        .reset_index()
    )
    print(f"\n{label.capitalize()} unmet distribution (by GT/boiler/TES states):")
    print(grouped.to_string(index=False))


def _print_abs_low(tagged: pd.DataFrame) -> None:
    total = len(tagged)
    low_df = tagged[tagged["abs_temp_low"]]
    print(f"\nabs_drive_temp_low in {len(low_df)} / {total} steps ({100*len(low_df)/max(1,total):.1f}%)")
    if low_df.empty:
        return
    summary = (
        low_df.groupby(["gt_state", "boiler_state", "tes_state"])
        .agg(
            steps=("timestamp", "count"),
            avg_t_hot=("t_tes_hot_k", "mean"),
            median_e_tes=("e_tes_mwh", "median"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


def main() -> None:
    args = _parse_args()
    ctx = load_context(args.env_config, args.train, args.eval)
    result = collect_episode(ctx, dataset=args.dataset, seed=args.seed, max_steps=args.max_steps)
    tagged = _tag_states(result.steps)
    print(f"Analyzing {len(tagged)} steps (terminated={result.terminated})")

    unmet_heat = tagged[tagged["energy_unmet_h_mwh"] > 1e-4]
    unmet_cool = tagged[tagged["energy_unmet_c_mwh"] > 1e-4]
    print(f"Heat unmet steps: {len(unmet_heat)}  Cool unmet steps: {len(unmet_cool)}")
    _summarize_unmet(unmet_heat, "heat")
    _summarize_unmet(unmet_cool, "cool")
    _print_abs_low(tagged)

    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        if not unmet_heat.empty:
            unmet_heat.to_csv(args.export_dir / "heat_unmet.csv", index=False)
        if not unmet_cool.empty:
            unmet_cool.to_csv(args.export_dir / "cool_unmet.csv", index=False)
        tagged.to_csv(args.export_dir / "full_tagged.csv", index=False)
        print(f"\nSaved tagged data under {args.export_dir}")


if __name__ == "__main__":
    main()
