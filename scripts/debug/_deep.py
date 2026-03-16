"""Deep dive per-step diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import collect_episode, load_context


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump detailed per-step diagnostics.")
    parser.add_argument("--env-config", type=Path, default=Path("src/cchp_physical_env/config/config.yaml"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/cchp_main_15min_2024.csv"))
    parser.add_argument("--eval", type=Path, default=Path("data/processed/cchp_main_15min_2025.csv"))
    parser.add_argument("--dataset", choices=("train", "eval"), default="eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--export-csv", type=Path, help="Optional path to write CSV output.")
    return parser.parse_args()


def _print_heat_section(df: pd.DataFrame) -> None:
    print("--- cost_unmet_h ---")
    unmet = df[df["energy_unmet_h_mwh"] > 1e-4]
    steps = len(df)
    print(f"Unmet steps: {len(unmet)} / {steps} ({100*len(unmet)/max(1,steps):.1f}%)")
    print(f"Unmet energy: {df['energy_unmet_h_mwh'].sum():.1f} MWh  demand: {df['energy_demand_h_mwh'].sum():.1f} MWh")
    if not unmet.empty:
        supply = unmet["q_hrsg_rec_mw"] + unmet["q_boiler_mw"] + unmet["q_tes_discharge_mw"]
        print(f"  avg GT output: {unmet['p_gt_mw'].mean():.3f} MW")
        print(f"  avg HRSG heat: {unmet['q_hrsg_rec_mw'].mean():.3f} MW")
        print(f"  avg boiler:    {unmet['q_boiler_mw'].mean():.3f} MW")
        print(f"  avg TES dis:   {unmet['q_tes_discharge_mw'].mean():.3f} MW")
        print(f"  avg supply vs demand: {supply.mean():.3f} MW vs {unmet['qh_dem_mw'].mean():.3f} MW")


def _print_abs_section(df: pd.DataFrame) -> None:
    print("\n--- abs_drive_temp_low ---")
    steps = len(df)
    print(f"abs_drive_temp_low_state: {df['abs_temp_low'].sum()} / {steps} ({100*df['abs_temp_low'].mean():.1f}%)")
    print(f"t_hot >= 70C: {(df['t_tes_hot_k'] >= 343.15).sum()}  t_hot >= 85C: {(df['t_tes_hot_k'] >= 358.15).sum()}")
    print("TES temperature quantiles:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        val = df["t_tes_hot_k"].quantile(q)
        print(f"  p{int(q*100):02d}: {val:.1f} K ({val-273.15:.1f} C)")
    print(f"  max: {df['t_tes_hot_k'].max():.1f} K ({df['t_tes_hot_k'].max()-273.15:.1f} C)")


def _print_tes_section(df: pd.DataFrame) -> None:
    print("\n--- TES ---")
    print(f"Charge steps (>0.01 MW): {(df['q_tes_charge_mw'] > 0.01).sum()}")
    print(f"Discharge steps (>0.01 MW): {(df['q_tes_discharge_mw'] > 0.01).sum()}")
    print(f"HRSG>0 steps: {(df['q_hrsg_rec_mw'] > 0.01).sum()}  avg HRSG: {df['q_hrsg_rec_mw'].mean():.3f} MW")
    hrsg_on = df[df["q_hrsg_rec_mw"] > 0.01]
    if not hrsg_on.empty:
        print(f"  When HRSG>0, TES charge steps: {(hrsg_on['q_tes_charge_mw'] > 0.01).sum()} / {len(hrsg_on)}")
        print(f"  TES energy when HRSG>0: {hrsg_on['e_tes_mwh'].mean():.2f} MWh")


def _print_gt_section(df: pd.DataFrame) -> None:
    print("\n--- Gas turbine ---")
    gt_starts = int(df["gt_started"].sum())
    print(f"GT starts: {gt_starts} ({gt_starts/365:.2f} per day)")
    print(f"GT=0 steps: {(df['p_gt_mw'] == 0.0).sum()} / {len(df)}")
    print(f"safety_gt_min_enforced: {df['gt_min_enforced'].sum()} steps")
    print(df["p_gt_mw"].describe().round(3).to_string())


def _print_abs_usage(df: pd.DataFrame) -> None:
    print("\n--- Cooling split ---")
    abs_on = df[df["q_abs_cool_mw"] > 0.01]
    print(f"Absorption on steps: {len(abs_on)} / {len(df)} ({100*len(abs_on)/max(1,len(df)):.1f}%)")
    print(f"Absorption avg MW: {df['q_abs_cool_mw'].mean():.3f}")
    print(f"Electric chiller avg MW: {df['q_ech_cool_mw'].mean():.3f}")
    print(f"Cooling demand: {df['energy_demand_c_mwh'].sum():.1f} MWh  unmet: {df['energy_unmet_c_mwh'].sum():.1f} MWh")


def main() -> None:
    args = _parse_args()
    ctx = load_context(args.env_config, args.train, args.eval)
    result = collect_episode(ctx, dataset=args.dataset, seed=args.seed, max_steps=args.max_steps)
    df = result.steps
    print(f"Collected {len(df)} steps (terminated={result.terminated}) for seed={args.seed}")
    _print_heat_section(df)
    _print_abs_section(df)
    _print_tes_section(df)
    _print_gt_section(df)
    _print_abs_usage(df)
    if args.export_csv:
        df.to_csv(args.export_csv, index=False)
        print(f"\nSaved CSV to {args.export_csv}")


if __name__ == "__main__":
    main()
