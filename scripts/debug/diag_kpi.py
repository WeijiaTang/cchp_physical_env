"""Rule policy KPI diagnostics with optional step limits."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import collect_episode, load_context


def _sep(title: str) -> None:
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print KPI diagnostics for the rule policy.")
    parser.add_argument("--env-config", type=Path, default=Path("src/cchp_physical_env/config/config.yaml"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/cchp_main_15min_2024.csv"))
    parser.add_argument("--eval", type=Path, default=Path("data/processed/cchp_main_15min_2025.csv"))
    parser.add_argument("--dataset", choices=("train", "eval"), default="eval")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the detailed KPI table.")
    parser.add_argument(
        "--summary-seeds",
        type=int,
        nargs="+",
        default=[0, 42, 123, 456, 999],
        help="Seeds for the variability block.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on steps per episode.")
    parser.add_argument("--export-csv", type=Path, help="Optional path to dump the per-step DataFrame.")
    return parser.parse_args()


def _print_env_basics(ctx) -> None:
    cfg = ctx.env_config
    print("Environment snapshot")
    print(f"  sigma_per_hour       = {cfg.sigma_per_hour}")
    print(f"  gt_start_cost        = {cfg.gt_start_cost} CNY/start")
    print(f"  gt_min_output_mw     = {cfg.gt_min_output_mw} MW")
    print(f"  e_tes_cap_mwh        = {cfg.e_tes_cap_mwh} MWh")
    print(f"  e_tes_init_mwh       = {cfg.e_tes_init_mwh} MWh")
    soc_init = cfg.e_tes_init_mwh / max(1e-9, cfg.e_tes_cap_mwh)
    t_init = cfg.hrsg_water_inlet_k + soc_init * 60.0
    print(f"  TES init SOC         = {soc_init:.2f}")
    print(f"  TES init temp        = {t_init:.1f} K ({t_init-273.15:.1f} C)")


def _summarize_seeds(ctx, seeds, args):
    _sep("0. Seed variability")
    summary_costs = []
    cache: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        result = collect_episode(ctx, dataset=args.dataset, seed=seed, max_steps=args.max_steps)
        df_seed = result.steps
        cache[seed] = df_seed
        tc = df_seed["cost_total"].sum()
        vr = df_seed["hrsg_cap_inv"].mean()
        gt_starts = df_seed["gt_started"].sum()
        summary_costs.append(tc)
        print(f"  seed={seed:4d}: total_cost={tc:>14,.0f}  hrsg_cap_inv_rate={vr:.4f}  gt_starts={gt_starts}")
    print(f"  StdDev(total_cost): {np.std(summary_costs):.1f}")
    return cache


def _detailed_tables(df: pd.DataFrame) -> None:
    n_steps = len(df)
    if n_steps == 0:
        print("No steps collected.")
        return

    _sep("1. HRSG capacity_invalid")
    hrsg_inv = df["hrsg_cap_inv"].sum()
    gt_off = (df["p_gt_mw"] == 0.0).sum()
    print(f"Steps: {n_steps}")
    print(f"hrsg_capacity_invalid count: {hrsg_inv}  ({100 * hrsg_inv / max(1, n_steps):.2f}%)")
    if hrsg_inv > 0:
        viol = df[df["hrsg_cap_inv"]]
        print(f"  GT=0 at violation: {(viol['p_gt_mw'] == 0.0).sum()}")
        print(f"  GT>0 at violation: {(viol['p_gt_mw'] > 0.0).sum()}")
    print(f"GT=0 steps: {gt_off}  ({100 * gt_off / max(1, n_steps):.1f}%)")
    print(f"GT>0 steps: {(df['p_gt_mw'] > 0.0).sum()}  ({100 * (df['p_gt_mw'] > 0.0).mean():.1f}%)")

    _sep("2. TES temperature / abs drive")
    print(f"abs_drive_temp_low_state: {df['abs_temp_low'].sum()} / {n_steps}  ({100*df['abs_temp_low'].mean():.1f}%)")
    print(f"t_hot >= 343.15K (70C): {(df['t_tes_hot_k'] >= 343.15).sum()}")
    print(f"t_hot >= 358.15K (85C): {(df['t_tes_hot_k'] >= 358.15).sum()}")
    print("\nTES temperature quantiles (K / C):")
    for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        t_val = df["t_tes_hot_k"].quantile(q)
        print(f"  p{int(q*100):02d}: {t_val:.2f} K  ({t_val-273.15:.1f} C)")
    t_max = df["t_tes_hot_k"].max()
    print(f"  max: {t_max:.2f} K  ({t_max-273.15:.1f} C)")

    _sep("3. TES and chiller usage")
    print(f"TES charge steps (>0.01 MW): {(df['q_tes_charge_mw'] > 0.01).sum()}")
    print(f"TES discharge steps (>0.01 MW): {(df['q_tes_discharge_mw'] > 0.01).sum()}")
    print(f"HRSG avg output: {df['q_hrsg_rec_mw'].mean():.3f} MW  total: {df['q_hrsg_rec_mw'].sum()*0.25:.1f} MWh")
    print(f"Boiler avg output: {df['q_boiler_mw'].mean():.3f} MW")
    print(f"Abs cooling avg: {df['q_abs_cool_mw'].mean():.3f} MW")
    print(f"ECH cooling avg: {df['q_ech_cool_mw'].mean():.3f} MW")
    print(f"TES charge total: {df['q_tes_charge_mw'].sum()*0.25:.1f} MWh")
    print(f"TES discharge total: {df['q_tes_discharge_mw'].sum()*0.25:.1f} MWh")

    _sep("4. GT operating stats")
    print(df["p_gt_mw"].describe().round(3).to_string())
    gt_starts = int(df["gt_started"].sum())
    print(f"GT starts: {gt_starts} per year ({gt_starts/365:.2f} /day)")
    print(f"GT zero output steps: {(df['p_gt_mw'] == 0.0).sum()} / {n_steps}")
    print(f"safety_gt_min_output_enforced: {df['gt_min_enforced'].sum()} steps")
    print(f"safety_gt_ramp_limited:        {df['gt_ramp_limited'].sum()} steps")

    _sep("5. cost_unmet_h breakdown")
    unmet_h = df[df["energy_unmet_h_mwh"] > 1e-4]
    total_unmet_h = df["energy_unmet_h_mwh"].sum()
    total_dem_h = df["energy_demand_h_mwh"].sum()
    print(f"Unmet steps: {len(unmet_h)} / {n_steps}  ({100*len(unmet_h)/max(1,n_steps):.1f}%)")
    print(f"Unmet energy: {total_unmet_h:.1f} MWh  demand: {total_dem_h:.1f} MWh")
    print(f"Heat reliability: {1 - total_unmet_h / max(1e-9, total_dem_h):.4f}")
    print(f"cost_unmet_h: {df['cost_unmet_h'].sum():,.0f} CNY")
    if len(unmet_h) > 0:
        supply = unmet_h["q_hrsg_rec_mw"] + unmet_h["q_boiler_mw"] + unmet_h["q_tes_discharge_mw"]
        print(f"  avg GT output: {unmet_h['p_gt_mw'].mean():.3f} MW")
        print(f"  avg HRSG heat: {unmet_h['q_hrsg_rec_mw'].mean():.3f} MW")
        print(f"  avg boiler:    {unmet_h['q_boiler_mw'].mean():.3f} MW")
        print(f"  avg TES dis:   {unmet_h['q_tes_discharge_mw'].mean():.3f} MW")
        print(f"  avg supply vs demand: {supply.mean():.3f} MW vs {unmet_h['qh_dem_mw'].mean():.3f} MW")

    _sep("6. cost_unmet_c breakdown")
    unmet_c = df[df["energy_unmet_c_mwh"] > 1e-4]
    total_unmet_c = df["energy_unmet_c_mwh"].sum()
    total_dem_c = df["energy_demand_c_mwh"].sum()
    print(f"Unmet steps: {len(unmet_c)} / {n_steps}  ({100*len(unmet_c)/max(1,n_steps):.1f}%)")
    print(f"Unmet energy: {total_unmet_c:.1f} MWh  demand: {total_dem_c:.1f} MWh")
    print(f"Cooling reliability: {1 - total_unmet_c / max(1e-9, total_dem_c):.4f}")
    print(f"cost_unmet_c: {df['cost_unmet_c'].sum():,.0f} CNY")

    _sep("7. Cost breakdown")
    total_cost = df["cost_total"].sum()
    for col in sorted(c for c in df.columns if c.startswith("cost_")):
        val = df[col].sum()
        pct = 100 * val / total_cost if total_cost > 0 else 0.0
        print(f"  {col:<32s}: {val:>14,.0f}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<32s}: {total_cost:>14,.0f}")


def main() -> None:
    args = _parse_args()
    ctx = load_context(args.env_config, args.train, args.eval)
    _print_env_basics(ctx)

    cache = _summarize_seeds(ctx, args.summary_seeds, args)
    if args.seed in cache:
        detailed_df = cache[args.seed]
    else:
        detailed_df = collect_episode(ctx, dataset=args.dataset, seed=args.seed, max_steps=args.max_steps).steps

    _detailed_tables(detailed_df)
    if args.export_csv:
        detailed_df.to_csv(args.export_csv, index=False)
        print(f"\nSaved per-step data to {args.export_csv}")


if __name__ == "__main__":
    main()
