"""Lightweight run summary for multiple seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import collect_episode, load_context


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate KPI stats across seeds.")
    parser.add_argument("--env-config", type=Path, default=Path("src/cchp_physical_env/config/config.yaml"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/cchp_main_15min_2024.csv"))
    parser.add_argument("--eval", type=Path, default=Path("data/processed/cchp_main_15min_2025.csv"))
    parser.add_argument("--dataset", choices=("train", "eval"), default="eval")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123, 456, 999])
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ctx = load_context(args.env_config, args.train, args.eval)
    print(f"Dataset={args.dataset}  steps_limit={args.max_steps or 'full_year'}")
    rows = []
    for seed in args.seeds:
        result = collect_episode(ctx, dataset=args.dataset, seed=seed, max_steps=args.max_steps)
        df = result.steps
        total_cost = df["cost_total"].sum()
        cost_unmet_h = df["cost_unmet_h"].sum()
        cost_unmet_c = df["cost_unmet_c"].sum()
        hrsg_inv = df["hrsg_cap_inv"].sum()
        gt_min = df["gt_min_enforced"].sum()
        heat_rel = 1 - df["energy_unmet_h_mwh"].sum() / max(1e-9, df["energy_demand_h_mwh"].sum())
        cool_rel = 1 - df["energy_unmet_c_mwh"].sum() / max(1e-9, df["energy_demand_c_mwh"].sum())
        rows.append(
            {
                "seed": seed,
                "steps": len(df),
                "total_cost": total_cost,
                "cost_unmet_h": cost_unmet_h,
                "cost_unmet_c": cost_unmet_c,
                "hrsg_capacity_invalid": hrsg_inv,
                "gt_min_enforced": gt_min,
                "heat_reliability": heat_rel,
                "cool_reliability": cool_rel,
                "gt_starts": df["gt_started"].sum(),
                "abs_temp_low_pct": df["abs_temp_low"].mean() * 100,
                "tes_temp_p50_c": df["t_tes_hot_k"].quantile(0.5) - 273.15,
            }
        )

    print("\n=== KPI snapshot ===")
    for r in rows:
        print(
            f"seed={r['seed']:4d} | total_cost={r['total_cost']:>13,.0f} | "
            f"heat_rel={r['heat_reliability']:.3f} | cool_rel={r['cool_reliability']:.3f} | "
            f"abs_low={r['abs_temp_low_pct']:.1f}% | hrsg_inv={r['hrsg_capacity_invalid']}"
        )

    cost_std = np.std([r["total_cost"] for r in rows])
    print(f"\nStdDev(total_cost)={cost_std:.1f}")


if __name__ == "__main__":
    main()
