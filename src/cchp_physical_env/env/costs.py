# Ref: docs/spec/task.md
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .cchp_env import EnvConfig


@dataclass(slots=True)
class CostBreakdown:
    cost_grid: float
    cost_gt_fuel: float
    cost_gt_om: float
    cost_carbon: float
    cost_bes_degr: float
    cost_boiler: float
    cost_unmet_e: float
    cost_unmet_h: float
    cost_unmet_c: float
    cost_viol: float
    cost_total: float
    reward: float


def compute_cost_breakdown(
    *,
    dt_h: float,
    price_e: float,
    price_gas: float,
    carbon_tax: float,
    grid_import_mw: float,
    grid_export_mw: float,
    p_curtail_mw: float,
    fuel_input_gt_effective_mw: float,
    p_gt_mw: float,
    gt_started: int,
    bes_degradation_cost: float,
    boiler_fuel_input_mw: float,
    p_unmet_e_mw: float,
    qh_unmet_mw: float,
    qc_unmet_mw: float,
    violation_count: int,
    config: "EnvConfig",
) -> CostBreakdown:
    cost_grid = (
        grid_import_mw * dt_h * price_e
        - grid_export_mw * dt_h * price_e * config.sell_price_ratio
        + p_curtail_mw * dt_h * config.penalty_curtail_per_mwh
    )
    cost_gt_fuel = fuel_input_gt_effective_mw * dt_h * price_gas
    cost_gt_om = p_gt_mw * dt_h * config.gt_om_var_cost_per_mwh + gt_started * config.gt_start_cost
    cost_bes_degr = max(0.0, float(bes_degradation_cost))
    cost_boiler = boiler_fuel_input_mw * dt_h * price_gas

    emission_gt_ton = fuel_input_gt_effective_mw * dt_h * config.gt_emission_ton_per_mwh_th
    emission_boiler_ton = boiler_fuel_input_mw * dt_h * config.boiler_emission_ton_per_mwh_th
    cost_carbon = (emission_gt_ton + emission_boiler_ton) * carbon_tax

    cost_unmet_e = p_unmet_e_mw * dt_h * config.penalty_unmet_e_per_mwh
    cost_unmet_h = qh_unmet_mw * dt_h * config.penalty_unmet_h_per_mwh
    cost_unmet_c = qc_unmet_mw * dt_h * config.penalty_unmet_c_per_mwh
    cost_viol = violation_count * config.penalty_violation_per_flag

    cost_total = (
        cost_grid
        + cost_gt_fuel
        + cost_gt_om
        + cost_carbon
        + cost_bes_degr
        + cost_boiler
        + cost_unmet_e
        + cost_unmet_h
        + cost_unmet_c
        + cost_viol
    )
    return CostBreakdown(
        cost_grid=cost_grid,
        cost_gt_fuel=cost_gt_fuel,
        cost_gt_om=cost_gt_om,
        cost_carbon=cost_carbon,
        cost_bes_degr=cost_bes_degr,
        cost_boiler=cost_boiler,
        cost_unmet_e=cost_unmet_e,
        cost_unmet_h=cost_unmet_h,
        cost_unmet_c=cost_unmet_c,
        cost_viol=cost_viol,
        cost_total=cost_total,
        reward=-cost_total,
    )
