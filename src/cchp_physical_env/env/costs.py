# Ref: docs/spec/task.md
"""
成本计算模块：将物理状态转换为成本拆分和 reward。

成本项定义：
- cost_grid_import: 电网购电成本（正）
- cost_grid_export_revenue: 电网外送收益（正，会从总成本中扣除）
- cost_grid_curtail: 弃电惩罚（正）
- cost_grid_export_penalty: 外送电额外惩罚（正）
- cost_grid: 电网相关净成本 = import - export_revenue + curtail + export_penalty
- cost_gt_fuel: 燃气轮机燃料成本（正）
- cost_gt_om: 燃气轮机运维成本（正）
- cost_carbon: 碳税成本（正）
- cost_bes_degr: 储电池降解成本（正）
- cost_boiler: 备用锅炉燃料成本（正）
- cost_unmet_e/h/c: 未满足负荷惩罚（正）
- cost_viol: 约束违反惩罚（正）

关键点：
- reward = -cost_total（取负值，RL 最大化 reward 等价于最小化成本）
- 外送电价格为 sell_price_ratio * price_e，且有 cap 和额外惩罚
- 外送电收益会从总成本中扣除，因此可能降低总成本
- 弃电和外送电都有惩罚，避免策略滥用外送

常见坑：
- cost_grid 可能为负（外送收益大于购电成本 + 惩罚）
- 不要直接用 cost_grid 作为 reward，要用 cost_total
- violation_count 来自诊断标志，需确保物理层正确设置
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .cchp_env import EnvConfig


@dataclass(slots=True)
class CostBreakdown:
    """
    成本拆分结果，包含所有成本项和 reward。

    所有成本项均为正值（表示成本），reward 为负值（取负成本）。
    外送电收益会从 cost_grid 中扣除，因此 cost_grid 可能为负。
    """

    cost_grid_import: float
    cost_grid_export_revenue: float
    cost_grid_curtail: float
    cost_grid_export_penalty: float
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
    cost_invalid_abs_request: float
    cost_gt_toggle: float
    cost_gt_delta: float
    cost_idle_heat_backup: float
    cost_idle_cool_backup: float
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
    invalid_abs_request_penalty: float,
    gt_toggle_penalty: float,
    gt_delta_penalty: float,
    idle_heat_backup_penalty: float,
    idle_cool_backup_penalty: float,
    config: "EnvConfig",
) -> CostBreakdown:
    """
    计算成本拆分和 reward。

    参数：
    - dt_h: 时间步长（小时）
    - price_e: 电价（元/MWh）
    - price_gas: 气价（元/MWh）
    - carbon_tax: 碳税（元/吨）
    - grid_import_mw: 电网购电功率（MW）
    - grid_export_mw: 电网外送功率（MW）
    - p_curtail_mw: 弃电功率（MW）
    - fuel_input_gt_effective_mw: GT 燃料输入（MW，考虑启动修正）
    - p_gt_mw: GT 出力（MW）
    - gt_started: GT 是否启动（0/1）
    - bes_degradation_cost: 储电池降解成本
    - boiler_fuel_input_mw: 锅炉燃料输入（MW）
    - p_unmet_e_mw: 未满足电负荷（MW）
    - qh_unmet_mw: 未满足热负荷（MW）
    - qc_unmet_mw: 未满足冷负荷（MW）
    - violation_count: 约束违反计数
    - config: 环境配置（包含惩罚系数等）

    返回：CostBreakdown，包含所有成本项和 reward
    """
    sell_price = float(price_e) * float(config.sell_price_ratio)
    if float(config.sell_price_cap_per_mwh) > 0.0:
        sell_price = min(sell_price, float(config.sell_price_cap_per_mwh))

    cost_grid_import = grid_import_mw * dt_h * price_e
    cost_grid_export_revenue = grid_export_mw * dt_h * sell_price
    cost_grid_curtail = p_curtail_mw * dt_h * config.penalty_curtail_per_mwh

    export_over_soft_cap_mw = max(0.0, grid_export_mw - float(config.grid_export_soft_cap_mw))
    cost_grid_export_penalty = (
        grid_export_mw * dt_h * float(config.penalty_export_per_mwh)
        + export_over_soft_cap_mw * dt_h * float(config.penalty_export_over_soft_cap_per_mwh)
    )

    cost_grid = (
        cost_grid_import
        - cost_grid_export_revenue
        + cost_grid_curtail
        + cost_grid_export_penalty
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
    cost_invalid_abs_request = max(0.0, float(invalid_abs_request_penalty))
    cost_gt_toggle = max(0.0, float(gt_toggle_penalty))
    cost_gt_delta = max(0.0, float(gt_delta_penalty))
    cost_idle_heat_backup = max(0.0, float(idle_heat_backup_penalty))
    cost_idle_cool_backup = max(0.0, float(idle_cool_backup_penalty))

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
        + cost_invalid_abs_request
        + cost_gt_toggle
        + cost_gt_delta
        + cost_idle_heat_backup
        + cost_idle_cool_backup
    )
    return CostBreakdown(
        cost_grid_import=cost_grid_import,
        cost_grid_export_revenue=cost_grid_export_revenue,
        cost_grid_curtail=cost_grid_curtail,
        cost_grid_export_penalty=cost_grid_export_penalty,
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
        cost_invalid_abs_request=cost_invalid_abs_request,
        cost_gt_toggle=cost_gt_toggle,
        cost_gt_delta=cost_gt_delta,
        cost_idle_heat_backup=cost_idle_heat_backup,
        cost_idle_cool_backup=cost_idle_cool_backup,
        cost_total=cost_total,
        reward=-cost_total,
    )
