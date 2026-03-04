# Ref: docs/spec/task.md (Task-ID: 010)
# Ref: docs/spec/architecture.md (Pattern: Physics Layer / Pyomo)
from __future__ import annotations

from dataclasses import dataclass

import pyomo.environ as pyo


@dataclass(slots=True)
class BalanceInputs:
    p_dem_mw: float
    p_re_mw: float
    qh_dem_mw: float
    qc_dem_mw: float
    q_hrsg_available_mw: float
    cop_abs_est: float
    cop_electric_est: float


def add_balance_constraints(model: pyo.ConcreteModel, inputs: BalanceInputs, *, enabled: bool) -> None:
    if not enabled:
        return

    cop_e = max(1e-6, inputs.cop_electric_est)
    cop_abs = max(0.0, inputs.cop_abs_est)

    # 电制冷耗电与制冷量线性关联（q = COP * p）
    model.electric_chiller_link = pyo.Constraint(
        expr=model.q_ech_cool_mw == cop_e * model.p_ech_mw
    )

    # 吸收式制冷输出上界（q <= COP * q_drive）
    model.absorption_cooling_limit = pyo.Constraint(
        expr=model.q_abs_cool_mw <= cop_abs * model.q_abs_drive_mw
    )

    # 电平衡：发电 + 储能 + 可再生 + 电网 + 缺供 = 负荷 + 电制冷 + 弃电
    model.electricity_balance = pyo.Constraint(
        expr=(
            model.p_gt_mw
            + model.p_bes_mw
            + inputs.p_re_mw
            + model.p_grid_mw
            + model.p_unmet_e_mw
            == inputs.p_dem_mw + model.p_ech_mw + model.p_curtail_mw
        )
    )

    # 热平衡：HRSG + 锅炉 + TES放热 + 热缺供 = 热负荷 + 吸收式驱动 + TES充热 + 弃热
    model.heat_balance = pyo.Constraint(
        expr=(
            inputs.q_hrsg_available_mw
            + model.q_boiler_mw
            + model.q_tes_discharge_mw
            + model.qh_unmet_mw
            == inputs.qh_dem_mw + model.q_abs_drive_mw + model.q_tes_charge_mw + model.q_heat_dump_mw
        )
    )

    # 冷平衡：吸收式 + 电制冷 + 冷缺供 = 冷负荷
    model.cooling_balance = pyo.Constraint(
        expr=model.q_abs_cool_mw + model.q_ech_cool_mw + model.qc_unmet_mw == inputs.qc_dem_mw
    )

