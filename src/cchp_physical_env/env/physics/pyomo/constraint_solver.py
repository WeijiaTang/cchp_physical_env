# Ref: docs/spec/task.md (Task-ID: 010)
# Ref: docs/spec/architecture.md (Pattern: Physics Layer / Pyomo)
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pyomo.environ as pyo

from .balance_constraints import BalanceInputs, add_balance_constraints
from .ramp_constraints import add_gt_ramp_constraint


def _clip(value: float, low: float, high: float) -> float:
    """
    将数值裁剪到指定的范围内
    
    该函数将输入值限制在 [low, high] 区间内。如果值小于 low，则返回 low；
    如果值大于 high，则返回 high；否则返回原始值。
    
    Args:
        value: 需要被裁剪的原始数值
        low: 裁剪范围的下界（最小值）
        high: 裁剪范围的上界（最大值）
    
    Returns:
        裁剪后的数值，保证在 [low, high] 范围内
    """
    return max(low, min(high, value))


GT_MIN_OUTPUT_EPS_MW = 1e-6


@dataclass(slots=True)
class ConstraintConfig:
    """
    约束配置类
    
    该类定义了物理约束求解器所需的所有参数，包括燃气轮机、储能、电网、热泵、吸收式制冷机、热储能等设备的容量限制和运行参数。
    
    属性:
        p_gt_cap_mw (float): 燃气轮机最大输出功率（兆瓦），默认值12.0
        gt_min_output_mw (float): 燃气轮机最小输出功率（兆瓦），默认值1.0
        gt_ramp_mw_per_step (float): 燃气轮机每步最大功率变化（兆瓦），默认值2.5
        
        p_bes_cap_mw (float): 储能系统最大功率（兆瓦），默认值4.0
        e_bes_cap_mwh (float): 储能系统最大能量（兆瓦时），默认值8.0
        bes_soc_min (float): 储能系统最小荷电状态，范围[0,1]，默认值0.10
        bes_soc_max (float): 储能系统最大荷电状态，范围[0,1]，默认值0.95
        bes_eta_charge (float): 储能充电效率，范围(0,1]，默认值0.95
        bes_eta_discharge (float): 储能放电效率，范围(0,1]，默认值0.95
        dt_hours (float): 时间步长（小时），默认值0.25
        
        grid_import_cap_mw (float): 电网最大进口功率（兆瓦），默认值30.0
        grid_export_cap_mw (float): 电网最大出口功率（兆瓦），默认值30.0
        
        q_boiler_cap_mw (float): 锅炉最大输出热量（兆瓦），默认值10.0
        q_ech_cap_mw (float): 电动压缩式制冷机最大输出冷量（兆瓦），默认值6.0
        q_abs_drive_cap_mw (float): 吸收式制冷机驱动热量最大输入（兆瓦），默认值5.0
        q_abs_cool_cap_mw (float): 吸收式制冷机最大输出冷量（兆瓦），默认值4.5
        q_tes_charge_cap_mw (float): 热储能充电最大功率（兆瓦），默认值8.0
        q_tes_discharge_cap_mw (float): 热储能放电最大功率（兆瓦），默认值8.0
        
        solver_name (str): 求解器名称，默认值"glpk"
        tracking_weight (float): 跟踪权重，默认值1.0
        unmet_penalty_weight (float): 未满足惩罚权重，默认值1000.0
        curtail_penalty_weight (float): 裁剪惩罚权重，默认值100.0
    """
    p_gt_cap_mw: float = 12.0
    gt_min_output_mw: float = 1.0
    gt_ramp_mw_per_step: float = 2.5

    p_bes_cap_mw: float = 4.0
    e_bes_cap_mwh: float = 8.0
    bes_soc_min: float = 0.10
    bes_soc_max: float = 0.95
    bes_eta_charge: float = 0.95
    bes_eta_discharge: float = 0.95
    dt_hours: float = 0.25

    grid_import_cap_mw: float = 30.0
    grid_export_cap_mw: float = 30.0

    q_boiler_cap_mw: float = 10.0
    q_ech_cap_mw: float = 6.0
    q_abs_drive_cap_mw: float = 5.0
    q_abs_cool_cap_mw: float = 4.5
    q_tes_charge_cap_mw: float = 8.0
    q_tes_discharge_cap_mw: float = 8.0

    solver_name: str = "glpk"
    tracking_weight: float = 1.0
    unmet_penalty_weight: float = 1_000.0
    curtail_penalty_weight: float = 100.0


@dataclass(slots=True)
class ConstraintInputs:
    p_dem_mw: float
    qh_dem_mw: float
    qc_dem_mw: float
    p_re_mw: float
    p_gt_prev_mw: float
    soc_bes: float
    action: Mapping[str, float]
    q_hrsg_available_mw: float
    cop_abs_est: float
    cop_electric_est: float
    tes_charge_feasible_mw: float
    tes_discharge_feasible_mw: float
    is_physics_mode: bool


class ConstraintSolver:
    """电/热/冷平衡 + ramp + 容量约束的一体化动作修正。"""

    def __init__(self, config: ConstraintConfig) -> None:
        self.config = config

    def _build_targets(self, action: Mapping[str, float], *, is_physics_mode: bool) -> dict[str, float | bool]:
        raw_u_gt = float(action.get("u_gt", 0.0))
        raw_u_bes = float(action.get("u_bes", 0.0))
        raw_u_boiler = float(action.get("u_boiler", 0.0))
        raw_u_abs = float(action.get("u_abs", 0.0))
        raw_u_ech = float(action.get("u_ech", 0.0))
        raw_u_tes = float(action.get("u_tes", 0.0))

        u_gt = _clip(raw_u_gt, -1.0, 1.0)
        u_bes = _clip(raw_u_bes, -1.0, 1.0)
        u_boiler = _clip(raw_u_boiler, 0.0, 1.0)
        u_abs = _clip(raw_u_abs, 0.0, 1.0)
        u_ech = _clip(raw_u_ech, 0.0, 1.0)
        u_tes = _clip(raw_u_tes, -1.0, 1.0)

        p_gt_target_raw = ((u_gt + 1.0) * 0.5) * self.config.p_gt_cap_mw
        p_gt_target = p_gt_target_raw
        gt_min_output_enforced = False
        if (
            is_physics_mode
            and p_gt_target > GT_MIN_OUTPUT_EPS_MW
            and p_gt_target < (self.config.gt_min_output_mw - GT_MIN_OUTPUT_EPS_MW)
        ):
            # 对连续控制更友好的最小侵入修正：
            # 一旦判定“开机”，就直接抬升到最小稳定出力，避免在 0 与 min_output 之间出现硬断点。
            p_gt_target = self.config.gt_min_output_mw
            gt_min_output_enforced = True

        return {
            "u_boiler_target": u_boiler,
            "u_abs_target": u_abs,
            "u_ech_target": u_ech,
            "u_tes_target": u_tes,
            "p_gt_target_mw": p_gt_target,
            "p_gt_target_mw_raw": p_gt_target_raw,
            "p_bes_target_mw": u_bes * self.config.p_bes_cap_mw,
            "q_boiler_target_mw": u_boiler * self.config.q_boiler_cap_mw,
            "q_abs_drive_target_mw": u_abs * self.config.q_abs_drive_cap_mw,
            "q_ech_target_mw": u_ech * self.config.q_ech_cap_mw,
            "gt_min_output_enforced": gt_min_output_enforced,
            "raw_u_gt_clipped": abs(raw_u_gt - u_gt) > 1e-9,
            "raw_u_bes_clipped": abs(raw_u_bes - u_bes) > 1e-9,
            "raw_u_boiler_clipped": abs(raw_u_boiler - u_boiler) > 1e-9,
            "raw_u_abs_clipped": abs(raw_u_abs - u_abs) > 1e-9,
            "raw_u_ech_clipped": abs(raw_u_ech - u_ech) > 1e-9,
            "raw_u_tes_clipped": abs(raw_u_tes - u_tes) > 1e-9,
        }

    def _bes_bounds(self, soc_bes: float, *, is_physics_mode: bool) -> tuple[float, float]:
        if not is_physics_mode:
            return -self.config.p_bes_cap_mw, self.config.p_bes_cap_mw

        max_discharge = (
            (soc_bes - self.config.bes_soc_min)
            * self.config.e_bes_cap_mwh
            * self.config.bes_eta_discharge
            / max(1e-6, self.config.dt_hours)
        )
        max_charge = (
            (self.config.bes_soc_max - soc_bes)
            * self.config.e_bes_cap_mwh
            / max(1e-6, self.config.bes_eta_charge * self.config.dt_hours)
        )
        p_bes_min = -max(0.0, min(self.config.p_bes_cap_mw, max_charge))
        p_bes_max = max(0.0, min(self.config.p_bes_cap_mw, max_discharge))
        return p_bes_min, p_bes_max

    def build_model(self, inputs: ConstraintInputs, targets: dict[str, float | bool]) -> pyo.ConcreteModel:
        m = pyo.ConcreteModel()

        p_bes_min, p_bes_max = self._bes_bounds(inputs.soc_bes, is_physics_mode=inputs.is_physics_mode)
        tes_charge_cap = min(self.config.q_tes_charge_cap_mw, max(0.0, inputs.tes_charge_feasible_mw))
        tes_discharge_cap = min(self.config.q_tes_discharge_cap_mw, max(0.0, inputs.tes_discharge_feasible_mw))

        m.p_gt_mw = pyo.Var(bounds=(0.0, self.config.p_gt_cap_mw))
        m.p_bes_mw = pyo.Var(bounds=(p_bes_min, p_bes_max))
        m.p_grid_mw = pyo.Var(bounds=(-self.config.grid_export_cap_mw, self.config.grid_import_cap_mw))

        m.p_ech_mw = pyo.Var(bounds=(0.0, self.config.q_ech_cap_mw / max(1e-6, inputs.cop_electric_est)))
        m.q_boiler_mw = pyo.Var(bounds=(0.0, self.config.q_boiler_cap_mw))
        m.q_abs_drive_mw = pyo.Var(bounds=(0.0, self.config.q_abs_drive_cap_mw))
        m.q_abs_cool_mw = pyo.Var(bounds=(0.0, self.config.q_abs_cool_cap_mw))
        m.q_ech_cool_mw = pyo.Var(bounds=(0.0, self.config.q_ech_cap_mw))
        m.q_tes_charge_mw = pyo.Var(bounds=(0.0, tes_charge_cap))
        m.q_tes_discharge_mw = pyo.Var(bounds=(0.0, tes_discharge_cap))

        # 平衡松弛变量：约束不可行时承接缺供/弃供。
        m.p_unmet_e_mw = pyo.Var(bounds=(0.0, None))
        m.p_curtail_mw = pyo.Var(bounds=(0.0, None))
        m.qh_unmet_mw = pyo.Var(bounds=(0.0, None))
        m.q_heat_dump_mw = pyo.Var(bounds=(0.0, None))
        m.qc_unmet_mw = pyo.Var(bounds=(0.0, None))

        add_gt_ramp_constraint(
            m,
            p_gt_prev_mw=inputs.p_gt_prev_mw,
            gt_ramp_mw_per_step=self.config.gt_ramp_mw_per_step,
            enabled=inputs.is_physics_mode,
        )
        add_balance_constraints(
            m,
            BalanceInputs(
                p_dem_mw=inputs.p_dem_mw,
                p_re_mw=inputs.p_re_mw,
                qh_dem_mw=inputs.qh_dem_mw,
                qc_dem_mw=inputs.qc_dem_mw,
                q_hrsg_available_mw=inputs.q_hrsg_available_mw,
                cop_abs_est=inputs.cop_abs_est,
                cop_electric_est=inputs.cop_electric_est,
            ),
            enabled=inputs.is_physics_mode,
        )

        # L1 动作跟踪（GLPK 可解）+ 缺供/弃供惩罚。
        m.dev_p_gt_pos = pyo.Var(bounds=(0.0, None))
        m.dev_p_gt_neg = pyo.Var(bounds=(0.0, None))
        m.dev_p_bes_pos = pyo.Var(bounds=(0.0, None))
        m.dev_p_bes_neg = pyo.Var(bounds=(0.0, None))
        m.dev_q_boiler_pos = pyo.Var(bounds=(0.0, None))
        m.dev_q_boiler_neg = pyo.Var(bounds=(0.0, None))
        m.dev_q_abs_pos = pyo.Var(bounds=(0.0, None))
        m.dev_q_abs_neg = pyo.Var(bounds=(0.0, None))
        m.dev_q_ech_pos = pyo.Var(bounds=(0.0, None))
        m.dev_q_ech_neg = pyo.Var(bounds=(0.0, None))
        m.dev_q_tes_charge_pos = pyo.Var(bounds=(0.0, None))
        m.dev_q_tes_charge_neg = pyo.Var(bounds=(0.0, None))
        m.dev_q_tes_dis_pos = pyo.Var(bounds=(0.0, None))
        m.dev_q_tes_dis_neg = pyo.Var(bounds=(0.0, None))

        u_tes_target = float(targets["u_tes_target"])
        q_tes_charge_target = max(0.0, -u_tes_target) * tes_charge_cap
        q_tes_discharge_target = max(0.0, u_tes_target) * tes_discharge_cap

        m.track_p_gt = pyo.Constraint(
            expr=m.p_gt_mw - float(targets["p_gt_target_mw"]) == m.dev_p_gt_pos - m.dev_p_gt_neg
        )
        m.track_p_bes = pyo.Constraint(
            expr=m.p_bes_mw - float(targets["p_bes_target_mw"]) == m.dev_p_bes_pos - m.dev_p_bes_neg
        )
        m.track_q_boiler = pyo.Constraint(
            expr=m.q_boiler_mw - float(targets["q_boiler_target_mw"]) == m.dev_q_boiler_pos - m.dev_q_boiler_neg
        )
        m.track_q_abs = pyo.Constraint(
            expr=m.q_abs_drive_mw - float(targets["q_abs_drive_target_mw"]) == m.dev_q_abs_pos - m.dev_q_abs_neg
        )
        m.track_q_ech = pyo.Constraint(
            expr=m.q_ech_cool_mw - float(targets["q_ech_target_mw"]) == m.dev_q_ech_pos - m.dev_q_ech_neg
        )
        m.track_q_tes_charge = pyo.Constraint(
            expr=m.q_tes_charge_mw - q_tes_charge_target == m.dev_q_tes_charge_pos - m.dev_q_tes_charge_neg
        )
        m.track_q_tes_discharge = pyo.Constraint(
            expr=m.q_tes_discharge_mw - q_tes_discharge_target == m.dev_q_tes_dis_pos - m.dev_q_tes_dis_neg
        )

        tracking_expr = (
            m.dev_p_gt_pos
            + m.dev_p_gt_neg
            + m.dev_p_bes_pos
            + m.dev_p_bes_neg
            + m.dev_q_boiler_pos
            + m.dev_q_boiler_neg
            + m.dev_q_abs_pos
            + m.dev_q_abs_neg
            + m.dev_q_ech_pos
            + m.dev_q_ech_neg
            + m.dev_q_tes_charge_pos
            + m.dev_q_tes_charge_neg
            + m.dev_q_tes_dis_pos
            + m.dev_q_tes_dis_neg
        )
        unmet_expr = m.p_unmet_e_mw + m.qh_unmet_mw + m.qc_unmet_mw
        curtail_expr = m.p_curtail_mw + m.q_heat_dump_mw

        m.obj = pyo.Objective(
            expr=(
                self.config.tracking_weight * tracking_expr
                + self.config.unmet_penalty_weight * unmet_expr
                + self.config.curtail_penalty_weight * curtail_expr
            ),
            sense=pyo.minimize,
        )
        return m

    def _model_to_solution(
        self,
        model: pyo.ConcreteModel,
        *,
        targets: dict[str, float | bool],
        inputs: ConstraintInputs,
        solver_used: str,
        used_fallback: bool,
        solver_status: str | None = None,
        solver_termination: str | None = None,
        solver_error: str | None = None,
    ) -> dict[str, float | dict[str, bool] | str | None]:
        p_gt_mw = float(pyo.value(model.p_gt_mw))
        p_bes_mw = float(pyo.value(model.p_bes_mw))
        p_grid_mw = float(pyo.value(model.p_grid_mw))
        p_ech_mw = float(pyo.value(model.p_ech_mw))

        q_boiler_mw = float(pyo.value(model.q_boiler_mw))
        q_abs_drive_mw = float(pyo.value(model.q_abs_drive_mw))
        q_abs_cool_mw = float(pyo.value(model.q_abs_cool_mw))
        q_ech_cool_mw = float(pyo.value(model.q_ech_cool_mw))
        q_tes_charge_mw = float(pyo.value(model.q_tes_charge_mw))
        q_tes_discharge_mw = float(pyo.value(model.q_tes_discharge_mw))

        u_boiler = q_boiler_mw / max(1e-6, self.config.q_boiler_cap_mw)
        u_abs = q_abs_drive_mw / max(1e-6, self.config.q_abs_drive_cap_mw)
        u_ech = q_ech_cool_mw / max(1e-6, self.config.q_ech_cap_mw)
        u_tes = (
            q_tes_discharge_mw / max(1e-6, self.config.q_tes_discharge_cap_mw)
            - q_tes_charge_mw / max(1e-6, self.config.q_tes_charge_cap_mw)
        )

        p_gt_target = float(targets["p_gt_target_mw_raw"])
        p_gt_delta = p_gt_mw - inputs.p_gt_prev_mw
        violation_flags = {
            "safety_gt_clipped": bool(targets["raw_u_gt_clipped"]),
            "safety_bes_clipped": bool(targets["raw_u_bes_clipped"]),
            "safety_boiler_clipped": bool(targets["raw_u_boiler_clipped"]),
            "safety_abs_clipped": bool(targets["raw_u_abs_clipped"]),
            "safety_ech_clipped": bool(targets["raw_u_ech_clipped"]),
            "safety_tes_clipped": bool(targets["raw_u_tes_clipped"]),
            "safety_gt_min_output_enforced": bool(targets["gt_min_output_enforced"]),
            "safety_gt_ramp_limited": inputs.is_physics_mode
            and abs(p_gt_mw - float(targets["p_gt_target_mw"])) > 1e-9,
            "constraint_solver_fallback": used_fallback,
            "constraint_solver_exception": solver_error is not None,
        }

        return {
            "p_gt_mw": p_gt_mw,
            "p_bes_mw": p_bes_mw,
            "p_grid_mw": p_grid_mw,
            "grid_import_mw": max(0.0, p_grid_mw),
            "grid_export_mw": max(0.0, -p_grid_mw),
            "p_ech_mw": p_ech_mw,
            "u_boiler": _clip(u_boiler, 0.0, 1.0),
            "u_abs": _clip(u_abs, 0.0, 1.0),
            "u_ech": _clip(u_ech, 0.0, 1.0),
            "u_tes": _clip(u_tes, -1.0, 1.0),
            "q_boiler_mw": q_boiler_mw,
            "q_abs_drive_mw": q_abs_drive_mw,
            "q_abs_cool_mw": q_abs_cool_mw,
            "q_ech_cool_mw": q_ech_cool_mw,
            "q_tes_charge_mw": q_tes_charge_mw,
            "q_tes_discharge_mw": q_tes_discharge_mw,
            "p_unmet_e_mw": float(pyo.value(model.p_unmet_e_mw)),
            "p_curtail_mw": float(pyo.value(model.p_curtail_mw)),
            "qh_unmet_mw": float(pyo.value(model.qh_unmet_mw)),
            "q_heat_dump_mw": float(pyo.value(model.q_heat_dump_mw)),
            "qc_unmet_mw": float(pyo.value(model.qc_unmet_mw)),
            "p_gt_target_mw": p_gt_target,
            "p_gt_applied_mw": p_gt_mw,
            "p_gt_ramp_delta_mw": p_gt_delta,
            "p_gt_ramp_limit_mw_per_step": self.config.gt_ramp_mw_per_step,
            "solver_used": solver_used,
            "solver_status": solver_status,
            "solver_termination": solver_termination,
            "solver_error": solver_error,
            "violation_flags": violation_flags,
        }

    def _build_projection_model(
        self, inputs: ConstraintInputs, targets: dict[str, float | bool]
    ) -> pyo.ConcreteModel:
        """
        无外部求解器时的线性投影模型（直接按边界与平衡近似修正）。
        """
        m = pyo.ConcreteModel()
        p_bes_min, p_bes_max = self._bes_bounds(inputs.soc_bes, is_physics_mode=inputs.is_physics_mode)
        p_gt_target = _clip(float(targets["p_gt_target_mw"]), 0.0, self.config.p_gt_cap_mw)
        if inputs.is_physics_mode:
            p_gt_target = _clip(
                p_gt_target,
                max(0.0, inputs.p_gt_prev_mw - self.config.gt_ramp_mw_per_step),
                min(self.config.p_gt_cap_mw, inputs.p_gt_prev_mw + self.config.gt_ramp_mw_per_step),
            )
        p_bes_target = _clip(float(targets["p_bes_target_mw"]), p_bes_min, p_bes_max)

        q_boiler = _clip(float(targets["q_boiler_target_mw"]), 0.0, self.config.q_boiler_cap_mw)
        q_abs_drive = _clip(float(targets["q_abs_drive_target_mw"]), 0.0, self.config.q_abs_drive_cap_mw)
        q_abs_cool = min(self.config.q_abs_cool_cap_mw, q_abs_drive * max(0.0, inputs.cop_abs_est))
        q_ech = _clip(float(targets["q_ech_target_mw"]), 0.0, self.config.q_ech_cap_mw)
        p_ech = q_ech / max(1e-6, inputs.cop_electric_est)

        u_tes = float(targets["u_tes_target"])
        q_tes_charge = max(0.0, -u_tes) * min(self.config.q_tes_charge_cap_mw, inputs.tes_charge_feasible_mw)
        q_tes_discharge = max(0.0, u_tes) * min(self.config.q_tes_discharge_cap_mw, inputs.tes_discharge_feasible_mw)

        p_grid_need = (inputs.p_dem_mw + p_ech) - (p_gt_target + p_bes_target + inputs.p_re_mw)
        p_grid = _clip(p_grid_need, -self.config.grid_export_cap_mw, self.config.grid_import_cap_mw)
        p_unmet = max(0.0, p_grid_need - self.config.grid_import_cap_mw)
        p_curtail = max(0.0, -p_grid_need - self.config.grid_export_cap_mw)

        heat_supply = inputs.q_hrsg_available_mw + q_boiler + q_tes_discharge
        heat_use = inputs.qh_dem_mw + q_abs_drive + q_tes_charge
        qh_unmet = max(0.0, heat_use - heat_supply)
        q_heat_dump = max(0.0, heat_supply - heat_use)

        qc_supply = q_abs_cool + q_ech
        qc_unmet = max(0.0, inputs.qc_dem_mw - qc_supply)

        # 组装为“伪模型”变量接口，复用统一提取逻辑。
        m.p_gt_mw = pyo.Param(initialize=p_gt_target, mutable=True)
        m.p_bes_mw = pyo.Param(initialize=p_bes_target, mutable=True)
        m.p_grid_mw = pyo.Param(initialize=p_grid, mutable=True)
        m.p_ech_mw = pyo.Param(initialize=p_ech, mutable=True)
        m.q_boiler_mw = pyo.Param(initialize=q_boiler, mutable=True)
        m.q_abs_drive_mw = pyo.Param(initialize=q_abs_drive, mutable=True)
        m.q_abs_cool_mw = pyo.Param(initialize=q_abs_cool, mutable=True)
        m.q_ech_cool_mw = pyo.Param(initialize=q_ech, mutable=True)
        m.q_tes_charge_mw = pyo.Param(initialize=q_tes_charge, mutable=True)
        m.q_tes_discharge_mw = pyo.Param(initialize=q_tes_discharge, mutable=True)
        m.p_unmet_e_mw = pyo.Param(initialize=p_unmet, mutable=True)
        m.p_curtail_mw = pyo.Param(initialize=p_curtail, mutable=True)
        m.qh_unmet_mw = pyo.Param(initialize=qh_unmet, mutable=True)
        m.q_heat_dump_mw = pyo.Param(initialize=q_heat_dump, mutable=True)
        m.qc_unmet_mw = pyo.Param(initialize=qc_unmet, mutable=True)
        return m

    def solve(self, inputs: ConstraintInputs) -> dict[str, float | dict[str, bool] | str | None]:
        targets = self._build_targets(inputs.action, is_physics_mode=inputs.is_physics_mode)
        solver_name = self.config.solver_name.strip().lower()

        # 说明：reward_only 模式不运行外部求解器，直接使用线性投影（projection）得到可行动作。
        # 这样可保证在“无物理闭环/无求解器依赖”时也能稳定执行训练与评估。
        if not inputs.is_physics_mode:
            pseudo_model = self._build_projection_model(inputs, targets)
            return self._model_to_solution(
                pseudo_model,
                targets=targets,
                inputs=inputs,
                solver_used="reward_only_projection",
                used_fallback=False,
                solver_status="not_run",
                solver_termination="not_run",
                solver_error=None,
            )

        # physics_in_loop 下，如果显式指定 projection/none，则同样跳过外部求解器。
        if solver_name in {"projection", "none"}:
            pseudo_model = self._build_projection_model(inputs, targets)
            return self._model_to_solution(
                pseudo_model,
                targets=targets,
                inputs=inputs,
                solver_used=f"{solver_name}_projection",
                used_fallback=False,
                solver_status="not_run",
                solver_termination="not_run",
                solver_error=None,
            )

        model = self.build_model(inputs=inputs, targets=targets)
        solver = pyo.SolverFactory(self.config.solver_name)
        if not solver.available(False):
            # 求解器不可用时，降级到 projection，保证流程不中断。
            pseudo_model = self._build_projection_model(inputs, targets)
            return self._model_to_solution(
                pseudo_model,
                targets=targets,
                inputs=inputs,
                solver_used=f"{self.config.solver_name}_fallback_projection",
                used_fallback=True,
                solver_status="unavailable",
                solver_termination="not_run",
                solver_error=f"solver_not_available:{self.config.solver_name}",
            )

        status: str | None = None
        termination: str | None = None
        solver_error: str | None = None
        used_fallback = False
        try:
            # 记录 solver 状态用于诊断：失败时仍会 fallback，但保留失败原因以便回溯。
            result = solver.solve(model, tee=False)
            status = str(result.solver.status).lower()
            termination = str(result.solver.termination_condition).lower()
            if ("ok" not in status) or ("optimal" not in termination and "feasible" not in termination):
                used_fallback = True
        except Exception as error:
            used_fallback = True
            status = "exception"
            termination = "exception"
            solver_error = f"{type(error).__name__}: {error}"

        if used_fallback:
            # fallback：外部求解器失败/非最优/异常时，使用 projection 返回“可执行”的近似动作。
            pseudo_model = self._build_projection_model(inputs, targets)
            return self._model_to_solution(
                pseudo_model,
                targets=targets,
                inputs=inputs,
                solver_used=f"{self.config.solver_name}_fallback_projection",
                used_fallback=True,
                solver_status=status,
                solver_termination=termination,
                solver_error=solver_error,
            )

        return self._model_to_solution(
            model,
            targets=targets,
            inputs=inputs,
            solver_used=self.config.solver_name,
            used_fallback=False,
            solver_status=status,
            solver_termination=termination,
            solver_error=None,
        )
