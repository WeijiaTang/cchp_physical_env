# Ref: docs/spec/task.md (Task-ID: 010)
# Ref: docs/spec/architecture.md (Pattern: Physics Layer / TESPy)
from __future__ import annotations

from dataclasses import dataclass, field


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class AbsChillerDesignPoint:
    q_drive_cap_mw: float = 5.0
    q_cool_cap_mw: float = 4.5
    t_drive_min_k: float = 358.15
    t_drive_ref_k: float = 378.15
    cop_nominal: float = 0.75
    cop_min_fraction: float = 0.50
    t_cooling_water_ref_k: float = 303.15
    t_evap_ref_k: float = 280.15
    cooling_water_approach_k: float = 5.0
    cooling_tower_wetbulb_depression_k: float = 4.0
    cooling_tower_range_design_k: float = 5.0
    cooling_tower_load_exponent: float = 0.60
    evap_temp_k: float = 280.15
    chilled_water_supply_k: float = 280.15
    chilled_water_return_k: float = 285.15
    cop_cooling_water_slope_per_k: float = 0.012
    cop_evap_slope_per_k: float = 0.010
    cop_partload_min_fraction: float = 0.85
    cop_partload_curve_exp: float = 1.20


@dataclass(slots=True)
class AbsChillerResult:
    q_drive_used_mw: float
    q_cool_mw: float
    cop_abs: float
    violation_flags: dict[str, bool] = field(default_factory=dict)


class AbsChillerNetwork:
    """吸收式制冷机：控制导向的多因素 COP 曲线。

    该模型仍是年度调度用的低阶代理模型，不是吸收器/发生器/冷凝器/
    蒸发器的详细循环求解。但相比单一驱动温度线性 COP，显式纳入了
    冷却水温度、蒸发温度和部分负荷修正，便于论文中透明披露。
    """

    def __init__(self, design: AbsChillerDesignPoint) -> None:
        self.design = design

    def estimate_cop(
        self,
        t_hot_k: float,
        *,
        t_cooling_water_k: float | None = None,
        t_evap_k: float | None = None,
        plr: float | None = None,
    ) -> float:
        if t_hot_k < self.design.t_drive_min_k:
            return 0.0
        if self.design.t_drive_ref_k <= self.design.t_drive_min_k + 1e-9:
            drive_cop = max(0.0, self.design.cop_nominal)
        else:
            scale = (t_hot_k - self.design.t_drive_min_k) / max(
                1e-6, self.design.t_drive_ref_k - self.design.t_drive_min_k
            )
            scale = _clip(scale, 0.0, 1.0)
            cop_floor = max(0.0, self.design.cop_nominal * self.design.cop_min_fraction)
            drive_cop = cop_floor + (self.design.cop_nominal - cop_floor) * scale

        cw_temp = self.design.t_cooling_water_ref_k if t_cooling_water_k is None else float(t_cooling_water_k)
        evap_temp = self.design.t_evap_ref_k if t_evap_k is None else float(t_evap_k)
        temp_factor = 1.0
        temp_factor -= max(0.0, self.design.cop_cooling_water_slope_per_k) * (
            cw_temp - self.design.t_cooling_water_ref_k
        )
        temp_factor += max(0.0, self.design.cop_evap_slope_per_k) * (
            evap_temp - self.design.t_evap_ref_k
        )
        temp_factor = _clip(temp_factor, self.design.cop_min_fraction, 1.15)

        partload_factor = 1.0
        if plr is not None:
            partload = _clip(float(plr), 0.0, 1.0)
            partload_factor = self.design.cop_partload_min_fraction + (
                1.0 - self.design.cop_partload_min_fraction
            ) * (partload ** max(1e-6, self.design.cop_partload_curve_exp))

        cop_floor = max(0.0, self.design.cop_nominal * self.design.cop_min_fraction)
        return _clip(drive_cop * temp_factor * partload_factor, cop_floor, self.design.cop_nominal)

    def estimate_cooling_water_temp(
        self,
        t_amb_k: float,
        *,
        heat_rejection_mw: float | None = None,
    ) -> float:
        """Estimate condenser/cooling-water temperature from a tower surrogate.

        The weather file only exposes dry-bulb temperature, so wet-bulb is
        approximated as dry-bulb minus a configurable depression. The returned
        temperature is the cooling water entering the ABS condenser/absorber.
        A load-dependent tower range term makes high cooling-rejection periods
        less optimistic than the old fixed dry-bulb-plus-approach rule.
        """

        wetbulb_k = float(t_amb_k) - max(0.0, self.design.cooling_tower_wetbulb_depression_k)
        q_rej_cap = max(1e-6, self.design.q_drive_cap_mw + self.design.q_cool_cap_mw)
        if heat_rejection_mw is None:
            load_ratio = 0.0
        else:
            load_ratio = _clip(float(heat_rejection_mw) / q_rej_cap, 0.0, 1.25)
        range_lift = max(0.0, self.design.cooling_tower_range_design_k) * (
            load_ratio ** max(1e-6, self.design.cooling_tower_load_exponent)
        )
        return wetbulb_k + max(0.0, self.design.cooling_water_approach_k) + range_lift

    def solve(
        self,
        *,
        q_drive_request_mw: float,
        t_hot_k: float,
        t_cooling_water_k: float | None = None,
        t_evap_k: float | None = None,
    ) -> AbsChillerResult:
        requested = max(0.0, q_drive_request_mw)
        if requested <= 1e-9:
            return AbsChillerResult(
                q_drive_used_mw=0.0,
                q_cool_mw=0.0,
                cop_abs=max(
                    0.0,
                    self.estimate_cop(
                        t_hot_k=t_hot_k,
                        t_cooling_water_k=t_cooling_water_k,
                        t_evap_k=t_evap_k,
                        plr=0.0,
                    ),
                ),
                violation_flags={
                    "abs_drive_temp_low": False,
                    "abs_drive_clipped": False,
                },
            )
        if t_hot_k < self.design.t_drive_min_k:
            return AbsChillerResult(
                q_drive_used_mw=0.0,
                q_cool_mw=0.0,
                cop_abs=0.0,
                violation_flags={
                    "abs_drive_temp_low": requested > 0.0,
                    "abs_drive_clipped": requested > 0.0,
                },
            )

        q_drive_used = _clip(requested, 0.0, self.design.q_drive_cap_mw)
        plr = q_drive_used / max(1e-6, self.design.q_drive_cap_mw)
        cw_temp = t_cooling_water_k
        cop_abs = self.estimate_cop(
            t_hot_k=t_hot_k,
            t_cooling_water_k=cw_temp,
            t_evap_k=t_evap_k,
            plr=plr,
        )
        if t_cooling_water_k is None:
            q_cool_guess = min(self.design.q_cool_cap_mw, q_drive_used * cop_abs)
            cw_temp = self.estimate_cooling_water_temp(
                t_amb_k=self.design.t_cooling_water_ref_k + self.design.cooling_tower_wetbulb_depression_k,
                heat_rejection_mw=q_drive_used + q_cool_guess,
            )
            cop_abs = self.estimate_cop(
                t_hot_k=t_hot_k,
                t_cooling_water_k=cw_temp,
                t_evap_k=t_evap_k,
                plr=plr,
            )
        q_cool = min(self.design.q_cool_cap_mw, q_drive_used * cop_abs)
        return AbsChillerResult(
            q_drive_used_mw=q_drive_used,
            q_cool_mw=q_cool,
            cop_abs=cop_abs,
            violation_flags={
                "abs_drive_clipped": requested > q_drive_used + 1e-9,
                "abs_cooling_water_hot": (cw_temp is not None)
                and float(cw_temp) > float(self.design.t_cooling_water_ref_k) + 8.0,
            },
        )
