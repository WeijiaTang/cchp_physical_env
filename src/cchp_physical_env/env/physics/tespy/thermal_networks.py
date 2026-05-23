# Ref: docs/spec/task.md (Task-ID: 010)
# Ref: docs/spec/architecture.md (Pattern: Physics Layer / TESPy)
from __future__ import annotations

from dataclasses import dataclass, field


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class BoilerResult:
    q_heat_mw: float
    fuel_input_mw: float
    violation_flags: dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True)
class ElectricChillerResult:
    q_cool_mw: float
    p_electric_mw: float
    cop_electric: float
    violation_flags: dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True)
class BackupBoiler:
    q_boiler_cap_mw: float = 10.0
    efficiency: float = 0.92

    def solve(self, *, u_boiler: float) -> BoilerResult:
        signal = _clip(u_boiler, 0.0, 1.0)
        q_heat = signal * self.q_boiler_cap_mw
        fuel_input = q_heat / max(1e-6, self.efficiency)
        return BoilerResult(
            q_heat_mw=q_heat,
            fuel_input_mw=fuel_input,
            violation_flags={"boiler_signal_clipped": abs(signal - u_boiler) > 1e-9},
        )


@dataclass(slots=True)
class ElectricChillerNetwork:
    """Electric chiller with condenser-water, chilled-water and PLR correction.

    The model is a dispatch-grade vapor-compression performance curve. It
    preserves fast algebraic annual rollouts while replacing the old dry-bulb
    only COP rule with a transparent lift-based correction:
    condenser-water temperature represents cooling-tower burden, chilled-water
    supply temperature represents evaporator burden, and PLR accounts for
    degradation away from the rated/IPLV region.
    """

    q_ech_cap_mw: float = 6.0
    cop_nominal: float = 5.2
    cop_floor: float = 3.0
    cop_temp_slope_per_k: float = 0.03
    cop_ref_temp_k: float = 303.15
    cop_partload_min_fraction: float = 0.72
    cop_partload_curve_exp: float = 1.15
    condenser_water_approach_k: float = 5.0
    cooling_tower_wetbulb_depression_k: float = 4.0
    cooling_tower_range_design_k: float = 5.0
    cooling_tower_load_exponent: float = 0.60
    chilled_water_supply_k: float = 280.15
    chilled_water_return_k: float = 285.15
    chilled_water_supply_ref_k: float = 280.15

    def estimate_condenser_water_temp(
        self,
        t_amb_k: float,
        *,
        heat_rejection_mw: float | None = None,
    ) -> float:
        wetbulb_k = float(t_amb_k) - max(0.0, self.cooling_tower_wetbulb_depression_k)
        if heat_rejection_mw is None:
            load_ratio = 0.0
        else:
            q_rej_cap = self.q_ech_cap_mw * (1.0 + 1.0 / max(1e-6, self.cop_nominal))
            load_ratio = _clip(float(heat_rejection_mw) / max(1e-6, q_rej_cap), 0.0, 1.25)
        range_lift = max(0.0, self.cooling_tower_range_design_k) * (
            load_ratio ** max(1e-6, self.cooling_tower_load_exponent)
        )
        return wetbulb_k + max(0.0, self.condenser_water_approach_k) + range_lift

    def estimate_cop(
        self,
        t_amb_k: float,
        plr: float | None = None,
        *,
        t_cond_in_k: float | None = None,
        t_chw_supply_k: float | None = None,
    ) -> float:
        cond_temp = (
            self.estimate_condenser_water_temp(t_amb_k)
            if t_cond_in_k is None
            else float(t_cond_in_k)
        )
        chw_supply = self.chilled_water_supply_k if t_chw_supply_k is None else float(t_chw_supply_k)
        lift_delta = (cond_temp - self.cop_ref_temp_k) - (
            chw_supply - self.chilled_water_supply_ref_k
        )
        cop = self.cop_nominal - self.cop_temp_slope_per_k * lift_delta
        cop = _clip(cop, self.cop_floor, self.cop_nominal)
        if plr is None:
            return cop
        partload = _clip(float(plr), 0.0, 1.0)
        factor = self.cop_partload_min_fraction + (1.0 - self.cop_partload_min_fraction) * (
            partload ** self.cop_partload_curve_exp
        )
        return max(self.cop_floor, cop * factor)

    def solve(
        self,
        *,
        u_ech: float,
        t_amb_k: float,
        t_cond_in_k: float | None = None,
        t_chw_supply_k: float | None = None,
    ) -> ElectricChillerResult:
        signal = _clip(u_ech, 0.0, 1.0)
        q_cool = signal * self.q_ech_cap_mw
        cop = self.estimate_cop(
            t_amb_k=t_amb_k,
            plr=signal,
            t_cond_in_k=t_cond_in_k,
            t_chw_supply_k=t_chw_supply_k,
        )
        if t_cond_in_k is None and q_cool > 1e-9:
            p_guess = q_cool / max(1e-6, cop)
            cond_temp = self.estimate_condenser_water_temp(
                t_amb_k=t_amb_k,
                heat_rejection_mw=q_cool + p_guess,
            )
            cop = self.estimate_cop(
                t_amb_k=t_amb_k,
                plr=signal,
                t_cond_in_k=cond_temp,
                t_chw_supply_k=t_chw_supply_k,
            )
        p_electric = q_cool / max(1e-6, cop)
        return ElectricChillerResult(
            q_cool_mw=q_cool,
            p_electric_mw=p_electric,
            cop_electric=cop,
            violation_flags={"ech_signal_clipped": abs(signal - u_ech) > 1e-9},
        )
