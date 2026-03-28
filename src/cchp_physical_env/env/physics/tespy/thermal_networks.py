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
    q_ech_cap_mw: float = 6.0
    cop_nominal: float = 3.5
    cop_floor: float = 2.0
    cop_temp_slope_per_k: float = 0.03
    cop_ref_temp_k: float = 298.15
    cop_partload_min_fraction: float = 0.72
    cop_partload_curve_exp: float = 1.15

    def estimate_cop(self, t_amb_k: float, plr: float | None = None) -> float:
        cop = self.cop_nominal - self.cop_temp_slope_per_k * (t_amb_k - self.cop_ref_temp_k)
        cop = _clip(cop, self.cop_floor, self.cop_nominal)
        if plr is None:
            return cop
        partload = _clip(float(plr), 0.0, 1.0)
        factor = self.cop_partload_min_fraction + (1.0 - self.cop_partload_min_fraction) * (
            partload ** self.cop_partload_curve_exp
        )
        return max(self.cop_floor, cop * factor)

    def solve(self, *, u_ech: float, t_amb_k: float) -> ElectricChillerResult:
        signal = _clip(u_ech, 0.0, 1.0)
        q_cool = signal * self.q_ech_cap_mw
        cop = self.estimate_cop(t_amb_k=t_amb_k, plr=signal)
        p_electric = q_cool / max(1e-6, cop)
        return ElectricChillerResult(
            q_cool_mw=q_cool,
            p_electric_mw=p_electric,
            cop_electric=cop,
            violation_flags={"ech_signal_clipped": abs(signal - u_ech) > 1e-9},
        )
