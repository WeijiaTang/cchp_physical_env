# Ref: docs/spec/task.md (Task-ID: 010)
# Ref: docs/spec/architecture.md (Pattern: Physics Layer / TESPy)
from __future__ import annotations

from dataclasses import dataclass, field


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class ThermalStorageConfig:
    e_min_mwh: float = 0.0
    e_max_mwh: float = 20.0
    e_init_mwh: float = 10.0
    sigma_per_hour: float = 0.001
    max_charge_mw: float = 8.0
    max_discharge_mw: float = 8.0
    t_return_k: float = 333.15
    t_supply_max_k: float = 393.15


@dataclass(slots=True)
class TESStepResult:
    q_charge_mw: float
    q_discharge_mw: float
    e_tes_mwh: float
    soc: float
    t_hot_k: float
    violation_flags: dict[str, bool] = field(default_factory=dict)


class ThermalStorageState:
    """TES 一阶能量态更新。"""

    def __init__(self, config: ThermalStorageConfig) -> None:
        self.config = config
        self.energy_mwh = _clip(config.e_init_mwh, config.e_min_mwh, config.e_max_mwh)

    def reset(self, energy_mwh: float | None = None) -> None:
        value = self.config.e_init_mwh if energy_mwh is None else energy_mwh
        self.energy_mwh = _clip(value, self.config.e_min_mwh, self.config.e_max_mwh)

    def max_feasible_charge_mw(self, dt_h: float) -> float:
        if dt_h <= 0.0:
            return 0.0
        headroom_mwh = max(0.0, self.config.e_max_mwh - self.energy_mwh)
        return min(self.config.max_charge_mw, headroom_mwh / dt_h)

    def max_feasible_discharge_mw(self, dt_h: float) -> float:
        if dt_h <= 0.0:
            return 0.0
        available_mwh = max(0.0, self.energy_mwh - self.config.e_min_mwh)
        return min(self.config.max_discharge_mw, available_mwh / dt_h)

    def hot_water_temperature_k(self) -> float:
        span = max(1e-6, self.config.e_max_mwh - self.config.e_min_mwh)
        soc = _clip((self.energy_mwh - self.config.e_min_mwh) / span, 0.0, 1.0)
        return self.config.t_return_k + soc * (self.config.t_supply_max_k - self.config.t_return_k)

    def apply(self, *, charge_request_mw: float, discharge_request_mw: float, dt_h: float) -> TESStepResult:
        feasible_charge = self.max_feasible_charge_mw(dt_h=dt_h)
        feasible_discharge = self.max_feasible_discharge_mw(dt_h=dt_h)
        charge_mw = _clip(max(0.0, charge_request_mw), 0.0, feasible_charge)
        discharge_mw = _clip(max(0.0, discharge_request_mw), 0.0, feasible_discharge)

        self.energy_mwh = self.energy_mwh * (1.0 - self.config.sigma_per_hour * dt_h)
        self.energy_mwh = _clip(self.energy_mwh, self.config.e_min_mwh, self.config.e_max_mwh)

        self.energy_mwh = self.energy_mwh + (charge_mw - discharge_mw) * dt_h
        self.energy_mwh = _clip(self.energy_mwh, self.config.e_min_mwh, self.config.e_max_mwh)

        span = max(1e-6, self.config.e_max_mwh - self.config.e_min_mwh)
        soc = _clip((self.energy_mwh - self.config.e_min_mwh) / span, 0.0, 1.0)
        return TESStepResult(
            q_charge_mw=charge_mw,
            q_discharge_mw=discharge_mw,
            e_tes_mwh=self.energy_mwh,
            soc=soc,
            t_hot_k=self.hot_water_temperature_k(),
            violation_flags={
                "tes_charge_clipped": charge_request_mw > charge_mw + 1e-9,
                "tes_discharge_clipped": discharge_request_mw > discharge_mw + 1e-9,
            },
        )

