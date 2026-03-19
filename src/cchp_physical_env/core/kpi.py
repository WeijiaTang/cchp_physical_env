# Ref: docs/spec/task.md
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class KPITracker:
    """统一累计成本、可靠性、违约与启停指标。"""

    step_count: int = 0
    total_reward: float = 0.0
    total_cost: float = 0.0
    costs: dict[str, float] = field(default_factory=dict)
    energies_mwh: dict[str, float] = field(default_factory=dict)
    violation_counts: dict[str, int] = field(default_factory=dict)
    violation_step_count: int = 0
    diagnostic_counts: dict[str, int] = field(default_factory=dict)
    diagnostic_step_count: int = 0
    starts: dict[str, int] = field(default_factory=dict)
    emissions_ton: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.step_count = 0
        self.total_reward = 0.0
        self.total_cost = 0.0
        self.costs = {
            "grid": 0.0,
            "grid_import": 0.0,
            "grid_export_revenue": 0.0,
            "grid_curtail": 0.0,
            "grid_export_penalty": 0.0,
            "gt_fuel": 0.0,
            "gt_om": 0.0,
            "carbon": 0.0,
            "bes_degr": 0.0,
            "boiler": 0.0,
            "unmet_e": 0.0,
            "unmet_h": 0.0,
            "unmet_c": 0.0,
            "viol": 0.0,
            "invalid_abs_request": 0.0,
        }
        self.energies_mwh = {
            "demand_e": 0.0,
            "demand_h": 0.0,
            "demand_c": 0.0,
            "unmet_e": 0.0,
            "unmet_h": 0.0,
            "unmet_c": 0.0,
        }
        self.violation_counts = {}
        self.violation_step_count = 0
        self.diagnostic_counts = {}
        self.diagnostic_step_count = 0
        self.starts = {"gt": 0, "boiler": 0, "ech": 0}
        self.emissions_ton = {"gt": 0.0, "boiler": 0.0, "total": 0.0}

    def record(self, reward: float, step_info: dict) -> None:
        self.step_count += 1
        self.total_reward += float(reward)
        self.total_cost += float(step_info["cost_total"])

        for key in self.costs:
            self.costs[key] += float(step_info.get(f"cost_{key}", 0.0))

        for key in self.energies_mwh:
            self.energies_mwh[key] += float(step_info.get(f"energy_{key}_mwh", 0.0))

        any_violation = False
        for key, flag in step_info.get("violation_flags", {}).items():
            if flag:
                self.violation_counts[key] = self.violation_counts.get(key, 0) + 1
                any_violation = True
        if any_violation:
            self.violation_step_count += 1

        any_diagnostic = False
        for key, flag in step_info.get("diagnostic_flags", {}).items():
            if flag:
                self.diagnostic_counts[key] = self.diagnostic_counts.get(key, 0) + 1
                any_diagnostic = True
        if any_diagnostic:
            self.diagnostic_step_count += 1

        for key in self.starts:
            self.starts[key] += int(step_info.get(f"{key}_started", 0))

        self.emissions_ton["gt"] += float(step_info.get("emission_gt_ton", 0.0))
        self.emissions_ton["boiler"] += float(step_info.get("emission_boiler_ton", 0.0))
        self.emissions_ton["total"] += float(step_info.get("emission_total_ton", 0.0))

    def summary(self) -> dict:
        heat_demand = max(1e-9, self.energies_mwh["demand_h"])
        cool_demand = max(1e-9, self.energies_mwh["demand_c"])
        elec_demand = max(1e-9, self.energies_mwh["demand_e"])
        violation_total = int(sum(self.violation_counts.values()))
        violation_rate = float(self.violation_step_count / max(1, self.step_count))
        diagnostic_total = int(sum(self.diagnostic_counts.values()))
        diagnostic_rate = float(self.diagnostic_step_count / max(1, self.step_count))

        return {
            "step_count": int(self.step_count),
            "total_reward": float(self.total_reward),
            "total_cost": float(self.total_cost),
            "cost_breakdown": {key: float(value) for key, value in self.costs.items()},
            "unmet_energy_mwh": {
                "electric": float(self.energies_mwh["unmet_e"]),
                "heat": float(self.energies_mwh["unmet_h"]),
                "cooling": float(self.energies_mwh["unmet_c"]),
            },
            "reliability": {
                "electric": float(max(0.0, 1.0 - self.energies_mwh["unmet_e"] / elec_demand)),
                "heat": float(max(0.0, 1.0 - self.energies_mwh["unmet_h"] / heat_demand)),
                "cooling": float(max(0.0, 1.0 - self.energies_mwh["unmet_c"] / cool_demand)),
            },
            "violation_total": violation_total,
            "violation_step_count": int(self.violation_step_count),
            "violation_rate": violation_rate,
            "violation_counts": {key: int(value) for key, value in self.violation_counts.items()},
            "diagnostic_total": diagnostic_total,
            "diagnostic_step_count": int(self.diagnostic_step_count),
            "diagnostic_rate": diagnostic_rate,
            "diagnostic_counts": {key: int(value) for key, value in self.diagnostic_counts.items()},
            "starts": {key: int(value) for key, value in self.starts.items()},
            "emissions_ton": {key: float(value) for key, value in self.emissions_ton.items()},
        }
