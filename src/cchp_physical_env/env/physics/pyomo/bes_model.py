# Ref: docs/spec/task.md (Task-ID: 010)
# Ref: docs/spec/architecture.md (Pattern: Physics Layer / Pyomo)
from __future__ import annotations


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def update_bes_soc(
    *,
    p_bes_mw: float,
    current_soc: float,
    dt_hours: float,
    e_bes_cap_mwh: float,
    soc_min: float,
    soc_max: float,
    eta_charge: float,
    eta_discharge: float,
    self_discharge_per_hour: float = 0.0,
    aux_equip_eff: float = 1.0,
) -> tuple[float, bool]:
    self_discharge_per_hour = max(0.0, self_discharge_per_hour)
    aux_equip_eff = max(1e-6, aux_equip_eff)
    current_energy = current_soc * e_bes_cap_mwh

    # 自放电：按每小时比例衰减当前能量。
    current_energy *= max(0.0, 1.0 - self_discharge_per_hour * dt_hours)

    if p_bes_mw >= 0.0:
        delta = -p_bes_mw * dt_hours / max(1e-6, eta_discharge * aux_equip_eff)
    else:
        delta = -p_bes_mw * dt_hours * eta_charge * aux_equip_eff

    current_energy_raw = current_energy + delta
    min_energy = soc_min * e_bes_cap_mwh
    max_energy = soc_max * e_bes_cap_mwh
    current_energy = _clip(current_energy_raw, min_energy, max_energy)
    new_soc = current_energy / max(1e-6, e_bes_cap_mwh)
    clipped = abs(current_energy - current_energy_raw) > 1e-9
    return new_soc, clipped


def compute_bes_degradation_cost(
    *,
    dt_hours: float,
    soc_before: float,
    soc_after: float,
    e_bes_cap_mwh: float,
    dod_battery_capex_per_mwh: float,
    dod_k_p: float,
    dod_n_fail_100: float,
    dod_add_calendar_age: bool,
    dod_battery_life_years: float,
) -> float:
    investment_cost = max(0.0, e_bes_cap_mwh) * max(0.0, dod_battery_capex_per_mwh)
    denominator = max(1e-6, 2.0 * dod_n_fail_100)
    k_p = max(1e-6, dod_k_p)
    numerator = abs((1.0 - soc_after) ** k_p - (1.0 - soc_before) ** k_p)
    degr_cost = investment_cost * numerator / denominator
    if dod_add_calendar_age:
        calendar_floor = investment_cost / max(1e-6, dod_battery_life_years) / 8760.0 * max(0.0, dt_hours)
        degr_cost = max(degr_cost, calendar_floor)
    return max(0.0, degr_cost)
