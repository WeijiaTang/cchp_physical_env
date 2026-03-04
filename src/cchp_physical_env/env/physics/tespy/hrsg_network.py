# Ref: docs/spec/task.md (Task-ID: 010)
# Ref: docs/spec/architecture.md (Pattern: Physics Layer / TESPy)
from __future__ import annotations

from dataclasses import dataclass, field
import math

from tespy.components import HeatExchanger, Sink, Source
from tespy.connections import Connection
from tespy.networks import Network


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class HRSGDesignPoint:
    ua_mw_per_k: float = 0.08
    m_water_kg_per_s: float = 45.0
    t_water_in_k: float = 333.15

    cp_exh_kj_per_kgk: float = 1.10
    cp_water_kj_per_kgk: float = 4.18
    t_w_out_max_k: float = 393.15
    t_exh_out_min_k: float = 380.15

    m_exh_ref_kg_per_s: float = 45.0
    k_a_flow_exponent: float = 0.8


@dataclass(slots=True)
class HRSGResult:
    q_rec_mw: float
    t_exh_out_k: float
    t_water_out_k: float
    epsilon: float
    ua_effective_mw_per_k: float
    violation_flags: dict[str, bool] = field(default_factory=dict)


class HRSGNetwork:
    """HRSG 换热器网络封装。"""

    def __init__(self, design: HRSGDesignPoint, *, build_tespy_topology: bool = True) -> None:
        self.design = design
        self._network: Network | None = None
        if build_tespy_topology:
            self.build_network()

    def build_network(self) -> Network:
        if self._network is not None:
            return self._network

        nw = Network(fluids=["Ar", "N2", "O2", "CO2", "H2O", "CH4"])
        nw.units.set_defaults(
            temperature="K",
            pressure="bar",
            enthalpy="kJ / kg",
            mass_flow="kg / s",
        )

        exh_in = Source("exhaust_source")
        water_in = Source("water_source")
        hx = HeatExchanger("hrsg")
        exh_out = Sink("exhaust_sink")
        water_out = Sink("water_sink")

        c_exh_in = Connection(exh_in, "out1", hx, "in1", label="exhaust_in")
        c_exh_out = Connection(hx, "out1", exh_out, "in1", label="exhaust_out")
        c_w_in = Connection(water_in, "out1", hx, "in2", label="water_in")
        c_w_out = Connection(hx, "out2", water_out, "in1", label="water_out")
        nw.add_conns(c_exh_in, c_exh_out, c_w_in, c_w_out)

        self._network = nw
        return nw

    def _effective_ua(self, m_exh_kg_per_s: float) -> float:
        ratio = max(1e-6, m_exh_kg_per_s) / max(1e-6, self.design.m_exh_ref_kg_per_s)
        return self.design.ua_mw_per_k * (ratio ** self.design.k_a_flow_exponent)

    def solve(
        self,
        *,
        m_exh_kg_per_s: float,
        t_exh_in_k: float,
        m_water_kg_per_s: float | None = None,
        t_water_in_k: float | None = None,
    ) -> HRSGResult:
        m_w = self.design.m_water_kg_per_s if m_water_kg_per_s is None else max(0.0, m_water_kg_per_s)
        t_w_in = self.design.t_water_in_k if t_water_in_k is None else t_water_in_k
        ua_effective = self._effective_ua(m_exh_kg_per_s)

        c_exh_mw_per_k = max(0.0, m_exh_kg_per_s * self.design.cp_exh_kj_per_kgk / 1000.0)
        c_w_mw_per_k = max(0.0, m_w * self.design.cp_water_kj_per_kgk / 1000.0)
        c_min = min(c_exh_mw_per_k, c_w_mw_per_k)
        c_max = max(c_exh_mw_per_k, c_w_mw_per_k)

        if c_min <= 0.0 or (t_exh_in_k - t_w_in) <= 0.0:
            return HRSGResult(
                q_rec_mw=0.0,
                t_exh_out_k=t_exh_in_k,
                t_water_out_k=t_w_in,
                epsilon=0.0,
                ua_effective_mw_per_k=ua_effective,
                violation_flags={
                    "hrsg_capacity_invalid": c_min <= 0.0,
                    "hrsg_temperature_lift_invalid": (t_exh_in_k - t_w_in) <= 0.0,
                },
            )

        c_ratio = c_min / c_max if c_max > 0.0 else 0.0
        ntu = ua_effective / max(1e-9, c_min)
        if abs(c_ratio - 1.0) < 1e-8:
            epsilon = ntu / (1.0 + ntu)
        else:
            exp_value = math.exp(-ntu * (1.0 - c_ratio))
            numerator = 1.0 - exp_value
            denominator = 1.0 - c_ratio * exp_value
            epsilon = numerator / denominator if abs(denominator) > 1e-12 else 0.0
        epsilon = _clip(epsilon, 0.0, 1.0)

        q_rec = max(0.0, epsilon * c_min * (t_exh_in_k - t_w_in))
        t_exh_out = t_exh_in_k - (q_rec / c_exh_mw_per_k) if c_exh_mw_per_k > 0.0 else t_exh_in_k
        t_w_out = t_w_in + (q_rec / c_w_mw_per_k) if c_w_mw_per_k > 0.0 else t_w_in

        return HRSGResult(
            q_rec_mw=q_rec,
            t_exh_out_k=t_exh_out,
            t_water_out_k=t_w_out,
            epsilon=epsilon,
            ua_effective_mw_per_k=ua_effective,
            violation_flags={
                "hrsg_water_outlet_overheat": t_w_out > self.design.t_w_out_max_k,
                "hrsg_exhaust_too_cold": t_exh_out < self.design.t_exh_out_min_k,
            },
        )
