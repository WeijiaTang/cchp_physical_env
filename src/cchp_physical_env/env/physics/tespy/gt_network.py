# Ref: docs/spec/task.md (Task-ID: 010)
# Ref: docs/spec/architecture.md (Pattern: Physics Layer / TESPy)
from __future__ import annotations

from dataclasses import dataclass, field

from tespy.components import CombustionChamber, Compressor, Sink, Source, Turbine
from tespy.connections import Connection
from tespy.networks import Network

from .solver_cache import SolverCache


GT_MIN_OUTPUT_EPS_MW = 1e-6


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class GTDesignPoint:
    """GT 设计工况与近似参数。"""

    p_gt_cap_mw: float = 12.0
    gt_eta_min: float = 0.26
    gt_eta_max: float = 0.36
    gt_eta_curve_exp: float = 1.0
    gt_eta_frac_25: float = 0.66
    gt_eta_frac_50: float = 0.79
    gt_eta_frac_75: float = 0.91
    gt_eta_frac_100: float = 1.00
    gas_lhv_mj_per_kg: float = 50.0
    gt_min_output_mw: float = 1.0

    m_exh_per_fuel_ratio: float = 65.0
    t_exh_offset_k: float = 220.0
    t_exh_slope_k_per_mw: float = 2.0
    t_exh_load_curve_exp: float = 1.0
    t_exh_min_k: float = 420.0
    t_exh_max_k: float = 850.0
    gt_exh_temp_k_25: float = 673.15
    gt_exh_temp_k_50: float = 733.15
    gt_exh_temp_k_75: float = 773.15
    gt_exh_temp_k_100: float = 793.15
    gt_exh_flow_frac_25: float = 0.42
    gt_exh_flow_frac_50: float = 0.62
    gt_exh_flow_frac_75: float = 0.82
    gt_exh_flow_frac_100: float = 1.00

    t_amb_k: float = 288.15
    p_amb_bar: float = 1.013


@dataclass(slots=True)
class GTResult:
    p_gt_mw: float
    eta_gt: float
    fuel_input_mw: float
    fuel_flow_kg_per_s: float
    t_exh_k: float
    m_exh_kg_per_s: float
    violation_flags: dict[str, bool] = field(default_factory=dict)


class GTNetwork:
    """
    GT 热力学网络接口。

    说明：
    - 提供 tespy 拓扑构建（用于后续严格式离设计点求解扩展）。
    - 当前在线 step 采用稳定的一阶近似（与旧模型可对齐，便于回归）。
    """

    def __init__(
        self,
        design: GTDesignPoint,
        *,
        cache: SolverCache | None = None,
        build_tespy_topology: bool = True,
    ) -> None:
        self.design = design
        self.cache = cache or SolverCache()
        self._network: Network | None = None
        if build_tespy_topology:
            self.build_network()

    @staticmethod
    def _interp_from_quarter_load_map(load_ratio: float, values: tuple[float, float, float, float]) -> float:
        """Piecewise-linear 25/50/75/100% load map interpolation."""

        anchors = (0.25, 0.50, 0.75, 1.00)
        x = _clip(float(load_ratio), 0.0, 1.0)
        if x <= anchors[0]:
            return float(values[0])
        for idx in range(1, len(anchors)):
            left_x = anchors[idx - 1]
            right_x = anchors[idx]
            if x <= right_x:
                weight = (x - left_x) / max(1e-9, right_x - left_x)
                return float(values[idx - 1] + weight * (values[idx] - values[idx - 1]))
        return float(values[-1])

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

        air = Source("air_source")
        fuel = Source("fuel_source")
        compressor = Compressor("compressor")
        combustor = CombustionChamber("combustion_chamber")
        turbine = Turbine("turbine")
        exhaust = Sink("exhaust_sink")

        c_air = Connection(air, "out1", compressor, "in1", label="air_to_compressor")
        c_comp = Connection(compressor, "out1", combustor, "in1", label="compressor_to_cc")
        c_fuel = Connection(fuel, "out1", combustor, "in2", label="fuel_to_cc")
        c_turb = Connection(combustor, "out1", turbine, "in1", label="cc_to_turbine")
        c_exh = Connection(turbine, "out1", exhaust, "in1", label="turbine_to_exhaust")
        nw.add_conns(c_air, c_comp, c_fuel, c_turb, c_exh)

        self._network = nw
        return nw

    def solve_design(self) -> dict[str, float]:
        key = (
            f"gt_design:"
            f"{self.design.p_gt_cap_mw:.6f}:"
            f"{self.design.gt_eta_min:.6f}:"
            f"{self.design.gt_eta_max:.6f}:"
            f"{self.design.t_amb_k:.6f}"
        )

        def _solver() -> dict[str, float]:
            result = self.solve_offdesign(
                p_gt_request_mw=self.design.p_gt_cap_mw,
                t_amb_k=self.design.t_amb_k,
            )
            return {
                "p_gt_mw": result.p_gt_mw,
                "eta_gt": result.eta_gt,
                "fuel_input_mw": result.fuel_input_mw,
                "fuel_flow_kg_per_s": result.fuel_flow_kg_per_s,
                "t_exh_k": result.t_exh_k,
                "m_exh_kg_per_s": result.m_exh_kg_per_s,
            }

        return self.cache.get_or_compute_design(key, _solver)

    def solve_offdesign(self, *, p_gt_request_mw: float, t_amb_k: float) -> GTResult:
        p_gt = _clip(p_gt_request_mw, 0.0, self.design.p_gt_cap_mw)
        if 0.0 < p_gt < self.design.gt_min_output_mw:
            if p_gt >= (self.design.gt_min_output_mw - GT_MIN_OUTPUT_EPS_MW):
                p_gt = self.design.gt_min_output_mw
            else:
                p_gt = 0.0

        if p_gt <= 0.0:
            return GTResult(
                p_gt_mw=0.0,
                eta_gt=0.0,
                fuel_input_mw=0.0,
                fuel_flow_kg_per_s=0.0,
                t_exh_k=_clip(
                    t_amb_k + self.design.t_exh_offset_k,
                    self.design.t_exh_min_k,
                    self.design.t_exh_max_k,
                ),
                m_exh_kg_per_s=0.0,
                violation_flags={},
            )

        load_ratio = _clip(p_gt / max(1e-6, self.design.p_gt_cap_mw), 0.0, 1.0)
        eta_frac_map = (
            self.design.gt_eta_frac_25,
            self.design.gt_eta_frac_50,
            self.design.gt_eta_frac_75,
            self.design.gt_eta_frac_100,
        )
        eta_frac = self._interp_from_quarter_load_map(load_ratio, eta_frac_map)
        # Keep the legacy exponent as a mild calibration knob around the
        # manufacturer-style part-load map rather than reverting to a hidden
        # one-parameter curve.
        eta_frac *= load_ratio ** (max(1e-6, self.design.gt_eta_curve_exp) - 1.0)
        eta_gt = _clip(self.design.gt_eta_max * eta_frac, self.design.gt_eta_min, self.design.gt_eta_max)
        fuel_input_mw = p_gt / max(1e-6, eta_gt)
        fuel_flow_kg_per_s = fuel_input_mw / max(1e-6, self.design.gas_lhv_mj_per_kg)
        full_load_fuel_flow = (
            self.design.p_gt_cap_mw / max(1e-6, self.design.gt_eta_max)
        ) / max(1e-6, self.design.gas_lhv_mj_per_kg)
        full_load_exhaust_flow = full_load_fuel_flow * self.design.m_exh_per_fuel_ratio
        flow_frac = self._interp_from_quarter_load_map(
            load_ratio,
            (
                self.design.gt_exh_flow_frac_25,
                self.design.gt_exh_flow_frac_50,
                self.design.gt_exh_flow_frac_75,
                self.design.gt_exh_flow_frac_100,
            ),
        )
        m_exh_kg_per_s = max(0.0, full_load_exhaust_flow * flow_frac)

        load_shape = load_ratio ** max(1e-6, self.design.t_exh_load_curve_exp)
        map_temp = self._interp_from_quarter_load_map(
            load_ratio,
            (
                self.design.gt_exh_temp_k_25,
                self.design.gt_exh_temp_k_50,
                self.design.gt_exh_temp_k_75,
                self.design.gt_exh_temp_k_100,
            ),
        )
        ambient_correction = float(t_amb_k) - float(self.design.t_amb_k)
        t_raw = map_temp + ambient_correction * 0.35
        # Preserve the legacy slope as a transparent small off-design tuning
        # term. With slope=0, the quarter-load map is used as-is.
        t_raw += self.design.t_exh_slope_k_per_mw * self.design.p_gt_cap_mw * (load_shape - 1.0)
        t_exh_k = _clip(t_raw, self.design.t_exh_min_k, self.design.t_exh_max_k)

        return GTResult(
            p_gt_mw=p_gt,
            eta_gt=eta_gt,
            fuel_input_mw=fuel_input_mw,
            fuel_flow_kg_per_s=fuel_flow_kg_per_s,
            t_exh_k=t_exh_k,
            m_exh_kg_per_s=m_exh_kg_per_s,
            violation_flags={
                "gt_exhaust_temp_clipped_low": t_raw < self.design.t_exh_min_k,
                "gt_exhaust_temp_clipped_high": t_raw > self.design.t_exh_max_k,
            },
        )


def apply_gt_startup_fuel_correction(
    *,
    fuel_input_gt_mw: float,
    gt_started: bool,
    startup_fuel_correction_ratio: float,
) -> tuple[float, float]:
    ratio = max(0.0, startup_fuel_correction_ratio)
    if not gt_started or fuel_input_gt_mw <= 0.0 or ratio <= 0.0:
        return fuel_input_gt_mw, 0.0
    startup_extra_mw = fuel_input_gt_mw * ratio
    effective_fuel_input_gt_mw = fuel_input_gt_mw + startup_extra_mw
    return effective_fuel_input_gt_mw, startup_extra_mw
