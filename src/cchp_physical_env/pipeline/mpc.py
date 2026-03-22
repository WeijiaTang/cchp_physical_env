from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..env.cchp_env import CCHPPhysicalEnv, EnvConfig

try:  # pragma: no cover - scipy 是可选依赖
    from scipy import optimize, sparse
except ModuleNotFoundError:  # pragma: no cover
    optimize = None
    sparse = None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class _PlannerState:
    p_gt_prev_mw: float
    gt_prev_on: bool
    gt_on_steps: int
    gt_off_steps: int
    bes_energy_mwh: float
    tes_energy_mwh: float


@dataclass(slots=True)
class _EpisodeCoefficients:
    p_dem_mw: np.ndarray
    qh_dem_mw: np.ndarray
    qc_dem_mw: np.ndarray
    p_re_mw: np.ndarray
    price_e: np.ndarray
    price_gas: np.ndarray
    carbon_tax: np.ndarray
    sell_price: np.ndarray
    ech_power_coeff: np.ndarray
    gt_fuel_coeff: np.ndarray
    gt_heat_coeff: np.ndarray


@dataclass(slots=True)
class _ActionPlan:
    p_gt_mw: float
    p_bes_mw: float
    q_boiler_mw: float
    q_ech_mw: float
    q_tes_mw: float


class BaseMPCPolicy:
    """共享 surrogate 的 MPC 基类。"""

    def __init__(self, *, config: EnvConfig, history_steps: int, seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        requested_horizon = max(1, int(history_steps))
        self.planning_horizon_steps = int(min(requested_horizon, 12))
        self.replan_interval_steps = int(min(4, self.planning_horizon_steps))
        self._env: CCHPPhysicalEnv | None = None
        self._episode_df: pd.DataFrame | None = None
        self._coeffs: _EpisodeCoefficients | None = None
        self._cached_plan_start: int | None = None
        self._cached_actions: list[dict[str, float]] = []
        self._rng = np.random.default_rng(self.seed)

    def bind_episode_context(
        self,
        *,
        env: CCHPPhysicalEnv,
        episode_df: pd.DataFrame,
        initial_observation: dict[str, float],
        seed: int,
    ) -> None:
        del initial_observation
        self._env = env
        self._episode_df = episode_df.reset_index(drop=True).copy()
        self._coeffs = self._build_episode_coefficients(env=env, episode_df=self._episode_df)
        self._cached_plan_start = None
        self._cached_actions = []
        self._rng = np.random.default_rng(self.seed + int(seed))

    def reset_episode(self, observation: dict[str, float]) -> None:
        del observation
        self._cached_plan_start = None
        self._cached_actions = []

    def policy_metadata(self) -> dict[str, Any]:
        return {
            "planner": self.__class__.__name__,
            "forecast_mode": "perfect_episode_slice_surrogate_v1",
            "planning_horizon_steps": int(self.planning_horizon_steps),
            "replan_interval_steps": int(self.replan_interval_steps),
            "abs_used": False,
        }

    def act(self, observation: dict[str, float]) -> dict[str, float]:
        env = self._require_env()
        current_step = int(env.current_step)
        if (
            self._cached_plan_start is not None
            and current_step >= self._cached_plan_start
            and current_step < self._cached_plan_start + len(self._cached_actions)
        ):
            return dict(self._cached_actions[current_step - self._cached_plan_start])

        plan = self._solve_plan(current_step=current_step, observation=observation)
        if not plan:
            return self._fallback_action(observation)

        self._cached_plan_start = current_step
        self._cached_actions = list(plan[: self.replan_interval_steps])
        if not self._cached_actions:
            return self._fallback_action(observation)
        return dict(self._cached_actions[0])

    def _require_env(self) -> CCHPPhysicalEnv:
        if self._env is None or self._episode_df is None or self._coeffs is None:
            raise RuntimeError("MPC policy 尚未绑定 episode context。")
        return self._env

    def _snapshot_state(self) -> _PlannerState:
        env = self._require_env()
        return _PlannerState(
            p_gt_prev_mw=float(env.gt_prev_p_mw),
            gt_prev_on=bool(env.gt_prev_on),
            gt_on_steps=int(env.gt_on_steps),
            gt_off_steps=int(env.gt_off_steps),
            bes_energy_mwh=float(env.bes_soc) * float(self.config.e_bes_cap_mwh),
            tes_energy_mwh=float(env.thermal_storage.energy_mwh),
        )

    def _build_episode_coefficients(
        self,
        *,
        env: CCHPPhysicalEnv,
        episode_df: pd.DataFrame,
    ) -> _EpisodeCoefficients:
        p_cap = max(1e-6, float(self.config.p_gt_cap_mw))
        price_e = episode_df["price_e"].to_numpy(dtype=float)
        sell_price = price_e * float(self.config.sell_price_ratio)
        if float(self.config.sell_price_cap_per_mwh) > 0.0:
            sell_price = np.minimum(sell_price, float(self.config.sell_price_cap_per_mwh))

        t_amb_array = episode_df["t_amb_k"].to_numpy(dtype=float)
        ech_power_coeff = np.zeros_like(t_amb_array, dtype=float)
        gt_fuel_coeff = np.zeros_like(t_amb_array, dtype=float)
        gt_heat_coeff = np.zeros_like(t_amb_array, dtype=float)
        for idx, t_amb_k in enumerate(t_amb_array):
            cop = max(1e-6, float(env.electric_chiller.estimate_cop(t_amb_k=float(t_amb_k))))
            ech_power_coeff[idx] = 1.0 / cop
            gt_result = env.gt_network.solve_offdesign(
                p_gt_request_mw=float(self.config.p_gt_cap_mw),
                t_amb_k=float(t_amb_k),
            )
            hrsg_result = env.hrsg_network.solve(
                m_exh_kg_per_s=float(gt_result.m_exh_kg_per_s),
                t_exh_in_k=float(gt_result.t_exh_k),
            )
            gt_fuel_coeff[idx] = float(gt_result.fuel_input_mw) / p_cap
            gt_heat_coeff[idx] = float(hrsg_result.q_rec_mw) / p_cap

        return _EpisodeCoefficients(
            p_dem_mw=episode_df["p_dem_mw"].to_numpy(dtype=float),
            qh_dem_mw=episode_df["qh_dem_mw"].to_numpy(dtype=float),
            qc_dem_mw=episode_df["qc_dem_mw"].to_numpy(dtype=float),
            p_re_mw=episode_df["pv_mw"].to_numpy(dtype=float) + episode_df["wt_mw"].to_numpy(dtype=float),
            price_e=price_e,
            price_gas=episode_df["price_gas"].to_numpy(dtype=float),
            carbon_tax=episode_df["carbon_tax"].to_numpy(dtype=float),
            sell_price=sell_price,
            ech_power_coeff=ech_power_coeff,
            gt_fuel_coeff=gt_fuel_coeff,
            gt_heat_coeff=gt_heat_coeff,
        )

    def _fallback_action(self, observation: dict[str, float]) -> dict[str, float]:
        p_dem = float(observation["p_dem_mw"])
        p_re = float(observation["pv_mw"]) + float(observation["wt_mw"])
        qh_dem = float(observation["qh_dem_mw"])
        qc_dem = float(observation["qc_dem_mw"])
        soc_bes = float(observation["soc_bes"])
        price_e = float(observation["price_e"])
        net_load = max(0.0, p_dem - p_re)

        if net_load <= 0.50 * float(self.config.p_gt_cap_mw):
            u_gt = -1.0
        else:
            gt_ratio = min(0.60, net_load / max(1e-6, float(self.config.p_gt_cap_mw)))
            u_gt = gt_ratio * 2.0 - 1.0

        if price_e >= 1200.0 and soc_bes > 0.35:
            u_bes = 0.3
        elif price_e <= 600.0 and soc_bes < 0.75:
            u_bes = -0.3
        else:
            u_bes = 0.0

        return {
            "u_gt": float(_clip(u_gt, -1.0, 1.0)),
            "u_bes": float(_clip(u_bes, -1.0, 1.0)),
            "u_boiler": float(
                _clip(qh_dem / max(1e-6, float(self.config.q_boiler_cap_mw)), 0.0, 1.0)
            ),
            "u_abs": 0.0,
            "u_ech": float(
                _clip(qc_dem / max(1e-6, float(self.config.q_ech_cap_mw)), 0.0, 1.0)
            ),
            "u_tes": 0.0,
        }

    def _horizon_length(self, current_step: int) -> int:
        episode_df = self._episode_df
        if episode_df is None:
            return 0
        return max(0, min(self.planning_horizon_steps, len(episode_df) - int(current_step)))

    def _action_to_env_dict(self, action: _ActionPlan) -> dict[str, float]:
        p_gt_cap = max(1e-6, float(self.config.p_gt_cap_mw))
        p_bes_cap = max(1e-6, float(self.config.p_bes_cap_mw))
        q_boiler_cap = max(1e-6, float(self.config.q_boiler_cap_mw))
        q_ech_cap = max(1e-6, float(self.config.q_ech_cap_mw))
        q_tes_charge_cap = max(1e-6, float(self.config.q_tes_charge_cap_mw))
        q_tes_discharge_cap = max(1e-6, float(self.config.q_tes_discharge_cap_mw))

        p_gt_mw = float(_clip(action.p_gt_mw, 0.0, float(self.config.p_gt_cap_mw)))
        u_gt = -1.0 if p_gt_mw <= 1e-9 else (2.0 * p_gt_mw / p_gt_cap) - 1.0
        p_bes_mw = float(
            _clip(action.p_bes_mw, -float(self.config.p_bes_cap_mw), float(self.config.p_bes_cap_mw))
        )
        q_boiler_mw = float(_clip(action.q_boiler_mw, 0.0, float(self.config.q_boiler_cap_mw)))
        q_ech_mw = float(_clip(action.q_ech_mw, 0.0, float(self.config.q_ech_cap_mw)))
        q_tes_mw = float(
            _clip(action.q_tes_mw, -float(self.config.q_tes_charge_cap_mw), float(self.config.q_tes_discharge_cap_mw))
        )
        if q_tes_mw >= 0.0:
            u_tes = q_tes_mw / q_tes_discharge_cap
        else:
            u_tes = q_tes_mw / q_tes_charge_cap

        return {
            "u_gt": float(_clip(u_gt, -1.0, 1.0)),
            "u_bes": float(_clip(p_bes_mw / p_bes_cap, -1.0, 1.0)),
            "u_boiler": float(_clip(q_boiler_mw / q_boiler_cap, 0.0, 1.0)),
            "u_abs": 0.0,
            "u_ech": float(_clip(q_ech_mw / q_ech_cap, 0.0, 1.0)),
            "u_tes": float(_clip(u_tes, -1.0, 1.0)),
        }

    def _simulate_sequence(
        self,
        *,
        state: _PlannerState,
        start_idx: int,
        plan: np.ndarray,
    ) -> tuple[float, list[_ActionPlan]]:
        coeffs = self._coeffs
        if coeffs is None:
            raise RuntimeError("episode coeffs 未初始化。")

        dt_h = float(self.config.dt_hours)
        bes_eta_charge = float(self.config.bes_eta_charge)
        bes_eta_discharge = float(self.config.bes_eta_discharge)
        bes_e_min = float(self.config.bes_soc_min) * float(self.config.e_bes_cap_mwh)
        bes_e_max = float(self.config.bes_soc_max) * float(self.config.e_bes_cap_mwh)
        tes_e_min = 0.0
        tes_e_max = float(self.config.e_tes_cap_mwh)
        tes_loss = max(0.0, 1.0 - float(self.config.sigma_per_hour) * dt_h)
        p_gt_cap = float(self.config.p_gt_cap_mw)
        gt_min = float(self.config.gt_min_output_mw)
        p_bes_cap = float(self.config.p_bes_cap_mw)
        q_boiler_cap = float(self.config.q_boiler_cap_mw)
        q_ech_cap = float(self.config.q_ech_cap_mw)
        q_tes_charge_cap = float(self.config.q_tes_charge_cap_mw)
        q_tes_discharge_cap = float(self.config.q_tes_discharge_cap_mw)
        ramp = max(0.0, float(self.config.gt_ramp_mw_per_step))
        min_on = max(0, int(round(float(self.config.gt_min_on_steps))))
        min_off = max(0, int(round(float(self.config.gt_min_off_steps))))

        p_gt_prev = float(state.p_gt_prev_mw)
        gt_prev_on = bool(state.gt_prev_on)
        gt_on_steps = int(state.gt_on_steps)
        gt_off_steps = int(state.gt_off_steps)
        bes_e = float(state.bes_energy_mwh)
        tes_e = float(state.tes_energy_mwh)
        total_cost = 0.0
        realized_actions: list[_ActionPlan] = []
        boiler_eff = max(1e-6, float(self._require_env().boiler.efficiency))

        for offset in range(int(plan.shape[0])):
            idx = int(start_idx + offset)
            p_gt_req = float(_clip(plan[offset, 0], 0.0, p_gt_cap))
            if p_gt_req > 1e-9:
                p_gt_req = max(gt_min, p_gt_req)
            p_gt_req = float(_clip(p_gt_req, max(0.0, p_gt_prev - ramp), min(p_gt_cap, p_gt_prev + ramp)))
            requested_on = p_gt_req > 1e-9

            if gt_prev_on and (not requested_on) and gt_on_steps < min_on:
                p_gt_req = max(gt_min, p_gt_prev)
                requested_on = True
            if (not gt_prev_on) and requested_on and gt_off_steps < min_off:
                p_gt_req = 0.0
                requested_on = False
            if requested_on and 0.0 < p_gt_req < gt_min:
                p_gt_req = gt_min

            gt_started = int(requested_on and not gt_prev_on)
            gt_toggled = int(requested_on != gt_prev_on)
            gt_delta = abs(p_gt_req - p_gt_prev)

            p_bes_signed = float(_clip(plan[offset, 1], -p_bes_cap, p_bes_cap))
            p_bes_charge = max(0.0, -p_bes_signed)
            p_bes_discharge = max(0.0, p_bes_signed)
            if dt_h > 0.0:
                bes_charge_limit = min(
                    p_bes_cap,
                    max(0.0, (bes_e_max - bes_e) / max(1e-6, bes_eta_charge * dt_h)),
                )
                bes_discharge_limit = min(
                    p_bes_cap,
                    max(0.0, (bes_e - bes_e_min) * bes_eta_discharge / max(1e-6, dt_h)),
                )
            else:
                bes_charge_limit = 0.0
                bes_discharge_limit = 0.0
            p_bes_charge = min(p_bes_charge, bes_charge_limit)
            p_bes_discharge = min(p_bes_discharge, bes_discharge_limit)
            p_bes_mw = p_bes_discharge - p_bes_charge
            bes_e = float(
                _clip(
                    bes_e + (bes_eta_charge * p_bes_charge - p_bes_discharge / max(1e-6, bes_eta_discharge)) * dt_h,
                    bes_e_min,
                    bes_e_max,
                )
            )

            q_boiler_mw = float(_clip(plan[offset, 2], 0.0, q_boiler_cap))
            q_ech_mw = float(_clip(plan[offset, 3], 0.0, q_ech_cap))
            q_tes_signed = float(_clip(plan[offset, 4], -q_tes_charge_cap, q_tes_discharge_cap))
            q_tes_charge = max(0.0, -q_tes_signed)
            q_tes_discharge = max(0.0, q_tes_signed)
            tes_charge_limit = min(q_tes_charge_cap, max(0.0, (tes_e_max - tes_e * tes_loss) / max(1e-6, dt_h)))
            tes_discharge_limit = min(q_tes_discharge_cap, max(0.0, tes_e / max(1e-6, dt_h)))
            q_tes_charge = min(q_tes_charge, tes_charge_limit)
            q_tes_discharge = min(q_tes_discharge, tes_discharge_limit)
            tes_e = float(
                _clip(
                    tes_e * tes_loss + (q_tes_charge - q_tes_discharge) * dt_h,
                    tes_e_min,
                    tes_e_max,
                )
            )
            q_tes_mw = q_tes_discharge - q_tes_charge

            q_hrsg_mw = max(0.0, float(coeffs.gt_heat_coeff[idx]) * p_gt_req)
            qh_need = float(coeffs.qh_dem_mw[idx]) + q_tes_charge
            qh_supply = q_hrsg_mw + q_boiler_mw + q_tes_discharge
            qh_unmet = max(0.0, qh_need - qh_supply)

            qc_need = float(coeffs.qc_dem_mw[idx])
            qc_unmet = max(0.0, qc_need - q_ech_mw)

            p_ech_mw = q_ech_mw * float(coeffs.ech_power_coeff[idx])
            p_local = p_gt_req + float(coeffs.p_re_mw[idx]) + p_bes_mw - p_ech_mw
            import_need = max(0.0, float(coeffs.p_dem_mw[idx]) - p_local)
            export_need = max(0.0, p_local - float(coeffs.p_dem_mw[idx]))
            grid_import = min(import_need, float(self.config.grid_import_cap_mw))
            grid_export = min(export_need, float(self.config.grid_export_cap_mw))
            p_unmet = max(0.0, import_need - grid_import)
            p_curtail = max(0.0, export_need - grid_export)
            export_over_soft = max(0.0, grid_export - float(self.config.grid_export_soft_cap_mw))

            cost_grid = (
                grid_import * dt_h * float(coeffs.price_e[idx])
                - grid_export * dt_h * float(coeffs.sell_price[idx])
                + p_curtail * dt_h * float(self.config.penalty_curtail_per_mwh)
                + grid_export * dt_h * float(self.config.penalty_export_per_mwh)
                + export_over_soft * dt_h * float(self.config.penalty_export_over_soft_cap_per_mwh)
            )
            gt_fuel_input = p_gt_req * float(coeffs.gt_fuel_coeff[idx])
            startup_extra_fuel = gt_fuel_input * float(self.config.gt_startup_fuel_correction_ratio) * gt_started
            boiler_fuel_input = q_boiler_mw / boiler_eff
            cost_gt = (
                (gt_fuel_input + startup_extra_fuel) * dt_h * float(coeffs.price_gas[idx])
                + p_gt_req * dt_h * float(self.config.gt_om_var_cost_per_mwh)
                + gt_started * float(self.config.gt_start_cost)
            )
            cost_carbon = (
                (gt_fuel_input + startup_extra_fuel)
                * dt_h
                * float(self.config.gt_emission_ton_per_mwh_th)
                * float(coeffs.carbon_tax[idx])
                + boiler_fuel_input
                * dt_h
                * float(self.config.boiler_emission_ton_per_mwh_th)
                * float(coeffs.carbon_tax[idx])
            )
            cost_boiler = boiler_fuel_input * dt_h * float(coeffs.price_gas[idx])
            cost_unmet = (
                p_unmet * dt_h * float(self.config.penalty_unmet_e_per_mwh)
                + qh_unmet * dt_h * float(self.config.penalty_unmet_h_per_mwh)
                + qc_unmet * dt_h * float(self.config.penalty_unmet_c_per_mwh)
            )
            cost_bes_degr = (p_bes_charge + p_bes_discharge) * dt_h * 2.0
            cost_toggle = gt_toggled * float(self.config.penalty_gt_toggle)
            cost_delta = gt_delta * float(self.config.penalty_gt_delta_mw)
            idle_heat_backup = (
                qh_unmet > float(self.config.heat_unmet_th_mw)
                and q_boiler_mw < float(self.config.heat_backup_idle_th_mw)
            )
            idle_cool_backup = (
                qc_unmet > float(self.config.cool_unmet_th_mw)
                and q_ech_mw < float(self.config.cool_backup_idle_th_mw)
            )
            total_cost += (
                cost_grid
                + cost_gt
                + cost_carbon
                + cost_boiler
                + cost_unmet
                + cost_bes_degr
                + cost_toggle
                + cost_delta
                + (float(self.config.penalty_idle_heat_backup) if idle_heat_backup else 0.0)
                + (float(self.config.penalty_idle_cool_backup) if idle_cool_backup else 0.0)
            )

            realized_actions.append(
                _ActionPlan(
                    p_gt_mw=float(p_gt_req),
                    p_bes_mw=float(p_bes_mw),
                    q_boiler_mw=float(q_boiler_mw),
                    q_ech_mw=float(q_ech_mw),
                    q_tes_mw=float(q_tes_mw),
                )
            )

            p_gt_prev = float(p_gt_req)
            if requested_on:
                gt_on_steps = gt_on_steps + 1 if gt_prev_on else 1
                gt_off_steps = 0
            else:
                gt_off_steps = gt_off_steps + 1 if not gt_prev_on else 1
                gt_on_steps = 0
            gt_prev_on = requested_on

        return float(total_cost), realized_actions

    def _solve_plan(self, *, current_step: int, observation: dict[str, float]) -> list[dict[str, float]]:
        raise NotImplementedError


@dataclass(slots=True)
class _MILPIndex:
    p_gt: slice
    y_gt: slice
    z_start: slice
    p_bes_charge: slice
    p_bes_discharge: slice
    q_boiler: slice
    q_ech: slice
    q_tes_charge: slice
    q_tes_discharge: slice
    grid_import: slice
    grid_export: slice
    p_curtail: slice
    p_unmet_e: slice
    qh_unmet: slice
    qc_unmet: slice
    export_over_soft: slice
    gt_toggle: slice
    gt_delta: slice
    e_bes: slice
    e_tes: slice
    n_vars: int

    @classmethod
    def build(cls, horizon: int) -> "_MILPIndex":
        cursor = 0

        def _next() -> slice:
            nonlocal cursor
            start = cursor
            cursor += int(horizon)
            return slice(start, cursor)

        return cls(
            p_gt=_next(),
            y_gt=_next(),
            z_start=_next(),
            p_bes_charge=_next(),
            p_bes_discharge=_next(),
            q_boiler=_next(),
            q_ech=_next(),
            q_tes_charge=_next(),
            q_tes_discharge=_next(),
            grid_import=_next(),
            grid_export=_next(),
            p_curtail=_next(),
            p_unmet_e=_next(),
            qh_unmet=_next(),
            qc_unmet=_next(),
            export_over_soft=_next(),
            gt_toggle=_next(),
            gt_delta=_next(),
            e_bes=_next(),
            e_tes=_next(),
            n_vars=cursor,
        )


class MILPMPCPolicy(BaseMPCPolicy):
    def __init__(self, *, config: EnvConfig, history_steps: int, seed: int) -> None:
        super().__init__(config=config, history_steps=history_steps, seed=seed)
        self.time_limit_seconds = 0.15
        self.relative_gap = 0.05

    def policy_metadata(self) -> dict[str, Any]:
        payload = super().policy_metadata()
        payload.update(
            {
                "optimizer": "scipy_milp",
                "time_limit_seconds": float(self.time_limit_seconds),
                "mip_relative_gap": float(self.relative_gap),
            }
        )
        return payload

    def _solve_plan(self, *, current_step: int, observation: dict[str, float]) -> list[dict[str, float]]:
        if optimize is None or sparse is None:
            return [self._fallback_action(observation)]

        horizon = self._horizon_length(current_step)
        if horizon <= 0:
            return [self._fallback_action(observation)]

        coeffs = self._coeffs
        if coeffs is None:
            return [self._fallback_action(observation)]

        state = self._snapshot_state()
        idx = _MILPIndex.build(horizon)
        n = idx.n_vars
        c = np.zeros(n, dtype=float)
        lower = np.zeros(n, dtype=float)
        upper = np.full(n, np.inf, dtype=float)
        integrality = np.zeros(n, dtype=int)

        dt_h = float(self.config.dt_hours)
        p_gt_cap = float(self.config.p_gt_cap_mw)
        p_bes_cap = float(self.config.p_bes_cap_mw)
        q_boiler_cap = float(self.config.q_boiler_cap_mw)
        q_ech_cap = float(self.config.q_ech_cap_mw)
        q_tes_charge_cap = float(self.config.q_tes_charge_cap_mw)
        q_tes_discharge_cap = float(self.config.q_tes_discharge_cap_mw)
        bes_e_min = float(self.config.bes_soc_min) * float(self.config.e_bes_cap_mwh)
        bes_e_max = float(self.config.bes_soc_max) * float(self.config.e_bes_cap_mwh)
        tes_e_max = float(self.config.e_tes_cap_mwh)
        tes_loss = max(0.0, 1.0 - float(self.config.sigma_per_hour) * dt_h)
        ramp = max(0.0, float(self.config.gt_ramp_mw_per_step))
        gt_min = float(self.config.gt_min_output_mw)
        boiler_eff = max(1e-6, float(self._require_env().boiler.efficiency))

        def _set_bounds(slice_obj: slice, lb: float, ub: float, *, binary: bool = False) -> None:
            lower[slice_obj] = lb
            upper[slice_obj] = ub
            if binary:
                integrality[slice_obj] = 1

        _set_bounds(idx.p_gt, 0.0, p_gt_cap)
        _set_bounds(idx.y_gt, 0.0, 1.0, binary=True)
        _set_bounds(idx.z_start, 0.0, 1.0, binary=True)
        _set_bounds(idx.p_bes_charge, 0.0, p_bes_cap)
        _set_bounds(idx.p_bes_discharge, 0.0, p_bes_cap)
        _set_bounds(idx.q_boiler, 0.0, q_boiler_cap)
        _set_bounds(idx.q_ech, 0.0, q_ech_cap)
        _set_bounds(idx.q_tes_charge, 0.0, q_tes_charge_cap)
        _set_bounds(idx.q_tes_discharge, 0.0, q_tes_discharge_cap)
        _set_bounds(idx.grid_import, 0.0, float(self.config.grid_import_cap_mw))
        _set_bounds(idx.grid_export, 0.0, float(self.config.grid_export_cap_mw))
        _set_bounds(idx.p_curtail, 0.0, np.inf)
        _set_bounds(idx.p_unmet_e, 0.0, np.inf)
        _set_bounds(idx.qh_unmet, 0.0, np.inf)
        _set_bounds(idx.qc_unmet, 0.0, np.inf)
        _set_bounds(idx.export_over_soft, 0.0, np.inf)
        _set_bounds(idx.gt_toggle, 0.0, 1.0, binary=True)
        _set_bounds(idx.gt_delta, 0.0, np.inf)
        _set_bounds(idx.e_bes, bes_e_min, bes_e_max)
        _set_bounds(idx.e_tes, 0.0, tes_e_max)

        for offset in range(horizon):
            data_idx = int(current_step + offset)
            c[idx.grid_import.start + offset] = dt_h * float(coeffs.price_e[data_idx])
            c[idx.grid_export.start + offset] = (
                dt_h * (float(self.config.penalty_export_per_mwh) - float(coeffs.sell_price[data_idx]))
            )
            c[idx.p_curtail.start + offset] = dt_h * float(self.config.penalty_curtail_per_mwh)
            c[idx.export_over_soft.start + offset] = dt_h * float(self.config.penalty_export_over_soft_cap_per_mwh)
            gt_fuel_coeff = float(coeffs.gt_fuel_coeff[data_idx])
            gas_price = float(coeffs.price_gas[data_idx])
            carbon_tax = float(coeffs.carbon_tax[data_idx])
            c[idx.p_gt.start + offset] = (
                dt_h * gt_fuel_coeff * gas_price
                + dt_h * float(self.config.gt_om_var_cost_per_mwh)
                + dt_h * gt_fuel_coeff * float(self.config.gt_emission_ton_per_mwh_th) * carbon_tax
            )
            c[idx.z_start.start + offset] = (
                float(self.config.gt_start_cost)
                + dt_h
                * gt_fuel_coeff
                * float(self.config.gt_startup_fuel_correction_ratio)
                * gas_price
            )
            c[idx.p_bes_charge.start + offset] = dt_h * 2.0
            c[idx.p_bes_discharge.start + offset] = dt_h * 2.0
            c[idx.q_boiler.start + offset] = (
                dt_h * gas_price / boiler_eff
                + dt_h * float(self.config.boiler_emission_ton_per_mwh_th) * carbon_tax / boiler_eff
            )
            c[idx.q_ech.start + offset] = dt_h * float(coeffs.ech_power_coeff[data_idx]) * float(
                coeffs.price_e[data_idx]
            )
            c[idx.p_unmet_e.start + offset] = dt_h * float(self.config.penalty_unmet_e_per_mwh)
            c[idx.qh_unmet.start + offset] = dt_h * float(self.config.penalty_unmet_h_per_mwh)
            c[idx.qc_unmet.start + offset] = dt_h * float(self.config.penalty_unmet_c_per_mwh)
            c[idx.gt_toggle.start + offset] = float(self.config.penalty_gt_toggle)
            c[idx.gt_delta.start + offset] = float(self.config.penalty_gt_delta_mw)

        row_values: list[float] = []
        row_indices: list[int] = []
        row_indptr = [0]
        lb_rows: list[float] = []
        ub_rows: list[float] = []

        def _append_row(coeff_map: dict[int, float], lb_value: float, ub_value: float) -> None:
            for col, value in coeff_map.items():
                if abs(value) <= 1e-12:
                    continue
                row_indices.append(int(col))
                row_values.append(float(value))
            row_indptr.append(len(row_indices))
            lb_rows.append(float(lb_value))
            ub_rows.append(float(ub_value))

        for offset in range(horizon):
            data_idx = int(current_step + offset)
            p_gt_col = idx.p_gt.start + offset
            y_gt_col = idx.y_gt.start + offset
            z_start_col = idx.z_start.start + offset
            p_bes_ch_col = idx.p_bes_charge.start + offset
            p_bes_dis_col = idx.p_bes_discharge.start + offset
            q_boiler_col = idx.q_boiler.start + offset
            q_ech_col = idx.q_ech.start + offset
            q_tes_ch_col = idx.q_tes_charge.start + offset
            q_tes_dis_col = idx.q_tes_discharge.start + offset
            grid_imp_col = idx.grid_import.start + offset
            grid_exp_col = idx.grid_export.start + offset
            curtail_col = idx.p_curtail.start + offset
            p_unmet_col = idx.p_unmet_e.start + offset
            qh_unmet_col = idx.qh_unmet.start + offset
            qc_unmet_col = idx.qc_unmet.start + offset
            export_over_col = idx.export_over_soft.start + offset
            gt_toggle_col = idx.gt_toggle.start + offset
            gt_delta_col = idx.gt_delta.start + offset
            e_bes_col = idx.e_bes.start + offset
            e_tes_col = idx.e_tes.start + offset

            prev_y_col = idx.y_gt.start + offset - 1 if offset > 0 else None
            prev_p_col = idx.p_gt.start + offset - 1 if offset > 0 else None
            prev_e_bes_col = idx.e_bes.start + offset - 1 if offset > 0 else None
            prev_e_tes_col = idx.e_tes.start + offset - 1 if offset > 0 else None

            _append_row({p_gt_col: 1.0, y_gt_col: -p_gt_cap}, -np.inf, 0.0)
            _append_row({p_gt_col: 1.0, y_gt_col: -gt_min}, 0.0, np.inf)

            startup_rhs = float(state.gt_prev_on) if offset == 0 else 0.0
            startup_coeffs = {z_start_col: 1.0, y_gt_col: -1.0}
            if prev_y_col is None:
                _append_row(startup_coeffs, -startup_rhs, np.inf)
            else:
                startup_coeffs[prev_y_col] = 1.0
                _append_row(startup_coeffs, 0.0, np.inf)

            toggle_rhs = float(state.gt_prev_on) if offset == 0 else 0.0
            toggle_up = {gt_toggle_col: 1.0, y_gt_col: -1.0}
            toggle_down = {gt_toggle_col: 1.0, y_gt_col: 1.0}
            if prev_y_col is None:
                _append_row(toggle_up, -toggle_rhs, np.inf)
                _append_row(toggle_down, toggle_rhs, np.inf)
            else:
                toggle_up[prev_y_col] = 1.0
                toggle_down[prev_y_col] = -1.0
                _append_row(toggle_up, 0.0, np.inf)
                _append_row(toggle_down, 0.0, np.inf)

            delta_rhs = float(state.p_gt_prev_mw) if offset == 0 else 0.0
            delta_up = {gt_delta_col: 1.0, p_gt_col: -1.0}
            delta_down = {gt_delta_col: 1.0, p_gt_col: 1.0}
            if prev_p_col is None:
                _append_row(delta_up, -delta_rhs, np.inf)
                _append_row(delta_down, delta_rhs, np.inf)
                _append_row({p_gt_col: 1.0}, max(0.0, state.p_gt_prev_mw - ramp), min(p_gt_cap, state.p_gt_prev_mw + ramp))
            else:
                delta_up[prev_p_col] = 1.0
                delta_down[prev_p_col] = -1.0
                _append_row(delta_up, 0.0, np.inf)
                _append_row(delta_down, 0.0, np.inf)
                _append_row({p_gt_col: 1.0, prev_p_col: -1.0}, -ramp, ramp)

            bes_coeffs = {
                e_bes_col: 1.0,
                p_bes_ch_col: -dt_h * float(self.config.bes_eta_charge),
                p_bes_dis_col: dt_h / max(1e-6, float(self.config.bes_eta_discharge)),
            }
            if prev_e_bes_col is None:
                _append_row(bes_coeffs, float(state.bes_energy_mwh), float(state.bes_energy_mwh))
            else:
                bes_coeffs[prev_e_bes_col] = -1.0
                _append_row(bes_coeffs, 0.0, 0.0)

            tes_coeffs = {
                e_tes_col: 1.0,
                q_tes_ch_col: -dt_h,
                q_tes_dis_col: dt_h,
            }
            if prev_e_tes_col is None:
                _append_row(tes_coeffs, tes_loss * float(state.tes_energy_mwh), tes_loss * float(state.tes_energy_mwh))
            else:
                tes_coeffs[prev_e_tes_col] = -tes_loss
                _append_row(tes_coeffs, 0.0, 0.0)

            _append_row(
                {
                    p_gt_col: 1.0,
                    p_bes_dis_col: 1.0,
                    p_bes_ch_col: -1.0,
                    q_ech_col: -float(coeffs.ech_power_coeff[data_idx]),
                    grid_imp_col: 1.0,
                    grid_exp_col: -1.0,
                    curtail_col: -1.0,
                    p_unmet_col: 1.0,
                },
                float(coeffs.p_dem_mw[data_idx] - coeffs.p_re_mw[data_idx]),
                float(coeffs.p_dem_mw[data_idx] - coeffs.p_re_mw[data_idx]),
            )
            _append_row(
                {
                    p_gt_col: float(coeffs.gt_heat_coeff[data_idx]),
                    q_boiler_col: 1.0,
                    q_tes_dis_col: 1.0,
                    q_tes_ch_col: -1.0,
                    qh_unmet_col: 1.0,
                },
                float(coeffs.qh_dem_mw[data_idx]),
                np.inf,
            )
            _append_row({q_ech_col: 1.0, qc_unmet_col: 1.0}, float(coeffs.qc_dem_mw[data_idx]), np.inf)
            _append_row({q_ech_col: 1.0}, -np.inf, float(coeffs.qc_dem_mw[data_idx]))
            _append_row(
                {
                    export_over_col: 1.0,
                    grid_exp_col: -1.0,
                },
                -float(self.config.grid_export_soft_cap_mw),
                np.inf,
            )

        remaining_on_lock = (
            max(0, int(round(float(self.config.gt_min_on_steps))) - int(state.gt_on_steps))
            if state.gt_prev_on
            else 0
        )
        remaining_off_lock = (
            max(0, int(round(float(self.config.gt_min_off_steps))) - int(state.gt_off_steps))
            if not state.gt_prev_on
            else 0
        )
        for offset in range(min(horizon, remaining_on_lock)):
            _append_row({idx.y_gt.start + offset: 1.0}, 1.0, 1.0)
        for offset in range(min(horizon, remaining_off_lock)):
            _append_row({idx.y_gt.start + offset: 1.0}, 0.0, 0.0)

        matrix = sparse.csr_matrix((row_values, row_indices, row_indptr), shape=(len(lb_rows), n))
        constraints = optimize.LinearConstraint(matrix, np.asarray(lb_rows), np.asarray(ub_rows))
        bounds = optimize.Bounds(lower, upper)

        try:
            result = optimize.milp(
                c=c,
                integrality=integrality,
                bounds=bounds,
                constraints=constraints,
                options={
                    "time_limit": float(self.time_limit_seconds),
                    "mip_rel_gap": float(self.relative_gap),
                    "presolve": True,
                    "disp": False,
                },
            )
        except Exception:
            return [self._fallback_action(observation)]

        if not getattr(result, "success", False) or result.x is None:
            return [self._fallback_action(observation)]

        x = np.asarray(result.x, dtype=float)
        plan_array = np.column_stack(
            [
                x[idx.p_gt],
                x[idx.p_bes_discharge] - x[idx.p_bes_charge],
                x[idx.q_boiler],
                x[idx.q_ech],
                x[idx.q_tes_discharge] - x[idx.q_tes_charge],
            ]
        )
        _, realized_actions = self._simulate_sequence(state=state, start_idx=current_step, plan=plan_array)
        return [self._action_to_env_dict(action) for action in realized_actions]


class GAMPCPolicy(BaseMPCPolicy):
    def __init__(self, *, config: EnvConfig, history_steps: int, seed: int) -> None:
        super().__init__(config=config, history_steps=history_steps, seed=seed)
        self.population_size = 18
        self.generations = 6
        self.elite_count = 4
        self.mutation_scale = 0.12
        self._previous_plan: np.ndarray | None = None

    def policy_metadata(self) -> dict[str, Any]:
        payload = super().policy_metadata()
        payload.update(
            {
                "optimizer": "real_coded_ga",
                "population_size": int(self.population_size),
                "generations": int(self.generations),
                "elite_count": int(self.elite_count),
                "mutation_scale": float(self.mutation_scale),
            }
        )
        return payload

    def reset_episode(self, observation: dict[str, float]) -> None:
        super().reset_episode(observation)
        self._previous_plan = None

    def _solve_plan(self, *, current_step: int, observation: dict[str, float]) -> list[dict[str, float]]:
        horizon = self._horizon_length(current_step)
        if horizon <= 0:
            return [self._fallback_action(observation)]

        state = self._snapshot_state()
        lower = np.tile(
            np.array(
                [
                    0.0,
                    -float(self.config.p_bes_cap_mw),
                    0.0,
                    0.0,
                    -float(self.config.q_tes_charge_cap_mw),
                ],
                dtype=float,
            ),
            (horizon, 1),
        )
        upper = np.tile(
            np.array(
                [
                    float(self.config.p_gt_cap_mw),
                    float(self.config.p_bes_cap_mw),
                    float(self.config.q_boiler_cap_mw),
                    float(self.config.q_ech_cap_mw),
                    float(self.config.q_tes_discharge_cap_mw),
                ],
                dtype=float,
            ),
            (horizon, 1),
        )

        population = self._initialize_population(
            horizon=horizon,
            lower=lower,
            upper=upper,
            observation=observation,
            current_step=current_step,
        )

        best_plan = population[0].copy()
        best_score = np.inf
        for _ in range(int(self.generations)):
            scores = np.asarray(
                [
                    self._simulate_sequence(state=state, start_idx=current_step, plan=np.asarray(candidate))[0]
                    for candidate in population
                ],
                dtype=float,
            )
            elite_idx = np.argsort(scores)[: int(self.elite_count)]
            elites = [population[int(i)].copy() for i in elite_idx]
            if float(scores[elite_idx[0]]) < float(best_score):
                best_score = float(scores[elite_idx[0]])
                best_plan = population[int(elite_idx[0])].copy()

            next_population = elites.copy()
            while len(next_population) < int(self.population_size):
                parent_a = population[int(self._tournament_select(scores))]
                parent_b = population[int(self._tournament_select(scores))]
                child = 0.5 * (parent_a + parent_b)
                noise = self._rng.normal(loc=0.0, scale=1.0, size=child.shape) * (
                    (upper - lower) * float(self.mutation_scale)
                )
                child = np.clip(child + noise, lower, upper)
                next_population.append(child)
            population = next_population[: int(self.population_size)]

        _, realized_actions = self._simulate_sequence(state=state, start_idx=current_step, plan=best_plan)
        self._previous_plan = best_plan.copy()
        return [self._action_to_env_dict(action) for action in realized_actions]

    def _initialize_population(
        self,
        *,
        horizon: int,
        lower: np.ndarray,
        upper: np.ndarray,
        observation: dict[str, float],
        current_step: int,
    ) -> list[np.ndarray]:
        population: list[np.ndarray] = []
        fallback_action = self._fallback_action(observation)
        heuristic_plan = np.tile(
            np.array(
                [
                    ((float(fallback_action["u_gt"]) + 1.0) * 0.5) * float(self.config.p_gt_cap_mw),
                    float(fallback_action["u_bes"]) * float(self.config.p_bes_cap_mw),
                    float(fallback_action["u_boiler"]) * float(self.config.q_boiler_cap_mw),
                    float(fallback_action["u_ech"]) * float(self.config.q_ech_cap_mw),
                    float(fallback_action["u_tes"])
                    * (
                        float(self.config.q_tes_discharge_cap_mw)
                        if float(fallback_action["u_tes"]) >= 0.0
                        else float(self.config.q_tes_charge_cap_mw)
                    ),
                ],
                dtype=float,
            ),
            (horizon, 1),
        )
        population.append(np.clip(heuristic_plan, lower, upper))

        if self._previous_plan is not None and len(self._previous_plan) > 0:
            shifted = np.vstack([self._previous_plan[1:], self._previous_plan[-1:]])
            if shifted.shape[0] >= horizon:
                population.append(np.clip(shifted[:horizon], lower, upper))

        if optimize is not None and sparse is not None:
            milp_policy = MILPMPCPolicy(config=self.config, history_steps=self.planning_horizon_steps, seed=self.seed)
            milp_policy.bind_episode_context(
                env=self._require_env(),
                episode_df=self._episode_df if self._episode_df is not None else pd.DataFrame(),
                initial_observation=observation,
                seed=self.seed,
            )
            milp_actions = milp_policy._solve_plan(current_step=current_step, observation=observation)
            if milp_actions:
                milp_plan = []
                for action_dict in milp_actions[:horizon]:
                    q_tes = float(action_dict["u_tes"])
                    q_tes_mw = q_tes * (
                        float(self.config.q_tes_discharge_cap_mw)
                        if q_tes >= 0.0
                        else float(self.config.q_tes_charge_cap_mw)
                    )
                    milp_plan.append(
                        [
                            ((float(action_dict["u_gt"]) + 1.0) * 0.5) * float(self.config.p_gt_cap_mw),
                            float(action_dict["u_bes"]) * float(self.config.p_bes_cap_mw),
                            float(action_dict["u_boiler"]) * float(self.config.q_boiler_cap_mw),
                            float(action_dict["u_ech"]) * float(self.config.q_ech_cap_mw),
                            q_tes_mw,
                        ]
                    )
                if milp_plan:
                    population.append(np.clip(np.asarray(milp_plan, dtype=float), lower, upper))

        while len(population) < int(self.population_size):
            candidate = self._rng.uniform(low=lower, high=upper)
            if population:
                anchor = population[len(population) % len(population)]
                candidate = np.clip(0.6 * candidate + 0.4 * anchor, lower, upper)
            population.append(candidate)
        return population[: int(self.population_size)]

    def _tournament_select(self, scores: np.ndarray, k: int = 3) -> int:
        candidates = self._rng.integers(low=0, high=len(scores), size=max(1, int(k)))
        best = int(candidates[0])
        for index in candidates[1:]:
            if float(scores[int(index)]) < float(scores[best]):
                best = int(index)
        return best
