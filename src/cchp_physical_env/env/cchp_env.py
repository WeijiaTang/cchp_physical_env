# Ref: docs/spec/task.md (Task-ID: 010)
# Ref: docs/spec/architecture.md (Pattern: Orchestrator + Physics Layer)
"""
CCHP 物理环境：Gym 风格的多能耦合调度环境。

本环境是"编排层"，只负责：
- 管理 episode 生命周期（reset/step）
- 构建 observation（外生变量 + 内生状态）
- 调用 physics 层求解设备状态
- 调用约束求解器投影动作到可行域
- 计算成本、惩罚、reward
- 追踪 KPI（violation/unmet/成本拆分）

Action 语义：
- u_gt: 燃气轮机出力设定点（-1~1，映射到 0~p_gt_cap_mw）
- u_bes: 储电池功率（-1~1，负为充电，正为放电）
- u_boiler: 备用锅炉出力设定点（0~1）
- u_abs: 吸收式制冷机驱动热设定点（0~1）
- u_ech: 电制冷机出力设定点（0~1）
- u_tes: 蓄热罐充放热设定点（-1~1，负为充热，正为放热）

Observation 语义：
- 外生变量：负荷/新能源/天气/价格（来自 CSV）
- 内生状态：GT/BES/TES 状态、SOC、温度等

约束模式（constraint_mode）：
- physics_in_loop: 动作先投影到可行域，再执行物理仿真
- reward_only: 动作直接执行，约束违反通过惩罚项反馈

常见坑：
- reward = -total_cost（取负值，RL 最大化 reward 等价于最小化成本）
- diagnostic_rate 仅统计控制相关诊断；环境/工况状态单独记入 state_diagnostic_rate
- 外送电价格为 sell_price_ratio * price_e，且有 cap 和额外惩罚
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ..core.data import FROZEN_COLUMNS
from ..core.kpi import KPITracker
from .costs import compute_cost_breakdown
from .observations import build_observation
from .physics.pyomo import (
    ConstraintConfig,
    ConstraintInputs,
    ConstraintSolver,
    compute_bes_degradation_cost,
    update_bes_soc,
)
from .physics.tespy import (
    AbsChillerDesignPoint,
    AbsChillerNetwork,
    BackupBoiler,
    ElectricChillerNetwork,
    GTDesignPoint,
    GTNetwork,
    HRSGDesignPoint,
    HRSGNetwork,
    ThermalStorageConfig,
    ThermalStorageState,
    apply_gt_startup_fuel_correction,
)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


GT_MIN_OUTPUT_EPS_MW = 1e-6


@dataclass(slots=True)
class EnvConfig:
    """
    环境配置：包含所有物理参数、约束参数、惩罚系数。

    配置来源：config/config.yaml（通过 config_loader 构建）
    不要在代码中硬编码这些值。

    关键参数说明：
    - constraint_mode: 约束处理方式（physics_in_loop / reward_only）
    - physics_backend: 物理求解后端（当前仅支持 tespy）
    - p_gt_cap_mw: 燃气轮机额定功率
    - p_bes_cap_mw / e_bes_cap_mwh: 储电池功率/容量
    - grid_export_cap_mw: 外送电上限
    - sell_price_ratio: 外送电价格系数
    - penalty_*: 各类惩罚项系数
    """
    dt_hours: float
    constraint_mode: str
    physics_backend: str

    ua_mw_per_k: float
    sigma_per_hour: float
    cop_nominal: float
    m_exh_per_fuel_ratio: float
    t_exh_offset_k: float
    t_exh_slope_k_per_mw: float

    p_gt_cap_mw: float
    gt_min_output_mw: float
    gt_ramp_mw_per_step: float
    gt_eta_min: float
    gt_eta_max: float
    gas_lhv_mj_per_kg: float
    gt_om_var_cost_per_mwh: float
    gt_start_cost: float
    gt_startup_fuel_correction_ratio: float
    gt_dynamic_om_enabled: bool
    gt_cycle_cost: float
    gt_cycle_hours: float

    p_bes_cap_mw: float
    e_bes_cap_mwh: float
    bes_soc_init: float
    bes_soc_min: float
    bes_soc_max: float
    bes_eta_charge: float
    bes_eta_discharge: float
    bes_self_discharge_per_hour: float
    bes_aux_equip_eff: float
    bes_init_strategy: str
    bes_dod_battery_capex_per_mwh: float
    bes_dod_k_p: float
    bes_dod_n_fail_100: float
    bes_dod_add_calendar_age: bool
    bes_dod_battery_life_years: float

    grid_import_cap_mw: float
    grid_export_cap_mw: float
    sell_price_ratio: float
    sell_price_cap_per_mwh: float
    penalty_curtail_per_mwh: float
    penalty_export_per_mwh: float
    grid_export_soft_cap_mw: float
    penalty_export_over_soft_cap_per_mwh: float

    gt_emission_ton_per_mwh_th: float
    boiler_emission_ton_per_mwh_th: float

    penalty_unmet_e_per_mwh: float
    penalty_unmet_h_per_mwh: float
    penalty_unmet_c_per_mwh: float
    penalty_violation_per_flag: float
    abs_gate_enabled: bool
    abs_gate_scale_k: float
    abs_deadzone_gate_th: float
    abs_deadzone_u_th: float
    abs_invalid_req_u_th: float
    abs_invalid_req_gate_th: float
    penalty_invalid_abs_request: float
    gt_action_smoothing_enabled: bool
    gt_min_on_steps: float
    gt_min_off_steps: float
    penalty_gt_toggle: float
    penalty_gt_delta_mw: float
    heat_unmet_th_mw: float
    cool_unmet_th_mw: float
    heat_backup_idle_th_mw: float
    cool_backup_idle_th_mw: float
    penalty_idle_heat_backup: float
    penalty_idle_cool_backup: float
    heat_backup_shield_enabled: bool
    heat_backup_shield_margin_mw: float

    hrsg_water_mass_flow_kg_per_s: float
    hrsg_water_inlet_k: float

    q_boiler_cap_mw: float
    q_ech_cap_mw: float
    e_tes_cap_mwh: float
    e_tes_init_mwh: float
    q_tes_charge_cap_mw: float
    q_tes_discharge_cap_mw: float
    q_abs_drive_cap_mw: float
    q_abs_cool_cap_mw: float

    oracle_mpc_abs_enabled: bool
    oracle_mpc_hard_reliability: bool
    oracle_mpc_max_unmet_e_mw: float
    oracle_mpc_max_unmet_h_mw: float
    oracle_mpc_max_unmet_c_mw: float
    oracle_mpc_hard_unmet_penalty_per_mwh: float
    oracle_mpc_heat_backup_repair_enabled: bool
    oracle_mpc_tes_terminal_reserve_mwh: float
    oracle_mpc_tes_terminal_reserve_penalty_per_mwh: float
    oracle_mpc_planning_horizon_steps: float
    oracle_mpc_replan_interval_steps: float
    oracle_mpc_time_limit_seconds: float
    oracle_mpc_mip_relative_gap: float
    oracle_ga_population_size: float
    oracle_ga_generations: float
    oracle_ga_elite_count: float
    oracle_ga_mutation_scale: float

    pyomo_solver: str
    pyomo_tracking_weight: float
    pyomo_unmet_penalty_weight: float
    pyomo_curtail_penalty_weight: float


class CCHPPhysicalEnv:
    """
    CCHP Gym 风格环境：只负责编排，物理与约束下沉到 physics 层。

    使用方式：
    1. 从 config.yaml 构建 EnvConfig
    2. 加载外生数据 CSV
    3. 创建环境实例
    4. 调用 reset() 开始 episode
    5. 循环调用 step(action) 直到 done

    Action 格式：dict[str, float]，键为 action_keys
    Observation 格式：dict[str, float]
    Reward：float，等于 -total_cost
    """

    action_keys = ("u_gt", "u_bes", "u_boiler", "u_abs", "u_ech", "u_tes")

    def __init__(self, exogenous_df: pd.DataFrame, config: EnvConfig | None = None, seed: int = 0):
        self._validate_schema(exogenous_df)
        if config is None:
            raise ValueError(
                "EnvConfig 不能为空：当前采用 Option-C 全量 yaml 配置模式，请通过 config_loader 构建 EnvConfig 并传入。"
            )
        self.config = config
        self.rng = np.random.default_rng(seed)

        self.full_df = exogenous_df.reset_index(drop=True).copy()
        self.episode_df = self.full_df
        self.current_step = 0

        self.gt_prev_p_mw = 0.0
        self.gt_prev_on = False
        self.gt_on_steps = 0
        self.gt_off_steps = max(0, int(round(float(self.config.gt_min_off_steps))))
        self.bes_soc = self._init_bes_soc()
        self.boiler_prev_on = False
        self.ech_prev_on = False

        self._init_physics_components()
        self.kpi = KPITracker()
        self.kpi.reset()

    @staticmethod
    def _validate_schema(df: pd.DataFrame) -> None:
        missing = [column for column in FROZEN_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"环境输入缺少冻结列: {missing}")

    def _init_bes_soc(self) -> float:
        strategy = self.config.bes_init_strategy.strip().lower()
        if strategy == "fixed":
            return float(
                _clip(self.config.bes_soc_init, self.config.bes_soc_min, self.config.bes_soc_max)
            )
        if strategy == "min":
            return float(self.config.bes_soc_min)
        if strategy == "max":
            return float(self.config.bes_soc_max)
        if strategy == "half":
            return float(
                _clip(0.5, self.config.bes_soc_min, self.config.bes_soc_max)
            )
        if strategy == "random":
            return float(self.rng.uniform(self.config.bes_soc_min, self.config.bes_soc_max))
        raise ValueError(f"不支持的 bes_init_strategy: {self.config.bes_init_strategy}")

    def _init_physics_components(self) -> None:
        backend = self.config.physics_backend.strip().lower()
        if backend != "tespy":
            raise ValueError(f"不支持的 physics_backend: {self.config.physics_backend}")

        self.gt_network = GTNetwork(
            GTDesignPoint(
                p_gt_cap_mw=self.config.p_gt_cap_mw,
                gt_eta_min=self.config.gt_eta_min,
                gt_eta_max=self.config.gt_eta_max,
                gas_lhv_mj_per_kg=self.config.gas_lhv_mj_per_kg,
                gt_min_output_mw=self.config.gt_min_output_mw,
                m_exh_per_fuel_ratio=self.config.m_exh_per_fuel_ratio,
                t_exh_offset_k=self.config.t_exh_offset_k,
                t_exh_slope_k_per_mw=self.config.t_exh_slope_k_per_mw,
            )
        )
        self.hrsg_network = HRSGNetwork(
            HRSGDesignPoint(
                ua_mw_per_k=self.config.ua_mw_per_k,
                m_water_kg_per_s=self.config.hrsg_water_mass_flow_kg_per_s,
                t_water_in_k=self.config.hrsg_water_inlet_k,
            )
        )
        self.abs_chiller = AbsChillerNetwork(
            AbsChillerDesignPoint(
                q_drive_cap_mw=self.config.q_abs_drive_cap_mw,
                q_cool_cap_mw=self.config.q_abs_cool_cap_mw,
                cop_nominal=self.config.cop_nominal,
            )
        )
        self.thermal_storage = ThermalStorageState(
            ThermalStorageConfig(
                e_max_mwh=self.config.e_tes_cap_mwh,
                e_init_mwh=self.config.e_tes_init_mwh,
                sigma_per_hour=self.config.sigma_per_hour,
                max_charge_mw=self.config.q_tes_charge_cap_mw,
                max_discharge_mw=self.config.q_tes_discharge_cap_mw,
            )
        )
        self.boiler = BackupBoiler(q_boiler_cap_mw=self.config.q_boiler_cap_mw)
        self.electric_chiller = ElectricChillerNetwork(q_ech_cap_mw=self.config.q_ech_cap_mw)
        self.constraint_solver = ConstraintSolver(
            ConstraintConfig(
                p_gt_cap_mw=self.config.p_gt_cap_mw,
                gt_min_output_mw=self.config.gt_min_output_mw,
                gt_ramp_mw_per_step=self.config.gt_ramp_mw_per_step,
                p_bes_cap_mw=self.config.p_bes_cap_mw,
                e_bes_cap_mwh=self.config.e_bes_cap_mwh,
                bes_soc_min=self.config.bes_soc_min,
                bes_soc_max=self.config.bes_soc_max,
                bes_eta_charge=self.config.bes_eta_charge,
                bes_eta_discharge=self.config.bes_eta_discharge,
                dt_hours=self.config.dt_hours,
                grid_import_cap_mw=self.config.grid_import_cap_mw,
                grid_export_cap_mw=self.config.grid_export_cap_mw,
                q_boiler_cap_mw=self.config.q_boiler_cap_mw,
                q_ech_cap_mw=self.config.q_ech_cap_mw,
                q_abs_drive_cap_mw=self.config.q_abs_drive_cap_mw,
                q_abs_cool_cap_mw=self.config.q_abs_cool_cap_mw,
                q_tes_charge_cap_mw=self.config.q_tes_charge_cap_mw,
                q_tes_discharge_cap_mw=self.config.q_tes_discharge_cap_mw,
                solver_name=self.config.pyomo_solver,
                tracking_weight=self.config.pyomo_tracking_weight,
                unmet_penalty_weight=self.config.pyomo_unmet_penalty_weight,
                curtail_penalty_weight=self.config.pyomo_curtail_penalty_weight,
            )
        )

    def set_episode(self, episode_df: pd.DataFrame) -> None:
        self._validate_schema(episode_df)
        self.episode_df = episode_df.reset_index(drop=True).copy()

    def _reset_states(self) -> None:
        self.current_step = 0
        self.gt_prev_p_mw = 0.0
        self.gt_prev_on = False
        self.gt_on_steps = 0
        self.gt_off_steps = max(0, int(round(float(self.config.gt_min_off_steps))))
        self.bes_soc = self._init_bes_soc()
        self.boiler_prev_on = False
        self.ech_prev_on = False
        self.thermal_storage.reset()
        self.kpi.reset()

    def _abs_drive_margin_k(self, *, tes_hot_k: float) -> float:
        return float(tes_hot_k - float(self.abs_chiller.design.t_drive_min_k))

    def _estimate_hrsg_recoverable_heat_mw(self, *, p_gt_mw: float, t_amb_k: float) -> float:
        gt_est = self.gt_network.solve_offdesign(
            p_gt_request_mw=max(0.0, float(p_gt_mw)),
            t_amb_k=float(t_amb_k),
        )
        hrsg_est = self.hrsg_network.solve(
            m_exh_kg_per_s=gt_est.m_exh_kg_per_s,
            t_exh_in_k=gt_est.t_exh_k,
        )
        return float(hrsg_est.q_rec_mw)

    def _build_heat_reliability_observation_features(self, *, row: pd.Series) -> dict[str, float]:
        q_hrsg_est_now_mw = self._estimate_hrsg_recoverable_heat_mw(
            p_gt_mw=float(self.gt_prev_p_mw),
            t_amb_k=float(row["t_amb_k"]),
        )
        q_tes_discharge_feasible_mw = float(
            self.thermal_storage.max_feasible_discharge_mw(self.config.dt_hours)
        )
        heat_deficit_if_boiler_off_mw = max(
            0.0,
            float(row["qh_dem_mw"]) - q_hrsg_est_now_mw - q_tes_discharge_feasible_mw,
        )
        heat_backup_min_needed_mw = min(
            float(self.config.q_boiler_cap_mw),
            heat_deficit_if_boiler_off_mw,
        )
        return {
            "q_hrsg_est_now_mw": float(q_hrsg_est_now_mw),
            "q_tes_discharge_feasible_mw": float(q_tes_discharge_feasible_mw),
            "heat_deficit_if_boiler_off_mw": float(heat_deficit_if_boiler_off_mw),
            "heat_backup_min_needed_mw": float(heat_backup_min_needed_mw),
        }

    def _compute_boiler_heat_lower_bound(
        self,
        *,
        qh_demand_mw: float,
        q_hrsg_available_mw: float,
        tes_discharge_support_mw: float,
    ) -> dict[str, float]:
        heat_deficit_if_boiler_off_mw = max(
            0.0,
            float(qh_demand_mw) - float(q_hrsg_available_mw) - float(tes_discharge_support_mw),
        )
        heat_backup_min_needed_mw = min(
            float(self.config.q_boiler_cap_mw),
            heat_deficit_if_boiler_off_mw + max(0.0, float(self.config.heat_backup_shield_margin_mw)),
        )
        q_boiler_cap = max(1e-6, float(self.config.q_boiler_cap_mw))
        return {
            "heat_deficit_if_boiler_off_mw": float(heat_deficit_if_boiler_off_mw),
            "heat_backup_min_needed_mw": float(heat_backup_min_needed_mw),
            "u_boiler_lower_bound": float(
                _clip(heat_backup_min_needed_mw / q_boiler_cap, 0.0, 1.0)
            ),
        }

    def _gt_ramp_headroom(self) -> tuple[float, float]:
        prev = float(max(0.0, self.gt_prev_p_mw))
        ramp = float(max(0.0, self.config.gt_ramp_mw_per_step))
        cap = float(max(0.0, self.config.p_gt_cap_mw))
        up_limit = min(cap, prev + ramp)
        down_limit = max(0.0, prev - ramp)
        return max(0.0, up_limit - prev), max(0.0, prev - down_limit)

    def _gt_target_mw_to_action(self, p_gt_target_mw: float) -> float:
        cap = max(1e-6, float(self.config.p_gt_cap_mw))
        normalized = 2.0 * (float(p_gt_target_mw) / cap) - 1.0
        return float(_clip(normalized, -1.0, 1.0))

    def _preprocess_action(
        self, action: Mapping[str, float], *, tes_hot_k: float
    ) -> tuple[dict[str, float], dict[str, float | bool]]:
        processed_action = {
            key: float(action.get(key, 0.0))
            for key in self.action_keys
        }

        drive_margin_k = self._abs_drive_margin_k(tes_hot_k=tes_hot_k)
        gate_scale_k = max(1e-6, float(self.config.abs_gate_scale_k))
        abs_gate = 1.0
        if bool(self.config.abs_gate_enabled):
            abs_gate = _sigmoid(drive_margin_k / gate_scale_k)
        u_abs_raw = float(_clip(processed_action.get("u_abs", 0.0), 0.0, 1.0))
        u_abs_effective = float(_clip(u_abs_raw * abs_gate, 0.0, 1.0))
        abs_deadzone_gate_th = float(max(0.0, self.config.abs_deadzone_gate_th))
        abs_deadzone_u_th = float(max(0.0, self.config.abs_deadzone_u_th))
        abs_deadzone_active = (abs_gate < abs_deadzone_gate_th) or (
            u_abs_effective < abs_deadzone_u_th
        )
        u_abs_phys = 0.0 if abs_deadzone_active else u_abs_effective
        processed_action["u_abs"] = float(_clip(u_abs_phys, 0.0, 1.0))

        invalid_abs_request_cost = 0.0
        invalid_abs_request_flag = False
        invalid_request_threshold = float(max(0.0, self.config.abs_invalid_req_u_th))
        low_gate_threshold = float(max(0.0, self.config.abs_invalid_req_gate_th))
        if u_abs_raw > invalid_request_threshold and abs_gate < low_gate_threshold:
            invalid_abs_request_flag = True
            request_excess = (u_abs_raw - invalid_request_threshold) / max(
                1e-6, 1.0 - invalid_request_threshold
            )
            gate_deficit = (low_gate_threshold - abs_gate) / max(1e-6, low_gate_threshold)
            invalid_abs_request_cost = float(
                max(0.0, request_excess) * max(0.0, gate_deficit) * self.config.penalty_invalid_abs_request
            )

        u_gt_raw = float(_clip(processed_action.get("u_gt", 0.0), -1.0, 1.0))
        p_gt_requested_mw_raw = ((u_gt_raw + 1.0) * 0.5) * float(self.config.p_gt_cap_mw)
        p_gt_requested_mw = p_gt_requested_mw_raw
        gt_action_smoothing_applied = False
        gt_min_on_enforced = False
        gt_min_off_enforced = False
        gt_min_output_mw = float(self.config.gt_min_output_mw)
        p_gt_target_below_min = (
            p_gt_requested_mw_raw > GT_MIN_OUTPUT_EPS_MW
            and p_gt_requested_mw_raw < (gt_min_output_mw - GT_MIN_OUTPUT_EPS_MW)
        )

        if bool(self.config.gt_action_smoothing_enabled):
            ramp_limit = float(max(0.0, self.config.gt_ramp_mw_per_step))
            low_limit = max(0.0, float(self.gt_prev_p_mw) - ramp_limit)
            high_limit = min(float(self.config.p_gt_cap_mw), float(self.gt_prev_p_mw) + ramp_limit)
            limited_target = float(_clip(p_gt_requested_mw, low_limit, high_limit))
            if abs(limited_target - p_gt_requested_mw) > 1e-9:
                gt_action_smoothing_applied = True
            p_gt_requested_mw = limited_target

        min_on_steps = max(0, int(round(float(self.config.gt_min_on_steps))))
        min_off_steps = max(0, int(round(float(self.config.gt_min_off_steps))))
        requested_on = p_gt_requested_mw > 1e-9

        if self.gt_prev_on and (not requested_on) and int(self.gt_on_steps) < min_on_steps:
            gt_min_on_enforced = True
            gt_action_smoothing_applied = True
            p_gt_requested_mw = max(float(self.config.gt_min_output_mw), float(self.gt_prev_p_mw))
            requested_on = True

        if (not self.gt_prev_on) and requested_on and int(self.gt_off_steps) < min_off_steps:
            gt_min_off_enforced = True
            gt_action_smoothing_applied = True
            p_gt_requested_mw = 0.0
            requested_on = False

        if requested_on and 0.0 < p_gt_requested_mw < float(self.config.gt_min_output_mw):
            p_gt_requested_mw = float(self.config.gt_min_output_mw)
            gt_action_smoothing_applied = True

        processed_action["u_gt"] = self._gt_target_mw_to_action(p_gt_requested_mw)
        gt_requested_delta_mw = float(abs(p_gt_requested_mw - float(self.gt_prev_p_mw)))
        gt_delta_cost = float(gt_requested_delta_mw * max(0.0, float(self.config.penalty_gt_delta_mw)))
        return processed_action, {
            "u_abs_raw": float(u_abs_raw),
            "u_abs_gate": float(abs_gate),
            "u_abs_effective": float(u_abs_effective),
            "u_abs_phys": float(u_abs_phys),
            "abs_deadzone_active": bool(abs_deadzone_active),
            "abs_drive_margin_k": float(drive_margin_k),
            "invalid_abs_request_cost": float(invalid_abs_request_cost),
            "invalid_abs_request_flag": bool(invalid_abs_request_flag),
            "u_gt_raw": float(u_gt_raw),
            "p_gt_requested_mw_raw": float(p_gt_requested_mw_raw),
            "p_gt_requested_mw_effective": float(p_gt_requested_mw),
            "gt_requested_delta_mw": float(gt_requested_delta_mw),
            "gt_delta_cost": float(gt_delta_cost),
            "gt_action_smoothing_applied": bool(gt_action_smoothing_applied),
            "gt_min_on_enforced": bool(gt_min_on_enforced),
            "gt_min_off_enforced": bool(gt_min_off_enforced),
            "p_gt_target_below_min": bool(p_gt_target_below_min),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        episode_df: pd.DataFrame | None = None,
        start_idx: int | None = None,
        episode_steps: int | None = None,
    ) -> tuple[dict[str, float], dict]:
        """
        重置环境，开始新的 episode。

        参数：
        - seed: 随机种子（可选）
        - episode_df: 自定义 episode 数据（可选）
        - start_idx: 从全年数据的哪个位置开始（可选）
        - episode_steps: episode 长度（可选）

        返回：(observation, info)
        - observation: 初始观测
        - info: episode 元信息（步数、起止时间）
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if episode_df is not None:
            self.set_episode(episode_df)
        elif start_idx is not None and episode_steps is not None:
            end_idx = start_idx + episode_steps
            self.set_episode(self.full_df.iloc[start_idx:end_idx].reset_index(drop=True))
        else:
            self.episode_df = self.full_df

        self._reset_states()
        first_row = self.episode_df.iloc[0]
        observation = self._build_observation(first_row)
        info = {
            "episode_steps": int(len(self.episode_df)),
            "episode_start": pd.to_datetime(self.episode_df["timestamp"].iloc[0]).isoformat(),
            "episode_end": pd.to_datetime(self.episode_df["timestamp"].iloc[-1]).isoformat(),
        }
        return observation, info

    def _is_physics_in_loop(self) -> bool:
        """判断是否为 physics_in_loop 约束模式。"""
        mode = self.config.constraint_mode.strip().lower()
        if mode not in {"physics_in_loop", "reward_only"}:
            raise ValueError(f"不支持的 constraint_mode: {self.config.constraint_mode}")
        return mode == "physics_in_loop"

    def _compute_grid_balance(self, *, p_net_mw: float, p_demand_mw: float) -> dict[str, float]:
        """
        计算电网平衡：输入/输出/未满足/弃电。

        参数：
        - p_net_mw: 本地净发电（GT + BES 放电 - BES 充电 - 电制冷）
        - p_demand_mw: 电负荷需求

        返回：dict 包含 grid_import_mw / grid_export_mw / p_unmet_e_mw / p_curtail_mw
        """
        import_need = max(0.0, p_demand_mw - p_net_mw)
        export_need = max(0.0, p_net_mw - p_demand_mw)
        grid_import = min(import_need, self.config.grid_import_cap_mw)
        grid_export = min(export_need, self.config.grid_export_cap_mw)
        p_unmet = max(0.0, import_need - grid_import)
        p_curtail = max(0.0, export_need - grid_export)
        return {
            "grid_import_mw": grid_import,
            "grid_export_mw": grid_export,
            "p_unmet_e_mw": p_unmet,
            "p_curtail_mw": p_curtail,
        }

    def step(self, action: Mapping[str, float]) -> tuple[dict[str, float], float, bool, bool, dict]:
        """
        执行一步动作，返回 (observation, reward, terminated, truncated, info)。

        内部流程：
        1. 从 action 中提取控制设定点
        2. 调用约束求解器投影到可行域（physics_in_loop 模式）
        3. 调用 physics 层求解设备状态（GT/HRSG/BES/TES/锅炉/制冷机）
        4. 计算电网平衡（输入/输出/未满足/弃电）
        5. 计算成本拆分（燃料/电网/惩罚/降解）
        6. 更新内生状态（SOC/温度/历史状态）
        7. 追踪 KPI（violation/unmet/成本）
        8. 构建 observation 和 info

        返回：
        - observation: dict[str, float]，下一时刻观测
        - reward: float，等于 -total_cost
        - terminated: bool，是否到达 episode 末尾
        - truncated: bool，是否被截断（当前未使用）
        - info: dict，包含诊断信息
        """
        if self.current_step >= len(self.episode_df):
            raise RuntimeError("episode 已结束，请先 reset。")

        row = self.episode_df.iloc[self.current_step]
        t_amb_k = float(row["t_amb_k"])
        is_physics_mode = self._is_physics_in_loop()
        t_hot_k = float(self.thermal_storage.hot_water_temperature_k())
        e_tes_mwh = float(self.thermal_storage.energy_mwh)
        processed_action, action_debug = self._preprocess_action(action, tes_hot_k=t_hot_k)

        # 说明：约束求解器需要一个“线性化的可用热量输入”（HRSG 回收热量）。
        # 这里先用当前 action 的 GT 目标做一次快速离线求解，得到 HRSG 的近似可用热量，
        # 再把这个值喂给约束求解器。
        # 注意：这一步不直接决定最终动作，只用于构造约束模型的输入。
        u_gt_guess = _clip(float(processed_action.get("u_gt", 0.0)), -1.0, 1.0)
        p_gt_guess = ((u_gt_guess + 1.0) * 0.5) * self.config.p_gt_cap_mw
        gt_guess = self.gt_network.solve_offdesign(p_gt_request_mw=p_gt_guess, t_amb_k=t_amb_k)
        hrsg_guess = self.hrsg_network.solve(
            m_exh_kg_per_s=gt_guess.m_exh_kg_per_s,
            t_exh_in_k=gt_guess.t_exh_k,
        )

        solver_result = self.constraint_solver.solve(
            ConstraintInputs(
                p_dem_mw=float(row["p_dem_mw"]),
                qh_dem_mw=float(row["qh_dem_mw"]),
                qc_dem_mw=float(row["qc_dem_mw"]),
                p_re_mw=float(row["pv_mw"]) + float(row["wt_mw"]),
                p_gt_prev_mw=self.gt_prev_p_mw,
                soc_bes=self.bes_soc,
                action=processed_action,
                q_hrsg_available_mw=hrsg_guess.q_rec_mw,
                cop_abs_est=self.abs_chiller.estimate_cop(self.thermal_storage.hot_water_temperature_k()),
                cop_electric_est=self.electric_chiller.estimate_cop(t_amb_k=t_amb_k),
                tes_charge_feasible_mw=self.thermal_storage.max_feasible_charge_mw(self.config.dt_hours),
                tes_discharge_feasible_mw=self.thermal_storage.max_feasible_discharge_mw(self.config.dt_hours),
                is_physics_mode=is_physics_mode,
            )
        )

        p_gt_mw = float(solver_result["p_gt_mw"])
        p_bes_mw = float(solver_result["p_bes_mw"])

        gt_result = self.gt_network.solve_offdesign(p_gt_request_mw=p_gt_mw, t_amb_k=t_amb_k)
        hrsg_result = self.hrsg_network.solve(
            m_exh_kg_per_s=gt_result.m_exh_kg_per_s,
            t_exh_in_k=gt_result.t_exh_k,
        )

        dt_h = self.config.dt_hours
        qh_demand_mw = float(row["qh_dem_mw"])
        qc_demand_mw = float(row["qc_dem_mw"])

        # 热侧：先把两种模式共用的中间量前置统一，减少重复。
        # physics_in_loop：TES 充放热功率受“当前状态可行域”约束（max_feasible_*）。
        # reward_only：TES 充放热功率按额定上限（cap）执行，不额外做可行域收缩。
        u_tes = float(solver_result["u_tes"])
        u_abs = float(solver_result["u_abs"])
        q_drive_req = u_abs * self.config.q_abs_drive_cap_mw
        if is_physics_mode:
            tes_discharge_limit = self.thermal_storage.max_feasible_discharge_mw(dt_h)
            tes_charge_limit = self.thermal_storage.max_feasible_charge_mw(dt_h)
        else:
            tes_discharge_limit = self.config.q_tes_discharge_cap_mw
            tes_charge_limit = self.config.q_tes_charge_cap_mw

        tes_discharge_req = max(0.0, u_tes) * tes_discharge_limit
        tes_charge_req = max(0.0, -u_tes) * tes_charge_limit
        heat_overcommit_flag = False

        u_boiler_solver = float(_clip(float(solver_result["u_boiler"]), 0.0, 1.0))
        boiler_lower_bound = self._compute_boiler_heat_lower_bound(
            qh_demand_mw=qh_demand_mw,
            q_hrsg_available_mw=float(hrsg_result.q_rec_mw),
            tes_discharge_support_mw=float(tes_discharge_req),
        )
        u_boiler_final = u_boiler_solver
        heat_backup_shield_applied = False
        if bool(self.config.heat_backup_shield_enabled) and is_physics_mode:
            shield_lower_bound = float(boiler_lower_bound["u_boiler_lower_bound"])
            if shield_lower_bound > u_boiler_final + 1e-9:
                u_boiler_final = shield_lower_bound
                heat_backup_shield_applied = True

        boiler_result = self.boiler.solve(u_boiler=u_boiler_final)
        ech_result = self.electric_chiller.solve(u_ech=float(solver_result["u_ech"]), t_amb_k=t_amb_k)

        if is_physics_mode:
            # physics_in_loop 口径：遵循“供热优先级”。
            # 先满足供暖负荷，再把剩余热量分配给吸收式制冷的驱动热，最后再尝试给 TES 充热。
            # 这样可保证热量分配不会出现“先给 TES/制冷导致供暖缺供”的非物理现象。
            q_heat_available = hrsg_result.q_rec_mw + boiler_result.q_heat_mw + tes_discharge_req
            qh_served = min(qh_demand_mw, q_heat_available)
            qh_unmet_mw = max(0.0, qh_demand_mw - qh_served)
            heat_after_heating = q_heat_available - qh_served

            q_drive_alloc = min(q_drive_req, max(0.0, heat_after_heating))
            abs_result = self.abs_chiller.solve(
                q_drive_request_mw=q_drive_alloc,
                t_hot_k=t_hot_k,
            )
            heat_after_drive = heat_after_heating - abs_result.q_drive_used_mw

            q_tes_charge = min(tes_charge_req, max(0.0, heat_after_drive))
            q_heat_dump_mw = max(0.0, heat_after_drive - q_tes_charge)
            tes_result = self.thermal_storage.apply(
                charge_request_mw=q_tes_charge,
                discharge_request_mw=tes_discharge_req,
                dt_h=dt_h,
            )
        else:
            # reward_only 口径：按动作直接驱动各设备，然后再做供需结算。
            # 该模式下不强制“供暖优先级”，因此可能出现热量使用总需求超过供给的情况。
            # heat_overcommit_flag 用于标记这种“热侧超配”，供奖励/诊断使用。
            abs_result = self.abs_chiller.solve(
                q_drive_request_mw=q_drive_req,
                t_hot_k=t_hot_k,
            )
            tes_result = self.thermal_storage.apply(
                charge_request_mw=tes_charge_req,
                discharge_request_mw=tes_discharge_req,
                dt_h=dt_h,
            )

            qh_supply = hrsg_result.q_rec_mw + boiler_result.q_heat_mw + tes_result.q_discharge_mw
            qh_usage = qh_demand_mw + abs_result.q_drive_used_mw + tes_result.q_charge_mw
            qh_unmet_mw = max(0.0, qh_usage - qh_supply)
            q_heat_dump_mw = max(0.0, qh_supply - qh_usage)
            heat_overcommit_flag = qh_usage > qh_supply + 1e-9

        qc_supply_mw = abs_result.q_cool_mw + ech_result.q_cool_mw
        qc_unmet_mw = max(0.0, qc_demand_mw - qc_supply_mw)

        p_re_mw = float(row["pv_mw"]) + float(row["wt_mw"])
        p_net_mw = p_gt_mw + p_re_mw + p_bes_mw - ech_result.p_electric_mw
        grid_balance = self._compute_grid_balance(p_net_mw=p_net_mw, p_demand_mw=float(row["p_dem_mw"]))

        bes_soc_before = float(self.bes_soc)
        self.bes_soc, bes_soc_clipped = update_bes_soc(
            p_bes_mw=p_bes_mw,
            current_soc=self.bes_soc,
            dt_hours=self.config.dt_hours,
            e_bes_cap_mwh=self.config.e_bes_cap_mwh,
            soc_min=self.config.bes_soc_min,
            soc_max=self.config.bes_soc_max,
            eta_charge=self.config.bes_eta_charge,
            eta_discharge=self.config.bes_eta_discharge,
            self_discharge_per_hour=self.config.bes_self_discharge_per_hour,
            aux_equip_eff=self.config.bes_aux_equip_eff,
        )
        bes_degradation_cost = compute_bes_degradation_cost(
            dt_hours=dt_h,
            soc_before=bes_soc_before,
            soc_after=float(self.bes_soc),
            e_bes_cap_mwh=self.config.e_bes_cap_mwh,
            dod_battery_capex_per_mwh=self.config.bes_dod_battery_capex_per_mwh,
            dod_k_p=self.config.bes_dod_k_p,
            dod_n_fail_100=self.config.bes_dod_n_fail_100,
            dod_add_calendar_age=self.config.bes_dod_add_calendar_age,
            dod_battery_life_years=self.config.bes_dod_battery_life_years,
        )

        gt_started = int((not self.gt_prev_on) and (p_gt_mw > 1e-9))
        gt_toggled = bool(self.gt_prev_on != (p_gt_mw > 1e-9))
        gt_on_steps_effective = int(self.gt_on_steps) + 1 if p_gt_mw > 1e-9 else 0
        gt_toggle_cost = float(max(0.0, float(self.config.penalty_gt_toggle)) if gt_toggled else 0.0)
        fuel_input_gt_effective_mw, startup_extra_fuel_mw = apply_gt_startup_fuel_correction(
            fuel_input_gt_mw=gt_result.fuel_input_mw,
            gt_started=bool(gt_started),
            startup_fuel_correction_ratio=self.config.gt_startup_fuel_correction_ratio,
        )

        violation_flags: dict[str, bool] = {}
        violation_flags.update(dict(solver_result["violation_flags"]))
        violation_flags.update(gt_result.violation_flags)
        violation_flags.update(hrsg_result.violation_flags)
        violation_flags.update(tes_result.violation_flags)
        violation_flags.update(abs_result.violation_flags)
        violation_flags.update(boiler_result.violation_flags)
        violation_flags.update(ech_result.violation_flags)
        violation_flags["bes_soc_clipped"] = bes_soc_clipped
        # 仅 reward_only 下记录 heat_overcommit：physics_in_loop 已按分配策略避免该情况。
        violation_flags["heat_overcommit_reward_only"] = (not is_physics_mode) and heat_overcommit_flag

        state_diagnostic_flags: dict[str, bool] = {
            "abs_drive_temp_low_state": t_hot_k < float(self.abs_chiller.design.t_drive_min_k),
            "abs_deadzone_active": bool(action_debug["abs_deadzone_active"]),
        }
        diagnostic_flags: dict[str, bool] = {
            "invalid_abs_request": bool(action_debug["invalid_abs_request_flag"]),
            "gt_action_smoothing_applied": bool(action_debug["gt_action_smoothing_applied"]),
            "gt_min_on_enforced": bool(action_debug["gt_min_on_enforced"]),
            "gt_min_off_enforced": bool(action_debug["gt_min_off_enforced"]),
            "gt_toggled": bool(gt_toggled),
            "p_gt_target_below_min": bool(action_debug["p_gt_target_below_min"]),
            "heat_backup_shield_applied": bool(heat_backup_shield_applied),
        }

        heat_unmet_step = qh_unmet_mw > float(max(0.0, self.config.heat_unmet_th_mw))
        cool_unmet_step = qc_unmet_mw > float(max(0.0, self.config.cool_unmet_th_mw))
        idle_heat_backup = heat_unmet_step and (
            float(boiler_result.q_heat_mw) < float(max(0.0, self.config.heat_backup_idle_th_mw))
        )
        idle_cool_backup = (
            cool_unmet_step
            and (
                bool(action_debug["abs_deadzone_active"])
                or float(action_debug["u_abs_gate"]) < float(max(0.0, self.config.abs_deadzone_gate_th))
            )
            and float(ech_result.q_cool_mw) < float(max(0.0, self.config.cool_backup_idle_th_mw))
        )
        diagnostic_flags["heat_unmet_step"] = bool(heat_unmet_step)
        diagnostic_flags["cool_unmet_step"] = bool(cool_unmet_step)
        diagnostic_flags["idle_heat_backup"] = bool(idle_heat_backup)
        diagnostic_flags["idle_cool_backup"] = bool(idle_cool_backup)

        idle_heat_backup_cost = (
            float(max(0.0, self.config.penalty_idle_heat_backup)) if idle_heat_backup else 0.0
        )
        idle_cool_backup_cost = (
            float(max(0.0, self.config.penalty_idle_cool_backup)) if idle_cool_backup else 0.0
        )

        violation_count = sum(1 for flag in violation_flags.values() if flag)
        cost_breakdown = compute_cost_breakdown(
            dt_h=dt_h,
            price_e=float(row["price_e"]),
            price_gas=float(row["price_gas"]),
            carbon_tax=float(row["carbon_tax"]),
            grid_import_mw=grid_balance["grid_import_mw"],
            grid_export_mw=grid_balance["grid_export_mw"],
            p_curtail_mw=grid_balance["p_curtail_mw"],
            fuel_input_gt_effective_mw=fuel_input_gt_effective_mw,
            p_gt_mw=p_gt_mw,
            gt_started=gt_started,
            gt_on_steps=gt_on_steps_effective,
            bes_degradation_cost=bes_degradation_cost,
            boiler_fuel_input_mw=boiler_result.fuel_input_mw,
            p_unmet_e_mw=grid_balance["p_unmet_e_mw"],
            qh_unmet_mw=qh_unmet_mw,
            qc_unmet_mw=qc_unmet_mw,
            violation_count=violation_count,
            invalid_abs_request_penalty=float(action_debug["invalid_abs_request_cost"]),
            gt_toggle_penalty=gt_toggle_cost,
            gt_delta_penalty=float(action_debug["gt_delta_cost"]),
            idle_heat_backup_penalty=idle_heat_backup_cost,
            idle_cool_backup_penalty=idle_cool_backup_cost,
            config=self.config,
        )

        emission_gt_ton = fuel_input_gt_effective_mw * dt_h * float(self.config.gt_emission_ton_per_mwh_th)
        emission_boiler_ton = float(boiler_result.fuel_input_mw) * dt_h * float(self.config.boiler_emission_ton_per_mwh_th)
        emission_total_ton = float(emission_gt_ton + emission_boiler_ton)

        boiler_started = int((not self.boiler_prev_on) and (boiler_result.q_heat_mw > 1e-9))
        ech_started = int((not self.ech_prev_on) and (ech_result.q_cool_mw > 1e-9))

        sell_price_raw = float(row["price_e"]) * float(self.config.sell_price_ratio)
        sell_price = (
            min(sell_price_raw, float(self.config.sell_price_cap_per_mwh))
            if float(self.config.sell_price_cap_per_mwh) > 0.0
            else sell_price_raw
        )
        grid_export_over_soft_cap_mw = max(
            0.0, float(grid_balance["grid_export_mw"]) - float(self.config.grid_export_soft_cap_mw)
        )

        step_info = {
            "timestamp": pd.to_datetime(row["timestamp"]).isoformat(),
            "cost_total": float(cost_breakdown.cost_total),
            "cost_grid_import": float(cost_breakdown.cost_grid_import),
            "cost_grid_export_revenue": float(cost_breakdown.cost_grid_export_revenue),
            "cost_grid_curtail": float(cost_breakdown.cost_grid_curtail),
            "cost_grid_export_penalty": float(cost_breakdown.cost_grid_export_penalty),
            "cost_grid": float(cost_breakdown.cost_grid),
            "cost_gt_fuel": float(cost_breakdown.cost_gt_fuel),
            "cost_gt_om": float(cost_breakdown.cost_gt_om),
            "cost_carbon": float(cost_breakdown.cost_carbon),
            "cost_bes_degr": float(cost_breakdown.cost_bes_degr),
            "cost_boiler": float(cost_breakdown.cost_boiler),
            "cost_unmet_e": float(cost_breakdown.cost_unmet_e),
            "cost_unmet_h": float(cost_breakdown.cost_unmet_h),
            "cost_unmet_c": float(cost_breakdown.cost_unmet_c),
            "cost_viol": float(cost_breakdown.cost_viol),
            "cost_invalid_abs_request": float(cost_breakdown.cost_invalid_abs_request),
            "cost_gt_toggle": float(cost_breakdown.cost_gt_toggle),
            "cost_gt_delta": float(cost_breakdown.cost_gt_delta),
            "cost_idle_heat_backup": float(cost_breakdown.cost_idle_heat_backup),
            "cost_idle_cool_backup": float(cost_breakdown.cost_idle_cool_backup),
            "fuel_input_gt_mw_raw": float(gt_result.fuel_input_mw),
            "fuel_input_gt_effective_mw": float(fuel_input_gt_effective_mw),
            "fuel_input_gt_startup_extra_mw": float(startup_extra_fuel_mw),
            "emission_gt_ton": float(emission_gt_ton),
            "emission_boiler_ton": float(emission_boiler_ton),
            "emission_total_ton": float(emission_total_ton),
            "t_tes_hot_k": float(t_hot_k),
            "e_tes_mwh": float(e_tes_mwh),
            "energy_demand_e_mwh": float(float(row["p_dem_mw"]) * dt_h),
            "energy_demand_h_mwh": float(qh_demand_mw * dt_h),
            "energy_demand_c_mwh": float(qc_demand_mw * dt_h),
            "energy_unmet_e_mwh": float(grid_balance["p_unmet_e_mw"] * dt_h),
            "energy_unmet_h_mwh": float(qh_unmet_mw * dt_h),
            "energy_unmet_c_mwh": float(qc_unmet_mw * dt_h),
            "qh_unmet_mw": float(qh_unmet_mw),
            "qc_unmet_mw": float(qc_unmet_mw),
            "price_e_buy": float(row["price_e"]),
            "price_e_sell": float(sell_price),
            "u_gt_raw": float(action_debug["u_gt_raw"]),
            "u_boiler_policy": float(_clip(float(processed_action.get("u_boiler", 0.0)), 0.0, 1.0)),
            "u_boiler_solver": float(u_boiler_solver),
            "u_boiler_final": float(u_boiler_final),
            "u_boiler_lower_bound": float(boiler_lower_bound["u_boiler_lower_bound"]),
            "heat_backup_min_needed_mw": float(boiler_lower_bound["heat_backup_min_needed_mw"]),
            "heat_deficit_if_boiler_off_mw": float(boiler_lower_bound["heat_deficit_if_boiler_off_mw"]),
            "heat_backup_shield_applied": bool(heat_backup_shield_applied),
            "p_gt_request_mw_raw": float(action_debug["p_gt_requested_mw_raw"]),
            "p_gt_request_mw_effective": float(action_debug["p_gt_requested_mw_effective"]),
            "gt_requested_delta_mw": float(action_debug["gt_requested_delta_mw"]),
            "p_gt_mw": float(p_gt_mw),
            "p_gt_target_mw": float(solver_result["p_gt_target_mw"]),
            "p_gt_applied_mw": float(solver_result["p_gt_applied_mw"]),
            "p_gt_ramp_delta_mw": float(solver_result["p_gt_ramp_delta_mw"]),
            "p_gt_ramp_limit_mw_per_step": float(solver_result["p_gt_ramp_limit_mw_per_step"]),
            "p_bes_mw": float(p_bes_mw),
            "p_re_mw": float(p_re_mw),
            "p_grid_import_mw": float(grid_balance["grid_import_mw"]),
            "p_grid_export_mw": float(grid_balance["grid_export_mw"]),
            "p_grid_export_over_soft_cap_mw": float(grid_export_over_soft_cap_mw),
            "p_ech_mw": float(ech_result.p_electric_mw),
            "q_hrsg_rec_mw": float(hrsg_result.q_rec_mw),
            "q_boiler_mw": float(boiler_result.q_heat_mw),
            "u_abs_raw": float(action_debug["u_abs_raw"]),
            "u_abs_gate": float(action_debug["u_abs_gate"]),
            "u_abs_effective": float(action_debug["u_abs_effective"]),
            "u_abs_phys": float(action_debug["u_abs_phys"]),
            "u_abs_deadzone_applied": bool(action_debug["abs_deadzone_active"]),
            "abs_drive_margin_k": float(action_debug["abs_drive_margin_k"]),
            "q_abs_cool_mw": float(abs_result.q_cool_mw),
            "q_ech_cool_mw": float(ech_result.q_cool_mw),
            "q_tes_discharge_feasible_mw": float(tes_discharge_limit),
            "q_tes_charge_mw": float(tes_result.q_charge_mw),
            "q_tes_discharge_mw": float(tes_result.q_discharge_mw),
            "q_heat_dump_mw": float(q_heat_dump_mw),
            "constraint_mode": self.config.constraint_mode,
            "physics_backend": self.config.physics_backend,
            "solver_used": str(solver_result["solver_used"]),
            "solver_status": str(solver_result["solver_status"]),
            "solver_termination": str(solver_result["solver_termination"]),
            "solver_error": solver_result["solver_error"],
            "violation_flags": violation_flags,
            "diagnostic_flags": diagnostic_flags,
            "state_diagnostic_flags": state_diagnostic_flags,
            "gt_started": gt_started,
            "boiler_started": boiler_started,
            "ech_started": ech_started,
        }

        self.kpi.record(reward=cost_breakdown.reward, step_info=step_info)

        self.gt_prev_p_mw = p_gt_mw
        self.gt_prev_on = p_gt_mw > 1e-9
        if self.gt_prev_on:
            self.gt_on_steps = int(self.gt_on_steps) + 1
            self.gt_off_steps = 0
        else:
            self.gt_on_steps = 0
            self.gt_off_steps = int(self.gt_off_steps) + 1
        self.boiler_prev_on = boiler_result.q_heat_mw > 1e-9
        self.ech_prev_on = ech_result.q_cool_mw > 1e-9

        self.current_step += 1
        terminated = self.current_step >= len(self.episode_df)
        truncated = False

        if terminated:
            next_observation = self._build_observation(self.episode_df.iloc[-1])
            step_info["episode_summary"] = self.kpi.summary()
        else:
            next_row = self.episode_df.iloc[self.current_step]
            next_observation = self._build_observation(next_row)

        return next_observation, float(cost_breakdown.reward), terminated, truncated, step_info

    def _build_observation(self, row: pd.Series) -> dict[str, float]:
        dt_h = float(self.config.dt_hours)
        long_threshold_steps = max(1, int(round(1.0 / max(1e-9, dt_h))))
        gt_ramp_headroom_up_mw, gt_ramp_headroom_down_mw = self._gt_ramp_headroom()
        tes_hot_k = float(self.thermal_storage.hot_water_temperature_k())
        min_on_steps = max(0, int(round(float(self.config.gt_min_on_steps))))
        min_off_steps = max(0, int(round(float(self.config.gt_min_off_steps))))
        if not self.gt_prev_on:
            gt_state = 0.0
        elif int(self.gt_on_steps) >= long_threshold_steps:
            gt_state = 2.0
        else:
            gt_state = 1.0
        gt_on_steps = int(self.gt_on_steps)
        gt_off_steps = int(self.gt_off_steps)
        gt_min_on_remaining_steps = max(0, min_on_steps - gt_on_steps) if self.gt_prev_on else 0
        gt_min_off_remaining_steps = max(0, min_off_steps - gt_off_steps) if (not self.gt_prev_on) else 0
        heat_obs = self._build_heat_reliability_observation_features(row=row)
        return build_observation(
            row=row,
            bes_soc=self.bes_soc,
            gt_prev_on=self.gt_prev_on,
            gt_state=float(gt_state),
            gt_on_steps=float(gt_on_steps),
            gt_off_steps=float(gt_off_steps),
            gt_min_on_remaining_steps=float(gt_min_on_remaining_steps),
            gt_min_off_remaining_steps=float(gt_min_off_remaining_steps),
            p_gt_prev_mw=float(self.gt_prev_p_mw),
            gt_ramp_headroom_up_mw=float(gt_ramp_headroom_up_mw),
            gt_ramp_headroom_down_mw=float(gt_ramp_headroom_down_mw),
            tes_energy_mwh=self.thermal_storage.energy_mwh,
            tes_hot_k=tes_hot_k,
            abs_drive_margin_k=float(self._abs_drive_margin_k(tes_hot_k=tes_hot_k)),
            q_hrsg_est_now_mw=float(heat_obs["q_hrsg_est_now_mw"]),
            q_tes_discharge_feasible_mw=float(heat_obs["q_tes_discharge_feasible_mw"]),
            heat_deficit_if_boiler_off_mw=float(heat_obs["heat_deficit_if_boiler_off_mw"]),
            heat_backup_min_needed_mw=float(heat_obs["heat_backup_min_needed_mw"]),
        )
