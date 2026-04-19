# Ref: docs/spec/task.md (Task-ID: 012)
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.data import EVAL_YEAR, TRAIN_YEAR, dump_statistics_json, make_episode_sampler
from ..core.reporting import write_paper_eval_artifacts
from ..env.cchp_env import CCHPPhysicalEnv, EnvConfig
from ..pipeline.sequence import (
    DEFAULT_SEQUENCE_ACTION_KEYS,
    DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS,
    build_feature_vector,
)
from .checkpoint import load_policy, resolve_torch_device, save_policy
from .projection_surrogate import build_projection_surrogate_network

_NORM_EPS = 1e-6
_BES_PRIOR_MIN_OPPORTUNITY = 0.05
_COST_KEYS = ("cost_e", "cost_h", "cost_c")
_DUAL_NAMES = ("lambda_e", "lambda_h", "lambda_c")
_ACTION_BOUNDS = {
    "u_gt": (-1.0, 1.0),
    "u_bes": (-1.0, 1.0),
    "u_boiler": (0.0, 1.0),
    "u_abs": (0.0, 1.0),
    "u_ech": (0.0, 1.0),
    "u_tes": (-1.0, 1.0),
}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)


def _json_payload_from_path(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_year(df: pd.DataFrame) -> int:
    years = sorted({int(value.year) for value in pd.to_datetime(df["timestamp"])})
    if len(years) != 1:
        raise ValueError(f"仅支持单年数据，当前年份集合: {years}")
    return years[0]


def _require_torch_modules():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.optim import AdamW
    except ModuleNotFoundError as error:  # pragma: no cover
        raise ModuleNotFoundError("未检测到 torch，无法训练 PAFC-TD3。") from error
    return torch, nn, F, AdamW


def _action_bounds_arrays(
    action_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    lows: list[float] = []
    highs: list[float] = []
    for key in action_keys:
        low, high = _ACTION_BOUNDS[str(key)]
        lows.append(float(low))
        highs.append(float(high))
    return np.asarray(lows, dtype=np.float32), np.asarray(highs, dtype=np.float32)


def _action_vector_to_dict(
    action_vector: np.ndarray | Sequence[float],
    *,
    action_keys: Sequence[str],
) -> dict[str, float]:
    vector = np.asarray(action_vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != len(action_keys):
        raise ValueError(
            f"动作维度不匹配：期望 {len(action_keys)}，实际 {vector.shape[0]}"
        )
    result: dict[str, float] = {}
    for index, key in enumerate(action_keys):
        low, high = _ACTION_BOUNDS[str(key)]
        result[str(key)] = float(np.clip(vector[index], low, high))
    return result


def _materialize_teacher_action_np(
    *,
    teacher_action: Mapping[str, float],
    action_keys: Sequence[str],
    obs_vector: np.ndarray,
    returns_executed_action: bool,
    project_action_exec_fn,
) -> tuple[np.ndarray, np.ndarray]:
    action_raw = np.asarray(
        [float(teacher_action[key]) for key in action_keys],
        dtype=np.float32,
    )
    if bool(returns_executed_action):
        return action_raw, action_raw.copy()
    action_exec = np.asarray(
        project_action_exec_fn(
            obs_vector=obs_vector,
            action_raw=action_raw,
        ),
        dtype=np.float32,
    )
    return action_raw, action_exec


def _extract_action_vector_from_info(
    info: Mapping[str, Any],
    *,
    prefix: str,
    action_keys: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [float(info[f"{prefix}_{key}"]) for key in action_keys],
        dtype=np.float32,
    )


def _extract_cost_vector(info: Mapping[str, Any]) -> np.ndarray:
    demand_e = max(_NORM_EPS, float(info.get("energy_demand_e_mwh", 0.0)))
    demand_h = max(_NORM_EPS, float(info.get("energy_demand_h_mwh", 0.0)))
    demand_c = max(_NORM_EPS, float(info.get("energy_demand_c_mwh", 0.0)))
    return np.asarray(
        [
            float(info.get("energy_unmet_e_mwh", 0.0)) / demand_e,
            float(info.get("energy_unmet_h_mwh", 0.0)) / demand_h,
            float(info.get("energy_unmet_c_mwh", 0.0)) / demand_c,
        ],
        dtype=np.float32,
    )


def _extract_gap_vector(info: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(info.get("projection_gap_l1", 0.0)),
            float(info.get("projection_gap_l2", 0.0)),
            float(info.get("projection_gap_max", 0.0)),
        ],
        dtype=np.float32,
    )


def _gt_off_deadband_mw(*, gt_min_output_mw: float, gt_off_deadband_ratio: float) -> float:
    ratio = float(np.clip(float(gt_off_deadband_ratio), 0.0, 1.0))
    return max(0.0, float(gt_min_output_mw)) * ratio


def _canonicalize_gt_target_np(
    *,
    p_gt_target_mw: float,
    p_gt_low_mw: float,
    p_gt_high_mw: float,
    gt_min_output_mw: float,
    gt_off_deadband_ratio: float,
) -> float:
    p_gt_target = float(np.clip(p_gt_target_mw, p_gt_low_mw, p_gt_high_mw))
    gt_min_output_mw = max(0.0, float(gt_min_output_mw))
    off_deadband_mw = _gt_off_deadband_mw(
        gt_min_output_mw=gt_min_output_mw,
        gt_off_deadband_ratio=gt_off_deadband_ratio,
    )
    if p_gt_target <= (off_deadband_mw + _NORM_EPS):
        return 0.0
    if 0.0 < p_gt_target < gt_min_output_mw:
        return gt_min_output_mw
    return p_gt_target


def _gt_proxy_support_multiplier_np(*, abs_ready: float, heat_support_need: float) -> float:
    abs_ready = float(np.clip(abs_ready, 0.0, 1.0))
    heat_support_need = float(np.clip(heat_support_need, 0.0, 1.0))
    return float(np.clip(0.65 + 0.15 * abs_ready + 0.20 * heat_support_need, 0.0, 1.0))


def _resolve_bes_price_thresholds_from_train_statistics(
    *,
    train_statistics: Mapping[str, Any],
) -> tuple[float, float]:
    stats = dict(train_statistics.get("stats", {}) or {})
    price_stats = dict(stats.get("price_e", {}) or {})
    price_low = max(0.0, _safe_float(price_stats.get("p05", 0.0)))
    price_high = max(price_low + 1.0, _safe_float(price_stats.get("p95", price_low + 1.0)))
    if not np.isfinite(price_high) or price_high <= price_low + _NORM_EPS:
        price_high = max(price_low + 1.0, _safe_float(price_stats.get("p50", price_low + 1.0)))
    return float(price_low), float(price_high)


def _bes_price_pressure_np(
    *,
    price_e: float,
    price_low_threshold: float,
    price_high_threshold: float,
) -> dict[str, float]:
    low_price = float(price_low_threshold)
    high_price = max(low_price + _NORM_EPS, float(price_high_threshold))
    mid_price = 0.5 * (low_price + high_price)
    charge_span = max(_NORM_EPS, mid_price - low_price)
    discharge_span = max(_NORM_EPS, high_price - mid_price)
    charge_pressure = float(np.clip((mid_price - float(price_e)) / charge_span, 0.0, 1.0))
    discharge_pressure = float(np.clip((float(price_e) - mid_price) / discharge_span, 0.0, 1.0))
    return {
        "mid_price": float(mid_price),
        "charge_pressure": float(charge_pressure),
        "discharge_pressure": float(discharge_pressure),
    }


def _bes_price_prior_target_np(
    *,
    price_e: float,
    soc_bes: float,
    price_low_threshold: float,
    price_high_threshold: float,
    charge_soc_ceiling: float,
    discharge_soc_floor: float,
    bes_soc_min: float,
    bes_soc_max: float,
    charge_u: float,
    discharge_u: float,
) -> dict[str, float | str]:
    price_pressure = _bes_price_pressure_np(
        price_e=price_e,
        price_low_threshold=price_low_threshold,
        price_high_threshold=price_high_threshold,
    )
    charge_headroom = float(
        np.clip(
            (float(charge_soc_ceiling) - float(soc_bes))
            / max(_NORM_EPS, float(charge_soc_ceiling) - float(bes_soc_min)),
            0.0,
            1.0,
        )
    )
    discharge_headroom = float(
        np.clip(
            (float(soc_bes) - float(discharge_soc_floor))
            / max(_NORM_EPS, float(bes_soc_max) - float(discharge_soc_floor)),
            0.0,
            1.0,
        )
    )
    charge_score = float(price_pressure["charge_pressure"] * charge_headroom)
    discharge_score = float(price_pressure["discharge_pressure"] * discharge_headroom)
    net_signal = float(discharge_score - charge_score)
    opportunity = float(abs(net_signal))
    if opportunity <= _BES_PRIOR_MIN_OPPORTUNITY:
        return {
            "target_u_bes": 0.0,
            "opportunity": 0.0,
            "charge_score": float(charge_score),
            "discharge_score": float(discharge_score),
            "mode": "idle",
        }
    if net_signal > 0.0:
        return {
            "target_u_bes": float(abs(discharge_u) * min(1.0, opportunity)),
            "opportunity": float(opportunity),
            "charge_score": float(charge_score),
            "discharge_score": float(discharge_score),
            "mode": "discharge",
        }
    if net_signal < 0.0:
        return {
            "target_u_bes": float(-abs(charge_u) * min(1.0, opportunity)),
            "opportunity": float(opportunity),
            "charge_score": float(charge_score),
            "discharge_score": float(discharge_score),
            "mode": "charge",
        }
    return {
        "target_u_bes": 0.0,
        "opportunity": 0.0,
        "charge_score": float(charge_score),
        "discharge_score": float(discharge_score),
        "mode": "idle",
    }


def _bes_prior_weight_multiplier_np(
    *,
    mode: str,
    charge_score: float,
    charge_weight: float,
    discharge_weight: float,
    charge_pressure_bonus: float,
) -> float:
    mode_label = str(mode)
    if mode_label == "charge":
        base_weight = max(0.0, float(charge_weight))
    elif mode_label == "discharge":
        base_weight = max(0.0, float(discharge_weight))
    else:
        base_weight = 1.0
    return float(
        base_weight
        * (
            1.0
            + max(0.0, float(charge_pressure_bonus)) * max(0.0, float(charge_score))
        )
    )


def _select_bes_full_year_target_np(
    *,
    source_label: str,
    prior_target_u_bes: float,
    prior_opportunity: float,
    teacher_u_bes: float,
    teacher_min_abs_u: float = 0.05,
    idle_teacher_abs_u_min: float = 0.25,
) -> dict[str, float | str | bool]:
    source = str(source_label).strip().lower()
    prior_target = float(prior_target_u_bes)
    opportunity = max(0.0, float(prior_opportunity))
    teacher_u = float(teacher_u_bes)
    teacher_abs = abs(teacher_u)
    prior_sign = 0.0 if abs(prior_target) <= _NORM_EPS else float(np.sign(prior_target))
    teacher_sign = 0.0 if teacher_abs <= _NORM_EPS else float(np.sign(teacher_u))
    if source.startswith("economic") and teacher_abs > max(_NORM_EPS, float(teacher_min_abs_u)):
        if opportunity > 0.0 and (prior_sign == 0.0 or teacher_sign == prior_sign):
            return {
                "target_u_bes": float(teacher_u),
                "mode": "discharge" if teacher_u > 0.0 else "charge",
                "used_teacher": True,
                "weight_bonus": float(max(opportunity, teacher_abs)),
            }
        if opportunity <= 0.0 and teacher_abs >= float(idle_teacher_abs_u_min):
            return {
                "target_u_bes": float(teacher_u),
                "mode": "discharge" if teacher_u > 0.0 else "charge",
                "used_teacher": True,
                "weight_bonus": float(teacher_abs),
            }
    if abs(prior_target) <= _NORM_EPS:
        return {
            "target_u_bes": 0.0,
            "mode": "idle",
            "used_teacher": False,
            "weight_bonus": float(opportunity),
        }
    return {
        "target_u_bes": float(prior_target),
        "mode": "discharge" if prior_target > 0.0 else "charge",
        "used_teacher": False,
        "weight_bonus": float(opportunity),
    }


def _bes_full_year_selection_priority_np(
    *,
    base_priority: float,
    source_label: str,
    used_teacher: bool,
    teacher_priority_boost: float = 0.0,
    economic_source_priority_bonus: float = 0.0,
) -> float:
    priority = max(0.0, float(base_priority))
    if str(source_label).strip().lower().startswith("economic"):
        priority += max(0.0, float(economic_source_priority_bonus))
    if bool(used_teacher):
        priority *= 1.0 + max(0.0, float(teacher_priority_boost))
    return float(priority)


def _allocate_bes_warm_start_mode_counts(
    *,
    requested_total: int,
    charge_available: int,
    discharge_available: int,
    idle_available: int,
) -> dict[str, int]:
    requested_total = max(0, int(requested_total))
    available = {
        "charge": max(0, int(charge_available)),
        "discharge": max(0, int(discharge_available)),
        "idle": max(0, int(idle_available)),
    }
    if requested_total <= 0:
        return {mode: 0 for mode in available}
    planned = {
        "charge": int(round(float(requested_total) * 0.40)),
        "discharge": int(round(float(requested_total) * 0.40)),
    }
    planned["idle"] = max(0, requested_total - planned["charge"] - planned["discharge"])
    selected = {
        mode: min(int(available[mode]), int(planned.get(mode, 0)))
        for mode in ("charge", "discharge", "idle")
    }
    remaining = max(0, requested_total - int(sum(selected.values())))
    while remaining > 0:
        progressed = False
        for mode in ("discharge", "charge", "idle"):
            if int(selected[mode]) >= int(available[mode]):
                continue
            selected[mode] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
    return {
        mode: int(selected[mode])
        for mode in ("charge", "discharge", "idle")
    }


def _allocate_bes_source_counts(
    *,
    requested_total: int,
    economic_available: int,
    other_available: int,
    economic_min_share: float,
) -> dict[str, int]:
    requested_total = max(0, int(requested_total))
    economic_available = max(0, int(economic_available))
    other_available = max(0, int(other_available))
    share = min(1.0, max(0.0, float(economic_min_share)))
    economic_target = min(economic_available, int(round(float(requested_total) * share)))
    other_target = min(other_available, max(0, requested_total - economic_target))
    remaining = max(0, requested_total - economic_target - other_target)
    while remaining > 0:
        progressed = False
        if economic_target < economic_available:
            economic_target += 1
            remaining -= 1
            progressed = True
        if remaining > 0 and other_target < other_available:
            other_target += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return {
        "economic": int(economic_target),
        "other": int(other_target),
    }


def _allocate_bes_teacher_target_counts(
    *,
    requested_total: int,
    teacher_available: int,
    other_available: int,
    teacher_min_share: float,
) -> dict[str, int]:
    requested_total = max(0, int(requested_total))
    teacher_available = max(0, int(teacher_available))
    other_available = max(0, int(other_available))
    share = min(1.0, max(0.0, float(teacher_min_share)))
    teacher_target = min(teacher_available, int(round(float(requested_total) * share)))
    other_target = min(other_available, max(0, requested_total - teacher_target))
    remaining = max(0, requested_total - teacher_target - other_target)
    while remaining > 0:
        progressed = False
        if teacher_target < teacher_available:
            teacher_target += 1
            remaining -= 1
            progressed = True
        if remaining > 0 and other_target < other_available:
            other_target += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return {
        "teacher": int(teacher_target),
        "other": int(other_target),
    }


def _select_temporal_priority_indices(
    *,
    indices: Sequence[int],
    priorities: Sequence[float],
    target_count: int,
) -> list[int]:
    target_count = max(0, int(target_count))
    if target_count <= 0:
        return []
    unique_sorted = np.asarray(
        sorted({int(idx) for idx in indices if int(idx) >= 0}),
        dtype=np.int64,
    )
    if unique_sorted.size == 0:
        return []
    if unique_sorted.size <= target_count:
        return [int(idx) for idx in unique_sorted.tolist()]
    priority_array = np.asarray(priorities, dtype=np.float32).reshape(-1)
    selected: list[int] = []
    for chunk in np.array_split(unique_sorted, target_count):
        if chunk.size <= 0:
            continue
        chunk_scores = priority_array[chunk]
        best_idx = int(chunk[int(np.argmax(chunk_scores))])
        selected.append(best_idx)
    return selected


def _gt_anchor_relax_signal_np(
    *,
    price_advantage: float,
    net_grid_need_ratio: float,
    undercommit_ratio: float,
    abs_ready: float,
    heat_support_need: float,
    projection_risk: float,
) -> float:
    profitable_pressure = max(
        float(np.clip(net_grid_need_ratio, 0.0, 1.0)),
        float(np.clip(undercommit_ratio, 0.0, 1.0)),
    )
    support_multiplier = _gt_proxy_support_multiplier_np(
        abs_ready=abs_ready,
        heat_support_need=heat_support_need,
    )
    return float(
        np.clip(
            2.5
            * float(np.clip(price_advantage, 0.0, 1.0))
            * profitable_pressure
            * support_multiplier
            * (1.0 - float(np.clip(projection_risk, 0.0, 1.0))),
            0.0,
            1.0,
        )
    )


def _economic_teacher_blend_weight_np(
    *,
    opportunity_score: float,
    safety_margin: float,
    disagreement_score: float,
) -> float:
    return float(
        np.clip(
            float(np.clip(opportunity_score, 0.0, 1.0))
            * (0.25 + 0.75 * float(np.clip(safety_margin, 0.0, 1.0)))
            * (0.25 + 0.75 * float(np.clip(disagreement_score, 0.0, 1.0))),
            0.0,
            1.0,
        )
    )


def _estimate_dispatch_proxy_np(
    *,
    observation: Mapping[str, float],
    action_exec: np.ndarray,
    action_index: Mapping[str, int],
    env_config: EnvConfig,
    gt_off_deadband_ratio: float,
) -> dict[str, float]:
    def _action_value(key: str, default: float = 0.0) -> float:
        if key not in action_index:
            return float(default)
        return _safe_float(action_exec[int(action_index[key])], default=default)

    dt_hours = max(_NORM_EPS, float(env_config.dt_hours))
    p_gt_cap_mw = max(_NORM_EPS, float(env_config.p_gt_cap_mw))
    q_boiler_cap_mw = max(_NORM_EPS, float(env_config.q_boiler_cap_mw))
    p_bes_cap_mw = max(_NORM_EPS, float(env_config.p_bes_cap_mw))
    q_ech_cap_mw = max(_NORM_EPS, float(env_config.q_ech_cap_mw))

    p_dem_mw = max(0.0, _safe_float(observation.get("p_dem_mw", 0.0)))
    pv_mw = max(0.0, _safe_float(observation.get("pv_mw", 0.0)))
    wt_mw = max(0.0, _safe_float(observation.get("wt_mw", 0.0)))
    price_e = max(0.0, _safe_float(observation.get("price_e", 0.0)))
    price_gas = max(0.0, _safe_float(observation.get("price_gas", 0.0)))
    t_amb_k = _safe_float(observation.get("t_amb_k", 298.15), default=298.15)
    p_gt_prev_mw = max(0.0, _safe_float(observation.get("p_gt_prev_mw", 0.0)))
    abs_margin_k = _safe_float(observation.get("abs_drive_margin_k", 0.0))
    u_abs_exec = float(np.clip(_action_value("u_abs"), 0.0, 1.0))

    u_gt_exec = float(np.clip(_action_value("u_gt"), -1.0, 1.0))
    p_gt_exec_mw = ((u_gt_exec + 1.0) * 0.5) * p_gt_cap_mw
    gt_load_ratio = float(np.clip(p_gt_exec_mw / p_gt_cap_mw, 0.0, 1.0))
    eta_gt = float(env_config.gt_eta_min) + (
        float(env_config.gt_eta_max) - float(env_config.gt_eta_min)
    ) * gt_load_ratio
    gt_fuel_cost = dt_hours * (p_gt_exec_mw / max(_NORM_EPS, eta_gt)) * price_gas
    gt_var_om_cost = dt_hours * p_gt_exec_mw * max(0.0, float(env_config.gt_om_var_cost_per_mwh))

    off_deadband_mw = _gt_off_deadband_mw(
        gt_min_output_mw=float(env_config.gt_min_output_mw),
        gt_off_deadband_ratio=float(gt_off_deadband_ratio),
    )
    gt_started = (
        p_gt_prev_mw <= (off_deadband_mw + _NORM_EPS)
        and p_gt_exec_mw > (off_deadband_mw + _NORM_EPS)
    )
    gt_start_cycle_cost = (
        float(env_config.gt_start_cost) + float(env_config.gt_cycle_cost)
    ) if gt_started else 0.0
    gt_delta_cost = (
        dt_hours
        * abs(p_gt_exec_mw - p_gt_prev_mw)
        * max(0.0, float(env_config.gt_om_var_cost_per_mwh))
    )

    u_bes_exec = float(np.clip(_action_value("u_bes"), -1.0, 1.0))
    p_bes_charge_mw = (
        max(0.0, -u_bes_exec)
        * p_bes_cap_mw
        / max(_NORM_EPS, float(env_config.bes_eta_charge))
    )
    p_bes_discharge_mw = (
        max(0.0, u_bes_exec)
        * p_bes_cap_mw
        * max(_NORM_EPS, float(env_config.bes_eta_discharge))
    )

    u_ech_exec = float(np.clip(_action_value("u_ech"), 0.0, 1.0))
    q_ech_proxy_mw = u_ech_exec * q_ech_cap_mw
    ech_cop_base = float(
        np.clip(
            float(env_config.cop_nominal) - 0.03 * (t_amb_k - 298.15),
            float(env_config.cop_nominal) * float(env_config.ech_cop_partload_min_fraction),
            float(env_config.cop_nominal),
        )
    )
    p_ech_proxy_mw = q_ech_proxy_mw / max(_NORM_EPS, ech_cop_base)

    u_boiler_exec = float(np.clip(_action_value("u_boiler"), 0.0, 1.0))
    q_boiler_exec_mw = u_boiler_exec * q_boiler_cap_mw
    boiler_cost = dt_hours * q_boiler_exec_mw * price_gas

    net_grid_mw = (
        p_dem_mw
        + p_ech_proxy_mw
        + p_bes_charge_mw
        - pv_mw
        - wt_mw
        - p_bes_discharge_mw
        - p_gt_exec_mw
    )
    grid_import_mw = max(0.0, net_grid_mw)
    grid_export_mw = max(0.0, -net_grid_mw)
    export_price = price_e * max(0.0, float(env_config.sell_price_ratio))
    if float(env_config.sell_price_cap_per_mwh) > 0.0:
        export_price = min(export_price, float(env_config.sell_price_cap_per_mwh))
    grid_cost = dt_hours * (grid_import_mw * price_e - grid_export_mw * export_price)

    abs_invalid_risk = float(
        np.clip(
            max(0.0, -abs_margin_k / max(_NORM_EPS, float(env_config.abs_gate_scale_k))) * u_abs_exec,
            0.0,
            1.0,
        )
    )
    return {
        "proxy_cost": float(
            grid_cost
            + gt_fuel_cost
            + gt_var_om_cost
            + gt_start_cycle_cost
            + gt_delta_cost
            + boiler_cost
        ),
        "grid_cost": float(grid_cost),
        "gt_fuel_cost": float(gt_fuel_cost),
        "gt_var_om_cost": float(gt_var_om_cost),
        "gt_start_cycle_cost": float(gt_start_cycle_cost),
        "gt_delta_cost": float(gt_delta_cost),
        "boiler_cost": float(boiler_cost),
        "abs_invalid_risk": float(abs_invalid_risk),
    }


def _economic_teacher_gate_decision_np(
    *,
    safe_reference_available: bool,
    proxy_advantage_ratio: float,
    abs_risk_gap: float,
    projection_gap: float,
    min_proxy_advantage_ratio: float,
    max_safe_abs_risk_gap: float,
    max_projection_gap: float,
) -> dict[str, Any]:
    proxy_advantage_ratio = float(np.clip(proxy_advantage_ratio, -1.0, 1.0))
    abs_risk_gap = float(abs_risk_gap)
    projection_gap = float(max(0.0, projection_gap))
    reasons: list[str] = []
    if projection_gap > float(max_projection_gap):
        reasons.append("projection_gap_high")
    if safe_reference_available and proxy_advantage_ratio < float(min_proxy_advantage_ratio):
        reasons.append("proxy_advantage_low")
    if safe_reference_available and abs_risk_gap > float(max_safe_abs_risk_gap):
        reasons.append("abs_risk_high")
    return {
        "accepted": len(reasons) == 0,
        "reasons": reasons,
        "safe_reference_available": bool(safe_reference_available),
        "proxy_advantage_ratio": float(proxy_advantage_ratio),
        "abs_risk_gap": float(abs_risk_gap),
        "projection_gap": float(projection_gap),
    }


def _economic_teacher_projection_gap_np(
    *,
    action_raw: np.ndarray,
    action_exec: np.ndarray,
    action_index: Mapping[str, int],
    teacher_mask: np.ndarray | None = None,
) -> float:
    if teacher_mask is not None:
        mask = np.asarray(teacher_mask, dtype=np.float32).reshape(-1)
        target_indices = [int(idx) for idx in np.flatnonzero(mask > 0.5).tolist()]
    else:
        target_indices = []
    if not target_indices:
        target_indices = [
            int(action_index[key])
            for key in ("u_gt", "u_bes", "u_tes")
            if key in action_index
        ]
    if not target_indices:
        return float(np.abs(action_exec - action_raw).mean())
    target_raw = np.asarray(action_raw, dtype=np.float32)[target_indices]
    target_exec = np.asarray(action_exec, dtype=np.float32)[target_indices]
    return float(np.abs(target_exec - target_raw).mean())


def _economic_teacher_target_mask_np(
    *,
    action_dim: int,
    action_index: Mapping[str, int],
    keys: Sequence[str] | None = None,
) -> np.ndarray:
    mask = np.zeros((int(action_dim),), dtype=np.float32)
    target_keys = tuple(keys) if keys is not None else ("u_gt", "u_bes", "u_tes")
    for key in target_keys:
        if key in action_index:
            mask[int(action_index[key])] = 1.0
    return mask


def _build_mixed_economic_teacher_target_np(
    *,
    observation: Mapping[str, float],
    safe_action_exec: np.ndarray,
    safe_proxy: Mapping[str, float],
    economic_action_raw: np.ndarray,
    economic_action_exec: np.ndarray,
    action_index: Mapping[str, int],
    env_config: EnvConfig,
    gt_off_deadband_ratio: float,
    min_proxy_advantage_ratio: float,
    gt_proxy_advantage_ratio_min: float,
    max_safe_abs_risk_gap: float,
    max_projection_gap: float,
    gt_projection_gap_max: float,
    bes_proxy_advantage_ratio_min: float,
    bes_price_low_threshold: float,
    bes_price_high_threshold: float,
    bes_charge_soc_ceiling: float,
    bes_discharge_soc_floor: float,
    bes_soc_min: float,
    bes_soc_max: float,
    bes_charge_u: float,
    bes_discharge_u: float,
    bes_price_opportunity_min: float,
    gt_abs_margin_guard_k: float,
    gt_qc_ratio_guard: float,
    gt_heat_backup_ratio_guard: float,
) -> dict[str, Any]:
    mixed_target = np.asarray(safe_action_exec, dtype=np.float32).copy()
    mixed_proxy = dict(safe_proxy)
    safe_abs_risk = float(safe_proxy.get("abs_invalid_risk", 0.0))
    mixed_mask = np.zeros_like(mixed_target, dtype=np.float32)
    swapped_dims: list[str] = []
    swapped_projection_gaps: list[float] = []
    abs_margin_k = _safe_float(observation.get("abs_drive_margin_k", 0.0))
    qc_dem_mw = max(0.0, _safe_float(observation.get("qc_dem_mw", 0.0)))
    heat_backup_min_needed_mw = max(0.0, _safe_float(observation.get("heat_backup_min_needed_mw", 0.0)))
    q_total_cooling_cap_mw = max(
        _NORM_EPS,
        _safe_float(getattr(env_config, "q_abs_cool_cap_mw", 0.0))
        + _safe_float(getattr(env_config, "q_ech_cap_mw", 0.0)),
    )
    qc_ratio = qc_dem_mw / q_total_cooling_cap_mw
    heat_backup_ratio = heat_backup_min_needed_mw / max(
        _NORM_EPS,
        _safe_float(getattr(env_config, "q_boiler_cap_mw", 0.0)),
    )
    bes_prior = None
    if "u_bes" in action_index:
        bes_prior = _bes_price_prior_target_np(
            price_e=_safe_float(observation.get("price_e", 0.0)),
            soc_bes=_safe_float(observation.get("soc_bes", 0.0)),
            price_low_threshold=float(bes_price_low_threshold),
            price_high_threshold=float(bes_price_high_threshold),
            charge_soc_ceiling=float(bes_charge_soc_ceiling),
            discharge_soc_floor=float(bes_discharge_soc_floor),
            bes_soc_min=float(bes_soc_min),
            bes_soc_max=float(bes_soc_max),
            charge_u=float(bes_charge_u),
            discharge_u=float(bes_discharge_u),
        )

    for key in ("u_gt", "u_bes", "u_tes"):
        if key not in action_index:
            continue
        dim_index = int(action_index[key])
        economic_value = float(economic_action_exec[dim_index])
        safe_value = float(mixed_target[dim_index])
        if abs(economic_value - safe_value) <= _NORM_EPS:
            continue
        if key == "u_gt":
            if (
                abs_margin_k <= float(gt_abs_margin_guard_k)
                or qc_ratio >= float(gt_qc_ratio_guard)
                or heat_backup_ratio >= float(gt_heat_backup_ratio_guard)
            ):
                continue
        if key == "u_tes":
            if (
                qc_ratio >= float(gt_qc_ratio_guard)
                or heat_backup_ratio >= float(gt_heat_backup_ratio_guard)
            ):
                continue
        local_min_proxy_advantage = float(min_proxy_advantage_ratio)
        local_projection_gap_max = float(max_projection_gap)
        bes_charge_investment_mode = False
        if key == "u_gt":
            local_min_proxy_advantage = min(
                float(min_proxy_advantage_ratio),
                float(max(0.0, gt_proxy_advantage_ratio_min)),
            )
            local_projection_gap_max = max(
                float(max_projection_gap),
                float(gt_projection_gap_max),
            )
        if key == "u_bes":
            local_min_proxy_advantage = min(
                float(min_proxy_advantage_ratio),
                float(max(0.0, bes_proxy_advantage_ratio_min)),
            )
            if bes_prior is not None and float(bes_prior.get("opportunity", 0.0)) >= float(
                bes_price_opportunity_min
            ):
                teacher_target_u = float(bes_prior.get("target_u_bes", 0.0))
                teacher_sign = 0.0 if abs(teacher_target_u) <= _NORM_EPS else float(np.sign(teacher_target_u))
                economic_sign = 0.0 if abs(economic_value) <= _NORM_EPS else float(np.sign(economic_value))
                if teacher_sign != 0.0 and economic_sign != 0.0 and teacher_sign != economic_sign:
                    continue
                bes_charge_investment_mode = bool(
                    str(bes_prior.get("mode", "")) == "charge"
                    and teacher_sign < 0.0
                    and economic_sign < 0.0
                )
        dim_projection_gap = abs(
            float(economic_action_exec[dim_index]) - float(economic_action_raw[dim_index])
        )
        if dim_projection_gap > float(local_projection_gap_max):
            continue
        candidate_target = mixed_target.copy()
        candidate_target[dim_index] = economic_value
        candidate_proxy = _estimate_dispatch_proxy_np(
            observation=observation,
            action_exec=candidate_target,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=float(gt_off_deadband_ratio),
        )
        candidate_advantage_ratio = float(
            np.clip(
                (float(mixed_proxy.get("proxy_cost", 0.0)) - float(candidate_proxy["proxy_cost"]))
                / max(1.0, abs(float(mixed_proxy.get("proxy_cost", 0.0)))),
                -1.0,
                1.0,
            )
        )
        candidate_abs_risk_gap = float(candidate_proxy["abs_invalid_risk"]) - safe_abs_risk
        if (not bes_charge_investment_mode) and candidate_advantage_ratio < float(local_min_proxy_advantage):
            continue
        if candidate_abs_risk_gap > float(max_safe_abs_risk_gap):
            continue
        mixed_target = candidate_target
        mixed_proxy = candidate_proxy
        mixed_mask[dim_index] = 1.0
        swapped_dims.append(str(key))
        swapped_projection_gaps.append(float(dim_projection_gap))

    return {
        "target": mixed_target.astype(np.float32, copy=False),
        "mask": mixed_mask.astype(np.float32, copy=False),
        "swapped_dims": list(swapped_dims),
        "swapped_dim_count": int(len(swapped_dims)),
        "projection_gap": float(np.mean(swapped_projection_gaps)) if swapped_projection_gaps else 0.0,
        "proxy_cost": float(mixed_proxy.get("proxy_cost", 0.0)),
        "abs_invalid_risk": float(mixed_proxy.get("abs_invalid_risk", 0.0)),
    }


def _build_observation_norm(
    *,
    feature_keys: Sequence[str],
    train_statistics: Mapping[str, Any],
    env_config: EnvConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    stats = dict(train_statistics.get("stats", {}) or {})
    offsets: list[float] = []
    scales: list[float] = []
    for key in feature_keys:
        offset = 0.0
        scale = 1.0
        if key in stats:
            entry = dict(stats.get(key, {}) or {})
            offset = float(entry.get("mean", 0.0))
            scale = float(entry.get("std", 1.0))
        elif key == "soc_bes":
            low = float(env_config.bes_soc_min)
            high = float(env_config.bes_soc_max)
            offset = 0.5 * (low + high)
            scale = 0.5 * max(_NORM_EPS, high - low)
        elif key == "gt_on":
            offset = 0.5
            scale = 0.5
        elif key == "gt_state":
            offset = 1.0
            scale = 1.0
        elif key == "p_gt_prev_mw":
            cap = float(env_config.p_gt_cap_mw)
            offset = 0.5 * cap
            scale = 0.5 * max(_NORM_EPS, cap)
        elif key in {"gt_ramp_headroom_up_mw", "gt_ramp_headroom_down_mw"}:
            ramp = float(env_config.gt_ramp_mw_per_step)
            offset = 0.5 * ramp
            scale = 0.5 * max(_NORM_EPS, ramp)
        elif key == "e_tes_mwh":
            cap = float(env_config.e_tes_cap_mwh)
            offset = 0.5 * cap
            scale = 0.5 * max(_NORM_EPS, cap)
        elif key == "t_tes_hot_k":
            offset = float(env_config.hrsg_water_inlet_k) + 30.0
            scale = 30.0
        elif key == "abs_drive_margin_k":
            scale = max(2.0, float(env_config.abs_gate_scale_k))
        elif key == "q_hrsg_est_now_mw":
            cap = float(env_config.q_boiler_cap_mw) + float(env_config.q_tes_discharge_cap_mw)
            offset = 0.5 * cap
            scale = 0.5 * max(_NORM_EPS, cap)
        elif key == "q_tes_discharge_feasible_mw":
            cap = float(env_config.q_tes_discharge_cap_mw)
            offset = 0.5 * cap
            scale = 0.5 * max(_NORM_EPS, cap)
        elif key in {"heat_deficit_if_boiler_off_mw", "heat_backup_min_needed_mw"}:
            cap = float(env_config.q_boiler_cap_mw)
            offset = 0.5 * cap
            scale = 0.5 * max(_NORM_EPS, cap)
        if (not np.isfinite(scale)) or abs(scale) < _NORM_EPS:
            scale = 1.0
        offsets.append(float(offset))
        scales.append(float(scale))
    return (
        {
            "kind": "affine_v1",
            "observation_feature_keys": list(feature_keys),
            "offset": list(offsets),
            "scale": list(scales),
        },
        np.asarray(offsets, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
    )


@dataclass(slots=True)
class PAFCTD3TrainConfig:
    projection_surrogate_checkpoint_path: str | Path
    episode_days: int = 7
    total_env_steps: int = 4096
    warmup_steps: int = 512
    replay_capacity: int = 50_000
    batch_size: int = 128
    updates_per_step: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    dual_lr: float = 5e-3
    dual_warmup_steps: int = 1024
    actor_delay: int = 2
    exploration_noise_std: float = 0.08
    target_policy_noise_std: float = 0.08
    target_noise_clip: float = 0.15
    gap_penalty_coef: float = 0.5
    exec_action_anchor_coef: float = 5.0
    exec_action_anchor_safe_floor: float = 0.2
    gt_off_deadband_ratio: float = 0.0
    abs_ready_focus_coef: float = 0.0
    invalid_abs_penalty_coef: float = 0.0
    economic_boiler_proxy_coef: float = 0.0
    economic_abs_tradeoff_coef: float = 0.0
    economic_gt_grid_proxy_coef: float = 0.25
    economic_teacher_distill_coef: float = 0.0
    economic_teacher_proxy_advantage_min: float = 0.02
    economic_teacher_gt_proxy_advantage_min: float = 0.01
    economic_teacher_bes_proxy_advantage_min: float = 0.002
    economic_teacher_max_safe_abs_risk_gap: float = 0.05
    economic_teacher_projection_gap_max: float = 0.20
    economic_teacher_gt_projection_gap_max: float = 1.0
    economic_teacher_bes_price_opportunity_min: float = 0.10
    economic_teacher_bes_anchor_preserve_scale: float = 0.85
    economic_teacher_warm_start_weight: float = 4.0
    economic_teacher_prefill_replay_boost: int = 2
    economic_teacher_gt_action_weight: float = 2.0
    economic_teacher_bes_action_weight: float = 1.5
    economic_teacher_tes_action_weight: float = 0.5
    economic_teacher_full_year_warm_start_samples: int = 4096
    economic_teacher_full_year_warm_start_epochs: int = 4
    economic_bes_distill_coef: float = 0.0
    economic_bes_prior_u: float = 0.35
    economic_bes_charge_u_scale: float = 1.8
    economic_bes_discharge_u_scale: float = 1.0
    economic_bes_charge_weight: float = 2.0
    economic_bes_discharge_weight: float = 1.0
    economic_bes_charge_pressure_bonus: float = 1.0
    economic_bes_charge_soc_ceiling: float = 0.75
    economic_bes_discharge_soc_floor: float = 0.35
    economic_bes_full_year_warm_start_samples: int = 4096
    economic_bes_full_year_warm_start_epochs: int = 2
    economic_bes_full_year_warm_start_u_weight: float = 4.0
    economic_bes_teacher_selection_priority_boost: float = 0.75
    economic_bes_economic_source_priority_bonus: float = 0.10
    economic_bes_economic_source_min_share: float = 0.75
    economic_bes_idle_economic_source_min_share: float = 0.75
    economic_bes_teacher_target_min_share: float = 0.0
    state_feasible_action_shaping_enabled: bool = False
    abs_min_on_gate_th: float = 0.75
    abs_min_on_u_margin: float = 0.02
    expert_prefill_policy: str = "easy_rule_abs"
    expert_prefill_checkpoint_path: str | Path = ""
    expert_prefill_economic_policy: str = "checkpoint"
    expert_prefill_economic_checkpoint_path: str | Path = ""
    expert_prefill_steps: int = 1024
    actor_warm_start_epochs: int = 2
    actor_warm_start_batch_size: int = 256
    actor_warm_start_lr: float = 1e-4
    expert_prefill_cooling_bias: float = 0.5
    expert_prefill_abs_replay_boost: int = 0
    expert_prefill_abs_exec_threshold: float = 0.05
    expert_prefill_abs_window_mining_candidates: int = 4
    dual_abs_margin_k: float = 1.25
    dual_qc_ratio_th: float = 0.55
    dual_heat_backup_ratio_th: float = 0.10
    dual_safe_abs_u_th: float = 0.60
    checkpoint_interval_steps: int = 0
    eval_window_pool_size: int = 12
    eval_window_count: int = 4
    best_gate_enabled: bool = True
    best_gate_electric_min: float = 1.0
    best_gate_heat_min: float = 0.99
    best_gate_cool_min: float = 0.99
    plateau_control_enabled: bool = False
    plateau_patience_evals: int = 2
    plateau_lr_decay_factor: float = 0.5
    plateau_min_actor_lr: float = 5e-5
    plateau_min_critic_lr: float = 1e-4
    plateau_early_stop_patience_evals: int = 2
    hidden_dims: tuple[int, ...] = (256, 256)
    seed: int = 42
    device: str = "auto"
    observation_keys: tuple[str, ...] = field(default_factory=tuple)
    action_keys: tuple[str, ...] = field(default_factory=tuple)
    cost_targets: tuple[float, float, float] = (0.0, 0.01, 0.01)

    def __post_init__(self) -> None:
        self.projection_surrogate_checkpoint_path = str(
            Path(self.projection_surrogate_checkpoint_path)
        )
        self.episode_days = int(self.episode_days)
        self.total_env_steps = int(self.total_env_steps)
        self.warmup_steps = int(self.warmup_steps)
        self.replay_capacity = int(self.replay_capacity)
        self.batch_size = int(self.batch_size)
        self.updates_per_step = int(self.updates_per_step)
        self.gamma = float(self.gamma)
        self.tau = float(self.tau)
        self.actor_lr = float(self.actor_lr)
        self.critic_lr = float(self.critic_lr)
        self.dual_lr = float(self.dual_lr)
        self.dual_warmup_steps = int(self.dual_warmup_steps)
        self.actor_delay = int(self.actor_delay)
        self.exploration_noise_std = float(self.exploration_noise_std)
        self.target_policy_noise_std = float(self.target_policy_noise_std)
        self.target_noise_clip = float(self.target_noise_clip)
        self.gap_penalty_coef = float(self.gap_penalty_coef)
        self.exec_action_anchor_coef = float(self.exec_action_anchor_coef)
        self.exec_action_anchor_safe_floor = float(self.exec_action_anchor_safe_floor)
        self.gt_off_deadband_ratio = float(self.gt_off_deadband_ratio)
        self.abs_ready_focus_coef = float(self.abs_ready_focus_coef)
        self.invalid_abs_penalty_coef = float(self.invalid_abs_penalty_coef)
        self.economic_boiler_proxy_coef = float(self.economic_boiler_proxy_coef)
        self.economic_abs_tradeoff_coef = float(self.economic_abs_tradeoff_coef)
        self.economic_gt_grid_proxy_coef = float(self.economic_gt_grid_proxy_coef)
        self.economic_teacher_distill_coef = float(self.economic_teacher_distill_coef)
        self.economic_teacher_proxy_advantage_min = float(
            self.economic_teacher_proxy_advantage_min
        )
        self.economic_teacher_gt_proxy_advantage_min = float(
            self.economic_teacher_gt_proxy_advantage_min
        )
        self.economic_teacher_bes_proxy_advantage_min = float(
            self.economic_teacher_bes_proxy_advantage_min
        )
        self.economic_teacher_max_safe_abs_risk_gap = float(
            self.economic_teacher_max_safe_abs_risk_gap
        )
        self.economic_teacher_projection_gap_max = float(
            self.economic_teacher_projection_gap_max
        )
        self.economic_teacher_gt_projection_gap_max = float(
            self.economic_teacher_gt_projection_gap_max
        )
        self.economic_teacher_bes_price_opportunity_min = float(
            self.economic_teacher_bes_price_opportunity_min
        )
        self.economic_teacher_bes_anchor_preserve_scale = float(
            self.economic_teacher_bes_anchor_preserve_scale
        )
        self.economic_teacher_warm_start_weight = float(
            self.economic_teacher_warm_start_weight
        )
        self.economic_teacher_prefill_replay_boost = int(
            self.economic_teacher_prefill_replay_boost
        )
        self.economic_teacher_gt_action_weight = float(
            self.economic_teacher_gt_action_weight
        )
        self.economic_teacher_bes_action_weight = float(
            self.economic_teacher_bes_action_weight
        )
        self.economic_teacher_tes_action_weight = float(
            self.economic_teacher_tes_action_weight
        )
        self.economic_teacher_full_year_warm_start_samples = int(
            self.economic_teacher_full_year_warm_start_samples
        )
        self.economic_teacher_full_year_warm_start_epochs = int(
            self.economic_teacher_full_year_warm_start_epochs
        )
        self.economic_bes_distill_coef = float(self.economic_bes_distill_coef)
        self.economic_bes_prior_u = float(self.economic_bes_prior_u)
        self.economic_bes_charge_u_scale = float(self.economic_bes_charge_u_scale)
        self.economic_bes_discharge_u_scale = float(self.economic_bes_discharge_u_scale)
        self.economic_bes_charge_weight = float(self.economic_bes_charge_weight)
        self.economic_bes_discharge_weight = float(self.economic_bes_discharge_weight)
        self.economic_bes_charge_pressure_bonus = float(self.economic_bes_charge_pressure_bonus)
        self.economic_bes_charge_soc_ceiling = float(self.economic_bes_charge_soc_ceiling)
        self.economic_bes_discharge_soc_floor = float(self.economic_bes_discharge_soc_floor)
        self.economic_bes_full_year_warm_start_samples = int(
            self.economic_bes_full_year_warm_start_samples
        )
        self.economic_bes_full_year_warm_start_epochs = int(
            self.economic_bes_full_year_warm_start_epochs
        )
        self.economic_bes_full_year_warm_start_u_weight = float(
            self.economic_bes_full_year_warm_start_u_weight
        )
        self.economic_bes_teacher_selection_priority_boost = float(
            self.economic_bes_teacher_selection_priority_boost
        )
        self.economic_bes_economic_source_priority_bonus = float(
            self.economic_bes_economic_source_priority_bonus
        )
        self.economic_bes_economic_source_min_share = float(
            self.economic_bes_economic_source_min_share
        )
        self.economic_bes_idle_economic_source_min_share = float(
            self.economic_bes_idle_economic_source_min_share
        )
        self.economic_bes_teacher_target_min_share = float(
            self.economic_bes_teacher_target_min_share
        )
        self.state_feasible_action_shaping_enabled = bool(self.state_feasible_action_shaping_enabled)
        self.abs_min_on_gate_th = float(self.abs_min_on_gate_th)
        self.abs_min_on_u_margin = float(self.abs_min_on_u_margin)
        self.expert_prefill_policy = str(self.expert_prefill_policy).strip().lower().replace("-", "_")
        self.expert_prefill_checkpoint_path = str(self.expert_prefill_checkpoint_path).strip()
        self.expert_prefill_economic_policy = (
            str(self.expert_prefill_economic_policy).strip().lower().replace("-", "_")
        )
        self.expert_prefill_economic_checkpoint_path = str(
            self.expert_prefill_economic_checkpoint_path
        ).strip()
        self.expert_prefill_steps = int(self.expert_prefill_steps)
        self.actor_warm_start_epochs = int(self.actor_warm_start_epochs)
        self.actor_warm_start_batch_size = int(self.actor_warm_start_batch_size)
        self.actor_warm_start_lr = float(self.actor_warm_start_lr)
        self.expert_prefill_cooling_bias = float(self.expert_prefill_cooling_bias)
        self.expert_prefill_abs_replay_boost = int(self.expert_prefill_abs_replay_boost)
        self.expert_prefill_abs_exec_threshold = float(self.expert_prefill_abs_exec_threshold)
        self.expert_prefill_abs_window_mining_candidates = int(self.expert_prefill_abs_window_mining_candidates)
        self.dual_abs_margin_k = float(self.dual_abs_margin_k)
        self.dual_qc_ratio_th = float(self.dual_qc_ratio_th)
        self.dual_heat_backup_ratio_th = float(self.dual_heat_backup_ratio_th)
        self.dual_safe_abs_u_th = float(self.dual_safe_abs_u_th)
        self.checkpoint_interval_steps = int(self.checkpoint_interval_steps)
        self.eval_window_pool_size = int(self.eval_window_pool_size)
        self.eval_window_count = int(self.eval_window_count)
        self.best_gate_enabled = bool(self.best_gate_enabled)
        self.best_gate_electric_min = float(self.best_gate_electric_min)
        self.best_gate_heat_min = float(self.best_gate_heat_min)
        self.best_gate_cool_min = float(self.best_gate_cool_min)
        self.plateau_control_enabled = bool(self.plateau_control_enabled)
        self.plateau_patience_evals = int(self.plateau_patience_evals)
        self.plateau_lr_decay_factor = float(self.plateau_lr_decay_factor)
        self.plateau_min_actor_lr = float(self.plateau_min_actor_lr)
        self.plateau_min_critic_lr = float(self.plateau_min_critic_lr)
        self.plateau_early_stop_patience_evals = int(self.plateau_early_stop_patience_evals)
        self.seed = int(self.seed)
        self.observation_keys = tuple(self.observation_keys) or DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS
        self.action_keys = tuple(self.action_keys) or DEFAULT_SEQUENCE_ACTION_KEYS
        self.hidden_dims = tuple(int(dim) for dim in self.hidden_dims)
        self.cost_targets = tuple(float(value) for value in self.cost_targets)
        if not Path(self.projection_surrogate_checkpoint_path).exists():
            raise FileNotFoundError(
                f"projection surrogate checkpoint 不存在: {self.projection_surrogate_checkpoint_path}"
            )
        if self.episode_days < 7 or self.episode_days > 30:
            raise ValueError("episode_days 必须在 [7,30]。")
        if self.total_env_steps <= 0:
            raise ValueError("total_env_steps 必须 > 0。")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps 必须 >= 0。")
        if self.replay_capacity <= 1:
            raise ValueError("replay_capacity 必须 > 1。")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须 > 0。")
        if self.updates_per_step <= 0:
            raise ValueError("updates_per_step 必须 > 0。")
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError("gamma 必须在 [0,1]。")
        if self.tau <= 0.0 or self.tau > 1.0:
            raise ValueError("tau 必须在 (0,1]。")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError("actor_lr / critic_lr 必须 > 0。")
        if self.dual_lr < 0.0:
            raise ValueError("dual_lr 必须 >= 0。")
        if self.dual_warmup_steps < 0:
            raise ValueError("dual_warmup_steps 必须 >= 0。")
        if self.exec_action_anchor_coef < 0.0:
            raise ValueError("exec_action_anchor_coef 必须 >= 0。")
        if self.exec_action_anchor_safe_floor < 0.0 or self.exec_action_anchor_safe_floor > 1.0:
            raise ValueError("exec_action_anchor_safe_floor 必须在 [0,1]。")
        if self.gt_off_deadband_ratio < 0.0 or self.gt_off_deadband_ratio > 1.0:
            raise ValueError("gt_off_deadband_ratio 必须在 [0,1]。")
        if self.abs_ready_focus_coef < 0.0:
            raise ValueError("abs_ready_focus_coef 必须 >= 0。")
        if self.invalid_abs_penalty_coef < 0.0:
            raise ValueError("invalid_abs_penalty_coef 必须 >= 0。")
        if self.economic_boiler_proxy_coef < 0.0:
            raise ValueError("economic_boiler_proxy_coef 必须 >= 0。")
        if self.economic_abs_tradeoff_coef < 0.0:
            raise ValueError("economic_abs_tradeoff_coef 必须 >= 0。")
        if self.economic_gt_grid_proxy_coef < 0.0:
            raise ValueError("economic_gt_grid_proxy_coef 必须 >= 0。")
        if self.economic_teacher_distill_coef < 0.0:
            raise ValueError("economic_teacher_distill_coef 必须 >= 0。")
        if (
            self.economic_teacher_proxy_advantage_min < 0.0
            or self.economic_teacher_proxy_advantage_min > 1.0
        ):
            raise ValueError("economic_teacher_proxy_advantage_min 必须在 [0,1]。")
        if (
            self.economic_teacher_gt_proxy_advantage_min < 0.0
            or self.economic_teacher_gt_proxy_advantage_min > 1.0
        ):
            raise ValueError("economic_teacher_gt_proxy_advantage_min 必须在 [0,1]。")
        if (
            self.economic_teacher_bes_proxy_advantage_min < 0.0
            or self.economic_teacher_bes_proxy_advantage_min > 1.0
        ):
            raise ValueError("economic_teacher_bes_proxy_advantage_min 必须在 [0,1]。")
        if (
            self.economic_teacher_max_safe_abs_risk_gap < 0.0
            or self.economic_teacher_max_safe_abs_risk_gap > 1.0
        ):
            raise ValueError("economic_teacher_max_safe_abs_risk_gap 必须在 [0,1]。")
        if self.economic_teacher_projection_gap_max < 0.0:
            raise ValueError("economic_teacher_projection_gap_max 必须 >= 0。")
        if self.economic_teacher_gt_projection_gap_max < 0.0:
            raise ValueError("economic_teacher_gt_projection_gap_max 必须 >= 0。")
        if (
            self.economic_teacher_bes_price_opportunity_min < 0.0
            or self.economic_teacher_bes_price_opportunity_min > 1.0
        ):
            raise ValueError("economic_teacher_bes_price_opportunity_min 必须在 [0,1]。")
        if (
            self.economic_teacher_bes_anchor_preserve_scale < 0.0
            or self.economic_teacher_bes_anchor_preserve_scale > 1.0
        ):
            raise ValueError("economic_teacher_bes_anchor_preserve_scale 必须在 [0,1]。")
        if self.economic_teacher_warm_start_weight < 0.0:
            raise ValueError("economic_teacher_warm_start_weight 必须 >= 0。")
        if self.economic_teacher_prefill_replay_boost < 0:
            raise ValueError("economic_teacher_prefill_replay_boost 必须 >= 0。")
        if self.economic_teacher_gt_action_weight < 0.0:
            raise ValueError("economic_teacher_gt_action_weight 必须 >= 0。")
        if self.economic_teacher_bes_action_weight < 0.0:
            raise ValueError("economic_teacher_bes_action_weight 必须 >= 0。")
        if self.economic_teacher_tes_action_weight < 0.0:
            raise ValueError("economic_teacher_tes_action_weight 必须 >= 0。")
        if self.economic_teacher_full_year_warm_start_samples < 0:
            raise ValueError("economic_teacher_full_year_warm_start_samples 必须 >= 0。")
        if self.economic_teacher_full_year_warm_start_epochs < 0:
            raise ValueError("economic_teacher_full_year_warm_start_epochs 必须 >= 0。")
        if self.economic_bes_distill_coef < 0.0:
            raise ValueError("economic_bes_distill_coef 必须 >= 0。")
        if self.economic_bes_prior_u < 0.0 or self.economic_bes_prior_u > 1.0:
            raise ValueError("economic_bes_prior_u 必须在 [0,1]。")
        if self.economic_bes_charge_u_scale < 0.0:
            raise ValueError("economic_bes_charge_u_scale 必须 >= 0。")
        if self.economic_bes_discharge_u_scale < 0.0:
            raise ValueError("economic_bes_discharge_u_scale 必须 >= 0。")
        if self.economic_bes_charge_weight < 0.0:
            raise ValueError("economic_bes_charge_weight 必须 >= 0。")
        if self.economic_bes_discharge_weight < 0.0:
            raise ValueError("economic_bes_discharge_weight 必须 >= 0。")
        if self.economic_bes_charge_pressure_bonus < 0.0:
            raise ValueError("economic_bes_charge_pressure_bonus 必须 >= 0。")
        if self.economic_bes_charge_soc_ceiling < 0.0 or self.economic_bes_charge_soc_ceiling > 1.0:
            raise ValueError("economic_bes_charge_soc_ceiling 必须在 [0,1]。")
        if self.economic_bes_discharge_soc_floor < 0.0 or self.economic_bes_discharge_soc_floor > 1.0:
            raise ValueError("economic_bes_discharge_soc_floor 必须在 [0,1]。")
        if self.economic_bes_discharge_soc_floor >= self.economic_bes_charge_soc_ceiling:
            raise ValueError(
                "economic_bes_discharge_soc_floor 必须小于 economic_bes_charge_soc_ceiling。"
            )
        if self.economic_bes_full_year_warm_start_samples < 0:
            raise ValueError("economic_bes_full_year_warm_start_samples 必须 >= 0。")
        if self.economic_bes_full_year_warm_start_epochs < 0:
            raise ValueError("economic_bes_full_year_warm_start_epochs 必须 >= 0。")
        if self.economic_bes_full_year_warm_start_u_weight < 0.0:
            raise ValueError("economic_bes_full_year_warm_start_u_weight 必须 >= 0。")
        if self.economic_bes_teacher_selection_priority_boost < 0.0:
            raise ValueError("economic_bes_teacher_selection_priority_boost 必须 >= 0。")
        if self.economic_bes_economic_source_priority_bonus < 0.0:
            raise ValueError("economic_bes_economic_source_priority_bonus 必须 >= 0。")
        if (
            self.economic_bes_economic_source_min_share < 0.0
            or self.economic_bes_economic_source_min_share > 1.0
        ):
            raise ValueError("economic_bes_economic_source_min_share 必须在 [0,1]。")
        if (
            self.economic_bes_idle_economic_source_min_share < 0.0
            or self.economic_bes_idle_economic_source_min_share > 1.0
        ):
            raise ValueError("economic_bes_idle_economic_source_min_share 必须在 [0,1]。")
        if (
            self.economic_bes_teacher_target_min_share < 0.0
            or self.economic_bes_teacher_target_min_share > 1.0
        ):
            raise ValueError("economic_bes_teacher_target_min_share 必须在 [0,1]。")
        if self.abs_min_on_gate_th < 0.0 or self.abs_min_on_gate_th > 1.0:
            raise ValueError("abs_min_on_gate_th 必须在 [0,1]。")
        if self.abs_min_on_u_margin < 0.0 or self.abs_min_on_u_margin > 1.0:
            raise ValueError("abs_min_on_u_margin 必须在 [0,1]。")
        if self.expert_prefill_policy not in {
            "rule",
            "easy_rule",
            "easy_rule_abs",
            "checkpoint",
            "checkpoint_dual",
        }:
            raise ValueError(
                "expert_prefill_policy 当前仅支持 "
                "rule/easy_rule/easy_rule_abs/checkpoint/checkpoint_dual。"
            )
        if self.expert_prefill_economic_policy not in {
            "checkpoint",
            "milp_mpc",
            "ga_mpc",
        }:
            raise ValueError(
                "expert_prefill_economic_policy 当前仅支持 "
                "checkpoint/milp_mpc/ga_mpc。"
            )
        if self.expert_prefill_policy in {"checkpoint", "checkpoint_dual"}:
            if len(self.expert_prefill_checkpoint_path) == 0:
                raise ValueError(
                    "expert_prefill_policy=checkpoint/checkpoint_dual 时必须提供 "
                    "expert_prefill_checkpoint_path。"
                )
            if not Path(self.expert_prefill_checkpoint_path).exists():
                raise FileNotFoundError(
                    f"expert_prefill_checkpoint_path 不存在: {self.expert_prefill_checkpoint_path}"
                )
        economic_checkpoint_required = self.expert_prefill_economic_policy == "checkpoint"
        if (
            economic_checkpoint_required
            and len(self.expert_prefill_economic_checkpoint_path) > 0
            and not Path(self.expert_prefill_economic_checkpoint_path).exists()
        ):
            raise FileNotFoundError(
                "expert_prefill_economic_checkpoint_path 不存在: "
                f"{self.expert_prefill_economic_checkpoint_path}"
            )
        if self.expert_prefill_policy == "checkpoint_dual" and economic_checkpoint_required:
            if len(self.expert_prefill_economic_checkpoint_path) == 0:
                raise ValueError(
                    "expert_prefill_policy=checkpoint_dual 且 "
                    "expert_prefill_economic_policy=checkpoint 时必须提供 "
                    "expert_prefill_economic_checkpoint_path。"
                )
        if (
            (
                self.economic_teacher_distill_coef > 0.0
                or (
                    self.economic_teacher_full_year_warm_start_samples > 0
                    and self.economic_teacher_full_year_warm_start_epochs > 0
                )
            )
            and economic_checkpoint_required
            and len(self.expert_prefill_economic_checkpoint_path) == 0
        ):
            raise ValueError(
                "economic teacher 启用时且 "
                "expert_prefill_economic_policy=checkpoint 时必须提供 "
                "expert_prefill_economic_checkpoint_path。"
            )
        if self.expert_prefill_steps < 0:
            raise ValueError("expert_prefill_steps 必须 >= 0。")
        if self.actor_warm_start_epochs < 0:
            raise ValueError("actor_warm_start_epochs 必须 >= 0。")
        if self.actor_warm_start_batch_size <= 0:
            raise ValueError("actor_warm_start_batch_size 必须 > 0。")
        if self.actor_warm_start_lr <= 0.0:
            raise ValueError("actor_warm_start_lr 必须 > 0。")
        if self.expert_prefill_cooling_bias < 0.0 or self.expert_prefill_cooling_bias > 1.0:
            raise ValueError("expert_prefill_cooling_bias 必须在 [0,1]。")
        if self.expert_prefill_abs_replay_boost < 0:
            raise ValueError("expert_prefill_abs_replay_boost 必须 >= 0。")
        if self.expert_prefill_abs_exec_threshold < 0.0 or self.expert_prefill_abs_exec_threshold > 1.0:
            raise ValueError("expert_prefill_abs_exec_threshold 必须在 [0,1]。")
        if self.expert_prefill_abs_window_mining_candidates < 0:
            raise ValueError("expert_prefill_abs_window_mining_candidates 必须 >= 0。")
        if self.dual_abs_margin_k < 0.0:
            raise ValueError("dual_abs_margin_k 必须 >= 0。")
        if self.dual_qc_ratio_th < 0.0:
            raise ValueError("dual_qc_ratio_th 必须 >= 0。")
        if self.dual_heat_backup_ratio_th < 0.0:
            raise ValueError("dual_heat_backup_ratio_th 必须 >= 0。")
        if self.dual_safe_abs_u_th < 0.0 or self.dual_safe_abs_u_th > 1.0:
            raise ValueError("dual_safe_abs_u_th 必须在 [0,1]。")
        if self.checkpoint_interval_steps < 0:
            raise ValueError("checkpoint_interval_steps 必须 >= 0（0 表示自动）。")
        if self.eval_window_pool_size < 0:
            raise ValueError("eval_window_pool_size 必须 >= 0。")
        if self.eval_window_count < 0:
            raise ValueError("eval_window_count 必须 >= 0。")
        if self.eval_window_pool_size == 0 and self.eval_window_count != 0:
            raise ValueError("eval_window_pool_size=0 时，eval_window_count 也必须为 0。")
        if self.eval_window_pool_size > 0 and self.eval_window_count > self.eval_window_pool_size:
            raise ValueError("eval_window_count 不能大于 eval_window_pool_size。")
        for name, value in {
            "best_gate_electric_min": self.best_gate_electric_min,
            "best_gate_heat_min": self.best_gate_heat_min,
            "best_gate_cool_min": self.best_gate_cool_min,
        }.items():
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} 必须在 [0,1]。")
        if self.plateau_patience_evals <= 0:
            raise ValueError("plateau_patience_evals 必须 > 0。")
        if self.plateau_lr_decay_factor <= 0.0 or self.plateau_lr_decay_factor > 1.0:
            raise ValueError("plateau_lr_decay_factor 必须在 (0,1]。")
        if self.plateau_min_actor_lr <= 0.0 or self.plateau_min_critic_lr <= 0.0:
            raise ValueError("plateau_min_actor_lr / plateau_min_critic_lr 必须 > 0。")
        if self.plateau_min_actor_lr > self.actor_lr:
            raise ValueError("plateau_min_actor_lr 不能大于 actor_lr。")
        if self.plateau_min_critic_lr > self.critic_lr:
            raise ValueError("plateau_min_critic_lr 不能大于 critic_lr。")
        if self.plateau_early_stop_patience_evals <= 0:
            raise ValueError("plateau_early_stop_patience_evals 必须 > 0。")
        if self.actor_delay <= 0:
            raise ValueError("actor_delay 必须 > 0。")
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims 必须全部 > 0。")
        if len(self.cost_targets) != 3:
            raise ValueError("cost_targets 必须包含 3 个元素。")


class _ReplayBuffer:
    def __init__(self, *, capacity: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action_raw = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.action_exec = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.teacher_action_exec = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.teacher_action_mask = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.teacher_available = np.zeros((self.capacity, 1), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.cost = np.zeros((self.capacity, 3), dtype=np.float32)
        self.gap = np.zeros((self.capacity, 3), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self._size = 0
        self._ptr = 0

    def add(
        self,
        *,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action_raw: np.ndarray,
        action_exec: np.ndarray,
        teacher_action_exec: np.ndarray | None = None,
        teacher_action_mask: np.ndarray | None = None,
        teacher_available: bool = False,
        reward: float,
        cost: np.ndarray,
        gap: np.ndarray,
        done: bool,
    ) -> None:
        index = self._ptr
        self.obs[index] = np.asarray(obs, dtype=np.float32)
        self.next_obs[index] = np.asarray(next_obs, dtype=np.float32)
        self.action_raw[index] = np.asarray(action_raw, dtype=np.float32)
        self.action_exec[index] = np.asarray(action_exec, dtype=np.float32)
        if teacher_action_exec is not None:
            self.teacher_action_exec[index] = np.asarray(teacher_action_exec, dtype=np.float32)
        else:
            self.teacher_action_exec[index].fill(0.0)
        if teacher_action_mask is not None:
            self.teacher_action_mask[index] = np.asarray(teacher_action_mask, dtype=np.float32)
        else:
            self.teacher_action_mask[index].fill(0.0)
        self.teacher_available[index, 0] = 1.0 if teacher_available else 0.0
        self.reward[index, 0] = float(reward)
        self.cost[index] = np.asarray(cost, dtype=np.float32).reshape(3)
        self.gap[index] = np.asarray(gap, dtype=np.float32).reshape(3)
        self.done[index, 0] = 1.0 if done else 0.0
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, *, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if self._size <= 0:
            raise ValueError("replay buffer 为空，无法采样。")
        indices = rng.integers(0, self._size, size=int(batch_size))
        return {
            "obs": self.obs[indices].copy(),
            "next_obs": self.next_obs[indices].copy(),
            "action_raw": self.action_raw[indices].copy(),
            "action_exec": self.action_exec[indices].copy(),
            "teacher_action_exec": self.teacher_action_exec[indices].copy(),
            "teacher_action_mask": self.teacher_action_mask[indices].copy(),
            "teacher_available": self.teacher_available[indices].copy(),
            "reward": self.reward[indices].copy(),
            "cost": self.cost[indices].copy(),
            "gap": self.gap[indices].copy(),
            "done": self.done[indices].copy(),
        }

    @property
    def size(self) -> int:
        return int(self._size)


def _build_mlp_layers(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    nn,
):
    layers: list[Any] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        width = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, width))
        layers.append(nn.ReLU())
        prev_dim = width
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


def build_pafc_actor_network(
    *,
    observation_dim: int,
    action_keys: Sequence[str],
    hidden_dims: Sequence[int] = (256, 256),
):
    torch, nn, _, _ = _require_torch_modules()
    action_low_np, action_high_np = _action_bounds_arrays(action_keys)

    class PAFCActor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = _build_mlp_layers(
                input_dim=int(observation_dim),
                output_dim=len(tuple(action_keys)),
                hidden_dims=tuple(hidden_dims),
                nn=nn,
            )
            self.register_buffer(
                "action_low",
                torch.as_tensor(action_low_np, dtype=torch.float32),
            )
            self.register_buffer(
                "action_high",
                torch.as_tensor(action_high_np, dtype=torch.float32),
            )

        def forward(self, observation):
            bounded = torch.tanh(self.net(observation))
            scale = 0.5 * (self.action_high - self.action_low)
            center = 0.5 * (self.action_high + self.action_low)
            return bounded * scale + center

    return PAFCActor()


def build_pafc_critic_network(
    *,
    observation_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int] = (256, 256),
    positive_output: bool = False,
):
    torch, nn, _, _ = _require_torch_modules()

    class PAFCCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev_dim = int(observation_dim) + int(action_dim)
            for hidden_dim in tuple(hidden_dims):
                width = int(hidden_dim)
                layers.append(nn.Linear(prev_dim, width))
                layers.append(nn.ReLU())
                prev_dim = width
            self.feature_net = nn.Sequential(*layers)
            self.output = nn.Linear(prev_dim, 1)
            self.positive_output = bool(positive_output)
            self.softplus = nn.Softplus() if self.positive_output else None

        def forward(self, observation, action):
            hidden = self.feature_net(torch.cat([observation, action], dim=-1))
            value = self.output(hidden)
            if self.positive_output and self.softplus is not None:
                return self.softplus(value)
            return value

    return PAFCCritic()


class FrozenProjectionSurrogate:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        env_config: EnvConfig,
        observation_keys: Sequence[str],
        action_keys: Sequence[str],
        device: str,
    ) -> None:
        torch, _, _, _ = _require_torch_modules()
        payload = load_policy(checkpoint_path, map_location=device)
        metadata = dict(payload["metadata"])
        self.metadata = metadata
        self.env_config = env_config
        self.observation_keys = tuple(str(key) for key in observation_keys)
        self.action_keys = tuple(str(key) for key in action_keys)
        self.observation_index = {
            key: idx for idx, key in enumerate(self.observation_keys)
        }
        self.action_index = {
            key: idx for idx, key in enumerate(self.action_keys)
        }
        self.feature_keys = tuple(str(key) for key in metadata.get("feature_keys", ()))
        self.target_keys = tuple(str(key) for key in metadata.get("target_keys", ()))
        self.model = build_projection_surrogate_network(
            input_dim=int(metadata["input_dim"]),
            output_dim=int(metadata["output_dim"]),
            hidden_dims=tuple(int(dim) for dim in metadata.get("hidden_dims", (256, 256))),
        ).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        feature_norm = dict(metadata.get("feature_norm", {}))
        self.offset = torch.as_tensor(
            np.asarray(feature_norm.get("offset", []), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self.scale = torch.as_tensor(
            np.asarray(feature_norm.get("scale", []), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self.scale = torch.where(self.scale.abs() < _NORM_EPS, torch.ones_like(self.scale), self.scale)
        action_low_np, action_high_np = _action_bounds_arrays(self.action_keys)
        self.action_low = torch.as_tensor(action_low_np, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high_np, dtype=torch.float32, device=device)
        target_indices: list[int] = []
        for key in self.action_keys:
            target_key = f"action_exec_{key}"
            if target_key not in self.target_keys:
                raise ValueError(f"projection surrogate 缺少目标列: {target_key}")
            target_indices.append(self.target_keys.index(target_key))
        self.target_indices = tuple(target_indices)

    def _obs_column(self, obs_batch, key: str):
        if key not in self.observation_index:
            raise ValueError(f"观测中缺少 surrogate 所需特征: {key}")
        return obs_batch[:, self.observation_index[key]]

    def _requested_gt_mw(self, action_raw):
        u_gt = action_raw[:, self.action_index["u_gt"]]
        return 0.5 * (u_gt + 1.0) * float(self.env_config.p_gt_cap_mw)

    def _state_feature(self, key: str, obs_batch, action_raw):
        torch, _, _, _ = _require_torch_modules()
        if key in self.observation_index:
            return self._obs_column(obs_batch, key)
        if key == "energy_demand_e_mwh":
            return self._obs_column(obs_batch, "p_dem_mw") * float(self.env_config.dt_hours)
        if key == "energy_demand_h_mwh":
            return self._obs_column(obs_batch, "qh_dem_mw") * float(self.env_config.dt_hours)
        if key == "energy_demand_c_mwh":
            return self._obs_column(obs_batch, "qc_dem_mw") * float(self.env_config.dt_hours)
        if key == "price_e_buy":
            return self._obs_column(obs_batch, "price_e")
        if key == "p_re_mw":
            return self._obs_column(obs_batch, "pv_mw") + self._obs_column(obs_batch, "wt_mw")
        if key == "p_gt_mw":
            return self._requested_gt_mw(action_raw)
        if key == "q_hrsg_rec_mw":
            return self._obs_column(obs_batch, "q_hrsg_est_now_mw")
        if key == "u_boiler_lower_bound":
            lower = self._obs_column(obs_batch, "heat_backup_min_needed_mw") / max(
                _NORM_EPS, float(self.env_config.q_boiler_cap_mw)
            )
            return torch.clamp(lower, 0.0, 1.0)
        if key == "u_abs_gate":
            if not bool(self.env_config.abs_gate_enabled):
                return torch.ones(obs_batch.shape[0], dtype=obs_batch.dtype, device=obs_batch.device)
            return torch.sigmoid(
                self._obs_column(obs_batch, "abs_drive_margin_k")
                / max(_NORM_EPS, float(self.env_config.abs_gate_scale_k))
            )
        if key == "p_gt_ramp_delta_mw":
            return torch.abs(self._requested_gt_mw(action_raw) - self._obs_column(obs_batch, "p_gt_prev_mw"))
        raise ValueError(
            f"当前 M2 trainer 不支持构造 surrogate 特征 `{key}`。"
            "请使用仅依赖当前 observation + raw action 的 surrogate checkpoint。"
        )

    def project(self, obs_batch, action_raw):
        torch, _, _, _ = _require_torch_modules()
        columns: list[Any] = []
        for key in self.feature_keys:
            if key.startswith("action_raw_"):
                action_key = key[len("action_raw_") :]
                columns.append(action_raw[:, self.action_index[action_key]])
            else:
                columns.append(self._state_feature(key, obs_batch, action_raw))
        features = torch.stack(columns, dim=1)
        normalized = (features - self.offset) / self.scale
        prediction = self.model(normalized)
        exec_action = prediction[:, list(self.target_indices)]
        return torch.clamp(exec_action, self.action_low, self.action_high)


@dataclass(slots=True)
class EasyRuleAbsPolicy:
    env_config: EnvConfig
    p_gt_cap_mw: float = 12.0
    q_boiler_cap_mw: float = 10.0
    q_ech_cap_mw: float = 6.0
    q_abs_cool_cap_mw: float = 3.0
    price_low_threshold: float = 600.0
    price_high_threshold: float = 1200.0

    def act(self, observation: dict[str, float]) -> dict[str, float]:
        p_dem = float(observation["p_dem_mw"])
        p_re = float(observation["pv_mw"]) + float(observation["wt_mw"])
        qh_dem = float(observation["qh_dem_mw"])
        qc_dem = float(observation["qc_dem_mw"])
        soc_bes = float(observation["soc_bes"])
        price_e = float(observation["price_e"])
        drive_margin_k = float(observation.get("abs_drive_margin_k", 0.0))
        heat_backup_need = float(observation.get("heat_backup_min_needed_mw", qh_dem))
        e_tes = float(observation.get("e_tes_mwh", 0.0))

        net_load = max(0.0, p_dem - p_re)
        coarse_gt_deadband = 0.45 * float(self.p_gt_cap_mw)
        if net_load <= coarse_gt_deadband:
            u_gt = -1.0
        else:
            gt_ratio = min(0.55, net_load / max(1e-6, float(self.p_gt_cap_mw)))
            u_gt = gt_ratio * 2.0 - 1.0

        if price_e >= self.price_high_threshold and soc_bes > 0.35:
            u_bes = 0.3
        elif price_e <= self.price_low_threshold and soc_bes < 0.75:
            u_bes = -0.3
        else:
            u_bes = 0.0

        boiler_follow = min(1.0, max(0.0, heat_backup_need / max(1e-6, float(self.q_boiler_cap_mw))))
        abs_cool_cap_mw = max(1e-6, float(self.q_abs_cool_cap_mw))
        q_ech_cap_mw = max(1e-6, float(self.q_ech_cap_mw))
        cooling_heavy = qc_dem > 0.25 * abs_cool_cap_mw
        if drive_margin_k > 0.0 and cooling_heavy:
            u_abs = float(np.clip(max(0.35, qc_dem / abs_cool_cap_mw), 0.0, 1.0))
            u_ech = float(np.clip(min(0.35, qc_dem / q_ech_cap_mw), 0.0, 1.0))
            u_boiler = max(float(boiler_follow), 0.35 if qh_dem > 0.1 else 0.15)
            u_tes = 0.35 if e_tes > 0.2 * float(self.env_config.e_tes_cap_mwh) else 0.0
        else:
            u_abs = 0.0
            u_ech = float(np.clip(qc_dem / q_ech_cap_mw, 0.0, 1.0))
            u_boiler = float(boiler_follow)
            u_tes = 0.0

        return {
            "u_gt": float(np.clip(u_gt, -1.0, 1.0)),
            "u_bes": float(np.clip(u_bes, -1.0, 1.0)),
            "u_boiler": float(np.clip(u_boiler, 0.0, 1.0)),
            "u_abs": float(np.clip(u_abs, 0.0, 1.0)),
            "u_ech": float(np.clip(u_ech, 0.0, 1.0)),
            "u_tes": float(np.clip(u_tes, -1.0, 1.0)),
        }


def _steps_per_day(*, env_config: EnvConfig) -> int:
    return max(1, int(round(24.0 / max(_NORM_EPS, float(env_config.dt_hours)))))


def _build_fixed_eval_episode_df(
    *,
    train_df: pd.DataFrame,
    env_config: EnvConfig,
    eval_episode_days: int,
) -> pd.DataFrame:
    steps = int(eval_episode_days) * _steps_per_day(env_config=env_config)
    if steps <= 0:
        raise ValueError("eval_episode_days 对应的评估步数必须 > 0。")
    if len(train_df) < steps:
        raise ValueError(
            f"训练集长度不足以截取固定评估片段：需要 {steps} 步，当前仅 {len(train_df)} 步。"
        )
    return train_df.tail(steps).reset_index(drop=True)


def _build_eval_window_pool(
    *,
    train_df: pd.DataFrame,
    env_config: EnvConfig,
    eval_episode_days: int,
    pool_size: int,
    window_count: int,
    seed: int,
) -> dict[str, Any] | None:
    if int(pool_size) <= 0 or int(window_count) <= 0:
        return None

    steps = int(eval_episode_days) * _steps_per_day(env_config=env_config)
    if steps <= 0:
        raise ValueError("eval_episode_days 对应的评估步数必须 > 0。")
    if len(train_df) < steps:
        raise ValueError(
            f"训练集长度不足以构建验证窗口池：需要 {steps} 步，当前仅 {len(train_df)} 步。"
        )

    max_start = len(train_df) - steps
    candidate_starts = list(range(0, max_start + 1, steps))
    if not candidate_starts or candidate_starts[-1] != max_start:
        candidate_starts.append(int(max_start))
    candidate_starts = sorted({int(value) for value in candidate_starts})
    if not candidate_starts:
        raise ValueError("无法从训练集构建验证窗口候选集。")

    pool_target_size = min(int(pool_size), len(candidate_starts))
    pool_positions = np.linspace(0, len(candidate_starts) - 1, num=pool_target_size)
    pool_candidate_indices = sorted({int(round(value)) for value in pool_positions})
    pool_indices = [candidate_starts[index] for index in pool_candidate_indices]

    window_target_count = min(int(window_count), len(pool_indices))
    pool_index_blocks = np.array_split(np.arange(len(pool_indices), dtype=int), window_target_count)
    rng = np.random.default_rng(int(seed))
    selected_pool_positions: list[int] = []
    for block in pool_index_blocks:
        if len(block) == 0:
            continue
        if len(block) == 1:
            selected_pool_positions.append(int(block[0]))
            continue
        choice = int(rng.integers(0, len(block)))
        selected_pool_positions.append(int(block[choice]))
    selected_pool_positions = sorted(set(selected_pool_positions))
    if not selected_pool_positions:
        selected_pool_positions = [0]

    windows_payload: list[dict[str, Any]] = []
    episode_dfs: list[pd.DataFrame] = []
    selected_indices: list[int] = []
    for window_index, start_idx in enumerate(pool_indices):
        end_idx = int(start_idx + steps)
        episode_df = train_df.iloc[start_idx:end_idx].reset_index(drop=True)
        selected = int(window_index) in selected_pool_positions
        if selected:
            episode_dfs.append(episode_df)
            selected_indices.append(int(window_index))
        windows_payload.append(
            {
                "window_index": int(window_index),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_timestamp": str(pd.to_datetime(episode_df["timestamp"].iloc[0]).isoformat()),
                "end_timestamp": str(pd.to_datetime(episode_df["timestamp"].iloc[-1]).isoformat()),
                "selected_for_eval": bool(selected),
            }
        )

    return {
        "mode": "fixed_multi_window_pool_v1",
        "pool_size": int(len(pool_indices)),
        "window_count": int(len(selected_indices)),
        "seed": int(seed),
        "episode_steps": int(steps),
        "episode_days": int(eval_episode_days),
        "selected_window_indices": selected_indices,
        "windows": windows_payload,
        "episode_dfs": tuple(episode_dfs),
    }


def _evaluate_predictor_on_episode_df(
    *,
    predictor,
    exogenous_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    env_config: EnvConfig,
    seed: int,
) -> dict[str, Any]:
    env = CCHPPhysicalEnv(exogenous_df=exogenous_df, config=env_config, seed=int(seed))
    observation, _ = env.reset(seed=int(seed), episode_df=episode_df)
    terminated = False
    total_reward = 0.0
    final_info: dict[str, Any] = {}

    while not terminated:
        action = predictor(observation)
        observation, reward, terminated, _, info = env.step(action)
        total_reward += float(reward)
        final_info = dict(info)

    summary = dict(final_info.get("episode_summary", env.kpi.summary()))
    summary["total_reward"] = float(total_reward)
    return summary


def _aggregate_eval_episode_summaries(
    episode_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(episode_summaries) == 0:
        raise ValueError("episode_summaries 不能为空。")

    rewards = np.asarray(
        [float(summary.get("total_reward", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    total_costs = np.asarray(
        [float(summary.get("total_cost", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    violation_rates = np.asarray(
        [float(summary.get("violation_rate", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    unmet_electric = np.asarray(
        [
            float((summary.get("unmet_energy_mwh") or {}).get("electric", 0.0))
            for summary in episode_summaries
        ],
        dtype=np.float64,
    )
    unmet_heat = np.asarray(
        [float((summary.get("unmet_energy_mwh") or {}).get("heat", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    unmet_cooling = np.asarray(
        [
            float((summary.get("unmet_energy_mwh") or {}).get("cooling", 0.0))
            for summary in episode_summaries
        ],
        dtype=np.float64,
    )
    electric_reliability = np.asarray(
        [float((summary.get("reliability") or {}).get("electric", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    heat_reliability = np.asarray(
        [float((summary.get("reliability") or {}).get("heat", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    cooling_reliability = np.asarray(
        [float((summary.get("reliability") or {}).get("cooling", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )

    cost_breakdown_keys = sorted(
        {
            str(key)
            for summary in episode_summaries
            for key in dict(summary.get("cost_breakdown", {}) or {}).keys()
        }
    )
    mean_cost_breakdown = {
        key: float(
            np.mean(
                [
                    float((summary.get("cost_breakdown") or {}).get(key, 0.0))
                    for summary in episode_summaries
                ]
            )
        )
        for key in cost_breakdown_keys
    }

    return {
        "episode_count": int(len(episode_summaries)),
        "mean_reward": float(rewards.mean()),
        "mean_total_cost": float(total_costs.mean()),
        "mean_violation_rate": float(violation_rates.mean()),
        "mean_unmet_electric_mwh": float(unmet_electric.mean()),
        "mean_unmet_heat_mwh": float(unmet_heat.mean()),
        "mean_unmet_cool_mwh": float(unmet_cooling.mean()),
        "reliability_mean": {
            "electric": float(electric_reliability.mean()),
            "heat": float(heat_reliability.mean()),
            "cooling": float(cooling_reliability.mean()),
        },
        "reliability_min": {
            "electric": float(electric_reliability.min()),
            "heat": float(heat_reliability.min()),
            "cooling": float(cooling_reliability.min()),
        },
        "mean_cost_breakdown": mean_cost_breakdown,
        "episode_reward": [float(value) for value in rewards.tolist()],
        "episode_total_cost": [float(value) for value in total_costs.tolist()],
        "episode_reliability_electric": [float(value) for value in electric_reliability.tolist()],
        "episode_reliability_heat": [float(value) for value in heat_reliability.tolist()],
        "episode_reliability_cooling": [float(value) for value in cooling_reliability.tolist()],
    }


def _build_reliability_gate_result(
    *,
    metrics: Mapping[str, Any],
    config: PAFCTD3TrainConfig,
) -> dict[str, Any]:
    thresholds = {
        "electric": float(config.best_gate_electric_min),
        "heat": float(config.best_gate_heat_min),
        "cooling": float(config.best_gate_cool_min),
    }
    if not bool(config.best_gate_enabled):
        return {
            "enabled": False,
            "passed": True,
            "thresholds": thresholds,
            "actual": dict(metrics.get("reliability_min", {})),
            "failed_metrics": [],
            "shortfall": {
                "electric": 0.0,
                "heat": 0.0,
                "cooling": 0.0,
                "total": 0.0,
                "max": 0.0,
            },
        }

    reliability_min = dict(metrics.get("reliability_min", {}))
    failed_metrics: list[dict[str, float | str]] = []
    shortfall = {"electric": 0.0, "heat": 0.0, "cooling": 0.0}
    for key, threshold in thresholds.items():
        actual = float(reliability_min.get(key, 0.0))
        delta = max(0.0, float(threshold) - actual)
        shortfall[str(key)] = float(delta)
        if delta > 1e-9:
            failed_metrics.append(
                {
                    "metric": str(key),
                    "actual": float(actual),
                    "threshold": float(threshold),
                    "shortfall": float(delta),
                }
            )
    total_shortfall = float(sum(float(value) for value in shortfall.values()))
    max_shortfall = float(max((float(value) for value in shortfall.values()), default=0.0))
    return {
        "enabled": True,
        "passed": len(failed_metrics) == 0,
        "thresholds": thresholds,
        "actual": {
            key: float(reliability_min.get(key, 0.0))
            for key in ("electric", "heat", "cooling")
        },
        "failed_metrics": failed_metrics,
        "shortfall": {
            "electric": float(shortfall["electric"]),
            "heat": float(shortfall["heat"]),
            "cooling": float(shortfall["cooling"]),
            "total": total_shortfall,
            "max": max_shortfall,
        },
    }


class PAFCTD3Trainer:
    def __init__(
        self,
        *,
        train_df: pd.DataFrame,
        train_statistics: dict[str, Any],
        env_config: EnvConfig,
        config: PAFCTD3TrainConfig,
        run_root: str | Path = "runs",
    ) -> None:
        year = sorted({int(value.year) for value in pd.to_datetime(train_df["timestamp"])})
        if year != [TRAIN_YEAR]:
            raise ValueError(f"PAFC-TD3 训练仅支持 {TRAIN_YEAR}，当前年份集合: {year}")

        self.torch, _, self.F, AdamW = _require_torch_modules()
        self.train_df = train_df
        self.train_statistics = train_statistics
        self.env_config = env_config
        self.config = config
        self.device = resolve_torch_device(config.device)
        try:
            self.torch.manual_seed(int(self.config.seed))
            if str(self.device).startswith("cuda") and getattr(self.torch, "cuda", None) is not None:
                self.torch.cuda.manual_seed_all(int(self.config.seed))
        except Exception:
            pass
        self.rng = np.random.default_rng(int(self.config.seed))

        self.observation_keys = tuple(self.config.observation_keys)
        self.action_keys = tuple(self.config.action_keys)
        self.observation_index = {str(key): idx for idx, key in enumerate(self.observation_keys)}
        self.action_index = {str(key): idx for idx, key in enumerate(self.action_keys)}
        self.action_low_np, self.action_high_np = _action_bounds_arrays(self.action_keys)
        self.action_low = self.torch.as_tensor(self.action_low_np, dtype=self.torch.float32, device=self.device)
        self.action_high = self.torch.as_tensor(self.action_high_np, dtype=self.torch.float32, device=self.device)
        economic_teacher_action_weights = np.zeros((len(self.action_keys),), dtype=np.float32)
        if "u_gt" in self.action_index:
            economic_teacher_action_weights[int(self.action_index["u_gt"])] = float(
                self.config.economic_teacher_gt_action_weight
            )
        if "u_bes" in self.action_index:
            economic_teacher_action_weights[int(self.action_index["u_bes"])] = float(
                self.config.economic_teacher_bes_action_weight
            )
        if "u_tes" in self.action_index:
            economic_teacher_action_weights[int(self.action_index["u_tes"])] = float(
                self.config.economic_teacher_tes_action_weight
            )
        if float(economic_teacher_action_weights.sum()) <= 0.0:
            economic_teacher_action_weights.fill(1.0)
        self.economic_teacher_action_weight_np = economic_teacher_action_weights.astype(
            np.float32,
            copy=True,
        )
        self.economic_teacher_action_weight = self.torch.as_tensor(
            economic_teacher_action_weights.reshape(1, -1),
            dtype=self.torch.float32,
            device=self.device,
        )
        (
            self.bes_price_low_threshold,
            self.bes_price_high_threshold,
        ) = _resolve_bes_price_thresholds_from_train_statistics(
            train_statistics=self.train_statistics,
        )

        self.observation_norm_payload, obs_offset_np, obs_scale_np = _build_observation_norm(
            feature_keys=self.observation_keys,
            train_statistics=self.train_statistics,
            env_config=self.env_config,
        )
        self.obs_offset = self.torch.as_tensor(obs_offset_np, dtype=self.torch.float32, device=self.device)
        self.obs_scale = self.torch.as_tensor(
            np.where(np.abs(obs_scale_np) < _NORM_EPS, 1.0, obs_scale_np),
            dtype=self.torch.float32,
            device=self.device,
        )
        obs_dim = len(self.observation_keys)
        action_dim = len(self.action_keys)

        self.actor = build_pafc_actor_network(
            observation_dim=obs_dim,
            action_keys=self.action_keys,
            hidden_dims=self.config.hidden_dims,
        ).to(self.device)
        self.actor_target = build_pafc_actor_network(
            observation_dim=obs_dim,
            action_keys=self.action_keys,
            hidden_dims=self.config.hidden_dims,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q1 = build_pafc_critic_network(
            observation_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
        ).to(self.device)
        self.q2 = build_pafc_critic_network(
            observation_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
        ).to(self.device)
        self.q1_target = build_pafc_critic_network(
            observation_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
        ).to(self.device)
        self.q2_target = build_pafc_critic_network(
            observation_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
        ).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.cost_critics = [
            build_pafc_critic_network(
                observation_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=self.config.hidden_dims,
                positive_output=True,
            ).to(self.device)
            for _ in range(3)
        ]
        self.cost_target_critics = [
            build_pafc_critic_network(
                observation_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=self.config.hidden_dims,
                positive_output=True,
            ).to(self.device)
            for _ in range(3)
        ]
        for target, source in zip(self.cost_target_critics, self.cost_critics):
            target.load_state_dict(source.state_dict())

        self.actor_optimizer = AdamW(self.actor.parameters(), lr=float(self.config.actor_lr))
        self.reward_critic_optimizer = AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=float(self.config.critic_lr),
        )
        cost_params: list[Any] = []
        for critic in self.cost_critics:
            cost_params.extend(list(critic.parameters()))
        self.cost_critic_optimizer = AdamW(cost_params, lr=float(self.config.critic_lr))
        self.current_actor_lr = float(self.config.actor_lr)
        self.current_critic_lr = float(self.config.critic_lr)

        self.replay = _ReplayBuffer(
            capacity=int(self.config.replay_capacity),
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        self.surrogate = FrozenProjectionSurrogate(
            checkpoint_path=self.config.projection_surrogate_checkpoint_path,
            env_config=self.env_config,
            observation_keys=self.observation_keys,
            action_keys=self.action_keys,
            device=self.device,
        )
        self.dual_lambdas = np.zeros(3, dtype=np.float32)
        self.dual_targets = np.asarray(self.config.cost_targets, dtype=np.float32)

        self.run_dir = self._create_run_directory(run_root=run_root)
        dump_statistics_json(
            self.train_statistics,
            self.run_dir / "train" / "train_statistics.json",
        )
        self.actor_checkpoint_path = self.run_dir / "checkpoints" / "pafc_td3_actor.pt"
        self.actor_checkpoint_json = self.run_dir / "checkpoints" / "pafc_td3_actor.json"
        self.last_actor_checkpoint_path = self.run_dir / "checkpoints" / "pafc_td3_actor_last.pt"
        self.last_actor_checkpoint_json = self.run_dir / "checkpoints" / "pafc_td3_actor_last.json"
        self.reward_actor_checkpoint_path = self.run_dir / "checkpoints" / "pafc_td3_actor_reward_leader.pt"
        self.reward_actor_checkpoint_json = self.run_dir / "checkpoints" / "pafc_td3_actor_reward_leader.json"
        self.selection_history_path = self.run_dir / "train" / "pafc_reliability_eval_history.jsonl"
        self.eval_windows_path = self.run_dir / "checkpoints" / "pafc_eval_windows.json"
        self.retained_checkpoints_dir = self.run_dir / "checkpoints" / "retained"
        self.retained_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        eval_window_pool = _build_eval_window_pool(
            train_df=self.train_df,
            env_config=self.env_config,
            eval_episode_days=int(self.config.episode_days),
            pool_size=int(self.config.eval_window_pool_size),
            window_count=int(self.config.eval_window_count),
            seed=int(self.config.seed),
        )
        if eval_window_pool is None:
            eval_episode_df = _build_fixed_eval_episode_df(
                train_df=self.train_df,
                env_config=self.env_config,
                eval_episode_days=int(self.config.episode_days),
            )
            self.eval_episode_dfs = (eval_episode_df,)
            self.eval_protocol = {
                "mode": "fixed_tail_v1",
                "eval_episode_days": int(self.config.episode_days),
                "eval_window_start": str(pd.to_datetime(eval_episode_df["timestamp"].iloc[0]).isoformat()),
                "eval_window_end": str(pd.to_datetime(eval_episode_df["timestamp"].iloc[-1]).isoformat()),
                "model_source_default": "best",
            }
        else:
            self.eval_episode_dfs = tuple(eval_window_pool["episode_dfs"])
            eval_windows_payload = {
                key: value
                for key, value in eval_window_pool.items()
                if key != "episode_dfs"
            }
            self.eval_windows_path.write_text(
                json.dumps(eval_windows_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.eval_protocol = {
                "mode": str(eval_window_pool["mode"]),
                "eval_episode_days": int(self.config.episode_days),
                "pool_size": int(eval_window_pool["pool_size"]),
                "window_count": int(eval_window_pool["window_count"]),
                "seed": int(eval_window_pool["seed"]),
                "selected_window_indices": list(eval_window_pool["selected_window_indices"]),
                "eval_windows_path": str(self.eval_windows_path.resolve()).replace("\\", "/"),
                "model_source_default": "best",
            }
        self.best_selection_snapshot: dict[str, Any] | None = None
        self.best_reward_snapshot: dict[str, Any] | None = None
        self.best_selection_rank = (
            0,
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
        )
        self.best_reward_mean = float("-inf")
        self.validation_eval_count = 0
        self.no_improve_evals = 0
        self.lr_decay_count = 0
        self.fine_tune_applied = False
        self.stop_requested = False
        self.stop_reason = ""
        self.plateau_events: list[dict[str, Any]] = []
        self.expert_prefill_summary: dict[str, Any] = self._base_expert_prefill_summary()
        self.expert_prefill_summary.update(
            {
                "steps": 0,
                "status": "not_run",
            }
        )
        self.actor_warm_start_summary: dict[str, Any] = {
            "enabled": bool(int(self.config.actor_warm_start_epochs) > 0),
            "epochs": int(self.config.actor_warm_start_epochs),
            "status": "not_run",
        }
        self.actor_bes_warm_start_summary: dict[str, Any] = {
            "enabled": bool(
                int(self.config.economic_bes_full_year_warm_start_samples) > 0
                and int(self.config.economic_bes_full_year_warm_start_epochs) > 0
            ),
            "samples": int(self.config.economic_bes_full_year_warm_start_samples),
            "epochs": int(self.config.economic_bes_full_year_warm_start_epochs),
            "u_weight": float(self.config.economic_bes_full_year_warm_start_u_weight),
            "charge_u_scale": float(self.config.economic_bes_charge_u_scale),
            "discharge_u_scale": float(self.config.economic_bes_discharge_u_scale),
            "charge_weight": float(self.config.economic_bes_charge_weight),
            "discharge_weight": float(self.config.economic_bes_discharge_weight),
            "charge_pressure_bonus": float(self.config.economic_bes_charge_pressure_bonus),
            "status": "not_run",
        }
        self.actor_teacher_full_year_warm_start_summary: dict[str, Any] = {
            "enabled": bool(
                int(self.config.economic_teacher_full_year_warm_start_samples) > 0
                and int(self.config.economic_teacher_full_year_warm_start_epochs) > 0
            ),
            "samples": int(self.config.economic_teacher_full_year_warm_start_samples),
            "epochs": int(self.config.economic_teacher_full_year_warm_start_epochs),
            "status": "not_run",
        }
        self.actor_init_summary: dict[str, Any] = {
            "enabled": False,
            "status": "not_run",
        }
        (
            self.economic_teacher_policy,
            self.economic_teacher_safe_policy,
            self.economic_teacher_distill_summary,
        ) = self._maybe_build_economic_teacher_policy()

    def _create_run_directory(self, run_root: str | Path) -> Path:
        root = Path(run_root)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root / f"{stamp}_train_pafc_td3"
        (run_dir / "train").mkdir(parents=True, exist_ok=True)
        (run_dir / "eval").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        return run_dir

    def _observation_to_vector(self, observation: Mapping[str, float]) -> np.ndarray:
        return build_feature_vector(
            observation=observation,
            feature_keys=self.observation_keys,
        ).astype(np.float32)

    def _normalize_observation_tensor(self, observation):
        return (observation - self.obs_offset) / self.obs_scale

    def _base_expert_prefill_summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(int(self.config.expert_prefill_steps) > 0),
            "policy": str(self.config.expert_prefill_policy),
            "checkpoint_path": str(self.config.expert_prefill_checkpoint_path),
            "economic_policy": str(self.config.expert_prefill_economic_policy),
            "economic_checkpoint_path": str(self.config.expert_prefill_economic_checkpoint_path),
            "cooling_bias": float(self.config.expert_prefill_cooling_bias),
            "abs_replay_boost": int(self.config.expert_prefill_abs_replay_boost),
            "economic_teacher_prefill_replay_boost": int(
                self.config.economic_teacher_prefill_replay_boost
            ),
            "abs_exec_threshold": float(self.config.expert_prefill_abs_exec_threshold),
            "abs_window_mining_candidates": int(self.config.expert_prefill_abs_window_mining_candidates),
            "dual_gate": {
                "abs_margin_k": float(self.config.dual_abs_margin_k),
                "qc_ratio_th": float(self.config.dual_qc_ratio_th),
                "heat_backup_ratio_th": float(self.config.dual_heat_backup_ratio_th),
                "safe_abs_u_th": float(self.config.dual_safe_abs_u_th),
            },
        }

    def _maybe_build_economic_teacher_policy(self) -> tuple[Any | None, Any | None, dict[str, Any]]:
        teacher_enabled = bool(
            float(self.config.economic_teacher_distill_coef) > 0.0
            or (
                int(self.config.economic_teacher_full_year_warm_start_samples) > 0
                and int(self.config.economic_teacher_full_year_warm_start_epochs) > 0
            )
        )
        summary = {
            "enabled": bool(teacher_enabled),
            "coef": float(self.config.economic_teacher_distill_coef),
            "policy": str(self.config.expert_prefill_economic_policy),
            "checkpoint_path": str(self.config.expert_prefill_economic_checkpoint_path),
            "safe_checkpoint_path": str(self.config.expert_prefill_checkpoint_path),
            "status": "disabled",
            "teacher": None,
            "safe_teacher": None,
            "prefill_target_steps": 0,
            "online_target_steps": 0,
            "candidate_steps": 0,
            "accepted_target_steps": 0,
            "rejected_target_steps": 0,
            "safe_reference_steps": 0,
            "full_target_steps": 0,
            "mixed_target_steps": 0,
            "mixed_target_dim_swap_count": 0,
            "mixed_target_dim_mean": 0.0,
            "mixed_target_dim_counts": {},
            "proxy_advantage_ratio_mean": 0.0,
            "projection_gap_mean": 0.0,
            "abs_risk_gap_mean": 0.0,
            "teacher_proxy_cost_mean": 0.0,
            "safe_proxy_cost_mean": 0.0,
            "rejection_reason_counts": {},
            "gate": {
                "proxy_advantage_min": float(self.config.economic_teacher_proxy_advantage_min),
                "gt_proxy_advantage_min": float(
                    self.config.economic_teacher_gt_proxy_advantage_min
                ),
                "bes_proxy_advantage_min": float(
                    self.config.economic_teacher_bes_proxy_advantage_min
                ),
                "max_safe_abs_risk_gap": float(self.config.economic_teacher_max_safe_abs_risk_gap),
                "projection_gap_max": float(self.config.economic_teacher_projection_gap_max),
                "gt_projection_gap_max": float(
                    self.config.economic_teacher_gt_projection_gap_max
                ),
                "bes_price_opportunity_min": float(
                    self.config.economic_teacher_bes_price_opportunity_min
                ),
            },
        }
        if not teacher_enabled:
            return None, None, summary
        teacher_policy, teacher_info = self._build_economic_teacher_policy(
            role="economic_distill",
        )
        safe_policy = None
        safe_checkpoint_path = str(self.config.expert_prefill_checkpoint_path).strip()
        if len(safe_checkpoint_path) > 0:
            safe_policy, safe_info = self._build_checkpoint_expert_policy(
                checkpoint_path=safe_checkpoint_path,
                role="economic_safe_compare",
            )
            summary["safe_teacher"] = dict(safe_info)
        summary["status"] = "ready"
        summary["teacher"] = dict(teacher_info)
        return teacher_policy, safe_policy, summary

    def _bind_policy_episode_context(
        self,
        *,
        policy,
        env: CCHPPhysicalEnv,
        observation: Mapping[str, float],
        episode_seed: int,
    ) -> None:
        bind_context_fn = getattr(policy, "bind_episode_context", None)
        if callable(bind_context_fn):
            bind_context_fn(
                env=env,
                episode_df=env.episode_df,
                initial_observation=dict(observation),
                seed=int(episode_seed),
            )

    def _record_economic_teacher_gate(self, *, decision: Mapping[str, Any]) -> None:
        summary = self.economic_teacher_distill_summary
        candidate_steps = int(summary.get("candidate_steps", 0)) + 1
        summary["candidate_steps"] = int(candidate_steps)
        if bool(decision.get("safe_reference_available", False)):
            summary["safe_reference_steps"] = int(summary.get("safe_reference_steps", 0)) + 1
        if bool(decision.get("accepted", False)):
            summary["accepted_target_steps"] = int(summary.get("accepted_target_steps", 0)) + 1
            target_mode = str(decision.get("target_mode", "")).strip().lower()
            if target_mode == "full":
                summary["full_target_steps"] = int(summary.get("full_target_steps", 0)) + 1
            elif target_mode == "mixed":
                summary["mixed_target_steps"] = int(summary.get("mixed_target_steps", 0)) + 1
                dim_count = int(decision.get("mixed_dim_count", 0))
                summary["mixed_target_dim_swap_count"] = int(
                    summary.get("mixed_target_dim_swap_count", 0)
                ) + dim_count
                mixed_steps = int(summary.get("mixed_target_steps", 0))
                summary["mixed_target_dim_mean"] = float(
                    int(summary.get("mixed_target_dim_swap_count", 0)) / max(1, mixed_steps)
                )
                dim_counts = dict(summary.get("mixed_target_dim_counts", {}) or {})
                for key in decision.get("mixed_dims", []):
                    label = str(key)
                    dim_counts[label] = int(dim_counts.get(label, 0)) + 1
                summary["mixed_target_dim_counts"] = dim_counts
        else:
            summary["rejected_target_steps"] = int(summary.get("rejected_target_steps", 0)) + 1
            counts = dict(summary.get("rejection_reason_counts", {}) or {})
            for reason in decision.get("reasons", []):
                label = str(reason)
                counts[label] = int(counts.get(label, 0)) + 1
            summary["rejection_reason_counts"] = counts
        summary["accepted_target_rate"] = float(
            int(summary.get("accepted_target_steps", 0)) / max(1, candidate_steps)
        )
        for key in (
            "proxy_advantage_ratio",
            "projection_gap",
            "abs_risk_gap",
            "teacher_proxy_cost",
            "safe_proxy_cost",
        ):
            if key not in decision:
                continue
            target_key = f"{key}_mean"
            previous = float(summary.get(target_key, 0.0))
            current = float(decision[key])
            summary[target_key] = float(previous + (current - previous) / max(1, candidate_steps))

    def _project_action_exec_np(
        self,
        *,
        obs_vector: np.ndarray,
        action_raw: np.ndarray,
    ) -> np.ndarray:
        obs_tensor = self.torch.as_tensor(
            np.asarray(obs_vector, dtype=np.float32).reshape(1, -1),
            dtype=self.torch.float32,
            device=self.device,
        )
        action_tensor = self.torch.as_tensor(
            np.asarray(action_raw, dtype=np.float32).reshape(1, -1),
            dtype=self.torch.float32,
            device=self.device,
        )
        with self.torch.no_grad():
            action_exec = self.surrogate.project(obs_tensor, action_tensor)
        return action_exec.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    def _get_economic_teacher_target_step(
        self,
        *,
        observation: Mapping[str, float],
        obs_vector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if self.economic_teacher_policy is None:
            return (
                np.zeros((len(self.action_keys),), dtype=np.float32),
                np.zeros((len(self.action_keys),), dtype=np.float32),
                False,
            )
        teacher_action = dict(self.economic_teacher_policy.act(dict(observation)))
        teacher_action_raw, teacher_action_exec = _materialize_teacher_action_np(
            teacher_action=teacher_action,
            action_keys=self.action_keys,
            obs_vector=obs_vector,
            returns_executed_action=bool(
                getattr(self.economic_teacher_policy, "returns_executed_action", False)
            ),
            project_action_exec_fn=self._project_action_exec_np,
        )
        teacher_proxy = _estimate_dispatch_proxy_np(
            observation=observation,
            action_exec=teacher_action_exec,
            action_index=self.action_index,
            env_config=self.env_config,
            gt_off_deadband_ratio=float(self.config.gt_off_deadband_ratio),
        )
        safe_reference_available = False
        teacher_target = teacher_action_exec.copy()
        teacher_mask = _economic_teacher_target_mask_np(
            action_dim=len(self.action_keys),
            action_index=self.action_index,
        )
        safe_proxy_cost = 0.0
        proxy_advantage_ratio = 1.0
        abs_risk_gap = 0.0
        target_mode = "full"
        mixed_dims: list[str] = []
        bes_charge_u, bes_discharge_u = self._resolve_bes_prior_u_pair()
        default_teacher_dim_count = int(
            _economic_teacher_target_mask_np(
                action_dim=len(self.action_keys),
                action_index=self.action_index,
            ).sum()
        )
        if self.economic_teacher_safe_policy is not None:
            safe_action = dict(self.economic_teacher_safe_policy.act(dict(observation)))
            safe_action_raw, safe_action_exec = _materialize_teacher_action_np(
                teacher_action=safe_action,
                action_keys=self.action_keys,
                obs_vector=obs_vector,
                returns_executed_action=bool(
                    getattr(self.economic_teacher_safe_policy, "returns_executed_action", False)
                ),
                project_action_exec_fn=self._project_action_exec_np,
            )
            safe_proxy = _estimate_dispatch_proxy_np(
                observation=observation,
                action_exec=safe_action_exec,
                action_index=self.action_index,
                env_config=self.env_config,
                gt_off_deadband_ratio=float(self.config.gt_off_deadband_ratio),
            )
            safe_reference_available = True
            safe_proxy_cost = float(safe_proxy["proxy_cost"])
            teacher_target = safe_action_exec.copy()
            teacher_mask = np.zeros((len(self.action_keys),), dtype=np.float32)
            target_mode = "mixed"
            proxy_advantage_ratio = 0.0
            abs_risk_gap = 0.0
            mixed_target = _build_mixed_economic_teacher_target_np(
                observation=observation,
                safe_action_exec=safe_action_exec,
                safe_proxy=safe_proxy,
                economic_action_raw=teacher_action_raw,
                economic_action_exec=teacher_action_exec,
                action_index=self.action_index,
                env_config=self.env_config,
                gt_off_deadband_ratio=float(self.config.gt_off_deadband_ratio),
                min_proxy_advantage_ratio=float(self.config.economic_teacher_proxy_advantage_min),
                gt_proxy_advantage_ratio_min=float(
                    self.config.economic_teacher_gt_proxy_advantage_min
                ),
                max_safe_abs_risk_gap=float(self.config.economic_teacher_max_safe_abs_risk_gap),
                max_projection_gap=float(self.config.economic_teacher_projection_gap_max),
                gt_projection_gap_max=float(
                    self.config.economic_teacher_gt_projection_gap_max
                ),
                bes_proxy_advantage_ratio_min=float(
                    self.config.economic_teacher_bes_proxy_advantage_min
                ),
                bes_price_low_threshold=float(self.bes_price_low_threshold),
                bes_price_high_threshold=float(self.bes_price_high_threshold),
                bes_charge_soc_ceiling=float(self.config.economic_bes_charge_soc_ceiling),
                bes_discharge_soc_floor=float(self.config.economic_bes_discharge_soc_floor),
                bes_soc_min=float(self.env_config.bes_soc_min),
                bes_soc_max=float(self.env_config.bes_soc_max),
                bes_charge_u=float(bes_charge_u),
                bes_discharge_u=float(bes_discharge_u),
                bes_price_opportunity_min=float(
                    self.config.economic_teacher_bes_price_opportunity_min
                ),
                gt_abs_margin_guard_k=float(self.config.dual_abs_margin_k),
                gt_qc_ratio_guard=float(self.config.dual_qc_ratio_th),
                gt_heat_backup_ratio_guard=float(self.config.dual_heat_backup_ratio_th),
            )
            if int(mixed_target["swapped_dim_count"]) > 0:
                teacher_target = np.asarray(mixed_target["target"], dtype=np.float32)
                teacher_mask = np.asarray(mixed_target["mask"], dtype=np.float32)
                if int(mixed_target["swapped_dim_count"]) >= default_teacher_dim_count:
                    target_mode = "full"
                else:
                    target_mode = "mixed"
                mixed_dims = [str(key) for key in mixed_target["swapped_dims"]]
                teacher_proxy = {
                    **teacher_proxy,
                    "proxy_cost": float(mixed_target["proxy_cost"]),
                    "abs_invalid_risk": float(mixed_target["abs_invalid_risk"]),
                }
                proxy_advantage_ratio = float(
                    np.clip(
                        (safe_proxy_cost - float(mixed_target["proxy_cost"]))
                        / max(1.0, abs(safe_proxy_cost)),
                        -1.0,
                        1.0,
                    )
                )
                abs_risk_gap = float(mixed_target["abs_invalid_risk"]) - float(safe_proxy["abs_invalid_risk"])
        projection_gap = _economic_teacher_projection_gap_np(
            action_raw=teacher_action_raw,
            action_exec=teacher_target,
            action_index=self.action_index,
            teacher_mask=teacher_mask,
        )
        proxy_advantage_limit = float(self.config.economic_teacher_proxy_advantage_min)
        projection_gap_limit = float(self.config.economic_teacher_projection_gap_max)
        if "u_gt" in self.action_index:
            gt_index = int(self.action_index["u_gt"])
            if gt_index < len(teacher_mask) and float(teacher_mask[gt_index]) > 0.5:
                proxy_advantage_limit = min(
                    proxy_advantage_limit,
                    float(self.config.economic_teacher_gt_proxy_advantage_min),
                )
                projection_gap_limit = max(
                    projection_gap_limit,
                    float(self.config.economic_teacher_gt_projection_gap_max),
                )
        decision = _economic_teacher_gate_decision_np(
            safe_reference_available=safe_reference_available,
            proxy_advantage_ratio=proxy_advantage_ratio,
            abs_risk_gap=abs_risk_gap,
            projection_gap=projection_gap,
            min_proxy_advantage_ratio=float(proxy_advantage_limit),
            max_safe_abs_risk_gap=float(self.config.economic_teacher_max_safe_abs_risk_gap),
            max_projection_gap=float(projection_gap_limit),
        )
        if safe_reference_available and int(np.count_nonzero(teacher_mask)) <= 0:
            reasons = list(decision.get("reasons", []))
            reasons.append("no_profitable_dims")
            decision = {
                **decision,
                "accepted": False,
                "reasons": reasons,
            }
        self._record_economic_teacher_gate(
            decision={
                **decision,
                "teacher_proxy_cost": float(teacher_proxy["proxy_cost"]),
                "safe_proxy_cost": float(safe_proxy_cost),
                "target_mode": str(target_mode),
                "mixed_dim_count": int(len(mixed_dims)),
                "mixed_dims": list(mixed_dims),
                "proxy_advantage_limit": float(proxy_advantage_limit),
                "projection_gap_limit": float(projection_gap_limit),
            }
        )
        if not bool(decision["accepted"]):
            return (
                np.zeros((len(self.action_keys),), dtype=np.float32),
                np.zeros((len(self.action_keys),), dtype=np.float32),
                False,
            )
        return teacher_target.astype(np.float32, copy=False), teacher_mask.astype(np.float32, copy=False), True

    def _apply_abs_cooling_blend_tensor(self, *, obs_batch, action_batch):
        shaped_columns = [
            action_batch[:, index : index + 1]
            for index in range(action_batch.shape[1])
        ]

        if "u_gt" in self.action_index and "p_gt_prev_mw" in self.observation_index:
            u_gt_index = int(self.action_index["u_gt"])
            p_gt_prev_index = int(self.observation_index["p_gt_prev_mw"])
            p_gt_prev = obs_batch[:, p_gt_prev_index : p_gt_prev_index + 1]
            p_gt_cap_mw = max(_NORM_EPS, float(self.env_config.p_gt_cap_mw))
            gt_min_output_mw = max(0.0, float(self.env_config.gt_min_output_mw))
            gt_off_deadband_mw = _gt_off_deadband_mw(
                gt_min_output_mw=gt_min_output_mw,
                gt_off_deadband_ratio=float(self.config.gt_off_deadband_ratio),
            )
            if (
                "gt_ramp_headroom_up_mw" in self.observation_index
                and "gt_ramp_headroom_down_mw" in self.observation_index
            ):
                ramp_up_index = int(self.observation_index["gt_ramp_headroom_up_mw"])
                ramp_down_index = int(self.observation_index["gt_ramp_headroom_down_mw"])
                ramp_up = obs_batch[:, ramp_up_index : ramp_up_index + 1]
                ramp_down = obs_batch[:, ramp_down_index : ramp_down_index + 1]
                p_gt_low = self.torch.clamp(p_gt_prev - ramp_down, 0.0, p_gt_cap_mw)
                p_gt_high = self.torch.clamp(p_gt_prev + ramp_up, 0.0, p_gt_cap_mw)
            else:
                p_gt_low = self.torch.zeros_like(p_gt_prev)
                p_gt_high = self.torch.full_like(p_gt_prev, p_gt_cap_mw)
            p_gt_target = ((shaped_columns[u_gt_index] + 1.0) * 0.5) * p_gt_cap_mw
            p_gt_target = self.torch.clamp(p_gt_target, p_gt_low, p_gt_high)
            p_gt_target = self.torch.where(
                p_gt_target <= (gt_off_deadband_mw + _NORM_EPS),
                self.torch.zeros_like(p_gt_target),
                p_gt_target,
            )
            p_gt_target = self.torch.where(
                (p_gt_target > (gt_off_deadband_mw + _NORM_EPS))
                & (p_gt_target < gt_min_output_mw),
                self.torch.full_like(p_gt_target, gt_min_output_mw),
                p_gt_target,
            )
            shaped_columns[u_gt_index] = self.torch.clamp(
                2.0 * (p_gt_target / p_gt_cap_mw) - 1.0,
                -1.0,
                1.0,
            )

        if bool(self.config.state_feasible_action_shaping_enabled):
            if "u_bes" in self.action_index and "soc_bes" in self.observation_index:
                u_bes_index = int(self.action_index["u_bes"])
                soc_index = int(self.observation_index["soc_bes"])
                soc = obs_batch[:, soc_index : soc_index + 1]
                dt_h = max(_NORM_EPS, float(self.env_config.dt_hours))
                p_bes_cap_mw = max(_NORM_EPS, float(self.env_config.p_bes_cap_mw))
                e_bes_cap_mwh = max(_NORM_EPS, float(self.env_config.e_bes_cap_mwh))
                bes_soc_min = float(self.env_config.bes_soc_min)
                bes_soc_max = float(self.env_config.bes_soc_max)
                bes_eta_charge = max(_NORM_EPS, float(self.env_config.bes_eta_charge))
                bes_eta_discharge = max(_NORM_EPS, float(self.env_config.bes_eta_discharge))
                max_discharge_u = self.torch.clamp(
                    ((soc - bes_soc_min) * e_bes_cap_mwh * bes_eta_discharge / dt_h) / p_bes_cap_mw,
                    0.0,
                    1.0,
                )
                max_charge_u = self.torch.clamp(
                    ((bes_soc_max - soc) * e_bes_cap_mwh / (bes_eta_charge * dt_h)) / p_bes_cap_mw,
                    0.0,
                    1.0,
                )
                shaped_columns[u_bes_index] = self.torch.clamp(
                    shaped_columns[u_bes_index],
                    -max_charge_u,
                    max_discharge_u,
                )

            if "u_tes" in self.action_index and "e_tes_mwh" in self.observation_index:
                u_tes_index = int(self.action_index["u_tes"])
                e_tes_index = int(self.observation_index["e_tes_mwh"])
                e_tes = obs_batch[:, e_tes_index : e_tes_index + 1]
                dt_h = max(_NORM_EPS, float(self.env_config.dt_hours))
                e_tes_cap_mwh = max(_NORM_EPS, float(self.env_config.e_tes_cap_mwh))
                q_tes_charge_cap_mw = max(_NORM_EPS, float(self.env_config.q_tes_charge_cap_mw))
                q_tes_discharge_cap_mw = max(_NORM_EPS, float(self.env_config.q_tes_discharge_cap_mw))
                if "q_tes_discharge_feasible_mw" in self.observation_index:
                    discharge_index = int(self.observation_index["q_tes_discharge_feasible_mw"])
                    discharge_feasible_mw = obs_batch[:, discharge_index : discharge_index + 1]
                else:
                    discharge_feasible_mw = self.torch.clamp(e_tes / dt_h, 0.0, q_tes_discharge_cap_mw)
                charge_headroom_mw = self.torch.clamp((e_tes_cap_mwh - e_tes) / dt_h, 0.0, q_tes_charge_cap_mw)
                u_tes_max = self.torch.clamp(discharge_feasible_mw / q_tes_discharge_cap_mw, 0.0, 1.0)
                u_tes_min = -self.torch.clamp(charge_headroom_mw / q_tes_charge_cap_mw, 0.0, 1.0)
                shaped_columns[u_tes_index] = self.torch.clamp(
                    shaped_columns[u_tes_index],
                    u_tes_min,
                    u_tes_max,
                )

        if "abs_drive_margin_k" not in self.observation_index:
            return self.torch.cat(shaped_columns, dim=1)
        if "u_abs" not in self.action_index or "u_ech" not in self.action_index:
            return self.torch.cat(shaped_columns, dim=1)
        margin_index = int(self.observation_index["abs_drive_margin_k"])
        u_abs_index = int(self.action_index["u_abs"])
        u_ech_index = int(self.action_index["u_ech"])
        gate_scale_k = max(_NORM_EPS, float(self.env_config.abs_gate_scale_k))
        transfer_ratio = float(
            max(_NORM_EPS, float(self.env_config.q_abs_cool_cap_mw))
            / max(_NORM_EPS, float(self.env_config.q_ech_cap_mw))
        )
        invalid_req_u_th = float(max(0.0, float(self.env_config.abs_invalid_req_u_th)))
        invalid_req_gate_th = float(max(0.0, float(self.env_config.abs_invalid_req_gate_th)))
        abs_deadzone_gate_th = float(max(0.0, float(self.env_config.abs_deadzone_gate_th)))
        abs_deadzone_u_th = float(max(0.0, float(self.env_config.abs_deadzone_u_th)))
        abs_effective_min_u = float(max(0.0, abs_deadzone_u_th + float(self.config.abs_min_on_u_margin)))
        abs_min_on_gate_th = float(max(0.0, float(self.config.abs_min_on_gate_th)))
        margin = obs_batch[:, margin_index : margin_index + 1]
        abs_gate = self.torch.sigmoid(margin / gate_scale_k)
        u_abs = shaped_columns[u_abs_index]
        u_ech = shaped_columns[u_ech_index]
        cooling_transfer_ratio = abs_gate * transfer_ratio

        u_abs_safe = u_abs
        if abs_deadzone_gate_th > 0.0:
            u_abs_safe = self.torch.where(
                abs_gate < abs_deadzone_gate_th,
                self.torch.zeros_like(u_abs_safe),
                u_abs_safe,
            )
        if invalid_req_gate_th > abs_deadzone_gate_th:
            u_abs_safe = self.torch.where(
                (abs_gate >= abs_deadzone_gate_th) & (abs_gate < invalid_req_gate_th),
                self.torch.minimum(u_abs_safe, self.torch.full_like(u_abs_safe, invalid_req_u_th)),
                u_abs_safe,
            )
        effective_before_min_on = u_abs_safe * abs_gate
        if abs_deadzone_u_th > 0.0:
            u_abs_safe = self.torch.where(
                (effective_before_min_on < abs_deadzone_u_th) & (abs_gate < abs_min_on_gate_th),
                self.torch.zeros_like(u_abs_safe),
                u_abs_safe,
            )
        suppressed_abs = self.torch.clamp(u_abs - u_abs_safe, 0.0, 1.0)
        u_ech_safe = self.torch.clamp(
            u_ech + suppressed_abs * cooling_transfer_ratio,
            0.0,
            1.0,
        )

        if abs_min_on_gate_th > 0.0 and abs_effective_min_u > 0.0:
            min_on_mask = abs_gate >= abs_min_on_gate_th
            target_u_abs = self.torch.clamp(
                self.torch.full_like(abs_gate, abs_effective_min_u)
                / self.torch.clamp(abs_gate, min=_NORM_EPS),
                0.0,
                1.0,
            )
            required_abs_increase = self.torch.clamp(target_u_abs - u_abs_safe, 0.0, 1.0)
            max_abs_increase_from_ech = self.torch.where(
                cooling_transfer_ratio > _NORM_EPS,
                u_ech_safe / self.torch.clamp(cooling_transfer_ratio, min=_NORM_EPS),
                self.torch.zeros_like(u_ech_safe),
            )
            abs_increase = self.torch.where(
                min_on_mask,
                self.torch.minimum(required_abs_increase, max_abs_increase_from_ech),
                self.torch.zeros_like(required_abs_increase),
            )
        else:
            abs_increase = self.torch.zeros_like(u_abs_safe)
        shaped_columns[u_abs_index] = self.torch.clamp(u_abs_safe + abs_increase, 0.0, 1.0)
        shaped_columns[u_ech_index] = self.torch.clamp(
            u_ech_safe - abs_increase * cooling_transfer_ratio,
            0.0,
            1.0,
        )
        return self.torch.cat(shaped_columns, dim=1)

    def _apply_abs_cooling_blend_np(
        self,
        *,
        observation_vector: np.ndarray,
        action_vector: np.ndarray,
    ) -> np.ndarray:
        blended = np.asarray(action_vector, dtype=np.float32).copy()

        if "u_bes" in self.action_index and "soc_bes" in self.observation_index:
            u_bes_index = int(self.action_index["u_bes"])
            soc_index = int(self.observation_index["soc_bes"])
            soc = float(observation_vector[soc_index])
            dt_h = max(_NORM_EPS, float(self.env_config.dt_hours))
            p_bes_cap_mw = max(_NORM_EPS, float(self.env_config.p_bes_cap_mw))
            e_bes_cap_mwh = max(_NORM_EPS, float(self.env_config.e_bes_cap_mwh))
            max_discharge_u = np.clip(
                ((soc - float(self.env_config.bes_soc_min)) * e_bes_cap_mwh * float(self.env_config.bes_eta_discharge) / dt_h)
                / p_bes_cap_mw,
                0.0,
                1.0,
            )
            max_charge_u = np.clip(
                ((float(self.env_config.bes_soc_max) - soc) * e_bes_cap_mwh / (max(_NORM_EPS, float(self.env_config.bes_eta_charge)) * dt_h))
                / p_bes_cap_mw,
                0.0,
                1.0,
            )
            blended[u_bes_index] = float(np.clip(float(blended[u_bes_index]), -max_charge_u, max_discharge_u))

        if "u_tes" in self.action_index and "e_tes_mwh" in self.observation_index:
            u_tes_index = int(self.action_index["u_tes"])
            e_tes_index = int(self.observation_index["e_tes_mwh"])
            e_tes = float(observation_vector[e_tes_index])
            dt_h = max(_NORM_EPS, float(self.env_config.dt_hours))
            e_tes_cap_mwh = max(_NORM_EPS, float(self.env_config.e_tes_cap_mwh))
            q_tes_charge_cap_mw = max(_NORM_EPS, float(self.env_config.q_tes_charge_cap_mw))
            q_tes_discharge_cap_mw = max(_NORM_EPS, float(self.env_config.q_tes_discharge_cap_mw))
            if "q_tes_discharge_feasible_mw" in self.observation_index:
                discharge_index = int(self.observation_index["q_tes_discharge_feasible_mw"])
                discharge_feasible_mw = float(observation_vector[discharge_index])
            else:
                discharge_feasible_mw = float(np.clip(e_tes / dt_h, 0.0, q_tes_discharge_cap_mw))
            charge_headroom_mw = float(np.clip((e_tes_cap_mwh - e_tes) / dt_h, 0.0, q_tes_charge_cap_mw))
            u_tes_max = float(np.clip(discharge_feasible_mw / q_tes_discharge_cap_mw, 0.0, 1.0))
            u_tes_min = -float(np.clip(charge_headroom_mw / q_tes_charge_cap_mw, 0.0, 1.0))
            blended[u_tes_index] = float(np.clip(float(blended[u_tes_index]), u_tes_min, u_tes_max))

        if "u_gt" in self.action_index and "p_gt_prev_mw" in self.observation_index:
            u_gt_index = int(self.action_index["u_gt"])
            p_gt_prev = float(observation_vector[int(self.observation_index["p_gt_prev_mw"])])
            p_gt_cap_mw = max(_NORM_EPS, float(self.env_config.p_gt_cap_mw))
            if (
                "gt_ramp_headroom_up_mw" in self.observation_index
                and "gt_ramp_headroom_down_mw" in self.observation_index
            ):
                ramp_up = float(observation_vector[int(self.observation_index["gt_ramp_headroom_up_mw"])])
                ramp_down = float(observation_vector[int(self.observation_index["gt_ramp_headroom_down_mw"])])
                p_gt_low = float(np.clip(p_gt_prev - ramp_down, 0.0, p_gt_cap_mw))
                p_gt_high = float(np.clip(p_gt_prev + ramp_up, 0.0, p_gt_cap_mw))
            else:
                p_gt_low = 0.0
                p_gt_high = p_gt_cap_mw
            p_gt_target = ((float(blended[u_gt_index]) + 1.0) * 0.5) * p_gt_cap_mw
            p_gt_target = _canonicalize_gt_target_np(
                p_gt_target_mw=p_gt_target,
                p_gt_low_mw=p_gt_low,
                p_gt_high_mw=p_gt_high,
                gt_min_output_mw=float(self.env_config.gt_min_output_mw),
                gt_off_deadband_ratio=float(self.config.gt_off_deadband_ratio),
            )
            blended[u_gt_index] = float(np.clip(2.0 * (p_gt_target / p_gt_cap_mw) - 1.0, -1.0, 1.0))

        if "abs_drive_margin_k" not in self.observation_index:
            return blended
        if "u_abs" not in self.action_index or "u_ech" not in self.action_index:
            return blended
        margin_index = int(self.observation_index["abs_drive_margin_k"])
        u_abs_index = int(self.action_index["u_abs"])
        u_ech_index = int(self.action_index["u_ech"])
        gate_scale_k = max(_NORM_EPS, float(self.env_config.abs_gate_scale_k))
        transfer_ratio = float(
            max(_NORM_EPS, float(self.env_config.q_abs_cool_cap_mw))
            / max(_NORM_EPS, float(self.env_config.q_ech_cap_mw))
        )
        invalid_req_u_th = float(max(0.0, float(self.env_config.abs_invalid_req_u_th)))
        invalid_req_gate_th = float(max(0.0, float(self.env_config.abs_invalid_req_gate_th)))
        abs_deadzone_gate_th = float(max(0.0, float(self.env_config.abs_deadzone_gate_th)))
        abs_deadzone_u_th = float(max(0.0, float(self.env_config.abs_deadzone_u_th)))
        abs_effective_min_u = float(max(0.0, abs_deadzone_u_th + float(self.config.abs_min_on_u_margin)))
        abs_min_on_gate_th = float(max(0.0, float(self.config.abs_min_on_gate_th)))
        margin = float(observation_vector[margin_index])
        abs_gate = float(1.0 / (1.0 + np.exp(-margin / gate_scale_k)))
        cooling_transfer_ratio = float(abs_gate * transfer_ratio)
        u_abs = float(blended[u_abs_index])
        u_ech = float(blended[u_ech_index])
        u_abs_safe = float(u_abs)
        if abs_deadzone_gate_th > 0.0 and abs_gate < abs_deadzone_gate_th:
            u_abs_safe = 0.0
        elif invalid_req_gate_th > abs_deadzone_gate_th and abs_gate < invalid_req_gate_th:
            u_abs_safe = float(min(u_abs_safe, invalid_req_u_th))
        if (u_abs_safe * abs_gate) < abs_deadzone_u_th and abs_gate < abs_min_on_gate_th:
            u_abs_safe = 0.0
        suppressed_abs = float(np.clip(u_abs - u_abs_safe, 0.0, 1.0))
        u_ech_safe = float(np.clip(u_ech + suppressed_abs * cooling_transfer_ratio, 0.0, 1.0))
        abs_increase = 0.0
        if abs_min_on_gate_th > 0.0 and abs_effective_min_u > 0.0 and abs_gate >= abs_min_on_gate_th:
            target_u_abs = float(np.clip(abs_effective_min_u / max(_NORM_EPS, abs_gate), 0.0, 1.0))
            required_abs_increase = float(np.clip(target_u_abs - u_abs_safe, 0.0, 1.0))
            if cooling_transfer_ratio > _NORM_EPS:
                max_abs_increase_from_ech = float(np.clip(u_ech_safe / cooling_transfer_ratio, 0.0, 1.0))
                abs_increase = float(min(required_abs_increase, max_abs_increase_from_ech))
        blended[u_abs_index] = float(np.clip(u_abs_safe + abs_increase, 0.0, 1.0))
        blended[u_ech_index] = float(np.clip(u_ech_safe - abs_increase * cooling_transfer_ratio, 0.0, 1.0))
        return blended

    def _resolve_prefill_checkpoint_artifact(
        self,
        *,
        checkpoint_path: str | Path,
    ) -> dict[str, Any]:
        entry_path = Path(str(checkpoint_path)).expanduser()
        if not entry_path.exists():
            raise FileNotFoundError(f"expert checkpoint 不存在: {entry_path}")

        if entry_path.suffix.lower() == ".json":
            payload = _json_payload_from_path(entry_path)
            artifact_type = str(payload.get("artifact_type", "")).strip().lower()
            if artifact_type == "pafc_td3_actor":
                resolved = payload.get("checkpoint_path")
                if not isinstance(resolved, str) or len(resolved.strip()) == 0:
                    raise ValueError("pafc_td3_actor.json 缺少 checkpoint_path。")
                resolved_path = Path(resolved).expanduser()
                if not resolved_path.exists():
                    raise FileNotFoundError(f"PAFC actor checkpoint 不存在: {resolved_path}")
                return {
                    "artifact_type": artifact_type,
                    "entry_path": entry_path,
                    "resolved_path": resolved_path,
                    "payload": payload,
                }
            if artifact_type == "sb3_policy":
                return {
                    "artifact_type": artifact_type,
                    "entry_path": entry_path,
                    "resolved_path": entry_path,
                    "payload": payload,
                }
            raise ValueError(
                "当前 expert_prefill checkpoint 仅支持 pafc_td3_actor(.json/.pt) 或 sb3 baseline_policy.json。"
            )

        if entry_path.suffix.lower() == ".pt":
            payload = load_policy(entry_path, map_location="cpu")
            metadata = dict(payload.get("metadata", {}) or {})
            artifact_type = str(metadata.get("artifact_type", "")).strip().lower()
            if artifact_type != "pafc_td3_actor":
                raise ValueError(
                    f"不支持的 PAFC expert checkpoint artifact_type: {artifact_type or 'unknown'}"
                )
            return {
                "artifact_type": artifact_type,
                "entry_path": entry_path,
                "resolved_path": entry_path,
                "payload": metadata,
            }

        raise ValueError(
            f"不支持的 expert checkpoint 后缀: {entry_path.suffix}。仅支持 .pt / .json。"
        )

    def _build_planner_expert_policy(
        self,
        *,
        planner_policy: str,
        role: str,
    ) -> tuple[Any, dict[str, Any]]:
        from ..pipeline.mpc import GAMPCPolicy, MILPMPCPolicy

        normalized_policy = str(planner_policy).strip().lower().replace("-", "_")
        planner_cls = {
            "milp_mpc": MILPMPCPolicy,
            "ga_mpc": GAMPCPolicy,
        }.get(normalized_policy)
        if planner_cls is None:
            raise ValueError(f"不支持的 economic planner teacher: {planner_policy}")
        planner = planner_cls(
            config=self.env_config,
            history_steps=max(1, int(round(float(self.env_config.oracle_mpc_planning_horizon_steps)))),
            seed=int(self.config.seed),
        )

        class _PlannerPrefillPolicy:
            returns_executed_action = True

            def __init__(self) -> None:
                self._last_decision: dict[str, Any] = {}

            def bind_episode_context(
                self,
                *,
                env: CCHPPhysicalEnv,
                episode_df: pd.DataFrame,
                initial_observation: dict[str, float],
                seed: int,
            ) -> None:
                planner.bind_episode_context(
                    env=env,
                    episode_df=episode_df,
                    initial_observation=initial_observation,
                    seed=seed,
                )
                self._last_decision = {}

            def reset_episode(self, observation: Mapping[str, float] | None = None) -> None:
                if observation is None:
                    observation = {}
                planner.reset_episode(dict(observation))
                self._last_decision = {}

            def act(self, observation: dict[str, float]) -> dict[str, float]:
                action = dict(planner.act(observation))
                self._last_decision = {
                    "source": str(role),
                    "artifact_type": "planner_policy",
                    "planner": str(normalized_policy),
                    "action_semantics": "executed",
                }
                return action

            def consume_last_decision(self) -> dict[str, Any]:
                return dict(self._last_decision)

        return _PlannerPrefillPolicy(), {
            "role": str(role),
            "artifact_type": "planner_policy",
            "planner": str(normalized_policy),
            "action_semantics": "executed",
            "planning_horizon_steps": int(round(float(self.env_config.oracle_mpc_planning_horizon_steps))),
            "replan_interval_steps": int(round(float(self.env_config.oracle_mpc_replan_interval_steps))),
            "oracle_mode": str(getattr(self.env_config, "oracle_mpc_mode", "")),
        }

    def _build_checkpoint_expert_policy(
        self,
        *,
        checkpoint_path: str | Path,
        role: str,
    ) -> tuple[Any, dict[str, Any]]:
        artifact = self._resolve_prefill_checkpoint_artifact(
            checkpoint_path=checkpoint_path,
        )
        artifact_type = str(artifact["artifact_type"])
        resolved_path = Path(artifact["resolved_path"])
        entry_path = Path(artifact["entry_path"])

        if artifact_type == "pafc_td3_actor":
            predictor, metadata = load_pafc_td3_predictor(
                checkpoint_path=resolved_path,
                device="cpu",
                env_config=self.env_config,
            )

            class _PAFCCheckpointPrefillPolicy:
                def __init__(self) -> None:
                    self._last_decision: dict[str, Any] = {}

                def reset_episode(self, observation: Mapping[str, float] | None = None) -> None:
                    del observation
                    self._last_decision = {}

                def act(self, observation: dict[str, float]) -> dict[str, float]:
                    action = dict(predictor(observation))
                    self._last_decision = {
                        "source": str(role),
                        "artifact_type": "pafc_td3_actor",
                    }
                    return action

                def consume_last_decision(self) -> dict[str, Any]:
                    return dict(self._last_decision)

            return _PAFCCheckpointPrefillPolicy(), {
                "role": str(role),
                "artifact_type": "pafc_td3_actor",
                "entry_path": str(entry_path.resolve()).replace("\\", "/"),
                "resolved_path": str(resolved_path.resolve()).replace("\\", "/"),
                "paper_model_label": str(metadata.get("paper_model_label", "")),
                "observation_keys": list(metadata.get("observation_keys", [])),
                "action_keys": list(metadata.get("action_keys", [])),
            }

        from .sb3 import (
            OBS_KEYS,
            RuleBasedDiscreteActionMapper,
            WindowBuffer,
            _action_vector_to_env_action,
            _action_vector_to_residual_delta,
            _build_observation_normalizer,
            _build_residual_expert_policy,
            _compose_residual_action,
            _observation_dict_to_vector,
            _require_sb3_modules,
            _resolve_checkpoint_sidecar_path,
            _resolve_sb3_eval_artifacts,
        )

        gym, spaces, PPO, SAC, TD3, DDPG, DQN, DummyVecEnv, VecNormalize = _require_sb3_modules()
        checkpoint_json_path = entry_path
        checkpoint_payload = dict(artifact["payload"])
        algo = str(checkpoint_payload.get("algo", "")).strip().lower()
        algo_cls = {
            "ppo": PPO,
            "sac": SAC,
            "td3": TD3,
            "ddpg": DDPG,
            "dqn": DQN,
        }.get(algo)
        if algo_cls is None:
            raise ValueError(f"未知 SB3 algo: {algo}")
        resolved_model_path, resolved_vecnormalize_path, resolved_model_source = _resolve_sb3_eval_artifacts(
            checkpoint_json=checkpoint_json_path,
            checkpoint_payload=checkpoint_payload,
            model_source="best",
        )
        history_steps = int(checkpoint_payload.get("history_steps", 1))
        if history_steps <= 0:
            raise ValueError("SB3 expert history_steps 必须 > 0。")
        observation_keys = tuple(
            str(item) for item in (checkpoint_payload.get("observation_keys") or OBS_KEYS)
        )
        residual_payload = checkpoint_payload.get("residual") or {}
        residual_enabled = bool(residual_payload.get("enabled", False))
        residual_policy_name = str(residual_payload.get("policy", "rule"))
        residual_scale = float(residual_payload.get("scale", 0.0))
        obs_norm = checkpoint_payload.get("obs_norm") or {}
        obs_norm_mode = ""
        if isinstance(obs_norm, dict):
            obs_norm_mode = str(obs_norm.get("mode", "")).strip().lower()
        train_statistics = None
        normalizer = None
        train_statistics_path = checkpoint_payload.get("train_statistics_path")
        needs_train_statistics = (
            algo == "dqn"
            or (resolved_vecnormalize_path is None and obs_norm_mode in {"zscore_affine_v1", "affine_v1"})
            or (residual_enabled and residual_policy_name == "rule")
        )
        if needs_train_statistics:
            if not train_statistics_path:
                raise ValueError("SB3 expert checkpoint 缺少 train_statistics_path。")
            resolved_train_statistics_path = _resolve_checkpoint_sidecar_path(
                checkpoint_json=checkpoint_json_path,
                path_value=str(train_statistics_path),
                artifact_label="SB3 训练统计文件（train_statistics.json）",
            )
            train_statistics = json.loads(
                resolved_train_statistics_path.read_text(encoding="utf-8")
            )
            if resolved_vecnormalize_path is None and obs_norm_mode in {"zscore_affine_v1", "affine_v1"}:
                normalizer = _build_observation_normalizer(
                    train_statistics=train_statistics,
                    env_config=self.env_config,
                    keys=observation_keys,
                )

        discrete_action_mapper = None
        if algo == "dqn":
            dqn_hparams = checkpoint_payload.get("dqn_hyperparameters") or {}
            discrete_action_mapper = RuleBasedDiscreteActionMapper(
                env_config=self.env_config,
                train_statistics=train_statistics if train_statistics is not None else {},
                action_mode=str(dqn_hparams.get("action_mode", "rb_v1")),
            )

        residual_policy = None
        if residual_enabled and discrete_action_mapper is None:
            residual_policy = _build_residual_expert_policy(
                policy_name=residual_policy_name,
                env_config=self.env_config,
                train_statistics=train_statistics,
            )

        observation_shape = (int(history_steps), int(len(observation_keys)))
        if discrete_action_mapper is None:
            action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(self.action_keys),),
                dtype=np.float32,
            )
        else:
            action_space = spaces.Discrete(discrete_action_mapper.action_count)
        observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=observation_shape,
            dtype=np.float32,
        )

        class _SB3PredictDummyEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self) -> None:
                super().__init__()
                self.observation_space = observation_space
                self.action_space = action_space

            def reset(self, *, seed: int | None = None, options: dict | None = None):
                del seed, options
                return np.zeros(observation_shape, dtype=np.float32), {}

            def step(self, action):
                del action
                return (
                    np.zeros(observation_shape, dtype=np.float32),
                    0.0,
                    True,
                    False,
                    {},
                )

        model_env = DummyVecEnv([_SB3PredictDummyEnv])
        vec_normalizer = None
        if resolved_vecnormalize_path is not None:
            vec_normalizer = VecNormalize.load(str(resolved_vecnormalize_path), model_env)
            vec_normalizer.training = False
            vec_normalizer.norm_reward = False
            model_env = vec_normalizer
        model = algo_cls.load(
            str(resolved_model_path),
            env=model_env,
            device="cpu",
        )

        class _SB3CheckpointPrefillPolicy:
            def __init__(self) -> None:
                self._buffer = WindowBuffer(
                    history_steps=int(history_steps),
                    obs_dim=len(observation_keys),
                )
                self._initialized = False
                self._last_decision: dict[str, Any] = {}

            def reset_episode(self, observation: Mapping[str, float] | None = None) -> None:
                del observation
                self._initialized = False
                self._last_decision = {}

            def act(self, observation: dict[str, float]) -> dict[str, float]:
                vector = _observation_dict_to_vector(observation, keys=observation_keys)
                if normalizer is not None:
                    vector = normalizer.apply(vector)
                if not self._initialized:
                    window = self._buffer.reset(vector)
                    self._initialized = True
                else:
                    window = self._buffer.push(vector)
                model_obs = window.reshape(1, *window.shape).astype(np.float32, copy=False)
                if vec_normalizer is not None:
                    model_obs = vec_normalizer.normalize_obs(model_obs)
                action, _ = model.predict(model_obs, deterministic=True)
                if discrete_action_mapper is not None:
                    action_dict = discrete_action_mapper.decode(action, observation)
                elif residual_policy is not None:
                    delta_action = _action_vector_to_residual_delta(action)
                    action_dict, _ = _compose_residual_action(
                        base_action=residual_policy.act(observation),
                        delta_action=delta_action,
                        residual_scale=float(residual_scale),
                    )
                else:
                    action_dict = _action_vector_to_env_action(action)
                self._last_decision = {
                    "source": str(role),
                    "artifact_type": "sb3_policy",
                    "algo": str(algo),
                    "paper_model_label": str(checkpoint_payload.get("paper_model_label", "")),
                }
                return dict(action_dict)

            def consume_last_decision(self) -> dict[str, Any]:
                return dict(self._last_decision)

        return _SB3CheckpointPrefillPolicy(), {
            "role": str(role),
            "artifact_type": "sb3_policy",
            "entry_path": str(entry_path.resolve()).replace("\\", "/"),
            "resolved_path": str(checkpoint_json_path.resolve()).replace("\\", "/"),
            "algo": str(algo),
            "paper_model_label": str(checkpoint_payload.get("paper_model_label", "")),
            "history_steps": int(history_steps),
            "model_source": str(resolved_model_source),
            "model_path": str(resolved_model_path.resolve()).replace("\\", "/"),
            "vecnormalize_path": (
                str(resolved_vecnormalize_path.resolve()).replace("\\", "/")
                if resolved_vecnormalize_path is not None
                else None
            ),
        }

    def _build_economic_teacher_policy(
        self,
        *,
        role: str,
    ) -> tuple[Any, dict[str, Any]]:
        economic_policy = str(self.config.expert_prefill_economic_policy)
        if economic_policy in {"milp_mpc", "ga_mpc"}:
            return self._build_planner_expert_policy(
                planner_policy=economic_policy,
                role=role,
            )
        checkpoint_path = str(self.config.expert_prefill_economic_checkpoint_path).strip()
        if len(checkpoint_path) == 0:
            raise ValueError(
                "expert_prefill_economic_policy=checkpoint 时必须提供 "
                "expert_prefill_economic_checkpoint_path。"
            )
        return self._build_checkpoint_expert_policy(
            checkpoint_path=checkpoint_path,
            role=role,
        )

    def _build_expert_policy(self) -> tuple[Any, dict[str, Any]]:
        from ..pipeline.runner import EasyRulePolicy, RulePolicy

        if str(self.config.expert_prefill_policy) == "checkpoint":
            return self._build_checkpoint_expert_policy(
                checkpoint_path=self.config.expert_prefill_checkpoint_path,
                role="primary",
            )
        if str(self.config.expert_prefill_policy) == "checkpoint_dual":
            safe_teacher, safe_info = self._build_checkpoint_expert_policy(
                checkpoint_path=self.config.expert_prefill_checkpoint_path,
                role="safe",
            )
            economic_teacher, economic_info = self._build_economic_teacher_policy(
                role="economic",
            )
            q_total_cooling_cap_mw = max(
                _NORM_EPS,
                float(self.env_config.q_abs_cool_cap_mw) + float(self.env_config.q_ech_cap_mw),
            )
            q_boiler_cap_mw = max(_NORM_EPS, float(self.env_config.q_boiler_cap_mw))
            dual_abs_margin_k = float(self.config.dual_abs_margin_k)
            dual_qc_ratio_th = float(self.config.dual_qc_ratio_th)
            dual_heat_backup_ratio_th = float(self.config.dual_heat_backup_ratio_th)
            dual_safe_abs_u_th = float(self.config.dual_safe_abs_u_th)

            class _DualCheckpointPrefillPolicy:
                def __init__(self) -> None:
                    self._last_decision: dict[str, Any] = {}

                def bind_episode_context(
                    self,
                    *,
                    env: CCHPPhysicalEnv,
                    episode_df: pd.DataFrame,
                    initial_observation: dict[str, float],
                    seed: int,
                ) -> None:
                    if hasattr(safe_teacher, "bind_episode_context"):
                        safe_teacher.bind_episode_context(
                            env=env,
                            episode_df=episode_df,
                            initial_observation=initial_observation,
                            seed=seed,
                        )
                    if hasattr(economic_teacher, "bind_episode_context"):
                        economic_teacher.bind_episode_context(
                            env=env,
                            episode_df=episode_df,
                            initial_observation=initial_observation,
                            seed=seed,
                        )
                    self._last_decision = {}

                def reset_episode(self, observation: Mapping[str, float] | None = None) -> None:
                    if hasattr(safe_teacher, "reset_episode"):
                        safe_teacher.reset_episode(observation=observation)
                    if hasattr(economic_teacher, "reset_episode"):
                        economic_teacher.reset_episode(observation=observation)
                    self._last_decision = {}

                def act(self, observation: dict[str, float]) -> dict[str, float]:
                    safe_action = dict(safe_teacher.act(observation))
                    economic_action = dict(economic_teacher.act(observation))
                    abs_margin_k = _safe_float(observation.get("abs_drive_margin_k", 0.0))
                    qc_ratio = _safe_float(observation.get("qc_dem_mw", 0.0)) / q_total_cooling_cap_mw
                    heat_backup_ratio = _safe_float(
                        observation.get("heat_backup_min_needed_mw", 0.0)
                    ) / q_boiler_cap_mw
                    safe_abs_u = _safe_float(safe_action.get("u_abs", 0.0))
                    gate_reasons: list[str] = []
                    if abs_margin_k <= dual_abs_margin_k:
                        gate_reasons.append("abs_margin_low")
                    if qc_ratio >= dual_qc_ratio_th:
                        gate_reasons.append("cooling_load_high")
                    if heat_backup_ratio >= dual_heat_backup_ratio_th:
                        gate_reasons.append("heat_backup_high")
                    if safe_abs_u >= dual_safe_abs_u_th:
                        gate_reasons.append("safe_abs_commit")
                    source = "safe" if gate_reasons else "economic"
                    selected_action = safe_action if source == "safe" else economic_action
                    self._last_decision = {
                        "source": str(source),
                        "artifact_type": "checkpoint_dual",
                        "gate_reasons": list(gate_reasons),
                        "abs_margin_k": float(abs_margin_k),
                        "qc_ratio": float(qc_ratio),
                        "heat_backup_ratio": float(heat_backup_ratio),
                        "safe_abs_u": float(safe_abs_u),
                    }
                    return dict(selected_action)

                def consume_last_decision(self) -> dict[str, Any]:
                    return dict(self._last_decision)

            return _DualCheckpointPrefillPolicy(), {
                "mode": "checkpoint_dual",
                "teachers": {
                    "safe": dict(safe_info),
                    "economic": dict(economic_info),
                },
            }
        if str(self.config.expert_prefill_policy) == "easy_rule_abs":
            return (
                EasyRuleAbsPolicy(
                    env_config=self.env_config,
                    p_gt_cap_mw=float(self.env_config.p_gt_cap_mw),
                    q_boiler_cap_mw=float(self.env_config.q_boiler_cap_mw),
                    q_ech_cap_mw=float(self.env_config.q_ech_cap_mw),
                    q_abs_cool_cap_mw=float(self.env_config.q_abs_cool_cap_mw),
                ),
                {"mode": "easy_rule_abs"},
            )
        if str(self.config.expert_prefill_policy) == "easy_rule":
            return (
                EasyRulePolicy(
                    p_gt_cap_mw=float(self.env_config.p_gt_cap_mw),
                    q_boiler_cap_mw=float(self.env_config.q_boiler_cap_mw),
                    q_ech_cap_mw=float(self.env_config.q_ech_cap_mw),
                ),
                {"mode": "easy_rule"},
            )
        return (
            RulePolicy(
                train_statistics=self.train_statistics,
                p_gt_cap_mw=float(self.env_config.p_gt_cap_mw),
                q_ech_cap_mw=float(self.env_config.q_ech_cap_mw),
                abs_drive_threshold_k=float(self.env_config.abs_t_drive_min_k),
            ),
            {"mode": "rule"},
        )

    def _build_bes_full_year_warm_start_policy(self) -> tuple[Any, dict[str, Any]]:
        safe_checkpoint_path = str(self.config.expert_prefill_checkpoint_path).strip()
        if len(safe_checkpoint_path) > 0:
            policy, info = self._build_checkpoint_expert_policy(
                checkpoint_path=safe_checkpoint_path,
                role="bes_full_year_safe",
            )
            summary = dict(info)
            summary["source"] = "safe_checkpoint"
            return policy, summary
        policy, info = self._build_expert_policy()
        summary = dict(info)
        summary["source"] = "expert_prefill_policy"
        return policy, summary

    def _build_bes_full_year_economic_policy(self) -> tuple[Any | None, dict[str, Any]]:
        economic_policy_name = str(self.config.expert_prefill_economic_policy).strip().lower()
        economic_checkpoint_path = str(self.config.expert_prefill_economic_checkpoint_path).strip()
        if economic_policy_name != "checkpoint" or len(economic_checkpoint_path) == 0:
            return None, {}
        safe_checkpoint_path = str(self.config.expert_prefill_checkpoint_path).strip()
        if safe_checkpoint_path == economic_checkpoint_path:
            return None, {}
        policy, info = self._build_checkpoint_expert_policy(
            checkpoint_path=economic_checkpoint_path,
            role="bes_full_year_economic",
        )
        summary = dict(info)
        summary["source"] = "economic_checkpoint"
        return policy, summary

    def _make_prefill_episode_df(self) -> pd.DataFrame:
        cooling_bias = float(self.config.expert_prefill_cooling_bias)
        episode_steps = int(self.config.episode_days) * 96
        max_start = max(0, int(len(self.train_df) - episode_steps))
        if max_start <= 0:
            return self.train_df.iloc[:episode_steps].reset_index(drop=True)

        if cooling_bias <= 0.0:
            start_index = int(self.rng.integers(0, max_start + 1))
        else:
            qc_series = pd.to_numeric(self.train_df["qc_dem_mw"], errors="coerce").fillna(0.0)
            rolling = qc_series.rolling(window=episode_steps, min_periods=episode_steps).mean()
            start_scores = (
                rolling.shift(-(episode_steps - 1))
                .iloc[: max_start + 1]
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            start_scores = np.maximum(start_scores, 0.0)
            base = np.full_like(start_scores, fill_value=1.0 / max(1, len(start_scores)), dtype=np.float64)
            if not np.isfinite(start_scores).any() or float(start_scores.sum()) <= 0.0:
                weights = base
            else:
                focused = start_scores / max(_NORM_EPS, float(start_scores.sum()))
                weights = (1.0 - cooling_bias) * base + cooling_bias * focused
                weights = weights / max(_NORM_EPS, float(weights.sum()))
            start_index = int(self.rng.choice(np.arange(len(weights), dtype=np.int64), p=weights))
        return self.train_df.iloc[start_index : start_index + episode_steps].reset_index(drop=True)

    def _rollout_expert_prefill_episode(
        self,
        *,
        expert_policy,
        expert_policy_info: dict[str, Any],
        episode_df: pd.DataFrame,
        episode_seed: int,
        max_steps: int,
        abs_exec_threshold: float,
        collect_teacher_targets: bool = True,
    ) -> dict[str, Any]:
        env = CCHPPhysicalEnv(
            exogenous_df=self.train_df,
            config=self.env_config,
            seed=int(episode_seed),
        )
        observation, _ = env.reset(
            seed=int(episode_seed),
            episode_df=episode_df,
        )
        self._bind_policy_episode_context(
            policy=expert_policy,
            env=env,
            observation=observation,
            episode_seed=episode_seed,
        )
        if hasattr(expert_policy, "reset_episode"):
            expert_policy.reset_episode(observation=observation)
        if collect_teacher_targets and self.economic_teacher_policy is not None:
            self._bind_policy_episode_context(
                policy=self.economic_teacher_policy,
                env=env,
                observation=observation,
                episode_seed=episode_seed,
            )
        if (
            collect_teacher_targets
            and self.economic_teacher_policy is not None
            and hasattr(self.economic_teacher_policy, "reset_episode")
        ):
            self.economic_teacher_policy.reset_episode(observation=observation)
        if collect_teacher_targets and self.economic_teacher_safe_policy is not None:
            self._bind_policy_episode_context(
                policy=self.economic_teacher_safe_policy,
                env=env,
                observation=observation,
                episode_seed=episode_seed,
            )
        if (
            collect_teacher_targets
            and self.economic_teacher_safe_policy is not None
            and hasattr(self.economic_teacher_safe_policy, "reset_episode")
        ):
            self.economic_teacher_safe_policy.reset_episode(observation=observation)
        transitions: list[dict[str, Any]] = []
        abs_exec_positive_count = 0
        bes_prior_override_count = 0
        bes_prior_charge_count = 0
        bes_prior_discharge_count = 0
        bes_charge_u, bes_discharge_u = self._resolve_bes_prior_u_pair()
        q_abs_cool_sum_mw = 0.0
        episode_steps = 0
        terminated = False
        teacher_source_counts: dict[str, int] = {}
        dual_gate_reason_counts: dict[str, int] = {}

        while (not terminated) and episode_steps < int(max_steps):
            obs_vector = self._observation_to_vector(observation)
            if collect_teacher_targets:
                teacher_action_exec, teacher_action_mask, teacher_available = (
                    self._get_economic_teacher_target_step(
                        observation=observation,
                        obs_vector=obs_vector,
                    )
                )
            else:
                teacher_action_exec = np.zeros((len(self.action_keys),), dtype=np.float32)
                teacher_action_mask = np.zeros((len(self.action_keys),), dtype=np.float32)
                teacher_available = False
            action_dict = dict(expert_policy.act(dict(observation)))
            decision_info = {}
            if hasattr(expert_policy, "consume_last_decision"):
                decision_info = dict(expert_policy.consume_last_decision() or {})
            teacher_source = str(
                decision_info.get("source")
                or expert_policy_info.get("mode")
                or self.config.expert_prefill_policy
            )
            teacher_source_counts[teacher_source] = teacher_source_counts.get(teacher_source, 0) + 1
            gate_reasons = decision_info.get("gate_reasons", [])
            if isinstance(gate_reasons, Sequence) and not isinstance(gate_reasons, (str, bytes)):
                for reason in gate_reasons:
                    label = str(reason)
                    dual_gate_reason_counts[label] = dual_gate_reason_counts.get(label, 0) + 1
            bes_prior = None
            if "u_bes" in action_dict:
                bes_prior = _bes_price_prior_target_np(
                    price_e=float(observation.get("price_e", 0.0)),
                    soc_bes=float(observation.get("soc_bes", 0.0)),
                    price_low_threshold=float(self.bes_price_low_threshold),
                    price_high_threshold=float(self.bes_price_high_threshold),
                    charge_soc_ceiling=float(self.config.economic_bes_charge_soc_ceiling),
                    discharge_soc_floor=float(self.config.economic_bes_discharge_soc_floor),
                    bes_soc_min=float(self.env_config.bes_soc_min),
                    bes_soc_max=float(self.env_config.bes_soc_max),
                    charge_u=float(bes_charge_u),
                    discharge_u=float(bes_discharge_u),
                )
                if float(bes_prior["opportunity"]) > 0.0:
                    action_dict["u_bes"] = float(bes_prior["target_u_bes"])
                    bes_prior_override_count += 1
                    if str(bes_prior["mode"]) == "charge":
                        bes_prior_charge_count += 1
                    elif str(bes_prior["mode"]) == "discharge":
                        bes_prior_discharge_count += 1
            action_raw = np.asarray(
                [float(action_dict[key]) for key in self.action_keys],
                dtype=np.float32,
            )
            next_observation, reward, terminated, _, info = env.step(action_dict)
            next_obs_vector = self._observation_to_vector(next_observation)
            action_exec = _extract_action_vector_from_info(
                info,
                prefix="action_exec",
                action_keys=self.action_keys,
            )
            cost = _extract_cost_vector(info)
            gap = _extract_gap_vector(info)
            q_abs_cool_mw = float(info.get("q_abs_cool_mw", 0.0))
            if float(action_exec[self.action_index["u_abs"]]) > float(abs_exec_threshold):
                abs_exec_positive_count += 1
            q_abs_cool_sum_mw += q_abs_cool_mw
            transitions.append(
                {
                    "obs": obs_vector.copy(),
                    "next_obs": next_obs_vector.copy(),
                    "action_raw": action_raw.copy(),
                    "action_exec": action_exec.copy(),
                    "teacher_action_exec": teacher_action_exec.copy(),
                    "teacher_action_mask": teacher_action_mask.copy(),
                    "teacher_available": bool(teacher_available),
                    "reward": float(reward),
                    "cost": cost.copy(),
                    "gap": gap.copy(),
                    "done": bool(terminated),
                    "q_abs_cool_mw": float(q_abs_cool_mw),
                    "teacher_source": str(teacher_source),
                    "gate_reasons": list(gate_reasons)
                    if isinstance(gate_reasons, Sequence) and not isinstance(gate_reasons, (str, bytes))
                    else [],
                }
            )
            observation = next_observation
            episode_steps += 1

        return {
            "bes_bidirectional_score": float(
                min(int(bes_prior_charge_count), int(bes_prior_discharge_count))
            ),
            "episode_df": episode_df,
            "transitions": transitions,
            "episode_steps": int(episode_steps),
            "abs_exec_positive_count": int(abs_exec_positive_count),
            "abs_exec_positive_rate": float(abs_exec_positive_count / max(1, episode_steps)),
            "bes_prior_override_count": int(bes_prior_override_count),
            "bes_prior_override_rate": float(bes_prior_override_count / max(1, episode_steps)),
            "bes_prior_charge_count": int(bes_prior_charge_count),
            "bes_prior_discharge_count": int(bes_prior_discharge_count),
            "q_abs_cool_sum_mw": float(q_abs_cool_sum_mw),
            "mining_score": float(
                abs_exec_positive_count
                + q_abs_cool_sum_mw
                + 0.10 * float(bes_prior_override_count)
                + 0.50 * float(min(int(bes_prior_charge_count), int(bes_prior_discharge_count)))
            ),
            "teacher_source_counts": {
                str(key): int(value) for key, value in teacher_source_counts.items()
            },
            "dual_gate_reason_counts": {
                str(key): int(value) for key, value in dual_gate_reason_counts.items()
            },
        }

    def _prefill_replay_with_expert(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target_steps = int(self.config.expert_prefill_steps)
        if target_steps <= 0:
            self.expert_prefill_summary = self._base_expert_prefill_summary()
            self.expert_prefill_summary.update(
                {
                    "enabled": False,
                    "steps": 0,
                    "status": "disabled",
                }
            )
            return (
                np.zeros((0, len(self.observation_keys)), dtype=np.float32),
                np.zeros((0, len(self.action_keys)), dtype=np.float32),
                np.zeros((0, len(self.action_keys)), dtype=np.float32),
                np.zeros((0, len(self.action_keys)), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
            )

        expert_policy, expert_policy_info = self._build_expert_policy()
        obs_rows: list[np.ndarray] = []
        action_exec_rows: list[np.ndarray] = []
        teacher_action_exec_rows: list[np.ndarray] = []
        teacher_action_mask_rows: list[np.ndarray] = []
        teacher_available_rows: list[np.ndarray] = []
        gap_rows: list[np.ndarray] = []
        cost_rows: list[np.ndarray] = []
        abs_positive_transition_rows: list[dict[str, Any]] = []
        teacher_target_transition_rows: list[dict[str, Any]] = []
        episode_qc_means: list[float] = []
        episode_start_timestamps: list[str] = []
        mining_score_rows: list[float] = []
        teacher_source_counts_total: dict[str, int] = {}
        dual_gate_reason_counts_total: dict[str, int] = {}
        bes_prior_override_total = 0
        bes_prior_charge_total = 0
        bes_prior_discharge_total = 0
        collected_steps = 0
        prefill_target_steps = 0
        episode_idx = 0
        abs_exec_threshold = float(self.config.expert_prefill_abs_exec_threshold)
        abs_window_mining_candidates = int(self.config.expert_prefill_abs_window_mining_candidates)
        prefill_window_step_cap = min(
            int(self.config.episode_days) * 96,
            max(96, int(target_steps // 4) if target_steps > 0 else 96),
        )

        while collected_steps < target_steps:
            selected_rollout: dict[str, Any] | None = None
            candidate_count = max(1, int(abs_window_mining_candidates))
            for candidate_idx in range(candidate_count):
                episode_df = self._make_prefill_episode_df()
                candidate_rollout = self._rollout_expert_prefill_episode(
                    expert_policy=expert_policy,
                    expert_policy_info=expert_policy_info,
                    episode_df=episode_df,
                    episode_seed=int(self.config.seed) + 20_000 + episode_idx * 100 + candidate_idx,
                    max_steps=prefill_window_step_cap,
                    abs_exec_threshold=abs_exec_threshold,
                )
                if (
                    selected_rollout is None
                    or float(candidate_rollout["mining_score"]) > float(selected_rollout["mining_score"])
                ):
                    selected_rollout = candidate_rollout
            if selected_rollout is None:
                raise RuntimeError("prefill episode mining 未能生成任何 rollout。")

            selected_episode_df = pd.DataFrame(selected_rollout["episode_df"]).reset_index(drop=True)
            episode_qc_means.append(
                float(pd.to_numeric(selected_episode_df["qc_dem_mw"], errors="coerce").fillna(0.0).mean())
            )
            episode_start_timestamps.append(str(selected_episode_df["timestamp"].iloc[0]))
            mining_score_rows.append(float(selected_rollout["mining_score"]))
            bes_prior_override_total += int(selected_rollout.get("bes_prior_override_count", 0))
            bes_prior_charge_total += int(selected_rollout.get("bes_prior_charge_count", 0))
            bes_prior_discharge_total += int(selected_rollout.get("bes_prior_discharge_count", 0))
            for transition in selected_rollout["transitions"]:
                if collected_steps >= target_steps:
                    break
                self.replay.add(
                    obs=transition["obs"],
                    next_obs=transition["next_obs"],
                    action_raw=transition["action_raw"],
                    action_exec=transition["action_exec"],
                    teacher_action_exec=transition["teacher_action_exec"],
                    teacher_action_mask=transition["teacher_action_mask"],
                    teacher_available=bool(transition["teacher_available"]),
                    reward=float(transition["reward"]),
                    cost=transition["cost"],
                    gap=transition["gap"],
                    done=bool(transition["done"]),
                )
                obs_rows.append(np.asarray(transition["obs"], dtype=np.float32).copy())
                action_exec_rows.append(np.asarray(transition["action_exec"], dtype=np.float32).copy())
                teacher_action_exec_rows.append(
                    np.asarray(transition["teacher_action_exec"], dtype=np.float32).copy()
                )
                teacher_action_mask_rows.append(
                    np.asarray(transition["teacher_action_mask"], dtype=np.float32).copy()
                )
                teacher_available_rows.append(
                    np.asarray([float(bool(transition["teacher_available"]))], dtype=np.float32)
                )
                gap_rows.append(np.asarray(transition["gap"], dtype=np.float32).copy())
                cost_rows.append(np.asarray(transition["cost"], dtype=np.float32).copy())
                teacher_source = str(transition.get("teacher_source", "")).strip()
                if teacher_source:
                    teacher_source_counts_total[teacher_source] = (
                        teacher_source_counts_total.get(teacher_source, 0) + 1
                    )
                if bool(transition.get("teacher_available", False)):
                    prefill_target_steps += 1
                    transition_teacher_mask = np.asarray(
                        transition["teacher_action_mask"],
                        dtype=np.float32,
                    ).copy()
                    if np.any(transition_teacher_mask > 0.5):
                        teacher_target_transition_rows.append(
                            {
                                "obs": np.asarray(transition["obs"], dtype=np.float32).copy(),
                                "next_obs": np.asarray(transition["next_obs"], dtype=np.float32).copy(),
                                "action_raw": np.asarray(transition["action_raw"], dtype=np.float32).copy(),
                                "action_exec": np.asarray(transition["action_exec"], dtype=np.float32).copy(),
                                "teacher_action_exec": np.asarray(
                                    transition["teacher_action_exec"],
                                    dtype=np.float32,
                                ).copy(),
                                "teacher_action_mask": transition_teacher_mask,
                                "teacher_available": bool(transition["teacher_available"]),
                                "reward": float(transition["reward"]),
                                "cost": np.asarray(transition["cost"], dtype=np.float32).copy(),
                                "gap": np.asarray(transition["gap"], dtype=np.float32).copy(),
                                "done": bool(transition["done"]),
                            }
                        )
                gate_reasons = transition.get("gate_reasons", [])
                if isinstance(gate_reasons, Sequence) and not isinstance(gate_reasons, (str, bytes)):
                    for reason in gate_reasons:
                        label = str(reason)
                        dual_gate_reason_counts_total[label] = (
                            dual_gate_reason_counts_total.get(label, 0) + 1
                        )
                if float(transition["action_exec"][self.action_index["u_abs"]]) > abs_exec_threshold:
                    abs_positive_transition_rows.append(
                        {
                            "obs": np.asarray(transition["obs"], dtype=np.float32).copy(),
                            "next_obs": np.asarray(transition["next_obs"], dtype=np.float32).copy(),
                            "action_raw": np.asarray(transition["action_raw"], dtype=np.float32).copy(),
                            "action_exec": np.asarray(transition["action_exec"], dtype=np.float32).copy(),
                            "teacher_action_exec": np.asarray(
                                transition["teacher_action_exec"],
                                dtype=np.float32,
                            ).copy(),
                            "teacher_action_mask": np.asarray(
                                transition["teacher_action_mask"],
                                dtype=np.float32,
                            ).copy(),
                            "teacher_available": bool(transition["teacher_available"]),
                            "reward": float(transition["reward"]),
                            "cost": np.asarray(transition["cost"], dtype=np.float32).copy(),
                            "gap": np.asarray(transition["gap"], dtype=np.float32).copy(),
                            "done": bool(transition["done"]),
                        }
                    )
                collected_steps += 1
            episode_idx += 1

        abs_replay_duplicates_added = 0
        teacher_replay_duplicates_added = 0
        abs_replay_boost = int(self.config.expert_prefill_abs_replay_boost)
        if abs_replay_boost > 0 and abs_positive_transition_rows:
            for transition in abs_positive_transition_rows:
                for _ in range(abs_replay_boost):
                    self.replay.add(
                        obs=transition["obs"],
                        next_obs=transition["next_obs"],
                        action_raw=transition["action_raw"],
                        action_exec=transition["action_exec"],
                        teacher_action_exec=transition["teacher_action_exec"],
                        teacher_action_mask=transition["teacher_action_mask"],
                        teacher_available=bool(transition["teacher_available"]),
                        reward=float(transition["reward"]),
                        cost=transition["cost"],
                        gap=transition["gap"],
                        done=bool(transition["done"]),
                    )
                    obs_rows.append(np.asarray(transition["obs"], dtype=np.float32).copy())
                    action_exec_rows.append(np.asarray(transition["action_exec"], dtype=np.float32).copy())
                    teacher_action_exec_rows.append(
                        np.asarray(transition["teacher_action_exec"], dtype=np.float32).copy()
                    )
                    teacher_action_mask_rows.append(
                        np.asarray(transition["teacher_action_mask"], dtype=np.float32).copy()
                    )
                    teacher_available_rows.append(
                        np.asarray([float(bool(transition["teacher_available"]))], dtype=np.float32)
                    )
                    abs_replay_duplicates_added += 1
        teacher_replay_boost = int(self.config.economic_teacher_prefill_replay_boost)
        if teacher_replay_boost > 0 and teacher_target_transition_rows:
            for transition in teacher_target_transition_rows:
                for _ in range(teacher_replay_boost):
                    self.replay.add(
                        obs=transition["obs"],
                        next_obs=transition["next_obs"],
                        action_raw=transition["action_raw"],
                        action_exec=transition["action_exec"],
                        teacher_action_exec=transition["teacher_action_exec"],
                        teacher_action_mask=transition["teacher_action_mask"],
                        teacher_available=bool(transition["teacher_available"]),
                        reward=float(transition["reward"]),
                        cost=transition["cost"],
                        gap=transition["gap"],
                        done=bool(transition["done"]),
                    )
                    obs_rows.append(np.asarray(transition["obs"], dtype=np.float32).copy())
                    action_exec_rows.append(np.asarray(transition["action_exec"], dtype=np.float32).copy())
                    teacher_action_exec_rows.append(
                        np.asarray(transition["teacher_action_exec"], dtype=np.float32).copy()
                    )
                    teacher_action_mask_rows.append(
                        np.asarray(transition["teacher_action_mask"], dtype=np.float32).copy()
                    )
                    teacher_available_rows.append(
                        np.asarray([float(bool(transition["teacher_available"]))], dtype=np.float32)
                    )
                    teacher_replay_duplicates_added += 1

        gap_mean = (
            np.mean(np.asarray(gap_rows, dtype=np.float32), axis=0).astype(float).tolist()
            if gap_rows
            else [0.0, 0.0, 0.0]
        )
        cost_mean = (
            np.mean(np.asarray(cost_rows, dtype=np.float32), axis=0).astype(float).tolist()
            if cost_rows
            else [0.0, 0.0, 0.0]
        )
        self.expert_prefill_summary = self._base_expert_prefill_summary()
        self.expert_prefill_summary.update(
            {
                "enabled": True,
                "steps": int(collected_steps),
                "status": "applied",
                "teachers": dict(expert_policy_info),
                "episode_count": int(episode_idx),
                "unique_window_count": int(len(set(episode_start_timestamps))),
                "window_step_cap": int(prefill_window_step_cap),
                "abs_exec_positive_count": int(len(abs_positive_transition_rows)),
                "abs_exec_positive_rate": float(len(abs_positive_transition_rows) / max(1, collected_steps)),
                "teacher_target_transition_count": int(len(teacher_target_transition_rows)),
                "teacher_target_transition_rate": float(
                    len(teacher_target_transition_rows) / max(1, collected_steps)
                ),
                "bes_prior_override_count": int(bes_prior_override_total),
                "bes_prior_override_rate": float(bes_prior_override_total / max(1, collected_steps)),
                "bes_prior_charge_count": int(bes_prior_charge_total),
                "bes_prior_discharge_count": int(bes_prior_discharge_total),
                "abs_replay_duplicates_added": int(abs_replay_duplicates_added),
                "teacher_replay_duplicates_added": int(teacher_replay_duplicates_added),
                "episode_qc_dem_mean": (
                    float(np.mean(np.asarray(episode_qc_means, dtype=np.float32)))
                    if episode_qc_means
                    else 0.0
                ),
                "selected_mining_score_mean": (
                    float(np.mean(np.asarray(mining_score_rows, dtype=np.float32)))
                    if mining_score_rows
                    else 0.0
                ),
                "gap_mean": {
                    "projection_gap_l1": float(gap_mean[0]),
                    "projection_gap_l2": float(gap_mean[1]),
                    "projection_gap_max": float(gap_mean[2]),
                },
                "cost_mean": {
                    key: float(value)
                    for key, value in zip(_COST_KEYS, cost_mean)
                },
            }
        )
        if self.economic_teacher_policy is not None:
            self.economic_teacher_distill_summary["prefill_target_steps"] = int(prefill_target_steps)
        if teacher_source_counts_total:
            self.expert_prefill_summary["teacher_source_counts"] = {
                str(key): int(value)
                for key, value in teacher_source_counts_total.items()
            }
            self.expert_prefill_summary["teacher_source_rates"] = {
                str(key): float(int(value) / max(1, collected_steps))
                for key, value in teacher_source_counts_total.items()
            }
        if dual_gate_reason_counts_total:
            self.expert_prefill_summary["dual_gate_reason_counts"] = {
                str(key): int(value)
                for key, value in dual_gate_reason_counts_total.items()
            }
        return (
            np.asarray(obs_rows, dtype=np.float32),
            np.asarray(action_exec_rows, dtype=np.float32),
            np.asarray(teacher_action_exec_rows, dtype=np.float32),
            np.asarray(teacher_action_mask_rows, dtype=np.float32),
            np.asarray(teacher_available_rows, dtype=np.float32),
        )

    def _build_actor_warm_start_targets(
        self,
        *,
        observations: np.ndarray,
        action_exec_targets: np.ndarray,
        teacher_action_exec: np.ndarray,
        teacher_action_mask: np.ndarray,
        teacher_available: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        warm_targets = np.asarray(action_exec_targets, dtype=np.float32).copy()
        sample_weight_boost = np.ones((len(warm_targets), 1), dtype=np.float32)
        summary = {
            "teacher_target_row_rate": 0.0,
            "teacher_target_dim_rate": 0.0,
            "bes_price_prior_row_rate": 0.0,
            "bes_price_prior_charge_rate": 0.0,
            "bes_price_prior_discharge_rate": 0.0,
            "sample_weight_boost_mean": 1.0,
            "price_low_threshold": float(self.bes_price_low_threshold),
            "price_high_threshold": float(self.bes_price_high_threshold),
            "charge_u_scale": float(self.config.economic_bes_charge_u_scale),
            "discharge_u_scale": float(self.config.economic_bes_discharge_u_scale),
            "charge_weight": float(self.config.economic_bes_charge_weight),
            "discharge_weight": float(self.config.economic_bes_discharge_weight),
            "charge_pressure_bonus": float(self.config.economic_bes_charge_pressure_bonus),
            "teacher_warm_start_weight": float(self.config.economic_teacher_warm_start_weight),
            "teacher_sample_weight_boost_mean": 0.0,
            "teacher_sample_weight_boost_max": 0.0,
        }
        if warm_targets.size == 0:
            return warm_targets, sample_weight_boost, summary

        teacher_exec = np.asarray(teacher_action_exec, dtype=np.float32)
        teacher_mask = np.asarray(teacher_action_mask, dtype=np.float32)
        teacher_flag = np.asarray(teacher_available, dtype=np.float32).reshape(-1, 1)
        teacher_rows = 0
        teacher_dims = 0
        teacher_sample_weight_boost_sum = 0.0
        teacher_sample_weight_boost_max = 0.0
        bes_prior_rows = 0
        bes_prior_charge_rows = 0
        bes_prior_discharge_rows = 0
        u_bes_index = self.action_index.get("u_bes")
        charge_u, discharge_u = self._resolve_bes_prior_u_pair()

        for row_index in range(len(warm_targets)):
            teacher_bes_override = False
            if (
                row_index < len(teacher_exec)
                and row_index < len(teacher_mask)
                and row_index < len(teacher_flag)
                and float(teacher_flag[row_index, 0]) > 0.5
            ):
                active_mask = teacher_mask[row_index] > 0.5
                if np.any(active_mask):
                    warm_targets[row_index, active_mask] = teacher_exec[row_index, active_mask]
                    teacher_rows += 1
                    teacher_dims += int(active_mask.sum())
                    teacher_dim_weight = float(
                        np.mean(self.economic_teacher_action_weight_np[active_mask])
                    )
                    teacher_row_boost = float(self.config.economic_teacher_warm_start_weight) * teacher_dim_weight
                    sample_weight_boost[row_index, 0] += teacher_row_boost
                    teacher_sample_weight_boost_sum += teacher_row_boost
                    teacher_sample_weight_boost_max = max(
                        teacher_sample_weight_boost_max,
                        teacher_row_boost,
                    )
                    if u_bes_index is not None:
                        teacher_bes_override = bool(active_mask[int(u_bes_index)])

            if u_bes_index is None or teacher_bes_override:
                continue
            prior = _bes_price_prior_target_np(
                price_e=float(observations[row_index, self.observation_index["price_e"]]),
                soc_bes=float(observations[row_index, self.observation_index["soc_bes"]]),
                price_low_threshold=float(self.bes_price_low_threshold),
                price_high_threshold=float(self.bes_price_high_threshold),
                charge_soc_ceiling=float(self.config.economic_bes_charge_soc_ceiling),
                discharge_soc_floor=float(self.config.economic_bes_discharge_soc_floor),
                bes_soc_min=float(self.env_config.bes_soc_min),
                bes_soc_max=float(self.env_config.bes_soc_max),
                charge_u=float(charge_u),
                discharge_u=float(discharge_u),
            )
            if float(prior["opportunity"]) <= 0.0:
                continue
            warm_targets[row_index, int(u_bes_index)] = float(prior["target_u_bes"])
            sample_weight_boost[row_index, 0] += float(prior["opportunity"]) * float(
                _bes_prior_weight_multiplier_np(
                    mode=str(prior["mode"]),
                    charge_score=float(prior.get("charge_score", 0.0)),
                    charge_weight=float(self.config.economic_bes_charge_weight),
                    discharge_weight=float(self.config.economic_bes_discharge_weight),
                    charge_pressure_bonus=float(self.config.economic_bes_charge_pressure_bonus),
                )
            )
            bes_prior_rows += 1
            if str(prior["mode"]) == "charge":
                bes_prior_charge_rows += 1
            elif str(prior["mode"]) == "discharge":
                bes_prior_discharge_rows += 1

        total_rows = max(1, len(warm_targets))
        total_dims = max(1, len(warm_targets) * len(self.action_keys))
        summary.update(
            {
                "teacher_target_row_rate": float(teacher_rows / total_rows),
                "teacher_target_dim_rate": float(teacher_dims / total_dims),
                "teacher_sample_weight_boost_mean": float(
                    teacher_sample_weight_boost_sum / max(1, teacher_rows)
                ),
                "teacher_sample_weight_boost_max": float(teacher_sample_weight_boost_max),
                "bes_price_prior_row_rate": float(bes_prior_rows / total_rows),
                "bes_price_prior_charge_rate": float(bes_prior_charge_rows / total_rows),
                "bes_price_prior_discharge_rate": float(bes_prior_discharge_rows / total_rows),
                "sample_weight_boost_mean": float(sample_weight_boost.mean()),
            }
        )
        return warm_targets, sample_weight_boost, summary

    def _warm_start_actor_from_expert(
        self,
        *,
        observations: np.ndarray,
        action_exec_targets: np.ndarray,
        teacher_action_exec: np.ndarray,
        teacher_action_mask: np.ndarray,
        teacher_available: np.ndarray,
    ) -> None:
        if str(self.config.expert_prefill_policy) == "checkpoint":
            artifact = self._resolve_prefill_checkpoint_artifact(
                checkpoint_path=self.config.expert_prefill_checkpoint_path,
            )
            if str(artifact["artifact_type"]) != "pafc_td3_actor":
                self.actor_init_summary = {
                    "enabled": False,
                    "status": "skipped_non_pafc_checkpoint",
                    "artifact_type": str(artifact["artifact_type"]),
                    "checkpoint_path": str(Path(artifact["entry_path"]).resolve()).replace("\\", "/"),
                }
            else:
                payload = load_policy(
                    artifact["resolved_path"],
                    map_location=str(self.device),
                )
                self.actor.load_state_dict(payload["state_dict"])
                self.actor_target.load_state_dict(self.actor.state_dict())
                metadata = dict(payload.get("metadata", {}) or {})
                dual_lambdas = dict(metadata.get("dual_lambdas", {}) or {})
                if all(name in dual_lambdas for name in _DUAL_NAMES):
                    self.dual_lambdas = np.asarray(
                        [float(dual_lambdas[name]) for name in _DUAL_NAMES],
                        dtype=np.float32,
                    )
                _, _, _, AdamW = _require_torch_modules()
                self.actor_optimizer = AdamW(self.actor.parameters(), lr=float(self.config.actor_lr))
                self.current_actor_lr = float(self.config.actor_lr)
                self.actor_init_summary = {
                    "enabled": True,
                    "status": "loaded",
                    "checkpoint_path": str(Path(artifact["resolved_path"]).resolve()).replace("\\", "/"),
                    "checkpoint_entry_path": str(Path(artifact["entry_path"]).resolve()).replace("\\", "/"),
                    "dual_lambdas_loaded": {
                        name: float(value)
                        for name, value in zip(_DUAL_NAMES, self.dual_lambdas)
                    },
                }
        elif str(self.config.expert_prefill_policy) == "checkpoint_dual":
            self.actor_init_summary = {
                "enabled": False,
                "status": "skipped_checkpoint_dual_prefill_only",
                "checkpoint_path": str(self.config.expert_prefill_checkpoint_path),
                "reason": "checkpoint_dual 采用混合 prefill/warm-start，避免直接把 actor 锁死在 safe teacher 上。",
            }
        if observations.size == 0 or action_exec_targets.size == 0:
            self.actor_warm_start_summary = {
                "enabled": False,
                "epochs": int(self.config.actor_warm_start_epochs),
                "status": "no_data",
            }
            return
        if int(self.config.actor_warm_start_epochs) <= 0:
            self.actor_warm_start_summary = {
                "enabled": False,
                "epochs": int(self.config.actor_warm_start_epochs),
                "status": "disabled",
                "samples": int(len(observations)),
            }
            return

        _, _, _, AdamW = _require_torch_modules()
        optimizer = AdamW(self.actor.parameters(), lr=float(self.config.actor_warm_start_lr))
        obs_tensor = self.torch.as_tensor(observations, dtype=self.torch.float32, device=self.device)
        warm_targets_np, sample_weight_boost_np, warm_target_summary = self._build_actor_warm_start_targets(
            observations=observations,
            action_exec_targets=action_exec_targets,
            teacher_action_exec=teacher_action_exec,
            teacher_action_mask=teacher_action_mask,
            teacher_available=teacher_available,
        )
        base_action_tensor = self.torch.as_tensor(
            action_exec_targets,
            dtype=self.torch.float32,
            device=self.device,
        )
        warm_action_tensor = self.torch.as_tensor(
            warm_targets_np,
            dtype=self.torch.float32,
            device=self.device,
        )
        delta_mask_tensor = self.torch.as_tensor(
            (np.abs(warm_targets_np - action_exec_targets) > 1e-6).astype(np.float32),
            dtype=self.torch.float32,
            device=self.device,
        )
        teacher_target_mask_tensor = self.torch.as_tensor(
            (
                np.asarray(teacher_action_mask, dtype=np.float32)
                * np.asarray(teacher_available, dtype=np.float32).reshape(-1, 1)
            ),
            dtype=self.torch.float32,
            device=self.device,
        )
        focus_weight_tensor = self._compute_abs_focus_weights(
            obs_batch=obs_tensor,
            action_exec_batch=base_action_tensor,
        ).detach()
        delta_weight_tensor = focus_weight_tensor * self.torch.as_tensor(
            sample_weight_boost_np,
            dtype=self.torch.float32,
            device=self.device,
        )
        batch_size = min(int(self.config.actor_warm_start_batch_size), int(obs_tensor.shape[0]))
        indices = np.arange(int(obs_tensor.shape[0]), dtype=np.int64)
        epoch_losses: list[float] = []

        for _ in range(int(self.config.actor_warm_start_epochs)):
            self.rng.shuffle(indices)
            batch_losses: list[float] = []
            for start in range(0, int(len(indices)), int(batch_size)):
                batch_indices = indices[start : start + int(batch_size)]
                batch_obs = obs_tensor[batch_indices]
                batch_base_target = base_action_tensor[batch_indices]
                batch_target = warm_action_tensor[batch_indices]
                batch_delta_mask = delta_mask_tensor[batch_indices]
                batch_teacher_target_mask = teacher_target_mask_tensor[batch_indices]
                batch_focus_weight = focus_weight_tensor[batch_indices]
                batch_delta_weight = delta_weight_tensor[batch_indices]
                prediction = self.actor(self._normalize_observation_tensor(batch_obs))
                prediction = self._apply_abs_cooling_blend_tensor(
                    obs_batch=batch_obs,
                    action_batch=prediction,
                )
                batch_base_mask = self.torch.clamp(
                    1.0 - batch_teacher_target_mask,
                    0.0,
                    1.0,
                )
                safe_loss = (
                    batch_focus_weight
                    * (
                        ((prediction - batch_base_target).pow(2) * batch_base_mask).sum(
                            dim=1,
                            keepdim=True,
                        )
                        / batch_base_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                    )
                )
                delta_loss = batch_delta_weight * (
                    ((prediction - batch_target).pow(2) * batch_delta_mask).sum(dim=1, keepdim=True)
                    / batch_delta_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                )
                loss = (safe_loss + delta_loss).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))
            epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = AdamW(self.actor.parameters(), lr=float(self.config.actor_lr))
        self.current_actor_lr = float(self.config.actor_lr)
        self.actor_warm_start_summary = {
            "enabled": True,
            "epochs": int(self.config.actor_warm_start_epochs),
            "samples": int(obs_tensor.shape[0]),
            "batch_size": int(batch_size),
            "lr": float(self.config.actor_warm_start_lr),
            "status": "applied",
            "abs_ready_focus_coef": float(self.config.abs_ready_focus_coef),
            "focus_weight_mean": float(focus_weight_tensor.mean().detach().cpu().item()),
            "focus_weight_max": float(delta_weight_tensor.max().detach().cpu().item()),
            "abs_exec_positive_rate": float(
                (warm_action_tensor[:, self.action_index["u_abs"]] > 0.01)
                .float()
                .mean()
                .detach()
                .cpu()
                .item()
            ),
            "loss_first": float(epoch_losses[0]) if epoch_losses else 0.0,
            "loss_last": float(epoch_losses[-1]) if epoch_losses else 0.0,
            "teacher_target_row_rate": float(warm_target_summary["teacher_target_row_rate"]),
            "teacher_target_dim_rate": float(warm_target_summary["teacher_target_dim_rate"]),
            "teacher_warm_start_weight": float(warm_target_summary["teacher_warm_start_weight"]),
            "teacher_sample_weight_boost_mean": float(
                warm_target_summary["teacher_sample_weight_boost_mean"]
            ),
            "teacher_sample_weight_boost_max": float(
                warm_target_summary["teacher_sample_weight_boost_max"]
            ),
            "bes_price_prior_row_rate": float(warm_target_summary["bes_price_prior_row_rate"]),
            "bes_price_prior_charge_rate": float(warm_target_summary["bes_price_prior_charge_rate"]),
            "bes_price_prior_discharge_rate": float(
                warm_target_summary["bes_price_prior_discharge_rate"]
            ),
            "sample_weight_boost_mean": float(warm_target_summary["sample_weight_boost_mean"]),
            "price_low_threshold": float(warm_target_summary["price_low_threshold"]),
            "price_high_threshold": float(warm_target_summary["price_high_threshold"]),
        }

    def _compute_economic_teacher_full_year_row_priority(
        self,
        *,
        obs_vector: np.ndarray,
        action_exec: np.ndarray,
        teacher_action_exec: np.ndarray,
        teacher_action_mask: np.ndarray,
    ) -> dict[str, float | bool]:
        active_mask = np.asarray(teacher_action_mask, dtype=np.float32).reshape(-1) > 0.5
        if not np.any(active_mask):
            return {
                "priority": 0.0,
                "sample_weight": 0.0,
                "dim_weight": 0.0,
                "delta_mean": 0.0,
                "gt_bonus": 0.0,
                "gt_selected": False,
            }
        action_exec_np = np.asarray(action_exec, dtype=np.float32).reshape(-1)
        teacher_action_exec_np = np.asarray(teacher_action_exec, dtype=np.float32).reshape(-1)
        active_weights = np.asarray(
            self.economic_teacher_action_weight_np[active_mask],
            dtype=np.float32,
        )
        dim_weight = float(active_weights.mean()) if active_weights.size > 0 else 1.0
        delta_mean = float(
            np.abs(teacher_action_exec_np[active_mask] - action_exec_np[active_mask]).mean()
        )
        priority = max(0.0, dim_weight + delta_mean)
        sample_weight = 1.0 + float(self.config.economic_teacher_warm_start_weight) * max(
            0.0,
            dim_weight,
        )
        gt_bonus = 0.0
        gt_selected = False
        u_gt_index = self.action_index.get("u_gt")
        if u_gt_index is not None and bool(active_mask[int(u_gt_index)]):
            gt_selected = True
            required_obs = {
                "p_dem_mw",
                "pv_mw",
                "wt_mw",
                "price_e",
                "price_gas",
            }
            if required_obs.issubset(self.observation_index):
                p_dem = float(obs_vector[self.observation_index["p_dem_mw"]])
                pv = float(obs_vector[self.observation_index["pv_mw"]])
                wt = float(obs_vector[self.observation_index["wt_mw"]])
                price_e = max(_NORM_EPS, float(obs_vector[self.observation_index["price_e"]]))
                price_gas = float(obs_vector[self.observation_index["price_gas"]])
                p_bes_charge_proxy_mw = 0.0
                p_bes_discharge_proxy_mw = 0.0
                if "u_bes" in self.action_index:
                    u_bes_exec = float(
                        np.clip(
                            teacher_action_exec_np[int(self.action_index["u_bes"])],
                            -1.0,
                            1.0,
                        )
                    )
                    p_bes_charge_proxy_mw = (
                        max(0.0, -u_bes_exec)
                        * float(self.env_config.p_bes_cap_mw)
                        / max(_NORM_EPS, float(self.env_config.bes_eta_charge))
                    )
                    p_bes_discharge_proxy_mw = (
                        max(0.0, u_bes_exec)
                        * float(self.env_config.p_bes_cap_mw)
                        * float(self.env_config.bes_eta_discharge)
                    )
                p_ech_proxy_mw = 0.0
                if "u_ech" in self.action_index:
                    u_ech_exec = float(
                        np.clip(
                            teacher_action_exec_np[int(self.action_index["u_ech"])],
                            0.0,
                            1.0,
                        )
                    )
                    if "t_amb_k" in self.observation_index:
                        t_amb_k = float(obs_vector[self.observation_index["t_amb_k"]])
                    else:
                        t_amb_k = float(getattr(self.env_config, "t_amb_design_k", 298.15))
                    q_ech_proxy_mw = u_ech_exec * float(self.env_config.q_ech_cap_mw)
                    ech_cop = np.clip(
                        float(self.env_config.cop_nominal) - 0.03 * (t_amb_k - 298.15),
                        float(self.env_config.cop_nominal)
                        * float(self.env_config.ech_cop_partload_min_fraction),
                        float(self.env_config.cop_nominal),
                    )
                    p_ech_proxy_mw = q_ech_proxy_mw / max(_NORM_EPS, float(ech_cop))
                p_gt_cap_mw = max(_NORM_EPS, float(self.env_config.p_gt_cap_mw))
                p_gt_exec = (
                    (float(teacher_action_exec_np[int(u_gt_index)]) + 1.0) * 0.5 * p_gt_cap_mw
                )
                gt_load_ratio = float(np.clip(p_gt_exec / p_gt_cap_mw, 0.0, 1.0))
                eta_gt = float(self.env_config.gt_eta_min) + (
                    float(self.env_config.gt_eta_max) - float(self.env_config.gt_eta_min)
                ) * gt_load_ratio
                gt_marginal_cost = price_gas / max(_NORM_EPS, float(eta_gt))
                net_grid_need_proxy_mw = max(
                    0.0,
                    p_dem + p_ech_proxy_mw + p_bes_charge_proxy_mw - pv - wt - p_bes_discharge_proxy_mw,
                )
                gt_need_ratio = float(np.clip(net_grid_need_proxy_mw / p_gt_cap_mw, 0.0, 1.0))
                gt_price_advantage = float(
                    np.clip((price_e - gt_marginal_cost) / max(_NORM_EPS, price_e), 0.0, 1.0)
                )
                gt_delta = abs(
                    float(teacher_action_exec_np[int(u_gt_index)])
                    - float(action_exec_np[int(u_gt_index)])
                )
                gt_bonus = 2.0 * gt_price_advantage * max(gt_need_ratio, gt_delta)
                priority += gt_bonus
                sample_weight += float(self.config.economic_teacher_warm_start_weight) * gt_bonus
        return {
            "priority": float(priority),
            "sample_weight": float(sample_weight),
            "dim_weight": float(dim_weight),
            "delta_mean": float(delta_mean),
            "gt_bonus": float(gt_bonus),
            "gt_selected": bool(gt_selected),
        }

    def _collect_economic_teacher_full_year_warm_start_dataset(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        empty_obs = np.zeros((0, len(self.observation_keys)), dtype=np.float32)
        empty_actions = np.zeros((0, len(self.action_keys)), dtype=np.float32)
        empty_weights = np.zeros((0, 1), dtype=np.float32)
        summary = {
            "enabled": bool(
                int(self.config.economic_teacher_full_year_warm_start_samples) > 0
                and int(self.config.economic_teacher_full_year_warm_start_epochs) > 0
            ),
            "requested_samples": int(self.config.economic_teacher_full_year_warm_start_samples),
            "epochs": int(self.config.economic_teacher_full_year_warm_start_epochs),
            "status": "disabled",
            "samples": 0,
        }
        if not bool(summary["enabled"]):
            return (
                empty_obs,
                empty_actions,
                empty_actions,
                empty_actions,
                empty_weights,
                summary,
            )
        if self.economic_teacher_policy is None:
            summary["status"] = "teacher_unavailable"
            return (
                empty_obs,
                empty_actions,
                empty_actions,
                empty_actions,
                empty_weights,
                summary,
            )

        expert_policy, expert_policy_info = self._build_expert_policy()
        rollout = self._rollout_expert_prefill_episode(
            expert_policy=expert_policy,
            expert_policy_info=expert_policy_info,
            episode_df=self.train_df.reset_index(drop=True),
            episode_seed=int(self.config.seed) + 930_000,
            max_steps=int(len(self.train_df)),
            abs_exec_threshold=float(self.config.expert_prefill_abs_exec_threshold),
            collect_teacher_targets=True,
        )
        transitions = tuple(rollout.get("transitions", ()))
        if not transitions:
            summary["status"] = "empty_rollout"
            return (
                empty_obs,
                empty_actions,
                empty_actions,
                empty_actions,
                empty_weights,
                summary,
            )

        priorities = np.zeros((len(transitions),), dtype=np.float32)
        sample_weights = np.zeros((len(transitions), 1), dtype=np.float32)
        candidate_indices: list[int] = []
        gt_available_count = 0
        bes_available_count = 0
        tes_available_count = 0
        gt_bonus_sum = 0.0
        dim_weight_sum = 0.0
        delta_mean_sum = 0.0

        for row_index, transition in enumerate(transitions):
            if not bool(transition.get("teacher_available", False)):
                continue
            teacher_mask = np.asarray(
                transition["teacher_action_mask"],
                dtype=np.float32,
            ).reshape(-1)
            active_mask = teacher_mask > 0.5
            if not np.any(active_mask):
                continue
            row_stats = self._compute_economic_teacher_full_year_row_priority(
                obs_vector=np.asarray(transition["obs"], dtype=np.float32),
                action_exec=np.asarray(transition["action_exec"], dtype=np.float32),
                teacher_action_exec=np.asarray(
                    transition["teacher_action_exec"],
                    dtype=np.float32,
                ),
                teacher_action_mask=teacher_mask,
            )
            priorities[row_index] = float(row_stats["priority"])
            sample_weights[row_index, 0] = float(row_stats["sample_weight"])
            candidate_indices.append(int(row_index))
            dim_weight_sum += float(row_stats["dim_weight"])
            delta_mean_sum += float(row_stats["delta_mean"])
            gt_bonus_sum += float(row_stats["gt_bonus"])
            if "u_gt" in self.action_index and active_mask[int(self.action_index["u_gt"])]:
                gt_available_count += 1
            if "u_bes" in self.action_index and active_mask[int(self.action_index["u_bes"])]:
                bes_available_count += 1
            if "u_tes" in self.action_index and active_mask[int(self.action_index["u_tes"])]:
                tes_available_count += 1

        if not candidate_indices:
            summary["status"] = "no_teacher_targets"
            return (
                empty_obs,
                empty_actions,
                empty_actions,
                empty_actions,
                empty_weights,
                summary,
            )

        selected_indices = _select_temporal_priority_indices(
            indices=candidate_indices,
            priorities=priorities,
            target_count=min(
                int(self.config.economic_teacher_full_year_warm_start_samples),
                len(candidate_indices),
            ),
        )
        obs_rows = np.asarray(
            [transitions[idx]["obs"] for idx in selected_indices],
            dtype=np.float32,
        )
        base_action_rows = np.asarray(
            [transitions[idx]["action_exec"] for idx in selected_indices],
            dtype=np.float32,
        )
        teacher_action_rows = np.asarray(
            [transitions[idx]["teacher_action_exec"] for idx in selected_indices],
            dtype=np.float32,
        )
        teacher_mask_rows = np.asarray(
            [transitions[idx]["teacher_action_mask"] for idx in selected_indices],
            dtype=np.float32,
        )
        selected_weight_rows = np.asarray(
            [sample_weights[idx] for idx in selected_indices],
            dtype=np.float32,
        ).reshape(-1, 1)
        summary.update(
            {
                "status": "ready",
                "samples": int(len(selected_indices)),
                "available_target_rows": int(len(candidate_indices)),
                "available_gt_count": int(gt_available_count),
                "available_bes_count": int(bes_available_count),
                "available_tes_count": int(tes_available_count),
                "sampled_gt_count": int(
                    (teacher_mask_rows[:, int(self.action_index["u_gt"])] > 0.5).sum()
                )
                if "u_gt" in self.action_index and len(selected_indices) > 0
                else 0,
                "sampled_bes_count": int(
                    (teacher_mask_rows[:, int(self.action_index["u_bes"])] > 0.5).sum()
                )
                if "u_bes" in self.action_index and len(selected_indices) > 0
                else 0,
                "sampled_tes_count": int(
                    (teacher_mask_rows[:, int(self.action_index["u_tes"])] > 0.5).sum()
                )
                if "u_tes" in self.action_index and len(selected_indices) > 0
                else 0,
                "priority_mean": float(np.asarray(priorities[selected_indices], dtype=np.float32).mean())
                if selected_indices
                else 0.0,
                "sample_weight_mean": float(selected_weight_rows.mean())
                if len(selected_weight_rows) > 0
                else 0.0,
                "sample_weight_max": float(selected_weight_rows.max())
                if len(selected_weight_rows) > 0
                else 0.0,
                "dim_weight_mean": float(dim_weight_sum / max(1, len(candidate_indices))),
                "delta_mean": float(delta_mean_sum / max(1, len(candidate_indices))),
                "gt_bonus_mean": float(gt_bonus_sum / max(1, len(candidate_indices))),
                "teacher_target_dim_rate": float(teacher_mask_rows.mean())
                if teacher_mask_rows.size > 0
                else 0.0,
            }
        )
        return (
            obs_rows,
            base_action_rows,
            teacher_action_rows,
            teacher_mask_rows,
            selected_weight_rows,
            summary,
        )

    def _warm_start_actor_from_economic_teacher_full_year(self) -> None:
        (
            observations,
            action_exec_targets,
            teacher_action_exec,
            teacher_action_mask,
            sample_weights,
            summary,
        ) = self._collect_economic_teacher_full_year_warm_start_dataset()
        if (
            observations.size == 0
            or action_exec_targets.size == 0
            or teacher_action_exec.size == 0
            or teacher_action_mask.size == 0
        ):
            self.actor_teacher_full_year_warm_start_summary = dict(summary)
            return

        _, _, _, AdamW = _require_torch_modules()
        optimizer = AdamW(self.actor.parameters(), lr=float(self.config.actor_warm_start_lr))
        obs_tensor = self.torch.as_tensor(observations, dtype=self.torch.float32, device=self.device)
        base_action_tensor = self.torch.as_tensor(
            action_exec_targets,
            dtype=self.torch.float32,
            device=self.device,
        )
        teacher_action_tensor = self.torch.as_tensor(
            teacher_action_exec,
            dtype=self.torch.float32,
            device=self.device,
        )
        teacher_mask_tensor = self.torch.as_tensor(
            teacher_action_mask,
            dtype=self.torch.float32,
            device=self.device,
        )
        sample_weight_tensor = self.torch.as_tensor(
            sample_weights,
            dtype=self.torch.float32,
            device=self.device,
        )
        batch_size = min(int(self.config.actor_warm_start_batch_size), int(obs_tensor.shape[0]))
        indices = np.arange(int(obs_tensor.shape[0]), dtype=np.int64)
        epoch_losses: list[float] = []
        epoch_teacher_losses: list[float] = []

        for _ in range(int(self.config.economic_teacher_full_year_warm_start_epochs)):
            self.rng.shuffle(indices)
            batch_losses: list[float] = []
            batch_teacher_losses: list[float] = []
            for start in range(0, int(len(indices)), int(batch_size)):
                batch_indices = indices[start : start + int(batch_size)]
                batch_obs = obs_tensor[batch_indices]
                batch_base_target = base_action_tensor[batch_indices]
                batch_teacher_target = teacher_action_tensor[batch_indices]
                batch_teacher_mask = teacher_mask_tensor[batch_indices]
                batch_sample_weight = sample_weight_tensor[batch_indices]
                prediction = self.actor(self._normalize_observation_tensor(batch_obs))
                prediction = self._apply_abs_cooling_blend_tensor(
                    obs_batch=batch_obs,
                    action_batch=prediction,
                )
                batch_safe_mask = self.torch.clamp(1.0 - batch_teacher_mask, 0.0, 1.0)
                safe_loss = (
                    ((prediction - batch_base_target).pow(2) * batch_safe_mask).sum(
                        dim=1,
                        keepdim=True,
                    )
                    / batch_safe_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                ).mean()
                teacher_sq = (
                    ((prediction - batch_teacher_target).pow(2) * batch_teacher_mask).sum(
                        dim=1,
                        keepdim=True,
                    )
                    / batch_teacher_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                )
                teacher_loss = (
                    (batch_sample_weight * teacher_sq).sum()
                    / batch_sample_weight.sum().clamp_min(1.0)
                )
                loss = safe_loss + teacher_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))
                batch_teacher_losses.append(float(teacher_loss.detach().cpu().item()))
            epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
            epoch_teacher_losses.append(float(np.mean(batch_teacher_losses)) if batch_teacher_losses else 0.0)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = AdamW(self.actor.parameters(), lr=float(self.config.actor_lr))
        self.current_actor_lr = float(self.config.actor_lr)
        final_summary = dict(summary)
        final_summary.update(
            {
                "status": "applied",
                "batch_size": int(batch_size),
                "lr": float(self.config.actor_warm_start_lr),
                "loss_first": float(epoch_losses[0]) if epoch_losses else 0.0,
                "loss_last": float(epoch_losses[-1]) if epoch_losses else 0.0,
                "teacher_loss_first": float(epoch_teacher_losses[0]) if epoch_teacher_losses else 0.0,
                "teacher_loss_last": float(epoch_teacher_losses[-1]) if epoch_teacher_losses else 0.0,
            }
        )
        self.actor_teacher_full_year_warm_start_summary = final_summary

    def _collect_bes_full_year_warm_start_dataset(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        empty_obs = np.zeros((0, len(self.observation_keys)), dtype=np.float32)
        empty_actions = np.zeros((0, len(self.action_keys)), dtype=np.float32)
        empty_targets = np.zeros((0, 1), dtype=np.float32)
        empty_weights = np.zeros((0, 1), dtype=np.float32)
        empty_anchor_weights = np.zeros((0, 1), dtype=np.float32)
        summary = {
            "enabled": bool(
                int(self.config.economic_bes_full_year_warm_start_samples) > 0
                and int(self.config.economic_bes_full_year_warm_start_epochs) > 0
            ),
            "requested_samples": int(self.config.economic_bes_full_year_warm_start_samples),
            "epochs": int(self.config.economic_bes_full_year_warm_start_epochs),
            "u_weight": float(self.config.economic_bes_full_year_warm_start_u_weight),
            "status": "disabled",
            "samples": 0,
        }
        if not bool(summary["enabled"]):
            return (
                empty_obs,
                empty_actions,
                empty_targets,
                empty_weights,
                empty_anchor_weights,
                summary,
            )
        required_obs = {"price_e", "soc_bes"}
        if not required_obs.issubset(self.observation_index) or "u_bes" not in self.action_index:
            summary["status"] = "missing_bes_features"
            return (
                empty_obs,
                empty_actions,
                empty_targets,
                empty_weights,
                empty_anchor_weights,
                summary,
            )

        all_transitions: list[dict[str, Any]] = []
        transition_source_labels: list[str] = []
        transition_anchor_weights: list[float] = []
        source_rollouts: dict[str, dict[str, Any]] = {}

        def _append_rollout(
            *,
            source_label: str,
            source_anchor_weight: float,
            expert_policy,
            expert_policy_info: dict[str, Any],
        ) -> dict[str, Any]:
            rollout = self._rollout_expert_prefill_episode(
                expert_policy=expert_policy,
                expert_policy_info=expert_policy_info,
                episode_df=self.train_df.reset_index(drop=True),
                episode_seed=int(self.config.seed) + 910_000 + 1000 * len(source_rollouts),
                max_steps=int(len(self.train_df)),
                abs_exec_threshold=float(self.config.expert_prefill_abs_exec_threshold),
                collect_teacher_targets=False,
            )
            transitions = tuple(rollout.get("transitions", ()))
            source_rollouts[str(source_label)] = {
                "teacher": dict(expert_policy_info),
                "steps": int(len(transitions)),
                "bes_prior_override_count": int(rollout.get("bes_prior_override_count", 0)),
                "bes_prior_charge_count": int(rollout.get("bes_prior_charge_count", 0)),
                "bes_prior_discharge_count": int(rollout.get("bes_prior_discharge_count", 0)),
                "anchor_weight": float(source_anchor_weight),
            }
            for transition in transitions:
                all_transitions.append(dict(transition))
                transition_source_labels.append(str(source_label))
                transition_anchor_weights.append(float(source_anchor_weight))
            return rollout

        safe_policy, safe_policy_info = self._build_bes_full_year_warm_start_policy()
        safe_rollout = _append_rollout(
            source_label="safe",
            source_anchor_weight=1.0,
            expert_policy=safe_policy,
            expert_policy_info=safe_policy_info,
        )
        total_charge_available = int(safe_rollout.get("bes_prior_charge_count", 0))
        total_discharge_available = int(safe_rollout.get("bes_prior_discharge_count", 0))
        economic_policy, economic_policy_info = self._build_bes_full_year_economic_policy()
        if economic_policy is not None:
            economic_rollout = _append_rollout(
                source_label="economic",
                source_anchor_weight=0.0,
                expert_policy=economic_policy,
                expert_policy_info=economic_policy_info,
            )
            total_charge_available += int(economic_rollout.get("bes_prior_charge_count", 0))
            total_discharge_available += int(economic_rollout.get("bes_prior_discharge_count", 0))
        requested_samples = int(self.config.economic_bes_full_year_warm_start_samples)
        desired_charge_target = int(round(float(requested_samples) * 0.40))
        desired_discharge_target = int(round(float(requested_samples) * 0.40))
        fallback_enabled = (
            total_charge_available < desired_charge_target
            or total_discharge_available < desired_discharge_target
        )
        if fallback_enabled:
            fallback_policy = EasyRuleAbsPolicy(
                env_config=self.env_config,
                p_gt_cap_mw=float(self.env_config.p_gt_cap_mw),
                q_boiler_cap_mw=float(self.env_config.q_boiler_cap_mw),
                q_ech_cap_mw=float(self.env_config.q_ech_cap_mw),
                q_abs_cool_cap_mw=float(self.env_config.q_abs_cool_cap_mw),
            )
            _append_rollout(
                source_label="fallback_bidirectional",
                source_anchor_weight=0.0,
                expert_policy=fallback_policy,
                expert_policy_info={
                    "mode": "easy_rule_abs",
                    "source": "fallback_bidirectional",
                },
            )

        total_available = int(len(all_transitions))
        summary.update(
            {
                "status": "no_rollout",
                "teachers": dict(source_rollouts),
                "rollout_steps": int(total_available),
                "fallback_bidirectional_enabled": bool(fallback_enabled),
            }
        )
        if total_available <= 0:
            return (
                empty_obs,
                empty_actions,
                empty_targets,
                empty_weights,
                empty_anchor_weights,
                summary,
            )

        price_index = int(self.observation_index["price_e"])
        soc_index = int(self.observation_index["soc_bes"])
        u_bes_index = int(self.action_index["u_bes"])
        charge_u, discharge_u = self._resolve_bes_prior_u_pair()
        prior_modes: list[str] = []
        mode_indices = {"charge": [], "discharge": [], "idle": []}
        source_mode_indices = {
            "charge": {"economic": [], "other": []},
            "discharge": {"economic": [], "other": []},
            "idle": {"economic": [], "other": []},
        }
        teacher_mode_indices = {
            "charge": {"teacher": [], "other": []},
            "discharge": {"teacher": [], "other": []},
            "idle": {"teacher": [], "other": []},
        }
        opportunity_scores = np.zeros((total_available,), dtype=np.float32)
        selection_priorities = np.zeros((total_available,), dtype=np.float32)
        target_u_rows = np.zeros((total_available, 1), dtype=np.float32)
        economic_teacher_override_flags = np.zeros((total_available,), dtype=np.float32)
        economic_teacher_override_count = 0
        economic_teacher_idle_override_count = 0
        for idx, transition in enumerate(all_transitions):
            obs_vector = np.asarray(transition["obs"], dtype=np.float32).reshape(-1)
            prior = _bes_price_prior_target_np(
                price_e=float(obs_vector[price_index]),
                soc_bes=float(obs_vector[soc_index]),
                price_low_threshold=float(self.bes_price_low_threshold),
                price_high_threshold=float(self.bes_price_high_threshold),
                charge_soc_ceiling=float(self.config.economic_bes_charge_soc_ceiling),
                discharge_soc_floor=float(self.config.economic_bes_discharge_soc_floor),
                bes_soc_min=float(self.env_config.bes_soc_min),
                bes_soc_max=float(self.env_config.bes_soc_max),
                charge_u=float(charge_u),
                discharge_u=float(discharge_u),
            )
            target_choice = _select_bes_full_year_target_np(
                source_label=str(transition_source_labels[int(idx)]),
                prior_target_u_bes=float(prior["target_u_bes"]),
                prior_opportunity=float(prior["opportunity"]),
                teacher_u_bes=float(
                    np.asarray(transition["action_raw"], dtype=np.float32).reshape(-1)[u_bes_index]
                ),
            )
            mode = str(target_choice["mode"])
            if mode not in mode_indices:
                mode = "idle"
            prior_modes.append(mode)
            mode_indices[mode].append(int(idx))
            source_bucket = (
                "economic"
                if str(transition_source_labels[int(idx)]).strip().lower().startswith("economic")
                else "other"
            )
            source_mode_indices[mode][source_bucket].append(int(idx))
            base_priority = float(prior["opportunity"]) * float(
                _bes_prior_weight_multiplier_np(
                    mode=str(prior["mode"]),
                    charge_score=float(prior.get("charge_score", 0.0)),
                    charge_weight=float(self.config.economic_bes_charge_weight),
                    discharge_weight=float(self.config.economic_bes_discharge_weight),
                    charge_pressure_bonus=float(self.config.economic_bes_charge_pressure_bonus),
                )
            )
            base_priority = max(
                float(base_priority),
                float(target_choice["weight_bonus"]),
            )
            opportunity_scores[idx] = float(base_priority)
            selection_priorities[idx] = _bes_full_year_selection_priority_np(
                base_priority=float(base_priority),
                source_label=str(transition_source_labels[int(idx)]),
                used_teacher=bool(target_choice["used_teacher"]),
                teacher_priority_boost=float(
                    self.config.economic_bes_teacher_selection_priority_boost
                ),
                economic_source_priority_bonus=float(
                    self.config.economic_bes_economic_source_priority_bonus
                ),
            )
            target_u_rows[idx, 0] = float(target_choice["target_u_bes"])
            if bool(target_choice["used_teacher"]):
                economic_teacher_override_flags[idx] = 1.0
                economic_teacher_override_count += 1
                if float(prior["opportunity"]) <= 0.0:
                    economic_teacher_idle_override_count += 1
                teacher_mode_indices[mode]["teacher"].append(int(idx))
            else:
                teacher_mode_indices[mode]["other"].append(int(idx))

        requested_samples = min(
            int(self.config.economic_bes_full_year_warm_start_samples),
            total_available,
        )
        mode_targets = _allocate_bes_warm_start_mode_counts(
            requested_total=requested_samples,
            charge_available=len(mode_indices["charge"]),
            discharge_available=len(mode_indices["discharge"]),
            idle_available=len(mode_indices["idle"]),
        )
        selected_indices: list[int] = []
        for mode in ("discharge", "charge", "idle"):
            teacher_targets = _allocate_bes_teacher_target_counts(
                requested_total=int(mode_targets[mode]),
                teacher_available=len(teacher_mode_indices[mode]["teacher"]),
                other_available=(
                    len(teacher_mode_indices[mode]["other"])
                ),
                teacher_min_share=float(self.config.economic_bes_teacher_target_min_share),
            )
            teacher_selected = _select_temporal_priority_indices(
                indices=teacher_mode_indices[mode]["teacher"],
                priorities=selection_priorities,
                target_count=int(teacher_targets["teacher"]),
            )
            selected_indices.extend(teacher_selected)
            teacher_selected_set = {int(idx) for idx in teacher_selected}
            remaining_economic_indices = [
                idx
                for idx in source_mode_indices[mode]["economic"]
                if int(idx) not in teacher_selected_set
            ]
            remaining_other_indices = [
                idx
                for idx in source_mode_indices[mode]["other"]
                if int(idx) not in teacher_selected_set
            ]
            source_min_share = float(
                self.config.economic_bes_idle_economic_source_min_share
                if mode == "idle"
                else self.config.economic_bes_economic_source_min_share
            )
            source_targets = _allocate_bes_source_counts(
                requested_total=int(max(0, mode_targets[mode] - len(teacher_selected))),
                economic_available=len(remaining_economic_indices),
                other_available=len(remaining_other_indices),
                economic_min_share=source_min_share,
            )
            selected_indices.extend(
                _select_temporal_priority_indices(
                    indices=remaining_economic_indices,
                    priorities=selection_priorities,
                    target_count=int(source_targets["economic"]),
                )
            )
            selected_indices.extend(
                _select_temporal_priority_indices(
                    indices=remaining_other_indices,
                    priorities=selection_priorities,
                    target_count=int(source_targets["other"]),
                )
            )
        selected_set = {int(idx) for idx in selected_indices}
        if len(selected_set) < requested_samples:
            remaining_indices = [
                idx for idx in range(total_available) if int(idx) not in selected_set
            ]
            fill_indices = _select_temporal_priority_indices(
                indices=remaining_indices,
                priorities=selection_priorities,
                target_count=int(requested_samples - len(selected_set)),
            )
            selected_set.update(int(idx) for idx in fill_indices)
        selected_indices = sorted(selected_set)
        if not selected_indices:
            summary["status"] = "no_selected_samples"
            return (
                empty_obs,
                empty_actions,
                empty_targets,
                empty_weights,
                empty_anchor_weights,
                summary,
            )

        obs_rows: list[np.ndarray] = []
        action_exec_rows: list[np.ndarray] = []
        anchor_weight_rows: list[np.ndarray] = []
        sampled_modes = {"charge": 0, "discharge": 0, "idle": 0}
        sampled_source_counts: dict[str, int] = {}
        replay_augmented_steps = 0
        for idx in selected_indices:
            transition = all_transitions[int(idx)]
            obs_rows.append(np.asarray(transition["obs"], dtype=np.float32).copy())
            action_exec_rows.append(np.asarray(transition["action_exec"], dtype=np.float32).copy())
            anchor_weight_rows.append(
                np.asarray([float(transition_anchor_weights[int(idx)])], dtype=np.float32)
            )
            sampled_modes[str(prior_modes[int(idx)])] += 1
            source_label = str(transition_source_labels[int(idx)])
            sampled_source_counts[source_label] = sampled_source_counts.get(source_label, 0) + 1
            self.replay.add(
                obs=np.asarray(transition["obs"], dtype=np.float32).copy(),
                next_obs=np.asarray(transition["next_obs"], dtype=np.float32).copy(),
                action_raw=np.asarray(transition["action_raw"], dtype=np.float32).copy(),
                action_exec=np.asarray(transition["action_exec"], dtype=np.float32).copy(),
                teacher_action_exec=np.asarray(
                    transition["teacher_action_exec"],
                    dtype=np.float32,
                ).copy(),
                teacher_action_mask=np.asarray(
                    transition["teacher_action_mask"],
                    dtype=np.float32,
                ).copy(),
                teacher_available=bool(transition["teacher_available"]),
                reward=float(transition["reward"]),
                cost=np.asarray(transition["cost"], dtype=np.float32).copy(),
                gap=np.asarray(transition["gap"], dtype=np.float32).copy(),
                done=bool(transition["done"]),
            )
            replay_augmented_steps += 1

        selected_indices_np = np.asarray(selected_indices, dtype=np.int64)
        sampled_economic_source_count = int(
            sum(
                int(value)
                for key, value in sampled_source_counts.items()
                if str(key).strip().lower().startswith("economic")
            )
        )
        selected_weights = (
            1.0
            + np.asarray(opportunity_scores[selected_indices_np], dtype=np.float32).reshape(-1, 1)
            + (
                np.abs(target_u_rows[selected_indices_np])
                > _NORM_EPS
            ).astype(np.float32)
            + np.asarray(
                economic_teacher_override_flags[selected_indices_np],
                dtype=np.float32,
            ).reshape(-1, 1)
        )
        summary.update(
            {
                "status": "ready",
                "samples": int(len(selected_indices)),
                "available_charge_count": int(len(mode_indices["charge"])),
                "available_discharge_count": int(len(mode_indices["discharge"])),
                "available_idle_count": int(len(mode_indices["idle"])),
                "sampled_charge_count": int(sampled_modes["charge"]),
                "sampled_discharge_count": int(sampled_modes["discharge"]),
                "sampled_idle_count": int(sampled_modes["idle"]),
                "sampled_opportunity_mean": float(
                    np.asarray(opportunity_scores[selected_indices_np], dtype=np.float32).mean()
                ),
                "sampled_selection_priority_mean": float(
                    np.asarray(selection_priorities[selected_indices_np], dtype=np.float32).mean()
                ),
                "sampled_target_abs_mean": float(
                    np.asarray(np.abs(target_u_rows[selected_indices_np]), dtype=np.float32).mean()
                ),
                "economic_teacher_override_count": int(economic_teacher_override_count),
                "economic_teacher_idle_override_count": int(
                    economic_teacher_idle_override_count
                ),
                "sampled_economic_teacher_target_count": int(
                    np.asarray(
                        economic_teacher_override_flags[selected_indices_np],
                        dtype=np.float32,
                    ).sum()
                ),
                "sample_weight_mean": float(selected_weights.mean()),
                "anchor_weight_mean": float(np.asarray(anchor_weight_rows, dtype=np.float32).mean()),
                "charge_u_scale": float(self.config.economic_bes_charge_u_scale),
                "discharge_u_scale": float(self.config.economic_bes_discharge_u_scale),
                "charge_weight": float(self.config.economic_bes_charge_weight),
                "discharge_weight": float(self.config.economic_bes_discharge_weight),
                "charge_pressure_bonus": float(self.config.economic_bes_charge_pressure_bonus),
                "teacher_selection_priority_boost": float(
                    self.config.economic_bes_teacher_selection_priority_boost
                ),
                "economic_source_priority_bonus": float(
                    self.config.economic_bes_economic_source_priority_bonus
                ),
                "economic_source_min_share": float(
                    self.config.economic_bes_economic_source_min_share
                ),
                "idle_economic_source_min_share": float(
                    self.config.economic_bes_idle_economic_source_min_share
                ),
                "teacher_target_min_share": float(
                    self.config.economic_bes_teacher_target_min_share
                ),
                "sampled_economic_source_count": int(sampled_economic_source_count),
                "sampled_economic_source_rate": float(
                    sampled_economic_source_count / max(1, len(selected_indices))
                ),
                "sampled_source_counts": {
                    str(key): int(value) for key, value in sampled_source_counts.items()
                },
                "replay_augmented_steps": int(replay_augmented_steps),
                "price_low_threshold": float(self.bes_price_low_threshold),
                "price_high_threshold": float(self.bes_price_high_threshold),
            }
        )
        return (
            np.asarray(obs_rows, dtype=np.float32),
            np.asarray(action_exec_rows, dtype=np.float32),
            np.asarray(target_u_rows[selected_indices_np], dtype=np.float32),
            np.asarray(selected_weights, dtype=np.float32),
            np.asarray(anchor_weight_rows, dtype=np.float32),
            summary,
        )

    def _warm_start_actor_from_bes_full_year(self) -> None:
        (
            observations,
            action_exec_targets,
            target_u_bes,
            sample_weights,
            anchor_weights,
            summary,
        ) = self._collect_bes_full_year_warm_start_dataset()
        if observations.size == 0 or action_exec_targets.size == 0 or target_u_bes.size == 0:
            self.actor_bes_warm_start_summary = dict(summary)
            return

        _, _, _, AdamW = _require_torch_modules()
        optimizer = AdamW(self.actor.parameters(), lr=float(self.config.actor_warm_start_lr))
        obs_tensor = self.torch.as_tensor(observations, dtype=self.torch.float32, device=self.device)
        base_action_tensor = self.torch.as_tensor(
            action_exec_targets,
            dtype=self.torch.float32,
            device=self.device,
        )
        target_u_tensor = self.torch.as_tensor(
            target_u_bes,
            dtype=self.torch.float32,
            device=self.device,
        )
        sample_weight_tensor = self.torch.as_tensor(
            sample_weights,
            dtype=self.torch.float32,
            device=self.device,
        )
        anchor_weight_tensor = self.torch.as_tensor(
            anchor_weights,
            dtype=self.torch.float32,
            device=self.device,
        )
        u_bes_index = int(self.action_index["u_bes"])
        batch_size = min(int(self.config.actor_warm_start_batch_size), int(obs_tensor.shape[0]))
        indices = np.arange(int(obs_tensor.shape[0]), dtype=np.int64)
        anchor_scale = self.torch.ones(
            (1, len(self.action_keys)),
            dtype=self.torch.float32,
            device=self.device,
        )
        anchor_scale[:, u_bes_index : u_bes_index + 1] = 0.0
        epoch_losses: list[float] = []
        epoch_bes_losses: list[float] = []

        for _ in range(int(self.config.economic_bes_full_year_warm_start_epochs)):
            self.rng.shuffle(indices)
            batch_losses: list[float] = []
            batch_bes_losses: list[float] = []
            for start in range(0, int(len(indices)), int(batch_size)):
                batch_indices = indices[start : start + int(batch_size)]
                batch_obs = obs_tensor[batch_indices]
                batch_base_target = base_action_tensor[batch_indices]
                batch_target_u = target_u_tensor[batch_indices]
                batch_weight = sample_weight_tensor[batch_indices]
                batch_anchor_weight = anchor_weight_tensor[batch_indices]
                prediction = self.actor(self._normalize_observation_tensor(batch_obs))
                prediction = self._apply_abs_cooling_blend_tensor(
                    obs_batch=batch_obs,
                    action_batch=prediction,
                )
                safe_loss = (
                    batch_anchor_weight
                    * ((prediction - batch_base_target).pow(2) * anchor_scale).mean(dim=1, keepdim=True)
                )
                safe_loss = safe_loss.mean()
                bes_prediction = prediction[:, u_bes_index : u_bes_index + 1]
                bes_loss = (
                    batch_weight * (bes_prediction - batch_target_u).pow(2)
                ).sum() / batch_weight.sum().clamp_min(1.0)
                loss = safe_loss + (
                    float(self.config.economic_bes_full_year_warm_start_u_weight) * bes_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))
                batch_bes_losses.append(float(bes_loss.detach().cpu().item()))
            epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
            epoch_bes_losses.append(float(np.mean(batch_bes_losses)) if batch_bes_losses else 0.0)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = AdamW(self.actor.parameters(), lr=float(self.config.actor_lr))
        self.current_actor_lr = float(self.config.actor_lr)
        final_summary = dict(summary)
        final_summary.update(
            {
                "status": "applied",
                "batch_size": int(batch_size),
                "lr": float(self.config.actor_warm_start_lr),
                "loss_first": float(epoch_losses[0]) if epoch_losses else 0.0,
                "loss_last": float(epoch_losses[-1]) if epoch_losses else 0.0,
                "bes_loss_first": float(epoch_bes_losses[0]) if epoch_bes_losses else 0.0,
                "bes_loss_last": float(epoch_bes_losses[-1]) if epoch_bes_losses else 0.0,
                "target_positive_rate": float(
                    (target_u_tensor > 0.0).float().mean().detach().cpu().item()
                ),
                "target_negative_rate": float(
                    (target_u_tensor < 0.0).float().mean().detach().cpu().item()
                ),
                "target_abs_mean": float(target_u_tensor.abs().mean().detach().cpu().item()),
            }
        )
        self.actor_bes_warm_start_summary = final_summary

    def _compute_abs_focus_weights(self, *, obs_batch, action_exec_batch):
        if float(self.config.abs_ready_focus_coef) <= 0.0:
            return self.torch.ones((obs_batch.shape[0], 1), dtype=obs_batch.dtype, device=obs_batch.device)
        if "abs_drive_margin_k" not in self.observation_index or "qc_dem_mw" not in self.observation_index:
            return self.torch.ones((obs_batch.shape[0], 1), dtype=obs_batch.dtype, device=obs_batch.device)
        if "u_abs" not in self.action_index:
            return self.torch.ones((obs_batch.shape[0], 1), dtype=obs_batch.dtype, device=obs_batch.device)

        margin_index = int(self.observation_index["abs_drive_margin_k"])
        qc_index = int(self.observation_index["qc_dem_mw"])
        u_abs_index = int(self.action_index["u_abs"])
        margin = obs_batch[:, margin_index : margin_index + 1]
        qc_dem = obs_batch[:, qc_index : qc_index + 1]
        u_abs_exec = action_exec_batch[:, u_abs_index : u_abs_index + 1]
        ready_score = self.torch.clamp(
            margin / max(_NORM_EPS, float(self.env_config.abs_gate_scale_k)),
            0.0,
            1.0,
        )
        cooling_score = self.torch.clamp(
            qc_dem / max(_NORM_EPS, float(self.env_config.q_abs_cool_cap_mw)),
            0.0,
            1.0,
        )
        abs_exec_score = self.torch.clamp(u_abs_exec, 0.0, 1.0)
        focus_score = self.torch.maximum(abs_exec_score, ready_score * cooling_score * abs_exec_score.clamp_min(0.25))
        return 1.0 + float(self.config.abs_ready_focus_coef) * focus_score

    def _resolve_bes_prior_u_pair(self) -> tuple[float, float]:
        base_u = float(abs(self.config.economic_bes_prior_u))
        charge_u = base_u * max(0.0, float(self.config.economic_bes_charge_u_scale))
        discharge_u = base_u * max(0.0, float(self.config.economic_bes_discharge_u_scale))
        return float(charge_u), float(discharge_u)

    def _compute_bes_price_prior_terms(self, *, obs_batch):
        required_obs = {"price_e", "soc_bes"}
        if not required_obs.issubset(self.observation_index) or "u_bes" not in self.action_index:
            return None
        price_index = int(self.observation_index["price_e"])
        soc_index = int(self.observation_index["soc_bes"])
        price_e = obs_batch[:, price_index : price_index + 1]
        soc_bes = obs_batch[:, soc_index : soc_index + 1]
        low_price = float(self.bes_price_low_threshold)
        high_price = max(low_price + _NORM_EPS, float(self.bes_price_high_threshold))
        mid_price = 0.5 * (low_price + high_price)
        charge_span = max(_NORM_EPS, mid_price - low_price)
        discharge_span = max(_NORM_EPS, high_price - mid_price)
        charge_headroom = self.torch.clamp(
            (float(self.config.economic_bes_charge_soc_ceiling) - soc_bes)
            / max(
                _NORM_EPS,
                float(self.config.economic_bes_charge_soc_ceiling) - float(self.env_config.bes_soc_min),
            ),
            0.0,
            1.0,
        )
        discharge_headroom = self.torch.clamp(
            (soc_bes - float(self.config.economic_bes_discharge_soc_floor))
            / max(
                _NORM_EPS,
                float(self.env_config.bes_soc_max) - float(self.config.economic_bes_discharge_soc_floor),
            ),
            0.0,
            1.0,
        )
        charge_pressure = self.torch.clamp((mid_price - price_e) / charge_span, 0.0, 1.0)
        discharge_pressure = self.torch.clamp((price_e - mid_price) / discharge_span, 0.0, 1.0)
        charge_score = charge_pressure * charge_headroom
        discharge_score = discharge_pressure * discharge_headroom
        net_signal = discharge_score - charge_score
        opportunity = net_signal.abs()
        active_mask = (opportunity > float(_BES_PRIOR_MIN_OPPORTUNITY)).to(dtype=obs_batch.dtype)
        opportunity = opportunity * active_mask
        charge_u, discharge_u = self._resolve_bes_prior_u_pair()
        target_u = (
            net_signal.clamp(min=0.0, max=1.0) * float(discharge_u)
            - (-net_signal).clamp(min=0.0, max=1.0) * float(charge_u)
        ) * active_mask
        mode_weight = self.torch.ones_like(target_u)
        mode_weight = mode_weight + (
            max(0.0, float(self.config.economic_bes_charge_weight)) - 1.0
        ) * (target_u < 0.0).to(dtype=obs_batch.dtype)
        mode_weight = mode_weight + (
            max(0.0, float(self.config.economic_bes_discharge_weight)) - 1.0
        ) * (target_u > 0.0).to(dtype=obs_batch.dtype)
        pressure_bonus = 1.0 + max(
            0.0,
            float(self.config.economic_bes_charge_pressure_bonus),
        ) * charge_score
        return {
            "target_u_bes": target_u,
            "opportunity": opportunity,
            "charge_score": charge_score,
            "discharge_score": discharge_score,
            "mode_weight": mode_weight,
            "pressure_bonus": pressure_bonus,
        }

    def _compute_bes_prior_distill_loss(
        self,
        *,
        obs_batch,
        action_raw_batch,
        action_exec_batch,
        gap_batch,
    ):
        if float(self.config.economic_bes_distill_coef) <= 0.0:
            zero = self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
            return zero, zero
        if "u_bes" not in self.action_index:
            zero = self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
            return zero, zero
        prior_terms = self._compute_bes_price_prior_terms(obs_batch=obs_batch)
        if prior_terms is None:
            zero = self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
            return zero, zero
        reliability_risk = self.torch.zeros_like(prior_terms["opportunity"])
        if "heat_backup_min_needed_mw" in self.observation_index:
            heat_backup_index = int(self.observation_index["heat_backup_min_needed_mw"])
            heat_backup_ratio = self.torch.clamp(
                obs_batch[:, heat_backup_index : heat_backup_index + 1]
                / max(_NORM_EPS, float(self.env_config.q_boiler_cap_mw)),
                0.0,
                1.0,
            )
            reliability_risk = self.torch.maximum(reliability_risk, heat_backup_ratio)
        if "abs_drive_margin_k" in self.observation_index:
            margin_index = int(self.observation_index["abs_drive_margin_k"])
            abs_risk = self.torch.clamp(
                -obs_batch[:, margin_index : margin_index + 1]
                / max(_NORM_EPS, float(self.env_config.abs_gate_scale_k)),
                0.0,
                1.0,
            )
            reliability_risk = self.torch.maximum(reliability_risk, abs_risk)
        if "u_abs" in self.action_index:
            u_abs_index = int(self.action_index["u_abs"])
            abs_commit = self.torch.clamp(
                action_exec_batch[:, u_abs_index : u_abs_index + 1],
                0.0,
                1.0,
            )
            reliability_risk = self.torch.maximum(reliability_risk, 0.5 * abs_commit)
        projection_risk = self.torch.clamp(gap_batch[:, :1].detach(), 0.0, 1.0)
        prior_weight = (
            prior_terms["opportunity"]
            * prior_terms["mode_weight"]
            * prior_terms["pressure_bonus"]
            * (1.0 - reliability_risk)
            * (1.0 - projection_risk)
        )
        prior_weight_sum = prior_weight.sum()
        if float(prior_weight_sum.detach().cpu().item()) <= 0.0:
            zero = self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
            return zero, prior_weight.mean()
        u_bes_index = int(self.action_index["u_bes"])
        prior_sq = (
            action_raw_batch[:, u_bes_index : u_bes_index + 1].clamp(-1.0, 1.0)
            - prior_terms["target_u_bes"]
        ).pow(2)
        prior_loss = (prior_weight * prior_sq).sum() / prior_weight_sum.clamp_min(1.0)
        return prior_loss, prior_weight.mean()

    def _compute_invalid_abs_request_penalty(self, *, obs_batch, action_raw_batch):
        if float(self.config.invalid_abs_penalty_coef) <= 0.0:
            return self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
        if "abs_drive_margin_k" not in self.observation_index or "u_abs" not in self.action_index:
            return self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
        margin_index = int(self.observation_index["abs_drive_margin_k"])
        u_abs_index = int(self.action_index["u_abs"])
        margin = obs_batch[:, margin_index : margin_index + 1]
        u_abs_raw = action_raw_batch[:, u_abs_index : u_abs_index + 1]
        invalid_score = self.torch.clamp(
            -margin / max(_NORM_EPS, float(self.env_config.abs_gate_scale_k)),
            0.0,
            1.0,
        )
        return (invalid_score * u_abs_raw.pow(2)).mean()

    def _compute_boiler_economic_proxy_penalty(self, *, obs_batch, action_exec_batch):
        if float(self.config.economic_boiler_proxy_coef) <= 0.0:
            return self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
        if "heat_backup_min_needed_mw" not in self.observation_index or "u_boiler" not in self.action_index:
            return self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
        heat_backup_index = int(self.observation_index["heat_backup_min_needed_mw"])
        u_boiler_index = int(self.action_index["u_boiler"])
        backup_needed_mw = obs_batch[:, heat_backup_index : heat_backup_index + 1]
        boiler_lower_bound = self.torch.clamp(
            backup_needed_mw / max(_NORM_EPS, float(self.env_config.q_boiler_cap_mw)),
            0.0,
            1.0,
        )
        u_boiler_exec = action_exec_batch[:, u_boiler_index : u_boiler_index + 1]
        boiler_excess = self.torch.clamp(u_boiler_exec - boiler_lower_bound, 0.0, 1.0)
        return boiler_excess.mean()

    def _compute_abs_ech_tradeoff_proxy_penalty(self, *, obs_batch, action_exec_batch):
        if float(self.config.economic_abs_tradeoff_coef) <= 0.0:
            return self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
        required_obs = {"abs_drive_margin_k", "heat_backup_min_needed_mw"}
        required_actions = {"u_abs", "u_ech"}
        if not required_obs.issubset(self.observation_index) or not required_actions.issubset(self.action_index):
            return self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
        margin_index = int(self.observation_index["abs_drive_margin_k"])
        heat_backup_index = int(self.observation_index["heat_backup_min_needed_mw"])
        u_abs_index = int(self.action_index["u_abs"])
        u_ech_index = int(self.action_index["u_ech"])
        margin = obs_batch[:, margin_index : margin_index + 1]
        ready_score = self.torch.sigmoid(
            margin / max(_NORM_EPS, float(self.env_config.abs_gate_scale_k))
        )
        heat_backup_ratio = self.torch.clamp(
            obs_batch[:, heat_backup_index : heat_backup_index + 1]
            / max(_NORM_EPS, float(self.env_config.q_boiler_cap_mw)),
            0.0,
            1.0,
        )
        heat_slack_score = 1.0 - heat_backup_ratio
        abs_cooling_equiv = (
            action_exec_batch[:, u_abs_index : u_abs_index + 1]
            * float(self.env_config.q_abs_cool_cap_mw)
            / max(_NORM_EPS, float(self.env_config.q_ech_cap_mw))
        )
        ech_exec = action_exec_batch[:, u_ech_index : u_ech_index + 1]
        missed_abs_substitution = self.torch.clamp(ech_exec - abs_cooling_equiv, 0.0, 1.0)
        return (ready_score * heat_slack_score * missed_abs_substitution).mean()

    def _compute_gt_grid_proxy_terms(self, *, obs_batch, action_exec_batch):
        required_obs = {
            "p_dem_mw",
            "pv_mw",
            "wt_mw",
            "price_e",
            "price_gas",
            "t_amb_k",
            "heat_backup_min_needed_mw",
            "abs_drive_margin_k",
        }
        required_actions = {"u_gt", "u_bes", "u_ech"}
        if (
            not required_obs.issubset(self.observation_index)
            or not required_actions.issubset(self.action_index)
        ):
            return None

        p_dem_index = int(self.observation_index["p_dem_mw"])
        pv_index = int(self.observation_index["pv_mw"])
        wt_index = int(self.observation_index["wt_mw"])
        price_e_index = int(self.observation_index["price_e"])
        price_gas_index = int(self.observation_index["price_gas"])
        t_amb_index = int(self.observation_index["t_amb_k"])
        backup_index = int(self.observation_index["heat_backup_min_needed_mw"])
        margin_index = int(self.observation_index["abs_drive_margin_k"])
        u_gt_index = int(self.action_index["u_gt"])
        u_bes_index = int(self.action_index["u_bes"])
        u_ech_index = int(self.action_index["u_ech"])

        p_dem = obs_batch[:, p_dem_index : p_dem_index + 1]
        pv = obs_batch[:, pv_index : pv_index + 1]
        wt = obs_batch[:, wt_index : wt_index + 1]
        price_e = obs_batch[:, price_e_index : price_e_index + 1]
        price_gas = obs_batch[:, price_gas_index : price_gas_index + 1]
        t_amb_k = obs_batch[:, t_amb_index : t_amb_index + 1]
        heat_backup_min_needed_mw = obs_batch[:, backup_index : backup_index + 1]
        abs_margin_k = obs_batch[:, margin_index : margin_index + 1]
        p_gt_cap_mw = max(_NORM_EPS, float(self.env_config.p_gt_cap_mw))
        q_boiler_cap_mw = max(_NORM_EPS, float(self.env_config.q_boiler_cap_mw))

        u_bes_exec = action_exec_batch[:, u_bes_index : u_bes_index + 1].clamp(-1.0, 1.0)
        p_bes_charge_proxy_mw = (
            (-u_bes_exec).clamp_min(0.0)
            * float(self.env_config.p_bes_cap_mw)
            / max(_NORM_EPS, float(self.env_config.bes_eta_charge))
        )
        p_bes_discharge_proxy_mw = (
            u_bes_exec.clamp_min(0.0)
            * float(self.env_config.p_bes_cap_mw)
            * float(self.env_config.bes_eta_discharge)
        )
        u_ech_exec = action_exec_batch[:, u_ech_index : u_ech_index + 1].clamp(0.0, 1.0)
        q_ech_proxy_mw = u_ech_exec * float(self.env_config.q_ech_cap_mw)
        ech_cop_base = (
            float(self.env_config.cop_nominal)
            - 0.03 * (t_amb_k - 298.15)
        ).clamp(
            min=float(self.env_config.cop_nominal) * float(self.env_config.ech_cop_partload_min_fraction),
            max=float(self.env_config.cop_nominal),
        )
        p_ech_proxy_mw = q_ech_proxy_mw / ech_cop_base.clamp_min(_NORM_EPS)

        net_grid_need_proxy_mw = (
            p_dem
            + p_ech_proxy_mw
            + p_bes_charge_proxy_mw
            - pv
            - wt
            - p_bes_discharge_proxy_mw
        ).clamp_min(0.0)
        p_gt_exec = (
            (action_exec_batch[:, u_gt_index : u_gt_index + 1] + 1.0)
            * 0.5
            * p_gt_cap_mw
        )
        gt_load_ratio = self.torch.clamp(
            p_gt_exec / p_gt_cap_mw,
            0.0,
            1.0,
        )
        eta_gt = float(self.env_config.gt_eta_min) + (
            float(self.env_config.gt_eta_max) - float(self.env_config.gt_eta_min)
        ) * gt_load_ratio
        gt_marginal_cost = price_gas / eta_gt.clamp_min(_NORM_EPS)

        price_advantage = self.torch.clamp(
            (price_e - gt_marginal_cost) / price_e.clamp_min(_NORM_EPS),
            0.0,
            1.0,
        )
        heat_support_need = self.torch.clamp(
            heat_backup_min_needed_mw / q_boiler_cap_mw,
            0.0,
            1.0,
        )
        abs_ready = self.torch.sigmoid(
            abs_margin_k / max(_NORM_EPS, float(self.env_config.abs_gate_scale_k))
        )
        support_multiplier = self.torch.clamp(
            0.65 + 0.15 * abs_ready + 0.20 * heat_support_need,
            0.0,
            1.0,
        )
        encourage_score = self.torch.clamp(price_advantage * support_multiplier, 0.0, 1.0)
        gt_target_proxy_mw = (
            net_grid_need_proxy_mw.clamp_max(p_gt_cap_mw) * encourage_score
        )
        undercommit_ratio = (gt_target_proxy_mw - p_gt_exec).clamp_min(0.0) / p_gt_cap_mw
        net_grid_need_ratio = self.torch.clamp(net_grid_need_proxy_mw / p_gt_cap_mw, 0.0, 1.0)
        return {
            "gt_target_proxy_mw": gt_target_proxy_mw,
            "p_gt_exec": p_gt_exec,
            "undercommit_ratio": undercommit_ratio,
            "price_advantage": price_advantage,
            "abs_ready": abs_ready,
            "heat_support_need": heat_support_need,
            "support_multiplier": support_multiplier,
            "net_grid_need_ratio": net_grid_need_ratio,
        }

    def _compute_gt_grid_economic_proxy_penalty(self, *, obs_batch, action_exec_batch):
        if float(self.config.economic_gt_grid_proxy_coef) <= 0.0:
            return self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
        terms = self._compute_gt_grid_proxy_terms(
            obs_batch=obs_batch,
            action_exec_batch=action_exec_batch,
        )
        if terms is None:
            return self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
        return terms["undercommit_ratio"].mean()

    def _compute_anchor_weight(self, *, obs_batch, action_exec_batch, gap_batch):
        base_weight = self._compute_abs_focus_weights(
            obs_batch=obs_batch,
            action_exec_batch=action_exec_batch,
        ).detach()
        gate_scale = max(_NORM_EPS, float(self.env_config.abs_gate_scale_k))
        reliability_risk = self.torch.zeros_like(base_weight)
        if "heat_backup_min_needed_mw" in self.observation_index:
            heat_backup_index = int(self.observation_index["heat_backup_min_needed_mw"])
            heat_backup_ratio = self.torch.clamp(
                obs_batch[:, heat_backup_index : heat_backup_index + 1]
                / max(_NORM_EPS, float(self.env_config.q_boiler_cap_mw)),
                0.0,
                1.0,
            )
            reliability_risk = self.torch.maximum(reliability_risk, heat_backup_ratio)
        if "abs_drive_margin_k" in self.observation_index:
            margin_index = int(self.observation_index["abs_drive_margin_k"])
            abs_risk = self.torch.clamp(
                -obs_batch[:, margin_index : margin_index + 1] / gate_scale,
                0.0,
                1.0,
            )
            reliability_risk = self.torch.maximum(reliability_risk, abs_risk)
        if "u_abs" in self.action_index:
            u_abs_index = int(self.action_index["u_abs"])
            abs_commit = self.torch.clamp(
                action_exec_batch[:, u_abs_index : u_abs_index + 1],
                0.0,
                1.0,
            )
            reliability_risk = self.torch.maximum(reliability_risk, 0.5 * abs_commit)
        projection_risk = self.torch.clamp(gap_batch[:, :1].detach(), 0.0, 1.0)
        anchor_mix = self.torch.clamp(
            float(self.config.exec_action_anchor_safe_floor)
            + (1.0 - float(self.config.exec_action_anchor_safe_floor))
            * self.torch.maximum(reliability_risk, projection_risk),
            0.0,
            1.0,
        )
        return base_weight * anchor_mix / (1.0 + projection_risk)

    def _compute_gt_anchor_dimension_scale(
        self,
        *,
        obs_batch,
        action_raw_batch,
        action_exec_batch,
        teacher_available_batch=None,
        teacher_action_mask_batch=None,
    ):
        anchor_scale = self.torch.ones_like(action_exec_batch)
        if (
            float(self.config.economic_gt_grid_proxy_coef) <= 0.0
            or ("u_gt" not in self.action_index and "u_bes" not in self.action_index)
        ):
            return anchor_scale
        terms = self._compute_gt_grid_proxy_terms(
            obs_batch=obs_batch,
            action_exec_batch=action_exec_batch,
        )
        if terms is None:
            return anchor_scale
        reliability_risk = self.torch.zeros(
            (obs_batch.shape[0], 1),
            dtype=obs_batch.dtype,
            device=obs_batch.device,
        )
        if "heat_backup_min_needed_mw" in self.observation_index:
            heat_backup_index = int(self.observation_index["heat_backup_min_needed_mw"])
            heat_backup_ratio = self.torch.clamp(
                obs_batch[:, heat_backup_index : heat_backup_index + 1]
                / max(_NORM_EPS, float(self.env_config.q_boiler_cap_mw)),
                0.0,
                1.0,
            )
            reliability_risk = self.torch.maximum(reliability_risk, heat_backup_ratio)
        if "abs_drive_margin_k" in self.observation_index:
            margin_index = int(self.observation_index["abs_drive_margin_k"])
            abs_risk = self.torch.clamp(
                -obs_batch[:, margin_index : margin_index + 1]
                / max(_NORM_EPS, float(self.env_config.abs_gate_scale_k)),
                0.0,
                1.0,
            )
            reliability_risk = self.torch.maximum(reliability_risk, abs_risk)
        projection_risk = self.torch.clamp(
            (action_raw_batch - action_exec_batch).abs().max(dim=1, keepdim=True).values.detach(),
            0.0,
            1.0,
        )
        gt_relax_signal = self.torch.clamp(
            2.5
            * terms["price_advantage"]
            * self.torch.maximum(terms["net_grid_need_ratio"], terms["undercommit_ratio"])
            * terms["support_multiplier"]
            * (1.0 - projection_risk),
            0.0,
            1.0,
        )
        gt_anchor_floor = float(np.clip(self.config.exec_action_anchor_safe_floor, 0.0, 1.0))
        if "u_gt" in self.action_index:
            gt_anchor_scale = self.torch.clamp(
                1.0 - (1.0 - gt_anchor_floor) * gt_relax_signal,
                gt_anchor_floor,
                1.0,
            )
            u_gt_index = int(self.action_index["u_gt"])
            anchor_scale[:, u_gt_index : u_gt_index + 1] = gt_anchor_scale
        if "u_bes" in self.action_index:
            bes_anchor_floor = float(np.clip(gt_anchor_floor * 0.5, 0.05, 1.0))
            bes_relax_signal = self.torch.clamp(
                1.75
                * terms["price_advantage"]
                * self.torch.maximum(terms["net_grid_need_ratio"], 0.5 * terms["undercommit_ratio"])
                * (1.0 - reliability_risk)
                * (1.0 - projection_risk),
                0.0,
                1.0,
            )
            u_bes_index = int(self.action_index["u_bes"])
            bes_prior_terms = self._compute_bes_price_prior_terms(obs_batch=obs_batch)
            if bes_prior_terms is not None:
                current_u_bes = action_exec_batch[:, u_bes_index : u_bes_index + 1].clamp(-1.0, 1.0)
                bes_prior_gap = self.torch.clamp(
                    (bes_prior_terms["target_u_bes"] - current_u_bes).abs(),
                    0.0,
                    1.0,
                )
                bes_prior_relax_signal = self.torch.clamp(
                    2.0
                    * bes_prior_terms["opportunity"]
                    * bes_prior_gap
                    * (1.0 - reliability_risk)
                    * (1.0 - projection_risk),
                    0.0,
                    1.0,
                )
                bes_relax_signal = self.torch.maximum(bes_relax_signal, bes_prior_relax_signal)
            bes_anchor_scale = self.torch.clamp(
                1.0 - (1.0 - bes_anchor_floor) * bes_relax_signal,
                bes_anchor_floor,
                1.0,
            )
            if teacher_available_batch is not None and teacher_action_mask_batch is not None:
                teacher_bes_available = self.torch.clamp(
                    teacher_available_batch,
                    0.0,
                    1.0,
                ) * (teacher_action_mask_batch[:, u_bes_index : u_bes_index + 1] > 0.5).to(
                    dtype=obs_batch.dtype
                )
                if float(teacher_bes_available.max().detach().cpu().item()) > 0.0:
                    teacher_bes_anchor = teacher_bes_available * float(
                        self.config.economic_teacher_bes_anchor_preserve_scale
                    )
                    bes_anchor_scale = self.torch.maximum(
                        bes_anchor_scale,
                        teacher_bes_anchor,
                    )
            anchor_scale[:, u_bes_index : u_bes_index + 1] = bes_anchor_scale
        return anchor_scale

    def _compute_economic_teacher_weight(
        self,
        *,
        obs_batch,
        action_exec_batch,
        teacher_action_exec_batch,
        teacher_action_mask_batch,
        gap_batch,
        teacher_available_batch,
    ):
        if float(self.config.economic_teacher_distill_coef) <= 0.0:
            return self.torch.zeros((obs_batch.shape[0], 1), dtype=obs_batch.dtype, device=obs_batch.device)
        teacher_mask = self.torch.clamp(teacher_available_batch, 0.0, 1.0)
        active_teacher_dims = (
            teacher_action_mask_batch * self.economic_teacher_action_weight
        ).sum(dim=1, keepdim=True)
        teacher_mask = teacher_mask * (active_teacher_dims > 0.0).to(dtype=obs_batch.dtype)
        if float(teacher_mask.max().detach().cpu().item()) <= 0.0:
            return teacher_mask
        terms = self._compute_gt_grid_proxy_terms(
            obs_batch=obs_batch,
            action_exec_batch=teacher_action_exec_batch,
        )
        if terms is None:
            opportunity_score = teacher_mask
        else:
            opportunity_score = self.torch.clamp(
                0.5 * terms["price_advantage"] + 0.5 * terms["net_grid_need_ratio"],
                0.0,
                1.0,
            )
        reliability_risk = self.torch.zeros_like(teacher_mask)
        if "heat_backup_min_needed_mw" in self.observation_index:
            heat_backup_index = int(self.observation_index["heat_backup_min_needed_mw"])
            heat_backup_ratio = self.torch.clamp(
                obs_batch[:, heat_backup_index : heat_backup_index + 1]
                / max(_NORM_EPS, float(self.env_config.q_boiler_cap_mw)),
                0.0,
                1.0,
            )
            reliability_risk = self.torch.maximum(reliability_risk, heat_backup_ratio)
        if "abs_drive_margin_k" in self.observation_index:
            margin_index = int(self.observation_index["abs_drive_margin_k"])
            abs_risk = self.torch.clamp(
                -obs_batch[:, margin_index : margin_index + 1]
                / max(_NORM_EPS, float(self.env_config.abs_gate_scale_k)),
                0.0,
                1.0,
            )
            reliability_risk = self.torch.maximum(reliability_risk, abs_risk)
        projection_risk = self.torch.clamp(gap_batch[:, :1].detach(), 0.0, 1.0)
        reliability_risk = self.torch.maximum(reliability_risk, projection_risk)
        safety_margin = self.torch.clamp(1.0 - reliability_risk, 0.0, 1.0)
        disagreement_score = self.torch.clamp(
            (teacher_action_exec_batch - action_exec_batch).abs().mean(dim=1, keepdim=True).detach(),
            0.0,
            1.0,
        )
        return teacher_mask * self.torch.clamp(
            (0.35 + 0.65 * opportunity_score)
            * (0.50 + 0.50 * safety_margin)
            * (0.50 + 0.50 * disagreement_score),
            0.0,
            1.0,
        )

    def _random_action(self) -> np.ndarray:
        return self.rng.uniform(
            low=self.action_low_np,
            high=self.action_high_np,
            size=(len(self.action_keys),),
        ).astype(np.float32)

    def _select_action(self, *, observation_vector: np.ndarray, explore: bool) -> np.ndarray:
        observation_tensor = self.torch.as_tensor(
            observation_vector.reshape(1, -1),
            dtype=self.torch.float32,
            device=self.device,
        )
        with self.torch.no_grad():
            action = self.actor(self._normalize_observation_tensor(observation_tensor))
        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if explore and self.config.exploration_noise_std > 0.0:
            noise = self.rng.normal(
                loc=0.0,
                scale=float(self.config.exploration_noise_std),
                size=action_np.shape,
            ).astype(np.float32)
            action_np = np.clip(action_np + noise, self.action_low_np, self.action_high_np)
        return self._apply_abs_cooling_blend_np(
            observation_vector=observation_vector,
            action_vector=action_np,
        )

    def _sample_target_noise(self, shape: tuple[int, ...]):
        noise = self.torch.randn(shape, dtype=self.torch.float32, device=self.device)
        noise = noise * float(self.config.target_policy_noise_std)
        clip = float(self.config.target_noise_clip)
        return self.torch.clamp(noise, -clip, clip)

    def _soft_update(self, *, source_model, target_model) -> None:
        tau = float(self.config.tau)
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    def _build_actor_metadata(
        self,
        *,
        total_env_steps: int,
        episode_idx: int,
        training_complete: bool,
        checkpoint_role: str = "candidate",
    ) -> dict[str, Any]:
        return {
            "artifact_type": "pafc_td3_actor",
            "policy_name": "pafc_td3",
            "observation_dim": int(len(self.observation_keys)),
            "action_dim": int(len(self.action_keys)),
            "observation_keys": list(self.observation_keys),
            "action_keys": list(self.action_keys),
            "hidden_dims": list(self.config.hidden_dims),
            "observation_norm": dict(self.observation_norm_payload),
            "projection_surrogate_checkpoint_path": str(
                Path(self.config.projection_surrogate_checkpoint_path).resolve()
            ).replace("\\", "/"),
            "train_year": TRAIN_YEAR,
            "seed": int(self.config.seed),
            "episode_days": int(self.config.episode_days),
            "total_env_steps": int(total_env_steps),
            "episode_idx": int(episode_idx),
            "training_complete": bool(training_complete),
            "checkpoint_role": str(checkpoint_role),
            "dual_lambdas": {
                name: float(value)
                for name, value in zip(_DUAL_NAMES, self.dual_lambdas)
            },
            "cost_critic_positive_output": True,
            "state_feasible_action_shaping_enabled": bool(self.config.state_feasible_action_shaping_enabled),
            "abs_cooling_blend_enabled": True,
            "abs_to_ech_transfer_ratio": float(
                max(_NORM_EPS, float(self.env_config.q_abs_cool_cap_mw))
                / max(_NORM_EPS, float(self.env_config.q_ech_cap_mw))
            ),
            "abs_gate_scale_k": float(self.env_config.abs_gate_scale_k),
            "abs_min_on_gate_th": float(self.config.abs_min_on_gate_th),
            "abs_min_on_u_margin": float(self.config.abs_min_on_u_margin),
            "abs_invalid_req_u_th": float(self.env_config.abs_invalid_req_u_th),
            "abs_invalid_req_gate_th": float(self.env_config.abs_invalid_req_gate_th),
            "abs_deadzone_gate_th": float(self.env_config.abs_deadzone_gate_th),
            "abs_deadzone_u_th": float(self.env_config.abs_deadzone_u_th),
            "dt_hours": float(self.env_config.dt_hours),
            "p_bes_cap_mw": float(self.env_config.p_bes_cap_mw),
            "e_bes_cap_mwh": float(self.env_config.e_bes_cap_mwh),
            "bes_soc_min": float(self.env_config.bes_soc_min),
            "bes_soc_max": float(self.env_config.bes_soc_max),
            "bes_eta_charge": float(self.env_config.bes_eta_charge),
            "bes_eta_discharge": float(self.env_config.bes_eta_discharge),
            "e_tes_cap_mwh": float(self.env_config.e_tes_cap_mwh),
            "q_tes_charge_cap_mw": float(self.env_config.q_tes_charge_cap_mw),
            "q_tes_discharge_cap_mw": float(self.env_config.q_tes_discharge_cap_mw),
            "p_gt_cap_mw": float(self.env_config.p_gt_cap_mw),
            "gt_min_output_mw": float(self.env_config.gt_min_output_mw),
            "dual_warmup_steps": int(self.config.dual_warmup_steps),
            "exec_action_anchor_coef": float(self.config.exec_action_anchor_coef),
            "exec_action_anchor_safe_floor": float(self.config.exec_action_anchor_safe_floor),
            "gt_off_deadband_ratio": float(self.config.gt_off_deadband_ratio),
            "abs_ready_focus_coef": float(self.config.abs_ready_focus_coef),
            "invalid_abs_penalty_coef": float(self.config.invalid_abs_penalty_coef),
            "economic_boiler_proxy_coef": float(self.config.economic_boiler_proxy_coef),
            "economic_abs_tradeoff_coef": float(self.config.economic_abs_tradeoff_coef),
            "economic_gt_grid_proxy_coef": float(self.config.economic_gt_grid_proxy_coef),
            "economic_teacher_distill_coef": float(self.config.economic_teacher_distill_coef),
            "economic_teacher_proxy_advantage_min": float(
                self.config.economic_teacher_proxy_advantage_min
            ),
            "economic_teacher_gt_proxy_advantage_min": float(
                self.config.economic_teacher_gt_proxy_advantage_min
            ),
            "economic_teacher_bes_proxy_advantage_min": float(
                self.config.economic_teacher_bes_proxy_advantage_min
            ),
            "economic_teacher_max_safe_abs_risk_gap": float(
                self.config.economic_teacher_max_safe_abs_risk_gap
            ),
            "economic_teacher_projection_gap_max": float(
                self.config.economic_teacher_projection_gap_max
            ),
            "economic_teacher_gt_projection_gap_max": float(
                self.config.economic_teacher_gt_projection_gap_max
            ),
            "economic_teacher_bes_price_opportunity_min": float(
                self.config.economic_teacher_bes_price_opportunity_min
            ),
            "economic_teacher_bes_anchor_preserve_scale": float(
                self.config.economic_teacher_bes_anchor_preserve_scale
            ),
            "economic_teacher_warm_start_weight": float(
                self.config.economic_teacher_warm_start_weight
            ),
            "economic_teacher_prefill_replay_boost": int(
                self.config.economic_teacher_prefill_replay_boost
            ),
            "economic_teacher_gt_action_weight": float(
                self.config.economic_teacher_gt_action_weight
            ),
            "economic_teacher_bes_action_weight": float(
                self.config.economic_teacher_bes_action_weight
            ),
            "economic_teacher_tes_action_weight": float(
                self.config.economic_teacher_tes_action_weight
            ),
            "economic_teacher_full_year_warm_start_samples": int(
                self.config.economic_teacher_full_year_warm_start_samples
            ),
            "economic_teacher_full_year_warm_start_epochs": int(
                self.config.economic_teacher_full_year_warm_start_epochs
            ),
            "economic_bes_distill_coef": float(self.config.economic_bes_distill_coef),
            "economic_bes_prior_u": float(self.config.economic_bes_prior_u),
            "economic_bes_charge_u_scale": float(self.config.economic_bes_charge_u_scale),
            "economic_bes_discharge_u_scale": float(self.config.economic_bes_discharge_u_scale),
            "economic_bes_charge_weight": float(self.config.economic_bes_charge_weight),
            "economic_bes_discharge_weight": float(self.config.economic_bes_discharge_weight),
            "economic_bes_charge_pressure_bonus": float(
                self.config.economic_bes_charge_pressure_bonus
            ),
            "economic_bes_charge_soc_ceiling": float(self.config.economic_bes_charge_soc_ceiling),
            "economic_bes_discharge_soc_floor": float(self.config.economic_bes_discharge_soc_floor),
            "economic_bes_full_year_warm_start_samples": int(
                self.config.economic_bes_full_year_warm_start_samples
            ),
            "economic_bes_full_year_warm_start_epochs": int(
                self.config.economic_bes_full_year_warm_start_epochs
            ),
            "economic_bes_full_year_warm_start_u_weight": float(
                self.config.economic_bes_full_year_warm_start_u_weight
            ),
            "economic_bes_teacher_selection_priority_boost": float(
                self.config.economic_bes_teacher_selection_priority_boost
            ),
            "economic_bes_economic_source_priority_bonus": float(
                self.config.economic_bes_economic_source_priority_bonus
            ),
            "economic_bes_economic_source_min_share": float(
                self.config.economic_bes_economic_source_min_share
            ),
            "economic_bes_idle_economic_source_min_share": float(
                self.config.economic_bes_idle_economic_source_min_share
            ),
            "economic_bes_teacher_target_min_share": float(
                self.config.economic_bes_teacher_target_min_share
            ),
            "bes_price_low_threshold": float(self.bes_price_low_threshold),
            "bes_price_high_threshold": float(self.bes_price_high_threshold),
            "expert_prefill_policy": str(self.config.expert_prefill_policy),
            "expert_prefill_checkpoint_path": str(self.config.expert_prefill_checkpoint_path),
            "expert_prefill_economic_policy": str(self.config.expert_prefill_economic_policy),
            "expert_prefill_economic_checkpoint_path": str(
                self.config.expert_prefill_economic_checkpoint_path
            ),
            "expert_prefill_steps": int(self.config.expert_prefill_steps),
            "expert_prefill_cooling_bias": float(self.config.expert_prefill_cooling_bias),
            "expert_prefill_abs_replay_boost": int(self.config.expert_prefill_abs_replay_boost),
            "expert_prefill_abs_exec_threshold": float(self.config.expert_prefill_abs_exec_threshold),
            "expert_prefill_abs_window_mining_candidates": int(self.config.expert_prefill_abs_window_mining_candidates),
            "dual_abs_margin_k": float(self.config.dual_abs_margin_k),
            "dual_qc_ratio_th": float(self.config.dual_qc_ratio_th),
            "dual_heat_backup_ratio_th": float(self.config.dual_heat_backup_ratio_th),
            "dual_safe_abs_u_th": float(self.config.dual_safe_abs_u_th),
            "actor_warm_start_epochs": int(self.config.actor_warm_start_epochs),
            "checkpoint_interval_steps": int(self.config.checkpoint_interval_steps),
            "eval_window_pool_size": int(self.config.eval_window_pool_size),
            "eval_window_count": int(self.config.eval_window_count),
            "best_gate_enabled": bool(self.config.best_gate_enabled),
            "best_gate_electric_min": float(self.config.best_gate_electric_min),
            "best_gate_heat_min": float(self.config.best_gate_heat_min),
            "best_gate_cool_min": float(self.config.best_gate_cool_min),
            "plateau_control_enabled": bool(self.config.plateau_control_enabled),
            "plateau_patience_evals": int(self.config.plateau_patience_evals),
            "plateau_lr_decay_factor": float(self.config.plateau_lr_decay_factor),
            "plateau_min_actor_lr": float(self.config.plateau_min_actor_lr),
            "plateau_min_critic_lr": float(self.config.plateau_min_critic_lr),
            "plateau_early_stop_patience_evals": int(self.config.plateau_early_stop_patience_evals),
        }

    def _write_actor_checkpoint_json(
        self,
        *,
        checkpoint_path: Path,
        json_path: Path,
        total_env_steps: int,
        episode_idx: int,
        training_complete: bool,
        checkpoint_role: str,
    ) -> None:
        payload = {
            "artifact_type": "pafc_td3_actor",
            "checkpoint_path": str(checkpoint_path.resolve()).replace("\\", "/"),
            "checkpoint_role": str(checkpoint_role),
            "summary_path": str((self.run_dir / "train" / "summary.json").resolve()).replace("\\", "/"),
            "history_path": str((self.run_dir / "train" / "history.csv").resolve()).replace("\\", "/"),
            "selection_history_path": str(self.selection_history_path.resolve()).replace("\\", "/"),
            "projection_surrogate_checkpoint_path": str(
                Path(self.config.projection_surrogate_checkpoint_path).resolve()
            ).replace("\\", "/"),
            "total_env_steps": int(total_env_steps),
            "episode_idx": int(episode_idx),
            "training_complete": bool(training_complete),
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _save_actor_checkpoint(
        self,
        *,
        checkpoint_path: Path | None = None,
        checkpoint_json_path: Path | None = None,
        total_env_steps: int,
        episode_idx: int,
        training_complete: bool,
        checkpoint_role: str = "candidate",
    ) -> Path:
        target_checkpoint_path = checkpoint_path or self.actor_checkpoint_path
        save_policy(
            model=self.actor,
            checkpoint_path=target_checkpoint_path,
            metadata=self._build_actor_metadata(
                total_env_steps=total_env_steps,
                episode_idx=episode_idx,
                training_complete=training_complete,
                checkpoint_role=checkpoint_role,
            ),
        )
        if checkpoint_json_path is not None:
            self._write_actor_checkpoint_json(
                checkpoint_path=target_checkpoint_path,
                json_path=checkpoint_json_path,
                total_env_steps=total_env_steps,
                episode_idx=episode_idx,
                training_complete=training_complete,
                checkpoint_role=checkpoint_role,
            )
        return target_checkpoint_path

    def _build_retained_checkpoint_path(
        self,
        *,
        total_env_steps: int,
        episode_idx: int,
    ) -> Path:
        return self.retained_checkpoints_dir / (
            f"pafc_td3_actor_step_{int(total_env_steps):07d}_ep_{int(episode_idx):04d}.pt"
        )

    def _append_selection_history(self, payload: Mapping[str, Any]) -> None:
        with self.selection_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")

    def _set_optimizer_lr(self, optimizer, learning_rate: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)

    def _apply_learning_rates(self, *, actor_lr: float, critic_lr: float) -> None:
        self._set_optimizer_lr(self.actor_optimizer, actor_lr)
        self._set_optimizer_lr(self.reward_critic_optimizer, critic_lr)
        self._set_optimizer_lr(self.cost_critic_optimizer, critic_lr)
        self.current_actor_lr = float(actor_lr)
        self.current_critic_lr = float(critic_lr)

    def _record_plateau_event(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        event = {
            key: (
                float(value)
                if isinstance(value, (int, float, np.integer, np.floating))
                else value
            )
            for key, value in dict(payload).items()
        }
        self.plateau_events.append(event)
        return event

    def _restore_actor_from_checkpoint(
        self,
        *,
        checkpoint_path: str | Path,
        actor_lr: float,
        restore_dual_lambdas: bool = True,
    ) -> dict[str, Any]:
        payload = load_policy(
            checkpoint_path,
            map_location=str(self.device),
        )
        self.actor.load_state_dict(payload["state_dict"])
        self.actor_target.load_state_dict(self.actor.state_dict())
        _, _, _, AdamW = _require_torch_modules()
        self.actor_optimizer = AdamW(self.actor.parameters(), lr=float(actor_lr))
        restored_dual_lambdas = False
        metadata = dict(payload.get("metadata", {}) or {})
        dual_lambdas = dict(metadata.get("dual_lambdas", {}) or {})
        if restore_dual_lambdas and all(name in dual_lambdas for name in _DUAL_NAMES):
            self.dual_lambdas = np.asarray(
                [float(dual_lambdas[name]) for name in _DUAL_NAMES],
                dtype=np.float32,
            )
            restored_dual_lambdas = True
        return {
            "checkpoint_path": str(Path(checkpoint_path).resolve()).replace("\\", "/"),
            "restored_dual_lambdas": bool(restored_dual_lambdas),
        }

    def _resolve_checkpoint_interval_steps(self) -> int:
        if int(self.config.checkpoint_interval_steps) > 0:
            return int(self.config.checkpoint_interval_steps)
        return max(1, min(2_000, max(500, int(self.config.total_env_steps // 4))))

    def _evaluate_checkpoint_candidate(
        self,
        *,
        checkpoint_path: Path,
        total_env_steps: int,
        episode_idx: int,
        training_complete: bool,
    ) -> dict[str, Any]:
        def predictor(observation: Mapping[str, float]) -> dict[str, float]:
            observation_vector = self._observation_to_vector(observation)
            action_vector = self._select_action(
                observation_vector=observation_vector,
                explore=False,
            )
            return _action_vector_to_dict(action_vector, action_keys=self.action_keys)

        episode_summaries = [
            _evaluate_predictor_on_episode_df(
                predictor=predictor,
                exogenous_df=self.train_df,
                episode_df=episode_df,
                env_config=self.env_config,
                seed=int(self.config.seed + 10_000 + idx),
            )
            for idx, episode_df in enumerate(self.eval_episode_dfs)
        ]
        metrics = _aggregate_eval_episode_summaries(episode_summaries)
        gate_result = _build_reliability_gate_result(metrics=metrics, config=self.config)
        shortfall = dict(gate_result.get("shortfall") or {})
        current_rank = (
            1 if bool(gate_result.get("passed", False)) else 0,
            -float(shortfall.get("total", 0.0)),
            -float(shortfall.get("max", 0.0)),
            -float(metrics.get("mean_total_cost", 0.0)),
            -float(metrics.get("mean_violation_rate", 0.0)),
            float(metrics.get("mean_reward", float("-inf"))),
        )
        reward_improved = float(metrics.get("mean_reward", float("-inf"))) > float(self.best_reward_mean)
        best_improved = bool(current_rank > self.best_selection_rank)
        snapshot = {
            "timesteps": int(total_env_steps),
            "episode_idx": int(episode_idx),
            "training_complete": bool(training_complete),
            "checkpoint_path": str(checkpoint_path.resolve()).replace("\\", "/"),
            "metrics": dict(metrics),
            "gate": dict(gate_result),
        }
        if reward_improved:
            self.best_reward_mean = float(metrics.get("mean_reward", float("-inf")))
            self.best_reward_snapshot = dict(snapshot)
            self._save_actor_checkpoint(
                checkpoint_path=self.reward_actor_checkpoint_path,
                checkpoint_json_path=self.reward_actor_checkpoint_json,
                total_env_steps=total_env_steps,
                episode_idx=episode_idx,
                training_complete=training_complete,
                checkpoint_role="reward_leader",
            )
        if best_improved:
            self.best_selection_rank = current_rank
            self.best_selection_snapshot = dict(snapshot)
            self._save_actor_checkpoint(
                checkpoint_path=self.actor_checkpoint_path,
                checkpoint_json_path=self.actor_checkpoint_json,
                total_env_steps=total_env_steps,
                episode_idx=episode_idx,
                training_complete=training_complete,
                checkpoint_role="best",
            )
        self.validation_eval_count = int(self.validation_eval_count + 1)
        history_item = {
            "evaluation_index": int(self.validation_eval_count),
            "timesteps": int(total_env_steps),
            "episode_idx": int(episode_idx),
            "training_complete": bool(training_complete),
            "checkpoint_path": str(checkpoint_path.resolve()).replace("\\", "/"),
            "mean_reward": float(metrics["mean_reward"]),
            "mean_total_cost": float(metrics["mean_total_cost"]),
            "mean_violation_rate": float(metrics["mean_violation_rate"]),
            "reliability_mean": dict(metrics["reliability_mean"]),
            "reliability_min": dict(metrics["reliability_min"]),
            "mean_cost_breakdown": dict(metrics["mean_cost_breakdown"]),
            "gate": dict(gate_result),
            "selected_as_best": bool(best_improved),
            "selected_as_reward_leader": bool(reward_improved),
        }
        self._append_selection_history(history_item)
        return history_item

    def _handle_plateau_after_eval(
        self,
        *,
        evaluation_item: Mapping[str, Any],
        total_env_steps: int,
        episode_idx: int,
    ) -> dict[str, Any] | None:
        improved = bool(evaluation_item.get("selected_as_best", False))
        if improved:
            self.no_improve_evals = 0
        else:
            self.no_improve_evals = int(self.no_improve_evals + 1)

        if not bool(self.config.plateau_control_enabled):
            return None

        plateau_event: dict[str, Any] | None = None
        if (not improved) and (not self.fine_tune_applied) and self.no_improve_evals >= int(self.config.plateau_patience_evals):
            new_actor_lr = max(
                float(self.config.plateau_min_actor_lr),
                float(self.current_actor_lr) * float(self.config.plateau_lr_decay_factor),
            )
            new_critic_lr = max(
                float(self.config.plateau_min_critic_lr),
                float(self.current_critic_lr) * float(self.config.plateau_lr_decay_factor),
            )
            if new_actor_lr < float(self.current_actor_lr) - 1e-12 or new_critic_lr < float(self.current_critic_lr) - 1e-12:
                old_actor_lr = float(self.current_actor_lr)
                old_critic_lr = float(self.current_critic_lr)
                rollback_info: dict[str, Any] | None = None
                if self.best_selection_snapshot is not None:
                    best_checkpoint_path = str(self.best_selection_snapshot.get("checkpoint_path", "") or "").strip()
                    if len(best_checkpoint_path) > 0 and Path(best_checkpoint_path).exists():
                        rollback_info = self._restore_actor_from_checkpoint(
                            checkpoint_path=best_checkpoint_path,
                            actor_lr=float(new_actor_lr),
                            restore_dual_lambdas=True,
                        )
                self._apply_learning_rates(actor_lr=new_actor_lr, critic_lr=new_critic_lr)
                self.lr_decay_count = int(self.lr_decay_count + 1)
                self.fine_tune_applied = True
                event_payload = {
                    "timesteps": int(total_env_steps),
                    "episode_idx": int(episode_idx),
                    "action": "low_lr_fine_tune",
                    "stale_evals": int(self.no_improve_evals),
                    "old_actor_lr": old_actor_lr,
                    "new_actor_lr": float(new_actor_lr),
                    "old_critic_lr": old_critic_lr,
                    "new_critic_lr": float(new_critic_lr),
                    "rollback_to_best": bool(rollback_info is not None),
                }
                if rollback_info is not None:
                    event_payload["rollback_checkpoint_path"] = str(rollback_info["checkpoint_path"])
                    event_payload["rollback_restored_dual_lambdas"] = bool(
                        rollback_info["restored_dual_lambdas"]
                    )
                    event_payload["rollback_best_timesteps"] = int(
                        self.best_selection_snapshot.get("timesteps", total_env_steps)
                    )
                plateau_event = self._record_plateau_event(event_payload)
                self.no_improve_evals = 0
            else:
                self.fine_tune_applied = True
        elif self.fine_tune_applied and self.no_improve_evals >= int(self.config.plateau_early_stop_patience_evals):
            self.stop_requested = True
            self.stop_reason = "plateau_after_low_lr_fine_tune"
            plateau_event = self._record_plateau_event(
                {
                    "timesteps": int(total_env_steps),
                    "episode_idx": int(episode_idx),
                    "action": "early_stop",
                    "stale_evals": int(self.no_improve_evals),
                    "actor_lr": float(self.current_actor_lr),
                    "critic_lr": float(self.current_critic_lr),
                }
            )
        return plateau_event

    def _update_networks(self, *, update_step: int) -> dict[str, float]:
        batch = self.replay.sample(batch_size=int(self.config.batch_size), rng=self.rng)
        obs = self.torch.as_tensor(batch["obs"], dtype=self.torch.float32, device=self.device)
        next_obs = self.torch.as_tensor(batch["next_obs"], dtype=self.torch.float32, device=self.device)
        action_exec = self.torch.as_tensor(batch["action_exec"], dtype=self.torch.float32, device=self.device)
        teacher_action_exec = self.torch.as_tensor(
            batch["teacher_action_exec"],
            dtype=self.torch.float32,
            device=self.device,
        )
        teacher_action_mask = self.torch.as_tensor(
            batch["teacher_action_mask"],
            dtype=self.torch.float32,
            device=self.device,
        )
        teacher_available = self.torch.as_tensor(
            batch["teacher_available"],
            dtype=self.torch.float32,
            device=self.device,
        )
        gap = self.torch.as_tensor(batch["gap"], dtype=self.torch.float32, device=self.device)
        reward = self.torch.as_tensor(batch["reward"], dtype=self.torch.float32, device=self.device)
        cost = self.torch.as_tensor(batch["cost"], dtype=self.torch.float32, device=self.device)
        done = self.torch.as_tensor(batch["done"], dtype=self.torch.float32, device=self.device)

        obs_norm = self._normalize_observation_tensor(obs)
        next_obs_norm = self._normalize_observation_tensor(next_obs)

        with self.torch.no_grad():
            next_action_raw = self.actor_target(next_obs_norm)
            next_action_raw = self.torch.clamp(
                next_action_raw + self._sample_target_noise(tuple(next_action_raw.shape)),
                self.action_low,
                self.action_high,
            )
            next_action_raw = self._apply_abs_cooling_blend_tensor(
                obs_batch=next_obs,
                action_batch=next_action_raw,
            )
            next_action_exec_hat = self.surrogate.project(next_obs, next_action_raw)
            reward_target = reward + float(self.config.gamma) * (1.0 - done) * self.torch.minimum(
                self.q1_target(next_obs_norm, next_action_exec_hat),
                self.q2_target(next_obs_norm, next_action_exec_hat),
            )
            cost_targets = [
                cost[:, idx : idx + 1]
                + float(self.config.gamma)
                * (1.0 - done)
                * self.cost_target_critics[idx](next_obs_norm, next_action_exec_hat).clamp_min(0.0)
                for idx in range(3)
            ]

        q1_pred = self.q1(obs_norm, action_exec)
        q2_pred = self.q2(obs_norm, action_exec)
        reward_critic_loss = self.F.mse_loss(q1_pred, reward_target) + self.F.mse_loss(
            q2_pred, reward_target
        )
        self.reward_critic_optimizer.zero_grad(set_to_none=True)
        reward_critic_loss.backward()
        self.reward_critic_optimizer.step()

        cost_predictions = [critic(obs_norm, action_exec) for critic in self.cost_critics]
        cost_critic_loss = sum(
            self.F.mse_loss(prediction, target)
            for prediction, target in zip(cost_predictions, cost_targets)
        )
        self.cost_critic_optimizer.zero_grad(set_to_none=True)
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()

        actor_loss_value = float("nan")
        gap_loss_value = float("nan")
        exec_anchor_loss_value = float("nan")
        invalid_abs_penalty_value = float("nan")
        boiler_proxy_penalty_value = float("nan")
        abs_tradeoff_penalty_value = float("nan")
        gt_grid_proxy_penalty_value = float("nan")
        gt_anchor_scale_value = float("nan")
        bes_anchor_scale_value = float("nan")
        bes_teacher_anchor_rate_value = float("nan")
        economic_teacher_loss_value = float("nan")
        economic_teacher_weight_value = float("nan")
        economic_teacher_target_rate_value = float("nan")
        economic_bes_distill_loss_value = float("nan")
        economic_bes_distill_weight_value = float("nan")
        reward_actor_value = float("nan")
        mean_constraint_value = float("nan")
        dual_scale_value = float(
            min(
                1.0,
                float(update_step) / max(1.0, float(self.config.dual_warmup_steps)),
            )
        ) if int(self.config.dual_warmup_steps) > 0 else 1.0
        if update_step % int(self.config.actor_delay) == 0:
            action_raw = self.actor(obs_norm)
            action_raw = self._apply_abs_cooling_blend_tensor(
                obs_batch=obs,
                action_batch=action_raw,
            )
            action_exec_hat = self.surrogate.project(obs, action_raw)
            reward_actor = self.torch.minimum(
                self.q1(obs_norm, action_exec_hat),
                self.q2(obs_norm, action_exec_hat),
            )
            constraint_predictions = self.torch.cat(
                [critic(obs_norm, action_exec_hat) for critic in self.cost_critics],
                dim=1,
            ).clamp_min(0.0)
            lambda_tensor = self.torch.as_tensor(
                (self.dual_lambdas * float(dual_scale_value)).reshape(1, -1),
                dtype=self.torch.float32,
                device=self.device,
            )
            gap_loss = self.F.mse_loss(action_raw, action_exec_hat)
            support_weight = self._compute_anchor_weight(
                obs_batch=obs,
                action_exec_batch=action_exec,
                gap_batch=gap,
            )
            gt_anchor_dimension_scale = self._compute_gt_anchor_dimension_scale(
                obs_batch=obs,
                action_raw_batch=action_raw,
                action_exec_batch=action_exec_hat,
                teacher_available_batch=teacher_available,
                teacher_action_mask_batch=teacher_action_mask,
            )
            exec_anchor_loss = (
                support_weight
                * (
                    gt_anchor_dimension_scale
                    * (action_exec_hat - action_exec).pow(2)
                ).mean(dim=1, keepdim=True)
            ).mean()
            invalid_abs_penalty = self._compute_invalid_abs_request_penalty(
                obs_batch=obs,
                action_raw_batch=action_raw,
            )
            boiler_proxy_penalty = self._compute_boiler_economic_proxy_penalty(
                obs_batch=obs,
                action_exec_batch=action_exec_hat,
            )
            abs_tradeoff_penalty = self._compute_abs_ech_tradeoff_proxy_penalty(
                obs_batch=obs,
                action_exec_batch=action_exec_hat,
            )
            gt_grid_proxy_penalty = self._compute_gt_grid_economic_proxy_penalty(
                obs_batch=obs,
                action_exec_batch=action_exec_hat,
            )
            economic_teacher_weight = self._compute_economic_teacher_weight(
                obs_batch=obs,
                action_exec_batch=action_exec_hat,
                teacher_action_exec_batch=teacher_action_exec,
                teacher_action_mask_batch=teacher_action_mask,
                gap_batch=gap,
                teacher_available_batch=teacher_available,
            )
            effective_teacher_mask = teacher_action_mask * self.economic_teacher_action_weight
            economic_teacher_sq = (
                (action_exec_hat - teacher_action_exec).pow(2) * effective_teacher_mask
            ).sum(dim=1, keepdim=True) / effective_teacher_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            economic_teacher_weight_sum = economic_teacher_weight.sum()
            economic_teacher_loss = (
                (economic_teacher_weight * economic_teacher_sq).sum()
                / economic_teacher_weight_sum.clamp_min(1.0)
            )
            economic_bes_distill_loss, economic_bes_distill_weight = self._compute_bes_prior_distill_loss(
                obs_batch=obs,
                action_raw_batch=action_raw,
                action_exec_batch=action_exec_hat,
                gap_batch=gap,
            )
            actor_loss = (
                -reward_actor.mean()
                + (lambda_tensor * constraint_predictions).sum(dim=1).mean()
                + float(self.config.gap_penalty_coef) * gap_loss
                + float(self.config.exec_action_anchor_coef) * exec_anchor_loss
                + float(self.config.invalid_abs_penalty_coef) * invalid_abs_penalty
                + float(self.config.economic_boiler_proxy_coef) * boiler_proxy_penalty
                + float(self.config.economic_abs_tradeoff_coef) * abs_tradeoff_penalty
                + float(self.config.economic_gt_grid_proxy_coef) * gt_grid_proxy_penalty
                + float(self.config.economic_teacher_distill_coef) * economic_teacher_loss
                + float(self.config.economic_bes_distill_coef) * economic_bes_distill_loss
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(source_model=self.actor, target_model=self.actor_target)
            self._soft_update(source_model=self.q1, target_model=self.q1_target)
            self._soft_update(source_model=self.q2, target_model=self.q2_target)
            for source, target in zip(self.cost_critics, self.cost_target_critics):
                self._soft_update(source_model=source, target_model=target)

            batch_cost_mean = cost.mean(dim=0).detach().cpu().numpy().astype(np.float32)
            if dual_scale_value >= 1.0:
                self.dual_lambdas = np.maximum(
                    0.0,
                    self.dual_lambdas
                    + float(self.config.dual_lr) * (batch_cost_mean - self.dual_targets),
                ).astype(np.float32)

            actor_loss_value = float(actor_loss.detach().cpu().item())
            gap_loss_value = float(gap_loss.detach().cpu().item())
            exec_anchor_loss_value = float(exec_anchor_loss.detach().cpu().item())
            invalid_abs_penalty_value = float(invalid_abs_penalty.detach().cpu().item())
            boiler_proxy_penalty_value = float(boiler_proxy_penalty.detach().cpu().item())
            abs_tradeoff_penalty_value = float(abs_tradeoff_penalty.detach().cpu().item())
            gt_grid_proxy_penalty_value = float(gt_grid_proxy_penalty.detach().cpu().item())
            if "u_gt" in self.action_index:
                u_gt_index = int(self.action_index["u_gt"])
                gt_anchor_scale_value = float(
                    gt_anchor_dimension_scale[:, u_gt_index : u_gt_index + 1]
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
            if "u_bes" in self.action_index:
                u_bes_index = int(self.action_index["u_bes"])
                bes_anchor_scale_value = float(
                    gt_anchor_dimension_scale[:, u_bes_index : u_bes_index + 1]
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
                teacher_bes_mask = (
                    self.torch.clamp(teacher_available, 0.0, 1.0)
                    * (teacher_action_mask[:, u_bes_index : u_bes_index + 1] > 0.5).to(
                        dtype=obs.dtype
                    )
                )
                bes_teacher_anchor_rate_value = float(
                    teacher_bes_mask.mean().detach().cpu().item()
                )
            economic_teacher_loss_value = float(economic_teacher_loss.detach().cpu().item())
            economic_teacher_weight_value = float(economic_teacher_weight.mean().detach().cpu().item())
            economic_teacher_target_rate_value = float(teacher_available.mean().detach().cpu().item())
            economic_bes_distill_loss_value = float(
                economic_bes_distill_loss.detach().cpu().item()
            )
            economic_bes_distill_weight_value = float(
                economic_bes_distill_weight.detach().cpu().item()
            )
            reward_actor_value = float(reward_actor.mean().detach().cpu().item())
            mean_constraint_value = float(constraint_predictions.mean().detach().cpu().item())

        return {
            "reward_critic_loss": float(reward_critic_loss.detach().cpu().item()),
            "cost_critic_loss": float(cost_critic_loss.detach().cpu().item()),
            "actor_loss": actor_loss_value,
            "actor_reward_q": reward_actor_value,
            "actor_gap_loss": gap_loss_value,
            "actor_exec_anchor_loss": exec_anchor_loss_value,
            "actor_invalid_abs_penalty": invalid_abs_penalty_value,
            "actor_boiler_proxy_penalty": boiler_proxy_penalty_value,
            "actor_abs_tradeoff_penalty": abs_tradeoff_penalty_value,
            "actor_gt_grid_proxy_penalty": gt_grid_proxy_penalty_value,
            "actor_gt_anchor_scale": gt_anchor_scale_value,
            "actor_bes_anchor_scale": bes_anchor_scale_value,
            "actor_bes_teacher_anchor_rate": bes_teacher_anchor_rate_value,
            "actor_economic_teacher_loss": economic_teacher_loss_value,
            "actor_economic_teacher_weight": economic_teacher_weight_value,
            "actor_economic_teacher_target_rate": economic_teacher_target_rate_value,
            "actor_economic_bes_distill_loss": economic_bes_distill_loss_value,
            "actor_economic_bes_distill_weight": economic_bes_distill_weight_value,
            "actor_constraint_mean": mean_constraint_value,
            "dual_scale": float(dual_scale_value),
            "lambda_e": float(self.dual_lambdas[0]),
            "lambda_h": float(self.dual_lambdas[1]),
            "lambda_c": float(self.dual_lambdas[2]),
        }

    def train(self) -> dict[str, Any]:
        train_start_time = time.perf_counter()
        (
            prefill_observations,
            prefill_action_exec,
            prefill_teacher_action_exec,
            prefill_teacher_action_mask,
            prefill_teacher_available,
        ) = self._prefill_replay_with_expert()
        self._warm_start_actor_from_expert(
            observations=prefill_observations,
            action_exec_targets=prefill_action_exec,
            teacher_action_exec=prefill_teacher_action_exec,
            teacher_action_mask=prefill_teacher_action_mask,
            teacher_available=prefill_teacher_available,
        )
        self._warm_start_actor_from_economic_teacher_full_year()
        self._warm_start_actor_from_bes_full_year()
        effective_warmup_steps = 0 if self.replay.size > 0 else int(self.config.warmup_steps)

        sampler = make_episode_sampler(
            self.train_df,
            episode_days=int(self.config.episode_days),
            seed=int(self.config.seed),
        )
        total_env_steps = 0
        episode_idx = 0
        checkpoint_every = self._resolve_checkpoint_interval_steps()
        next_checkpoint_step = checkpoint_every
        history_rows: list[dict[str, Any]] = []
        latest_update_metrics: dict[str, float] = {
            "reward_critic_loss": float("nan"),
            "cost_critic_loss": float("nan"),
            "actor_loss": float("nan"),
            "actor_reward_q": float("nan"),
            "actor_gap_loss": float("nan"),
            "actor_exec_anchor_loss": float("nan"),
            "actor_invalid_abs_penalty": float("nan"),
            "actor_boiler_proxy_penalty": float("nan"),
            "actor_abs_tradeoff_penalty": float("nan"),
            "actor_gt_grid_proxy_penalty": float("nan"),
            "actor_gt_anchor_scale": float("nan"),
            "actor_bes_anchor_scale": float("nan"),
            "actor_bes_teacher_anchor_rate": float("nan"),
            "actor_economic_teacher_loss": float("nan"),
            "actor_economic_teacher_weight": float("nan"),
            "actor_economic_teacher_target_rate": float("nan"),
            "actor_economic_bes_distill_loss": float("nan"),
            "actor_economic_bes_distill_weight": float("nan"),
            "actor_constraint_mean": float("nan"),
            "dual_scale": 1.0,
            "lambda_e": 0.0,
            "lambda_h": 0.0,
            "lambda_c": 0.0,
        }

        initial_checkpoint_path = self._save_actor_checkpoint(
            checkpoint_path=self._build_retained_checkpoint_path(
                total_env_steps=total_env_steps,
                episode_idx=episode_idx,
            ),
            total_env_steps=total_env_steps,
            episode_idx=episode_idx,
            training_complete=False,
            checkpoint_role="candidate_init",
        )
        self._evaluate_checkpoint_candidate(
            checkpoint_path=initial_checkpoint_path,
            total_env_steps=total_env_steps,
            episode_idx=episode_idx,
            training_complete=False,
        )
        self._handle_plateau_after_eval(
            evaluation_item={"selected_as_best": True},
            total_env_steps=total_env_steps,
            episode_idx=episode_idx,
        )

        while total_env_steps < int(self.config.total_env_steps) and (not self.stop_requested):
            _, episode_df = next(sampler)
            env = CCHPPhysicalEnv(
                exogenous_df=self.train_df,
                config=self.env_config,
                seed=int(self.config.seed + episode_idx),
            )
            observation, _ = env.reset(
                seed=int(self.config.seed + episode_idx),
                episode_df=episode_df,
            )
            if self.economic_teacher_policy is not None:
                self._bind_policy_episode_context(
                    policy=self.economic_teacher_policy,
                    env=env,
                    observation=observation,
                    episode_seed=int(self.config.seed + episode_idx),
                )
            if self.economic_teacher_policy is not None and hasattr(self.economic_teacher_policy, "reset_episode"):
                self.economic_teacher_policy.reset_episode(observation=observation)
            if self.economic_teacher_safe_policy is not None:
                self._bind_policy_episode_context(
                    policy=self.economic_teacher_safe_policy,
                    env=env,
                    observation=observation,
                    episode_seed=int(self.config.seed + episode_idx),
                )
            if self.economic_teacher_safe_policy is not None and hasattr(self.economic_teacher_safe_policy, "reset_episode"):
                self.economic_teacher_safe_policy.reset_episode(observation=observation)
            terminated = False
            episode_reward = 0.0
            episode_cost_sum = np.zeros(3, dtype=np.float64)
            episode_gap_sum = np.zeros(3, dtype=np.float64)
            episode_steps = 0

            while (not terminated) and total_env_steps < int(self.config.total_env_steps) and (not self.stop_requested):
                obs_vector = self._observation_to_vector(observation)
                teacher_action_exec, teacher_action_mask, teacher_available = self._get_economic_teacher_target_step(
                    observation=observation,
                    obs_vector=obs_vector,
                )
                if total_env_steps < int(effective_warmup_steps):
                    action_raw = self._random_action()
                else:
                    action_raw = self._select_action(
                        observation_vector=obs_vector,
                        explore=True,
                    )
                env_action = _action_vector_to_dict(action_raw, action_keys=self.action_keys)
                next_observation, reward, terminated, _, info = env.step(env_action)
                next_obs_vector = self._observation_to_vector(next_observation)
                action_exec = _extract_action_vector_from_info(
                    info,
                    prefix="action_exec",
                    action_keys=self.action_keys,
                )
                cost = _extract_cost_vector(info)
                gap = _extract_gap_vector(info)
                self.replay.add(
                    obs=obs_vector,
                    next_obs=next_obs_vector,
                    action_raw=action_raw,
                    action_exec=action_exec,
                    teacher_action_exec=teacher_action_exec,
                    teacher_action_mask=teacher_action_mask,
                    teacher_available=bool(teacher_available),
                    reward=float(reward),
                    cost=cost,
                    gap=gap,
                    done=bool(terminated),
                )
                if teacher_available and self.economic_teacher_policy is not None:
                    self.economic_teacher_distill_summary["online_target_steps"] = int(
                        self.economic_teacher_distill_summary.get("online_target_steps", 0)
                    ) + 1

                observation = next_observation
                total_env_steps += 1
                episode_steps += 1
                episode_reward += float(reward)
                episode_cost_sum += cost.astype(np.float64)
                episode_gap_sum += gap.astype(np.float64)

                if (
                    self.replay.size >= int(self.config.batch_size)
                    and total_env_steps >= int(effective_warmup_steps)
                ):
                    for _ in range(int(self.config.updates_per_step)):
                        latest_update_metrics = self._update_networks(update_step=total_env_steps)

                if total_env_steps >= next_checkpoint_step:
                    retained_checkpoint_path = self._save_actor_checkpoint(
                        checkpoint_path=self._build_retained_checkpoint_path(
                            total_env_steps=total_env_steps,
                            episode_idx=episode_idx,
                        ),
                        total_env_steps=total_env_steps,
                        episode_idx=episode_idx,
                        training_complete=False,
                        checkpoint_role="candidate",
                    )
                    evaluation_item = self._evaluate_checkpoint_candidate(
                        checkpoint_path=retained_checkpoint_path,
                        total_env_steps=total_env_steps,
                        episode_idx=episode_idx,
                        training_complete=False,
                    )
                    self._handle_plateau_after_eval(
                        evaluation_item=evaluation_item,
                        total_env_steps=total_env_steps,
                        episode_idx=episode_idx,
                    )
                    next_checkpoint_step += checkpoint_every
                    if self.stop_requested:
                        break

            mean_cost = episode_cost_sum / max(1, episode_steps)
            mean_gap = episode_gap_sum / max(1, episode_steps)
            history_rows.append(
                {
                    "episode_idx": int(episode_idx),
                    "episode_steps": int(episode_steps),
                    "total_env_steps": int(total_env_steps),
                    "reward_sum": float(episode_reward),
                    "reward_mean": float(episode_reward / max(1, episode_steps)),
                    "cost_e_mean": float(mean_cost[0]),
                    "cost_h_mean": float(mean_cost[1]),
                    "cost_c_mean": float(mean_cost[2]),
                    "projection_gap_l1_mean": float(mean_gap[0]),
                    "projection_gap_l2_mean": float(mean_gap[1]),
                    "projection_gap_max_mean": float(mean_gap[2]),
                    "replay_size": int(self.replay.size),
                    "actor_lr_current": float(self.current_actor_lr),
                    "critic_lr_current": float(self.current_critic_lr),
                    "plateau_no_improve_evals": int(self.no_improve_evals),
                    "plateau_fine_tune_applied": bool(self.fine_tune_applied),
                    "plateau_stop_requested": bool(self.stop_requested),
                    **latest_update_metrics,
                }
            )
            episode_idx += 1

        pd.DataFrame(history_rows).to_csv(self.run_dir / "train" / "history.csv", index=False)
        final_checkpoint_path = self._save_actor_checkpoint(
            checkpoint_path=self.last_actor_checkpoint_path,
            checkpoint_json_path=self.last_actor_checkpoint_json,
            total_env_steps=total_env_steps,
            episode_idx=episode_idx,
            training_complete=True,
            checkpoint_role="last",
        )
        final_eval_item = self._evaluate_checkpoint_candidate(
            checkpoint_path=final_checkpoint_path,
            total_env_steps=total_env_steps,
            episode_idx=episode_idx,
            training_complete=True,
        )
        self._handle_plateau_after_eval(
            evaluation_item=final_eval_item,
            total_env_steps=total_env_steps,
            episode_idx=episode_idx,
        )
        if self.best_selection_snapshot is None:
            self._save_actor_checkpoint(
                checkpoint_path=self.actor_checkpoint_path,
                checkpoint_json_path=self.actor_checkpoint_json,
                total_env_steps=total_env_steps,
                episode_idx=episode_idx,
                training_complete=True,
                checkpoint_role="best_fallback",
            )
            self.best_selection_snapshot = {
                "timesteps": int(total_env_steps),
                "episode_idx": int(episode_idx),
                "training_complete": True,
                "checkpoint_path": str(self.actor_checkpoint_path.resolve()).replace("\\", "/"),
                "metrics": {},
                "gate": {"enabled": bool(self.config.best_gate_enabled), "passed": False},
                "fallback_reason": "no_validation_triggered",
            }
        last_row = history_rows[-1] if history_rows else {}
        train_wall_time_s = float(time.perf_counter() - train_start_time)
        summary = {
            "mode": "train",
            "policy": "pafc_td3",
            "train_year": TRAIN_YEAR,
            "run_dir": str(self.run_dir.resolve()).replace("\\", "/"),
            "actor_checkpoint_path": str(self.actor_checkpoint_path.resolve()).replace("\\", "/"),
            "actor_checkpoint_json_path": str(self.actor_checkpoint_json.resolve()).replace("\\", "/"),
            "last_actor_checkpoint_path": str(self.last_actor_checkpoint_path.resolve()).replace("\\", "/"),
            "last_actor_checkpoint_json_path": str(self.last_actor_checkpoint_json.resolve()).replace("\\", "/"),
            "best_reward_actor_checkpoint_path": str(self.reward_actor_checkpoint_path.resolve()).replace("\\", "/"),
            "best_reward_actor_checkpoint_json_path": str(self.reward_actor_checkpoint_json.resolve()).replace("\\", "/"),
            "selection_history_path": str(self.selection_history_path.resolve()).replace("\\", "/"),
            "projection_surrogate_checkpoint_path": str(
                Path(self.config.projection_surrogate_checkpoint_path).resolve()
            ).replace("\\", "/"),
            "device": str(self.device),
            "episodes": int(episode_idx),
            "total_env_steps": int(total_env_steps),
            "train_wall_time_s": train_wall_time_s,
            "train_steps_per_second": float(float(total_env_steps) / max(_NORM_EPS, train_wall_time_s)),
            "episode_days": int(self.config.episode_days),
            "warmup_steps": int(self.config.warmup_steps),
            "effective_warmup_steps": int(effective_warmup_steps),
            "batch_size": int(self.config.batch_size),
            "updates_per_step": int(self.config.updates_per_step),
            "gamma": float(self.config.gamma),
            "tau": float(self.config.tau),
            "actor_lr": float(self.config.actor_lr),
            "critic_lr": float(self.config.critic_lr),
            "dual_lr": float(self.config.dual_lr),
            "dual_warmup_steps": int(self.config.dual_warmup_steps),
            "gap_penalty_coef": float(self.config.gap_penalty_coef),
            "exec_action_anchor_coef": float(self.config.exec_action_anchor_coef),
            "exec_action_anchor_safe_floor": float(self.config.exec_action_anchor_safe_floor),
            "gt_off_deadband_ratio": float(self.config.gt_off_deadband_ratio),
            "abs_ready_focus_coef": float(self.config.abs_ready_focus_coef),
            "invalid_abs_penalty_coef": float(self.config.invalid_abs_penalty_coef),
            "economic_boiler_proxy_coef": float(self.config.economic_boiler_proxy_coef),
            "economic_abs_tradeoff_coef": float(self.config.economic_abs_tradeoff_coef),
            "economic_gt_grid_proxy_coef": float(self.config.economic_gt_grid_proxy_coef),
            "economic_teacher_distill_coef": float(self.config.economic_teacher_distill_coef),
            "economic_teacher_proxy_advantage_min": float(
                self.config.economic_teacher_proxy_advantage_min
            ),
            "economic_teacher_gt_proxy_advantage_min": float(
                self.config.economic_teacher_gt_proxy_advantage_min
            ),
            "economic_teacher_bes_proxy_advantage_min": float(
                self.config.economic_teacher_bes_proxy_advantage_min
            ),
            "economic_teacher_max_safe_abs_risk_gap": float(
                self.config.economic_teacher_max_safe_abs_risk_gap
            ),
            "economic_teacher_projection_gap_max": float(
                self.config.economic_teacher_projection_gap_max
            ),
            "economic_teacher_gt_projection_gap_max": float(
                self.config.economic_teacher_gt_projection_gap_max
            ),
            "economic_teacher_bes_price_opportunity_min": float(
                self.config.economic_teacher_bes_price_opportunity_min
            ),
            "economic_teacher_bes_anchor_preserve_scale": float(
                self.config.economic_teacher_bes_anchor_preserve_scale
            ),
            "economic_teacher_warm_start_weight": float(
                self.config.economic_teacher_warm_start_weight
            ),
            "economic_teacher_prefill_replay_boost": int(
                self.config.economic_teacher_prefill_replay_boost
            ),
            "economic_teacher_gt_action_weight": float(
                self.config.economic_teacher_gt_action_weight
            ),
            "economic_teacher_bes_action_weight": float(
                self.config.economic_teacher_bes_action_weight
            ),
            "economic_teacher_tes_action_weight": float(
                self.config.economic_teacher_tes_action_weight
            ),
            "economic_teacher_full_year_warm_start_samples": int(
                self.config.economic_teacher_full_year_warm_start_samples
            ),
            "economic_teacher_full_year_warm_start_epochs": int(
                self.config.economic_teacher_full_year_warm_start_epochs
            ),
            "economic_bes_distill_coef": float(self.config.economic_bes_distill_coef),
            "economic_bes_prior_u": float(self.config.economic_bes_prior_u),
            "economic_bes_charge_u_scale": float(self.config.economic_bes_charge_u_scale),
            "economic_bes_discharge_u_scale": float(self.config.economic_bes_discharge_u_scale),
            "economic_bes_charge_weight": float(self.config.economic_bes_charge_weight),
            "economic_bes_discharge_weight": float(self.config.economic_bes_discharge_weight),
            "economic_bes_charge_pressure_bonus": float(
                self.config.economic_bes_charge_pressure_bonus
            ),
            "economic_bes_charge_soc_ceiling": float(self.config.economic_bes_charge_soc_ceiling),
            "economic_bes_discharge_soc_floor": float(self.config.economic_bes_discharge_soc_floor),
            "economic_bes_full_year_warm_start_samples": int(
                self.config.economic_bes_full_year_warm_start_samples
            ),
            "economic_bes_full_year_warm_start_epochs": int(
                self.config.economic_bes_full_year_warm_start_epochs
            ),
            "economic_bes_full_year_warm_start_u_weight": float(
                self.config.economic_bes_full_year_warm_start_u_weight
            ),
            "economic_bes_teacher_selection_priority_boost": float(
                self.config.economic_bes_teacher_selection_priority_boost
            ),
            "economic_bes_economic_source_priority_bonus": float(
                self.config.economic_bes_economic_source_priority_bonus
            ),
            "economic_bes_economic_source_min_share": float(
                self.config.economic_bes_economic_source_min_share
            ),
            "economic_bes_idle_economic_source_min_share": float(
                self.config.economic_bes_idle_economic_source_min_share
            ),
            "economic_bes_teacher_target_min_share": float(
                self.config.economic_bes_teacher_target_min_share
            ),
            "state_feasible_action_shaping_enabled": bool(self.config.state_feasible_action_shaping_enabled),
            "abs_cooling_blend_enabled": True,
            "abs_to_ech_transfer_ratio": float(
                max(_NORM_EPS, float(self.env_config.q_abs_cool_cap_mw))
                / max(_NORM_EPS, float(self.env_config.q_ech_cap_mw))
            ),
            "abs_gate_scale_k": float(self.env_config.abs_gate_scale_k),
            "abs_min_on_gate_th": float(self.config.abs_min_on_gate_th),
            "abs_min_on_u_margin": float(self.config.abs_min_on_u_margin),
            "abs_invalid_req_u_th": float(self.env_config.abs_invalid_req_u_th),
            "abs_invalid_req_gate_th": float(self.env_config.abs_invalid_req_gate_th),
            "abs_deadzone_gate_th": float(self.env_config.abs_deadzone_gate_th),
            "abs_deadzone_u_th": float(self.env_config.abs_deadzone_u_th),
            "dt_hours": float(self.env_config.dt_hours),
            "p_bes_cap_mw": float(self.env_config.p_bes_cap_mw),
            "e_bes_cap_mwh": float(self.env_config.e_bes_cap_mwh),
            "bes_soc_min": float(self.env_config.bes_soc_min),
            "bes_soc_max": float(self.env_config.bes_soc_max),
            "bes_eta_charge": float(self.env_config.bes_eta_charge),
            "bes_eta_discharge": float(self.env_config.bes_eta_discharge),
            "e_tes_cap_mwh": float(self.env_config.e_tes_cap_mwh),
            "q_tes_charge_cap_mw": float(self.env_config.q_tes_charge_cap_mw),
            "q_tes_discharge_cap_mw": float(self.env_config.q_tes_discharge_cap_mw),
            "p_gt_cap_mw": float(self.env_config.p_gt_cap_mw),
            "gt_min_output_mw": float(self.env_config.gt_min_output_mw),
            "bes_price_low_threshold": float(self.bes_price_low_threshold),
            "bes_price_high_threshold": float(self.bes_price_high_threshold),
            "expert_prefill_policy": str(self.config.expert_prefill_policy),
            "expert_prefill_checkpoint_path": str(self.config.expert_prefill_checkpoint_path),
            "expert_prefill_economic_policy": str(self.config.expert_prefill_economic_policy),
            "expert_prefill_economic_checkpoint_path": str(
                self.config.expert_prefill_economic_checkpoint_path
            ),
            "expert_prefill_steps": int(self.config.expert_prefill_steps),
            "expert_prefill_cooling_bias": float(self.config.expert_prefill_cooling_bias),
            "expert_prefill_abs_replay_boost": int(self.config.expert_prefill_abs_replay_boost),
            "expert_prefill_abs_exec_threshold": float(self.config.expert_prefill_abs_exec_threshold),
            "expert_prefill_abs_window_mining_candidates": int(self.config.expert_prefill_abs_window_mining_candidates),
            "dual_abs_margin_k": float(self.config.dual_abs_margin_k),
            "dual_qc_ratio_th": float(self.config.dual_qc_ratio_th),
            "dual_heat_backup_ratio_th": float(self.config.dual_heat_backup_ratio_th),
            "dual_safe_abs_u_th": float(self.config.dual_safe_abs_u_th),
            "actor_warm_start_epochs": int(self.config.actor_warm_start_epochs),
            "actor_warm_start_batch_size": int(self.config.actor_warm_start_batch_size),
            "actor_warm_start_lr": float(self.config.actor_warm_start_lr),
            "eval_window_pool_size": int(self.config.eval_window_pool_size),
            "eval_window_count": int(self.config.eval_window_count),
            "best_gate_enabled": bool(self.config.best_gate_enabled),
            "best_gate_electric_min": float(self.config.best_gate_electric_min),
            "best_gate_heat_min": float(self.config.best_gate_heat_min),
            "best_gate_cool_min": float(self.config.best_gate_cool_min),
            "validation_eval_count": int(self.validation_eval_count),
            "training_early_stopped": bool(self.stop_requested),
            "training_stop_reason": str(self.stop_reason),
            "checkpoint_interval_steps": int(checkpoint_every),
            "current_actor_lr": float(self.current_actor_lr),
            "current_critic_lr": float(self.current_critic_lr),
            "hidden_dims": list(self.config.hidden_dims),
            "observation_keys": list(self.observation_keys),
            "action_keys": list(self.action_keys),
            "observation_norm": dict(self.observation_norm_payload),
            "cost_targets": {
                key: float(value)
                for key, value in zip(_COST_KEYS, self.dual_targets)
            },
            "dual_lambdas": {
                name: float(value)
                for name, value in zip(_DUAL_NAMES, self.dual_lambdas)
            },
            "cost_critic_positive_output": True,
            "validation_protocol": dict(self.eval_protocol),
            "best_model_selection": {
                "mode": "reliability_shortfall_then_cost_then_reward_v1",
                "selected": dict(self.best_selection_snapshot) if self.best_selection_snapshot is not None else None,
                "reward_leader": dict(self.best_reward_snapshot) if self.best_reward_snapshot is not None else None,
            },
            "plateau_control": {
                "enabled": bool(self.config.plateau_control_enabled),
                "patience_evals": int(self.config.plateau_patience_evals),
                "lr_decay_factor": float(self.config.plateau_lr_decay_factor),
                "min_actor_lr": float(self.config.plateau_min_actor_lr),
                "min_critic_lr": float(self.config.plateau_min_critic_lr),
                "early_stop_patience_evals": int(self.config.plateau_early_stop_patience_evals),
                "current_actor_lr": float(self.current_actor_lr),
                "current_critic_lr": float(self.current_critic_lr),
                "lr_decay_count": int(self.lr_decay_count),
                "no_improve_evals": int(self.no_improve_evals),
                "fine_tune_applied": bool(self.fine_tune_applied),
                "stopped_early": bool(self.stop_requested),
                "stop_reason": str(self.stop_reason),
                "events": list(self.plateau_events),
            },
            "expert_prefill": dict(self.expert_prefill_summary),
            "economic_teacher_distill": dict(self.economic_teacher_distill_summary),
            "actor_init": dict(self.actor_init_summary),
            "actor_warm_start": dict(self.actor_warm_start_summary),
            "actor_teacher_full_year_warm_start": dict(
                self.actor_teacher_full_year_warm_start_summary
            ),
            "actor_bes_warm_start": dict(self.actor_bes_warm_start_summary),
            "replay_schema": {
                "obs": list(self.observation_keys),
                "action_raw": list(self.action_keys),
                "action_exec": list(self.action_keys),
            "teacher_action_exec": list(self.action_keys),
            "teacher_action_mask": list(self.action_keys),
            "teacher_available": ["teacher_available"],
                "reward": ["reward"],
                "cost": list(_COST_KEYS),
                "projection_gap": [
                    "projection_gap_l1",
                    "projection_gap_l2",
                    "projection_gap_max",
                ],
                "next_obs": list(self.observation_keys),
                "done": ["done"],
            },
            "final_metrics": {
                key: (
                    float(value)
                    if isinstance(value, (int, float, np.integer, np.floating))
                    else value
                )
                for key, value in last_row.items()
                if key != "episode_idx"
            },
        }
        (self.run_dir / "train" / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary


def train_pafc_td3(
    *,
    train_df: pd.DataFrame,
    train_statistics: dict[str, Any],
    env_config: EnvConfig,
    trainer_config: PAFCTD3TrainConfig,
    run_root: str | Path = "runs",
) -> dict[str, Any]:
    trainer = PAFCTD3Trainer(
        train_df=train_df,
        train_statistics=train_statistics,
        env_config=env_config,
        config=trainer_config,
        run_root=run_root,
    )
    return trainer.train()


def load_pafc_td3_predictor(
    *,
    checkpoint_path: str | Path,
    device: str = "auto",
    env_config: EnvConfig | None = None,
):
    torch, _, _, _ = _require_torch_modules()
    target_device = resolve_torch_device(device)
    payload = load_policy(checkpoint_path, map_location=target_device)
    metadata = dict(payload["metadata"])
    observation_keys = tuple(str(key) for key in metadata.get("observation_keys", ()))
    action_keys = tuple(str(key) for key in metadata.get("action_keys", ()))
    actor = build_pafc_actor_network(
        observation_dim=int(metadata["observation_dim"]),
        action_keys=action_keys,
        hidden_dims=tuple(int(dim) for dim in metadata.get("hidden_dims", (256, 256))),
    ).to(target_device)
    actor.load_state_dict(payload["state_dict"])
    actor.eval()
    obs_norm = dict(metadata.get("observation_norm", {}))
    offset = np.asarray(obs_norm.get("offset", []), dtype=np.float32)
    scale = np.asarray(obs_norm.get("scale", []), dtype=np.float32)
    scale = np.where(np.abs(scale) < _NORM_EPS, 1.0, scale)
    observation_index = {key: idx for idx, key in enumerate(observation_keys)}
    action_index = {key: idx for idx, key in enumerate(action_keys)}
    abs_cooling_blend_enabled = bool(metadata.get("abs_cooling_blend_enabled", False))
    abs_to_ech_transfer_ratio = float(metadata.get("abs_to_ech_transfer_ratio", 0.0))
    gate_scale_k = max(_NORM_EPS, float(metadata.get("abs_gate_scale_k", 2.0)))
    abs_min_on_gate_th = float(metadata.get("abs_min_on_gate_th", 0.0))
    abs_min_on_u_margin = float(metadata.get("abs_min_on_u_margin", 0.0))
    abs_invalid_req_u_th = float(metadata.get("abs_invalid_req_u_th", 0.0))
    abs_invalid_req_gate_th = float(metadata.get("abs_invalid_req_gate_th", 0.0))
    abs_deadzone_gate_th = float(
        metadata.get(
            "abs_deadzone_gate_th",
            float(env_config.abs_deadzone_gate_th) if env_config is not None else 0.0,
        )
    )
    abs_deadzone_u_th = float(metadata.get("abs_deadzone_u_th", 0.0))
    dt_hours = float(
        metadata.get("dt_hours", float(env_config.dt_hours) if env_config is not None else 0.25)
    )
    p_bes_cap_mw = float(
        metadata.get("p_bes_cap_mw", float(env_config.p_bes_cap_mw) if env_config is not None else 4.0)
    )
    e_bes_cap_mwh = float(
        metadata.get("e_bes_cap_mwh", float(env_config.e_bes_cap_mwh) if env_config is not None else 8.0)
    )
    bes_soc_min = float(
        metadata.get("bes_soc_min", float(env_config.bes_soc_min) if env_config is not None else 0.1)
    )
    bes_soc_max = float(
        metadata.get("bes_soc_max", float(env_config.bes_soc_max) if env_config is not None else 0.95)
    )
    bes_eta_charge = float(
        metadata.get("bes_eta_charge", float(env_config.bes_eta_charge) if env_config is not None else 0.95)
    )
    bes_eta_discharge = float(
        metadata.get(
            "bes_eta_discharge",
            float(env_config.bes_eta_discharge) if env_config is not None else 0.95,
        )
    )
    e_tes_cap_mwh = float(
        metadata.get("e_tes_cap_mwh", float(env_config.e_tes_cap_mwh) if env_config is not None else 20.0)
    )
    q_tes_charge_cap_mw = float(
        metadata.get(
            "q_tes_charge_cap_mw",
            float(env_config.q_tes_charge_cap_mw) if env_config is not None else 8.0,
        )
    )
    q_tes_discharge_cap_mw = float(
        metadata.get(
            "q_tes_discharge_cap_mw",
            float(env_config.q_tes_discharge_cap_mw) if env_config is not None else 8.0,
        )
    )
    p_gt_cap_mw = float(
        metadata.get("p_gt_cap_mw", float(env_config.p_gt_cap_mw) if env_config is not None else 12.0)
    )
    gt_min_output_mw = float(
        metadata.get(
            "gt_min_output_mw",
            float(env_config.gt_min_output_mw) if env_config is not None else 1.0,
        )
    )
    gt_off_deadband_ratio = float(metadata.get("gt_off_deadband_ratio", 0.0))

    def predictor(observation: Mapping[str, float] | np.ndarray | Sequence[float]):
        if isinstance(observation, Mapping):
            missing = [key for key in observation_keys if key not in observation]
            if missing:
                raise ValueError(f"predictor 输入缺少 observation 键: {missing}")
            obs_vector = build_feature_vector(
                observation=observation,
                feature_keys=observation_keys,
            ).astype(np.float32)
        else:
            obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs_vector.shape[0] != len(observation_keys):
            raise ValueError(
                f"observation 维度不匹配：期望 {len(observation_keys)}，实际 {obs_vector.shape[0]}"
            )
        normalized = ((obs_vector - offset) / scale).astype(np.float32)
        with torch.no_grad():
            tensor = torch.as_tensor(
                normalized.reshape(1, -1),
                dtype=torch.float32,
                device=target_device,
            )
            action = actor(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
        if bool(metadata.get("state_feasible_action_shaping_enabled", False)):
            if "u_bes" in action_index and "soc_bes" in observation_index:
                u_bes_index = int(action_index["u_bes"])
                soc = float(obs_vector[int(observation_index["soc_bes"])])
                max_discharge_u = float(
                    np.clip(
                        ((soc - bes_soc_min) * e_bes_cap_mwh * max(_NORM_EPS, bes_eta_discharge) / max(_NORM_EPS, dt_hours))
                        / max(_NORM_EPS, p_bes_cap_mw),
                        0.0,
                        1.0,
                    )
                )
                max_charge_u = float(
                    np.clip(
                        ((bes_soc_max - soc) * e_bes_cap_mwh / max(_NORM_EPS, bes_eta_charge * dt_hours))
                        / max(_NORM_EPS, p_bes_cap_mw),
                        0.0,
                        1.0,
                    )
                )
                action[u_bes_index] = float(np.clip(float(action[u_bes_index]), -max_charge_u, max_discharge_u))
            if "u_tes" in action_index and "e_tes_mwh" in observation_index:
                u_tes_index = int(action_index["u_tes"])
                e_tes = float(obs_vector[int(observation_index["e_tes_mwh"])])
                if "q_tes_discharge_feasible_mw" in observation_index:
                    discharge_feasible_mw = float(obs_vector[int(observation_index["q_tes_discharge_feasible_mw"])])
                else:
                    discharge_feasible_mw = float(np.clip(e_tes / max(_NORM_EPS, dt_hours), 0.0, q_tes_discharge_cap_mw))
                charge_headroom_mw = float(
                    np.clip((e_tes_cap_mwh - e_tes) / max(_NORM_EPS, dt_hours), 0.0, q_tes_charge_cap_mw)
                )
                u_tes_max = float(np.clip(discharge_feasible_mw / max(_NORM_EPS, q_tes_discharge_cap_mw), 0.0, 1.0))
                u_tes_min = -float(np.clip(charge_headroom_mw / max(_NORM_EPS, q_tes_charge_cap_mw), 0.0, 1.0))
                action[u_tes_index] = float(np.clip(float(action[u_tes_index]), u_tes_min, u_tes_max))
        if "u_gt" in action_index and "p_gt_prev_mw" in observation_index:
            u_gt_index = int(action_index["u_gt"])
            p_gt_prev = float(obs_vector[int(observation_index["p_gt_prev_mw"])])
            if (
                "gt_ramp_headroom_up_mw" in observation_index
                and "gt_ramp_headroom_down_mw" in observation_index
            ):
                ramp_up = float(obs_vector[int(observation_index["gt_ramp_headroom_up_mw"])])
                ramp_down = float(obs_vector[int(observation_index["gt_ramp_headroom_down_mw"])])
                p_gt_low = float(np.clip(p_gt_prev - ramp_down, 0.0, p_gt_cap_mw))
                p_gt_high = float(np.clip(p_gt_prev + ramp_up, 0.0, p_gt_cap_mw))
            else:
                p_gt_low = 0.0
                p_gt_high = p_gt_cap_mw
            p_gt_target = ((float(action[u_gt_index]) + 1.0) * 0.5) * max(_NORM_EPS, p_gt_cap_mw)
            p_gt_target = _canonicalize_gt_target_np(
                p_gt_target_mw=p_gt_target,
                p_gt_low_mw=p_gt_low,
                p_gt_high_mw=p_gt_high,
                gt_min_output_mw=gt_min_output_mw,
                gt_off_deadband_ratio=gt_off_deadband_ratio,
            )
            action[u_gt_index] = float(
                np.clip(2.0 * (p_gt_target / max(_NORM_EPS, p_gt_cap_mw)) - 1.0, -1.0, 1.0)
            )
        if (
            abs_cooling_blend_enabled
            and "abs_drive_margin_k" in observation_index
            and "u_abs" in action_index
            and "u_ech" in action_index
        ):
            margin = float(obs_vector[int(observation_index["abs_drive_margin_k"])])
            abs_gate = float(1.0 / (1.0 + np.exp(-margin / gate_scale_k)))
            u_abs_index = int(action_index["u_abs"])
            u_ech_index = int(action_index["u_ech"])
            u_abs = float(action[u_abs_index])
            cooling_transfer_ratio = float(abs_gate * abs_to_ech_transfer_ratio)
            u_abs_safe = float(u_abs)
            if abs_deadzone_gate_th > 0.0 and abs_gate < abs_deadzone_gate_th:
                u_abs_safe = 0.0
            elif abs_invalid_req_gate_th > abs_deadzone_gate_th and abs_gate < abs_invalid_req_gate_th:
                u_abs_safe = float(min(u_abs_safe, abs_invalid_req_u_th))
            if (u_abs_safe * abs_gate) < abs_deadzone_u_th and abs_gate < abs_min_on_gate_th:
                u_abs_safe = 0.0
            suppressed_abs = float(np.clip(u_abs - u_abs_safe, 0.0, 1.0))
            u_ech_safe = float(
                np.clip(
                    float(action[u_ech_index]) + suppressed_abs * cooling_transfer_ratio,
                    0.0,
                    1.0,
                )
            )
            abs_increase = 0.0
            abs_effective_min_u = float(max(0.0, abs_deadzone_u_th + abs_min_on_u_margin))
            if abs_min_on_gate_th > 0.0 and abs_effective_min_u > 0.0 and abs_gate >= abs_min_on_gate_th:
                target_u_abs = float(np.clip(abs_effective_min_u / max(_NORM_EPS, abs_gate), 0.0, 1.0))
                required_abs_increase = float(np.clip(target_u_abs - u_abs_safe, 0.0, 1.0))
                if cooling_transfer_ratio > _NORM_EPS:
                    max_abs_increase_from_ech = float(
                        np.clip(u_ech_safe / cooling_transfer_ratio, 0.0, 1.0)
                    )
                    abs_increase = float(min(required_abs_increase, max_abs_increase_from_ech))
            action[u_abs_index] = float(np.clip(u_abs_safe + abs_increase, 0.0, 1.0))
            action[u_ech_index] = float(
                np.clip(u_ech_safe - abs_increase * cooling_transfer_ratio, 0.0, 1.0)
            )
        return _action_vector_to_dict(action, action_keys=action_keys)

    return predictor, metadata


def evaluate_pafc_td3(
    eval_df: pd.DataFrame,
    *,
    run_dir: str | Path,
    checkpoint_path: str | Path,
    seed: int = 42,
    device: str = "auto",
    config: EnvConfig,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    year = _extract_year(eval_df)
    if year != EVAL_YEAR:
        raise ValueError(f"评估必须使用 {EVAL_YEAR}，当前年份 {year}")
    if config is None:
        raise ValueError("config 不能为空：当前采用 Option-C 全量 yaml 配置模式，请先构建 EnvConfig 并传入。")

    output_run_dir = Path(run_dir)
    (output_run_dir / "eval").mkdir(parents=True, exist_ok=True)

    predictor, metadata = load_pafc_td3_predictor(
        checkpoint_path=checkpoint_path,
        device=device,
        env_config=config,
    )
    env = CCHPPhysicalEnv(exogenous_df=eval_df, config=config, seed=seed)
    observation, _ = env.reset(seed=seed, episode_df=eval_df)
    terminated = False
    total_reward = 0.0
    step_rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}

    while not terminated:
        action = predictor(observation)
        observation, reward, terminated, _, info = env.step(action)
        total_reward += float(reward)
        final_info = dict(info)
        log_row = {
            key: value
            for key, value in info.items()
            if key not in {"violation_flags", "diagnostic_flags", "state_diagnostic_flags"}
        }
        log_row["violation_flags_json"] = json.dumps(
            info.get("violation_flags", {}),
            ensure_ascii=False,
        )
        log_row["diagnostic_flags_json"] = json.dumps(
            info.get("diagnostic_flags", {}),
            ensure_ascii=False,
        )
        log_row["state_diagnostic_flags_json"] = json.dumps(
            info.get("state_diagnostic_flags", {}),
            ensure_ascii=False,
        )
        step_rows.append(log_row)

    summary = dict(final_info.get("episode_summary", env.kpi.summary()))
    summary["mode"] = "eval"
    summary["year"] = EVAL_YEAR
    summary["policy"] = "pafc_td3"
    summary["seed"] = int(seed)
    summary["device"] = str(device)
    summary["total_reward"] = float(total_reward)
    summary["eval_wall_time_s"] = float(time.perf_counter() - start_time)
    summary["eval_steps_per_second"] = float(
        len(step_rows) / max(1e-9, float(summary["eval_wall_time_s"]))
    )
    summary["checkpoint_path"] = str(Path(checkpoint_path).resolve()).replace("\\", "/")
    summary["projection_surrogate_checkpoint_path"] = str(
        metadata.get("projection_surrogate_checkpoint_path", "")
    )
    summary["dual_lambdas"] = dict(metadata.get("dual_lambdas", {}) or {})
    summary["policy_details"] = {
        "observation_keys": list(metadata.get("observation_keys", [])),
        "action_keys": list(metadata.get("action_keys", [])),
        "hidden_dims": list(metadata.get("hidden_dims", [])),
        "episode_days_train": metadata.get("episode_days"),
        "train_total_env_steps": metadata.get("total_env_steps"),
        "state_feasible_action_shaping_enabled": bool(
            metadata.get("state_feasible_action_shaping_enabled", False)
        ),
        "abs_cooling_blend_enabled": bool(metadata.get("abs_cooling_blend_enabled", False)),
        "abs_to_ech_transfer_ratio": float(metadata.get("abs_to_ech_transfer_ratio", 0.0)),
        "abs_min_on_gate_th": float(metadata.get("abs_min_on_gate_th", 0.0)),
        "abs_min_on_u_margin": float(metadata.get("abs_min_on_u_margin", 0.0)),
    }

    (output_run_dir / "eval" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    step_df = pd.DataFrame(step_rows)
    step_df.to_csv(output_run_dir / "eval" / "step_log.csv", index=False)
    write_paper_eval_artifacts(
        output_run_dir / "eval",
        summary=summary,
        step_log=step_df,
        dt_h=float(config.dt_hours),
    )
    return summary
