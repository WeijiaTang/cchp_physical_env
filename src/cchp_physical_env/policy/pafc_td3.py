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
_GT_PRIOR_MIN_OPPORTUNITY = 0.05
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


def _blend_surrogate_action_proxy(
    *,
    action_exec_hat,
    fallback_action,
    trust_weight,
):
    return trust_weight * action_exec_hat + (1.0 - trust_weight) * fallback_action


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


def _normalize_action_key_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
    elif isinstance(value, (list, tuple, set)):
        tokens = [str(token).strip() for token in value if str(token).strip()]
    else:
        tokens = [str(value).strip()]
    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = str(token).strip().lower().replace("-", "_")
        if not key or key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    return tuple(normalized)


def _observation_vector_to_dict(
    observation_vector: np.ndarray | Sequence[float],
    *,
    observation_keys: Sequence[str],
) -> dict[str, float]:
    vector = np.asarray(observation_vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != len(observation_keys):
        raise ValueError(
            f"观测维度不匹配：期望 {len(observation_keys)}，实际 {vector.shape[0]}"
        )
    return {
        str(key): float(vector[index])
        for index, key in enumerate(observation_keys)
    }


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


def _gt_price_prior_target_np(
    *,
    price_e: float,
    price_gas: float,
    p_dem_mw: float,
    pv_mw: float,
    wt_mw: float,
    t_amb_k: float,
    heat_backup_min_needed_mw: float,
    abs_drive_margin_k: float,
    qc_dem_mw: float,
    p_gt_prev_mw: float,
    env_config,
    gt_off_deadband_ratio: float,
    price_low_threshold: float,
    price_high_threshold: float,
    u_bes: float = 0.0,
    u_ech: float = 0.0,
    carbon_tax: float = 0.0,
) -> dict[str, float | str]:
    price_pressure = _bes_price_pressure_np(
        price_e=price_e,
        price_low_threshold=price_low_threshold,
        price_high_threshold=price_high_threshold,
    )
    p_gt_cap_mw = max(_NORM_EPS, float(env_config.p_gt_cap_mw))
    gt_min_output_mw = max(0.0, float(env_config.gt_min_output_mw))
    gt_min_ratio = float(np.clip(gt_min_output_mw / p_gt_cap_mw, 0.0, 1.0))
    q_boiler_cap_mw = max(_NORM_EPS, float(env_config.q_boiler_cap_mw))
    q_abs_cool_cap_mw = max(_NORM_EPS, float(env_config.q_abs_cool_cap_mw))
    q_ech_cap_mw = max(_NORM_EPS, float(env_config.q_ech_cap_mw))
    gate_scale_k = max(_NORM_EPS, float(env_config.abs_gate_scale_k))
    u_bes = float(np.clip(u_bes, -1.0, 1.0))
    u_ech = float(np.clip(u_ech, 0.0, 1.0))
    p_bes_charge_proxy_mw = (
        max(0.0, -u_bes)
        * float(env_config.p_bes_cap_mw)
        / max(_NORM_EPS, float(env_config.bes_eta_charge))
    )
    p_bes_discharge_proxy_mw = (
        max(0.0, u_bes)
        * float(env_config.p_bes_cap_mw)
        * float(env_config.bes_eta_discharge)
    )
    q_ech_proxy_mw = u_ech * q_ech_cap_mw
    ech_cop = np.clip(
        float(env_config.cop_nominal) - 0.03 * (float(t_amb_k) - 298.15),
        float(env_config.cop_nominal) * float(env_config.ech_cop_partload_min_fraction),
        float(env_config.cop_nominal),
    )
    p_ech_proxy_mw = q_ech_proxy_mw / max(_NORM_EPS, float(ech_cop))
    net_grid_need_proxy_mw = max(
        0.0,
        float(p_dem_mw)
        + p_ech_proxy_mw
        + p_bes_charge_proxy_mw
        - float(pv_mw)
        - float(wt_mw)
        - p_bes_discharge_proxy_mw,
    )
    net_grid_need_ratio = float(np.clip(net_grid_need_proxy_mw / p_gt_cap_mw, 0.0, 1.0))
    heat_support_need = float(
        np.clip(float(heat_backup_min_needed_mw) / q_boiler_cap_mw, 0.0, 1.0)
    )
    qc_need_ratio = float(
        np.clip(float(qc_dem_mw) / max(q_abs_cool_cap_mw, q_ech_cap_mw), 0.0, 1.0)
    )
    abs_ready = float(1.0 / (1.0 + np.exp(-float(abs_drive_margin_k) / gate_scale_k)))
    cool_support_need = float(np.clip(qc_need_ratio * max(abs_ready, 0.35), 0.0, 1.0))
    prev_gt_ratio = float(np.clip(float(p_gt_prev_mw) / p_gt_cap_mw, 0.0, 1.0))
    eta_ref_ratio = float(
        np.clip(max(gt_min_ratio, 0.5 * (prev_gt_ratio + net_grid_need_ratio)), 0.0, 1.0)
    )
    eta_ref = float(env_config.gt_eta_min) + (
        float(env_config.gt_eta_max) - float(env_config.gt_eta_min)
    ) * eta_ref_ratio
    gt_dispatch_basis_mw = max(gt_min_output_mw, 0.25 * p_gt_cap_mw, _NORM_EPS)
    startup_basis_mwh = max(
        _NORM_EPS,
        float(getattr(env_config, "dt_hours", 0.25)) * gt_dispatch_basis_mw,
    )
    startup_off_factor = float(
        np.clip(1.0 - float(p_gt_prev_mw) / gt_dispatch_basis_mw, 0.0, 1.0)
    )
    gt_marginal_cost = (
        float(price_gas) / max(_NORM_EPS, float(eta_ref))
        + float(getattr(env_config, "gt_om_var_cost_per_mwh", 0.0))
        + float(max(0.0, float(carbon_tax)))
        * float(getattr(env_config, "gt_emission_ton_per_mwh_th", 0.0))
        / max(_NORM_EPS, float(eta_ref))
        + startup_off_factor
        * (
            float(getattr(env_config, "gt_start_cost", 0.0))
            + float(getattr(env_config, "gt_cycle_cost", 0.0))
        )
        / startup_basis_mwh
    )
    price_advantage = float(
        np.clip(
            (float(price_e) - gt_marginal_cost)
            / max(_NORM_EPS, max(float(price_e), gt_marginal_cost)),
            -1.0,
            1.0,
        )
    )
    market_commit = max(float(price_pressure["discharge_pressure"]), max(0.0, price_advantage))
    market_off = max(float(price_pressure["charge_pressure"]), max(0.0, -price_advantage))
    net_grid_absorb_ratio = float(
        np.clip(
            (net_grid_need_ratio - 0.75 * gt_min_ratio)
            / max(_NORM_EPS, 1.0 - 0.75 * gt_min_ratio),
            0.0,
            1.0,
        )
    )
    cogen_support_need = max(heat_support_need, 0.55 * cool_support_need)
    support_commit_floor = float(
        np.clip(
            (0.55 * heat_support_need + 0.25 * cool_support_need)
            * (0.20 + 0.80 * max(0.0, price_advantage)),
            0.0,
            1.0,
        )
    )
    commit_score = float(
        np.clip(
            max(
                market_commit * max(net_grid_absorb_ratio, 0.40 * cogen_support_need),
                support_commit_floor,
            ),
            0.0,
            1.0,
        )
    )
    low_load_relief = float(
        np.clip(
            (gt_min_ratio - net_grid_need_ratio) / max(_NORM_EPS, gt_min_ratio),
            0.0,
            1.0,
        )
    )
    off_score = float(
        np.clip(
            (market_off + 0.35 * low_load_relief)
            * max(0.0, 1.0 - max(net_grid_absorb_ratio, 0.75 * cogen_support_need))
            * max(0.0, 1.0 - 0.15 * prev_gt_ratio),
            0.0,
            1.0,
        )
    )
    opportunity = float(max(commit_score, off_score))
    if opportunity <= _GT_PRIOR_MIN_OPPORTUNITY:
        return {
            "target_u_gt": 0.0,
            "opportunity": 0.0,
            "commit_score": float(commit_score),
            "off_score": float(off_score),
            "net_grid_need_ratio": float(net_grid_need_ratio),
            "net_grid_absorb_ratio": float(net_grid_absorb_ratio),
            "heat_support_need": float(heat_support_need),
            "cool_support_need": float(cool_support_need),
            "cogen_support_need": float(cogen_support_need),
            "price_advantage": float(max(0.0, price_advantage)),
            "target_load_ratio": 0.0,
            "mode_weight": 1.0,
            "mode": "idle",
        }
    commit_margin = float(np.clip(commit_score - off_score, 0.0, 1.0))
    if off_score >= max(commit_score, _GT_PRIOR_MIN_OPPORTUNITY) or commit_margin <= max(
        _GT_PRIOR_MIN_OPPORTUNITY,
        0.5 * gt_min_ratio,
    ):
        return {
            "target_u_gt": -1.0,
            "opportunity": float(off_score),
            "commit_score": float(commit_score),
            "off_score": float(off_score),
            "net_grid_need_ratio": float(net_grid_need_ratio),
            "net_grid_absorb_ratio": float(net_grid_absorb_ratio),
            "heat_support_need": float(heat_support_need),
            "cool_support_need": float(cool_support_need),
            "cogen_support_need": float(cogen_support_need),
            "price_advantage": float(max(0.0, price_advantage)),
            "target_load_ratio": 0.0,
            "mode_weight": float(1.0 + 0.75 * off_score),
            "mode": "off",
        }
    on_signal = max(net_grid_absorb_ratio, cogen_support_need, market_commit)
    dispatch_floor_ratio = float(
        np.clip(
            gt_min_ratio + 0.12 * on_signal,
            gt_min_ratio,
            max(gt_min_ratio, 0.35),
        )
    )
    dispatch_cap_ratio = float(
        np.clip(
            0.25 + 0.75 * on_signal,
            dispatch_floor_ratio,
            1.0,
        )
    )
    dispatch_strength = float(np.clip(commit_score * on_signal, 0.0, 1.0))
    target_load_ratio = float(
        np.clip(
            dispatch_floor_ratio
            + (dispatch_cap_ratio - dispatch_floor_ratio) * dispatch_strength,
            0.0,
            1.0,
        )
    )
    p_gt_target_mw = _canonicalize_gt_target_np(
        p_gt_target_mw=target_load_ratio * p_gt_cap_mw,
        p_gt_low_mw=0.0,
        p_gt_high_mw=float(env_config.p_gt_cap_mw),
        gt_min_output_mw=gt_min_output_mw,
        gt_off_deadband_ratio=gt_off_deadband_ratio,
    )
    target_u_gt = float(np.clip(2.0 * (p_gt_target_mw / p_gt_cap_mw) - 1.0, -1.0, 1.0))
    return {
        "target_u_gt": float(target_u_gt),
        "opportunity": float(commit_score),
        "commit_score": float(commit_score),
        "off_score": float(off_score),
        "net_grid_need_ratio": float(net_grid_need_ratio),
        "net_grid_absorb_ratio": float(net_grid_absorb_ratio),
        "heat_support_need": float(heat_support_need),
        "cool_support_need": float(cool_support_need),
        "cogen_support_need": float(cogen_support_need),
        "price_advantage": float(max(0.0, price_advantage)),
        "target_load_ratio": float(target_load_ratio),
        "mode_weight": float(1.0 + 0.25 * dispatch_strength + 0.35 * commit_margin),
        "mode": "on",
    }


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


def _bes_warm_start_cooling_guard_active_np(
    *,
    abs_drive_margin_k: float,
    qc_dem_mw: float,
    q_total_cooling_cap_mw: float,
    abs_margin_guard_k: float,
    qc_ratio_guard: float,
) -> bool:
    qc_ratio = max(0.0, float(qc_dem_mw)) / max(_NORM_EPS, float(q_total_cooling_cap_mw))
    return bool(
        float(abs_drive_margin_k) <= float(abs_margin_guard_k)
        and qc_ratio >= float(qc_ratio_guard)
    )


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


def _allocate_gt_warm_start_mode_counts(
    *,
    requested_total: int,
    on_available: int,
    off_available: int,
    idle_available: int,
) -> dict[str, int]:
    requested_total = max(0, int(requested_total))
    available = {
        "on": max(0, int(on_available)),
        "off": max(0, int(off_available)),
        "idle": max(0, int(idle_available)),
    }
    if requested_total <= 0:
        return {mode: 0 for mode in available}
    planned = {
        "off": int(round(float(requested_total) * 0.60)),
        "on": int(round(float(requested_total) * 0.35)),
    }
    planned["idle"] = max(0, requested_total - planned["off"] - planned["on"])
    selected = {
        mode: min(int(available[mode]), int(planned.get(mode, 0)))
        for mode in ("on", "off", "idle")
    }
    remaining = max(0, requested_total - int(sum(selected.values())))
    while remaining > 0:
        progressed = False
        for mode in ("off", "on", "idle"):
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
        for mode in ("on", "off", "idle")
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


def _select_temporal_priority_indices_by_season(
    *,
    indices: Sequence[int],
    priorities: Sequence[float],
    season_by_index: Mapping[int, str],
    target_count: int,
) -> list[int]:
    target_total = max(0, int(target_count))
    if target_total <= 0:
        return []
    unique_indices = sorted({int(idx) for idx in indices if int(idx) >= 0})
    if len(unique_indices) <= target_total:
        return unique_indices

    season_groups: dict[str, list[int]] = {
        season: []
        for season in ("winter", "spring", "summer", "autumn")
    }
    for idx in unique_indices:
        season = str(season_by_index.get(int(idx), "summer")).strip().lower()
        if season not in season_groups:
            season = "summer"
        season_groups[season].append(int(idx))
    counts = _allocate_eval_window_season_counts(
        requested_total=target_total,
        available_by_season={season: len(values) for season, values in season_groups.items()},
    )
    selected: list[int] = []
    selected_set: set[int] = set()
    for season in ("summer", "winter", "spring", "autumn"):
        season_selected = _select_temporal_priority_indices(
            indices=season_groups[season],
            priorities=priorities,
            target_count=int(counts.get(season, 0)),
        )
        for idx in season_selected:
            idx_int = int(idx)
            if idx_int in selected_set:
                continue
            selected.append(idx_int)
            selected_set.add(idx_int)
    if len(selected) < target_total:
        remaining = [
            int(idx)
            for idx in unique_indices
            if int(idx) not in selected_set
        ]
        top_up = _select_temporal_priority_indices(
            indices=remaining,
            priorities=priorities,
            target_count=int(target_total - len(selected)),
        )
        for idx in top_up:
            idx_int = int(idx)
            if idx_int in selected_set:
                continue
            selected.append(idx_int)
            selected_set.add(idx_int)
    return sorted(selected)


def _allocate_economic_teacher_action_counts(
    *,
    requested_total: int,
    gt_available: int,
    bes_available: int,
    tes_available: int,
) -> dict[str, int]:
    requested = max(0, int(requested_total))
    available = {
        "gt": max(0, int(gt_available)),
        "bes": max(0, int(bes_available)),
        "tes": max(0, int(tes_available)),
    }
    selected = {key: 0 for key in available}
    if requested <= 0:
        return selected

    if available["bes"] > 0:
        bes_floor = min(
            available["bes"],
            max(
                1,
                int(round(float(requested) * 0.25)),
            ),
        )
        selected["bes"] = int(bes_floor)
    if available["tes"] > 0:
        tes_floor = min(
            available["tes"],
            max(
                1,
                int(round(float(requested) * 0.10)),
            ),
        )
        selected["tes"] = int(tes_floor)
    selected["gt"] = min(
        available["gt"],
        max(0, requested - selected["bes"] - selected["tes"]),
    )

    total_selected = int(sum(selected.values()))
    if total_selected > requested:
        overflow = int(total_selected - requested)
        for key in ("gt", "tes", "bes"):
            if overflow <= 0:
                break
            reducible = min(int(selected[key]), int(overflow))
            selected[key] -= reducible
            overflow -= reducible

    remaining = max(0, requested - int(sum(selected.values())))
    while remaining > 0:
        progressed = False
        for key in ("gt", "bes", "tes"):
            if int(selected[key]) >= int(available[key]):
                continue
            selected[key] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
    return {
        key: int(value)
        for key, value in selected.items()
    }


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


def _gt_teacher_direction_consistent_np(
    *,
    teacher_u_gt: float,
    prior_mode: str,
) -> bool:
    teacher_gt_on = float(teacher_u_gt) > -0.95
    normalized_mode = str(prior_mode).strip().lower()
    if teacher_gt_on:
        return normalized_mode == "on"
    return normalized_mode != "on"


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
            if (
                abs_margin_k <= float(gt_abs_margin_guard_k)
                and qc_ratio >= float(gt_qc_ratio_guard)
                and economic_value < -_NORM_EPS
            ):
                continue
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


def _economic_teacher_safe_preserve_weights_np(
    *,
    action_keys: Sequence[str],
) -> np.ndarray:
    """Preserve cooling and thermal control dimensions during GT/BES teacher fitting."""

    weights = np.ones((len(action_keys),), dtype=np.float32)
    for index, key in enumerate(action_keys):
        normalized_key = str(key).strip().lower()
        if normalized_key in {"u_abs", "u_ech"}:
            weights[index] = 4.0
        elif normalized_key == "u_boiler":
            weights[index] = 3.0
        elif normalized_key == "u_tes":
            weights[index] = 2.0
    return weights


def _build_surrogate_audit_report(
    *,
    obs_batch: np.ndarray,
    predicted_exec_batch: np.ndarray,
    actual_exec_batch: np.ndarray,
    observation_index: Mapping[str, int],
    action_keys: Sequence[str],
    low_abs_margin_threshold: float,
    high_cooling_ratio_threshold: float,
    q_total_cooling_cap_mw: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    obs_np = np.asarray(obs_batch, dtype=np.float32)
    predicted_np = np.asarray(predicted_exec_batch, dtype=np.float32)
    actual_np = np.asarray(actual_exec_batch, dtype=np.float32)
    if (
        obs_np.ndim != 2
        or predicted_np.ndim != 2
        or actual_np.ndim != 2
        or len(obs_np) <= 0
        or predicted_np.shape != actual_np.shape
        or predicted_np.shape[0] != obs_np.shape[0]
    ):
        return {}, pd.DataFrame()

    action_key_list = tuple(str(key) for key in action_keys)
    signed_error = predicted_np - actual_np
    abs_error = np.abs(signed_error)
    step_gap_l1 = abs_error.sum(axis=1)
    step_gap_l2 = np.sqrt(np.square(signed_error).sum(axis=1))
    step_gap_max = abs_error.max(axis=1)

    total_count = int(obs_np.shape[0])
    segment_masks: dict[str, np.ndarray] = {
        "overall": np.ones((total_count,), dtype=bool),
    }
    threshold_summary: dict[str, float] = {}
    low_margin_mask: np.ndarray | None = None
    high_cooling_mask: np.ndarray | None = None
    if "abs_drive_margin_k" in observation_index:
        low_margin_mask = (
            obs_np[:, int(observation_index["abs_drive_margin_k"])]
            <= float(low_abs_margin_threshold)
        )
        segment_masks["low_abs_margin"] = low_margin_mask
        threshold_summary["low_abs_margin_k"] = float(low_abs_margin_threshold)
    if "qc_dem_mw" in observation_index and q_total_cooling_cap_mw > _NORM_EPS:
        qc_ratio = (
            obs_np[:, int(observation_index["qc_dem_mw"])]
            / max(_NORM_EPS, float(q_total_cooling_cap_mw))
        )
        high_cooling_mask = qc_ratio >= float(high_cooling_ratio_threshold)
        segment_masks["high_cooling_ratio"] = high_cooling_mask
        threshold_summary["high_cooling_ratio"] = float(high_cooling_ratio_threshold)
    if low_margin_mask is not None and high_cooling_mask is not None:
        segment_masks["low_abs_margin_high_cooling"] = low_margin_mask & high_cooling_mask

    segment_rows: list[dict[str, Any]] = []
    segment_summary: dict[str, Any] = {}
    for segment_name, mask in segment_masks.items():
        count = int(mask.sum())
        if count <= 0:
            continue
        row: dict[str, Any] = {
            "segment": str(segment_name),
            "sample_count": count,
            "sample_rate": float(count / max(1, total_count)),
            "surrogate_gap_l1__mean": float(step_gap_l1[mask].mean()),
            "surrogate_gap_l2__mean": float(step_gap_l2[mask].mean()),
            "surrogate_gap_max__mean": float(step_gap_max[mask].mean()),
        }
        mae_by_action: dict[str, float] = {}
        rmse_by_action: dict[str, float] = {}
        bias_by_action: dict[str, float] = {}
        for action_index, action_key in enumerate(action_key_list):
            mae = float(abs_error[mask, action_index].mean())
            rmse = float(np.sqrt(np.square(signed_error[mask, action_index]).mean()))
            bias = float(signed_error[mask, action_index].mean())
            row[f"{action_key}__mae"] = mae
            row[f"{action_key}__rmse"] = rmse
            row[f"{action_key}__bias"] = bias
            mae_by_action[str(action_key)] = mae
            rmse_by_action[str(action_key)] = rmse
            bias_by_action[str(action_key)] = bias
        segment_rows.append(row)
        worst_action = max(mae_by_action.items(), key=lambda item: float(item[1]))[0]
        segment_summary[str(segment_name)] = {
            "sample_count": count,
            "sample_rate": float(count / max(1, total_count)),
            "surrogate_gap_l1_mean": float(row["surrogate_gap_l1__mean"]),
            "surrogate_gap_l2_mean": float(row["surrogate_gap_l2__mean"]),
            "surrogate_gap_max_mean": float(row["surrogate_gap_max__mean"]),
            "mae_by_action": mae_by_action,
            "rmse_by_action": rmse_by_action,
            "bias_by_action": bias_by_action,
            "worst_action_by_mae": str(worst_action),
        }

    overall_summary = segment_summary.get("overall", {})
    focused_summary = (
        segment_summary.get("low_abs_margin_high_cooling")
        or segment_summary.get("high_cooling_ratio")
        or segment_summary.get("low_abs_margin")
        or {}
    )
    return (
        {
            "sample_count": total_count,
            "thresholds": threshold_summary,
            "segments": segment_summary,
            "overall_worst_action_by_mae": str(
                overall_summary.get("worst_action_by_mae", "")
            ),
            "focused_worst_action_by_mae": str(
                focused_summary.get("worst_action_by_mae", "")
            ),
            "overall_mae_by_action": dict(overall_summary.get("mae_by_action", {})),
            "focused_mae_by_action": dict(focused_summary.get("mae_by_action", {})),
        },
        pd.DataFrame(segment_rows),
    )


def _build_surrogate_actor_trust_scale_np(
    *,
    action_keys: Sequence[str],
    overall_mae_by_action: Mapping[str, float] | None,
    focused_mae_by_action: Mapping[str, float] | None,
    trust_coef: float,
    trust_min_scale: float,
    focused_mix: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    action_key_list = tuple(str(key) for key in action_keys)
    ones = np.ones((1, len(action_key_list)), dtype=np.float32)
    coef = max(0.0, float(trust_coef))
    min_scale = float(np.clip(float(trust_min_scale), 0.0, 1.0))
    focus_weight = float(np.clip(float(focused_mix), 0.0, 1.0))
    if coef <= 0.0 or len(action_key_list) <= 0:
        return (
            ones,
            {
                "enabled": False,
                "status": "disabled",
                "coef": float(coef),
                "min_scale": float(min_scale),
                "focused_mix": float(focus_weight),
                "mae_by_action": {key: 0.0 for key in action_key_list},
                "trust_by_action": {key: 1.0 for key in action_key_list},
            },
        )

    overall = dict(overall_mae_by_action or {})
    focused = dict(focused_mae_by_action or {})
    overall_vector = np.asarray(
        [max(0.0, float(overall.get(key, 0.0))) for key in action_key_list],
        dtype=np.float32,
    )
    if focused:
        focused_vector = np.asarray(
            [max(0.0, float(focused.get(key, overall.get(key, 0.0)))) for key in action_key_list],
            dtype=np.float32,
        )
        blended_mae = (1.0 - focus_weight) * overall_vector + focus_weight * focused_vector
    else:
        blended_mae = overall_vector
    raw_scale = 1.0 - coef * blended_mae
    clipped_scale = np.clip(raw_scale, min_scale, 1.0).astype(np.float32, copy=False)
    if not np.isfinite(clipped_scale).all():
        clipped_scale = np.ones((len(action_key_list),), dtype=np.float32)
    scale_matrix = clipped_scale.reshape(1, -1)
    weakest_action = ""
    if len(action_key_list) > 0:
        weakest_action = str(action_key_list[int(np.argmin(clipped_scale))])
    return (
        scale_matrix,
        {
            "enabled": True,
            "status": "applied",
            "coef": float(coef),
            "min_scale": float(min_scale),
            "focused_mix": float(focus_weight),
            "mae_by_action": {
                key: float(value) for key, value in zip(action_key_list, blended_mae.tolist())
            },
            "trust_by_action": {
                key: float(value) for key, value in zip(action_key_list, clipped_scale.tolist())
            },
            "weakest_action_by_trust": str(weakest_action),
        },
    )


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
    episode_days: int = 14
    total_env_steps: int = 262_144
    warmup_steps: int = 4_096
    actor_warmup_steps: int = 0
    replay_capacity: int = 100_000
    batch_size: int = 256
    updates_per_step: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    dual_lr: float = 5e-3
    dual_warmup_steps: int = 8_192
    actor_delay: int = 2
    exploration_noise_std: float = 0.06
    target_policy_noise_std: float = 0.06
    target_noise_clip: float = 0.12
    gap_penalty_coef: float = 0.2
    exec_action_anchor_coef: float = 1.5
    exec_action_anchor_safe_floor: float = 0.05
    gt_off_deadband_ratio: float = 0.0
    abs_ready_focus_coef: float = 0.25
    invalid_abs_penalty_coef: float = 0.25
    economic_boiler_proxy_coef: float = 0.10
    economic_abs_tradeoff_coef: float = 0.05
    economic_gt_grid_proxy_coef: float = 0.50
    economic_gt_distill_coef: float = 0.10
    economic_teacher_distill_coef: float = 0.25
    economic_teacher_safe_preserve_coef: float = 1.0
    economic_teacher_safe_preserve_low_margin_boost: float = 0.75
    economic_teacher_safe_preserve_high_cooling_boost: float = 1.0
    economic_teacher_safe_preserve_joint_boost: float = 1.0
    economic_teacher_mismatch_focus_coef: float = 0.0
    economic_teacher_mismatch_focus_min_scale: float = 0.75
    economic_teacher_mismatch_focus_max_scale: float = 2.5
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
    economic_gt_full_year_warm_start_samples: int = 0
    economic_gt_full_year_warm_start_epochs: int = 0
    economic_gt_full_year_warm_start_u_weight: float = 0.0
    economic_bes_distill_coef: float = 0.15
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
    economic_bes_warm_start_economic_anchor_weight: float = 0.0
    economic_bes_warm_start_fallback_anchor_weight: float = 0.0
    economic_bes_teacher_selection_priority_boost: float = 0.75
    economic_bes_economic_source_priority_bonus: float = 0.10
    economic_bes_economic_source_min_share: float = 0.75
    economic_bes_idle_economic_source_min_share: float = 0.75
    economic_bes_teacher_target_min_share: float = 0.0
    economic_bes_anchor_max_scale: float = 1.0
    surrogate_actor_trust_coef: float = 0.60
    surrogate_actor_trust_min_scale: float = 0.10
    actor_low_trust_raw_fallback_keys: tuple[str, ...] = field(default_factory=tuple)
    state_feasible_action_shaping_enabled: bool = True
    abs_min_on_gate_th: float = 0.75
    abs_min_on_u_margin: float = 0.02
    expert_prefill_policy: str = "easy_rule_abs"
    expert_prefill_checkpoint_path: str | Path = ""
    expert_prefill_economic_policy: str = "checkpoint"
    expert_prefill_economic_checkpoint_path: str | Path = ""
    frozen_action_keys: tuple[str, ...] = field(default_factory=tuple)
    frozen_action_safe_checkpoint_path: str | Path = ""
    gt_safe_action_delta_clip: float = 0.0
    bes_safe_action_delta_clip: float = 0.0
    boiler_safe_action_delta_clip: float = 0.0
    tes_safe_action_delta_clip: float = 0.0
    abs_safe_action_delta_clip: float = 0.0
    ech_safe_action_delta_clip: float = 0.0
    expert_prefill_steps: int = 4_096
    actor_warm_start_epochs: int = 4
    actor_warm_start_batch_size: int = 256
    actor_warm_start_lr: float = 1e-4
    expert_prefill_cooling_bias: float = 0.5
    expert_prefill_abs_replay_boost: int = 0
    expert_prefill_abs_exec_threshold: float = 0.05
    expert_prefill_abs_window_mining_candidates: int = 8
    dual_abs_margin_k: float = 1.25
    dual_qc_ratio_th: float = 0.55
    dual_heat_backup_ratio_th: float = 0.10
    dual_safe_abs_u_th: float = 0.60
    checkpoint_interval_steps: int = 16_384
    eval_window_pool_size: int = 16
    eval_window_count: int = 8
    best_gate_enabled: bool = True
    best_gate_electric_min: float = 1.0
    best_gate_heat_min: float = 0.99
    best_gate_cool_min: float = 0.99
    plateau_control_enabled: bool = True
    plateau_patience_evals: int = 4
    plateau_lr_decay_factor: float = 0.5
    plateau_min_actor_lr: float = 2.5e-5
    plateau_min_critic_lr: float = 1e-4
    plateau_early_stop_patience_evals: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
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
        self.actor_warmup_steps = int(self.actor_warmup_steps)
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
        self.economic_gt_distill_coef = float(self.economic_gt_distill_coef)
        self.economic_teacher_distill_coef = float(self.economic_teacher_distill_coef)
        self.economic_teacher_safe_preserve_coef = float(self.economic_teacher_safe_preserve_coef)
        self.economic_teacher_safe_preserve_low_margin_boost = float(
            self.economic_teacher_safe_preserve_low_margin_boost
        )
        self.economic_teacher_safe_preserve_high_cooling_boost = float(
            self.economic_teacher_safe_preserve_high_cooling_boost
        )
        self.economic_teacher_safe_preserve_joint_boost = float(
            self.economic_teacher_safe_preserve_joint_boost
        )
        self.economic_teacher_mismatch_focus_coef = float(
            self.economic_teacher_mismatch_focus_coef
        )
        self.economic_teacher_mismatch_focus_min_scale = float(
            self.economic_teacher_mismatch_focus_min_scale
        )
        self.economic_teacher_mismatch_focus_max_scale = float(
            self.economic_teacher_mismatch_focus_max_scale
        )
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
        self.economic_gt_full_year_warm_start_samples = int(
            self.economic_gt_full_year_warm_start_samples
        )
        self.economic_gt_full_year_warm_start_epochs = int(
            self.economic_gt_full_year_warm_start_epochs
        )
        self.economic_gt_full_year_warm_start_u_weight = float(
            self.economic_gt_full_year_warm_start_u_weight
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
        self.economic_bes_warm_start_economic_anchor_weight = float(
            self.economic_bes_warm_start_economic_anchor_weight
        )
        self.economic_bes_warm_start_fallback_anchor_weight = float(
            self.economic_bes_warm_start_fallback_anchor_weight
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
        self.economic_bes_anchor_max_scale = float(self.economic_bes_anchor_max_scale)
        self.surrogate_actor_trust_coef = float(self.surrogate_actor_trust_coef)
        self.surrogate_actor_trust_min_scale = float(
            self.surrogate_actor_trust_min_scale
        )
        self.actor_low_trust_raw_fallback_keys = _normalize_action_key_tuple(
            self.actor_low_trust_raw_fallback_keys
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
        self.frozen_action_keys = _normalize_action_key_tuple(self.frozen_action_keys)
        self.frozen_action_safe_checkpoint_path = str(
            self.frozen_action_safe_checkpoint_path
        ).strip()
        self.gt_safe_action_delta_clip = float(self.gt_safe_action_delta_clip)
        self.bes_safe_action_delta_clip = float(self.bes_safe_action_delta_clip)
        self.boiler_safe_action_delta_clip = float(self.boiler_safe_action_delta_clip)
        self.tes_safe_action_delta_clip = float(self.tes_safe_action_delta_clip)
        self.abs_safe_action_delta_clip = float(self.abs_safe_action_delta_clip)
        self.ech_safe_action_delta_clip = float(self.ech_safe_action_delta_clip)
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
        unknown_frozen_keys = sorted(set(self.frozen_action_keys) - set(self.action_keys))
        if unknown_frozen_keys:
            raise ValueError(f"frozen_action_keys 包含未知动作键: {unknown_frozen_keys}")
        if self.frozen_action_keys and len(self.frozen_action_safe_checkpoint_path) == 0:
            raise ValueError(
                "frozen_action_keys 非空时必须提供 frozen_action_safe_checkpoint_path。"
            )
        if self.tes_safe_action_delta_clip > 0.0 and len(self.frozen_action_safe_checkpoint_path) == 0:
            raise ValueError(
                "tes_safe_action_delta_clip > 0 时必须提供 frozen_action_safe_checkpoint_path。"
            )
        if self.gt_safe_action_delta_clip > 0.0 and len(self.frozen_action_safe_checkpoint_path) == 0:
            raise ValueError(
                "gt_safe_action_delta_clip > 0 时必须提供 frozen_action_safe_checkpoint_path。"
            )
        if self.bes_safe_action_delta_clip > 0.0 and len(self.frozen_action_safe_checkpoint_path) == 0:
            raise ValueError(
                "bes_safe_action_delta_clip > 0 时必须提供 frozen_action_safe_checkpoint_path。"
            )
        if self.boiler_safe_action_delta_clip > 0.0 and len(self.frozen_action_safe_checkpoint_path) == 0:
            raise ValueError(
                "boiler_safe_action_delta_clip > 0 时必须提供 frozen_action_safe_checkpoint_path。"
            )
        if self.abs_safe_action_delta_clip > 0.0 and len(self.frozen_action_safe_checkpoint_path) == 0:
            raise ValueError(
                "abs_safe_action_delta_clip > 0 时必须提供 frozen_action_safe_checkpoint_path。"
            )
        if self.ech_safe_action_delta_clip > 0.0 and len(self.frozen_action_safe_checkpoint_path) == 0:
            raise ValueError(
                "ech_safe_action_delta_clip > 0 时必须提供 frozen_action_safe_checkpoint_path。"
            )
        if len(self.frozen_action_keys) >= len(self.action_keys):
            raise ValueError("frozen_action_keys 不能冻结全部动作维度。")
        if not Path(self.projection_surrogate_checkpoint_path).exists():
            raise FileNotFoundError(
                f"projection surrogate checkpoint 不存在: {self.projection_surrogate_checkpoint_path}"
            )
        if self.frozen_action_keys and not Path(self.frozen_action_safe_checkpoint_path).exists():
            raise FileNotFoundError(
                "frozen_action_safe_checkpoint_path 不存在: "
                f"{self.frozen_action_safe_checkpoint_path}"
            )
        if self.tes_safe_action_delta_clip < 0.0 or self.tes_safe_action_delta_clip > 1.0:
            raise ValueError("tes_safe_action_delta_clip 必须在 [0,1]。")
        if self.tes_safe_action_delta_clip > 0.0 and "u_tes" not in self.action_keys:
            raise ValueError("tes_safe_action_delta_clip > 0 时动作空间必须包含 u_tes。")
        if self.gt_safe_action_delta_clip < 0.0 or self.gt_safe_action_delta_clip > 1.0:
            raise ValueError("gt_safe_action_delta_clip 必须在 [0,1]。")
        if self.gt_safe_action_delta_clip > 0.0 and "u_gt" not in self.action_keys:
            raise ValueError("gt_safe_action_delta_clip > 0 时动作空间必须包含 u_gt。")
        if self.bes_safe_action_delta_clip < 0.0 or self.bes_safe_action_delta_clip > 1.0:
            raise ValueError("bes_safe_action_delta_clip 必须在 [0,1]。")
        if self.bes_safe_action_delta_clip > 0.0 and "u_bes" not in self.action_keys:
            raise ValueError("bes_safe_action_delta_clip > 0 时动作空间必须包含 u_bes。")
        if self.boiler_safe_action_delta_clip < 0.0 or self.boiler_safe_action_delta_clip > 1.0:
            raise ValueError("boiler_safe_action_delta_clip 必须在 [0,1]。")
        if self.boiler_safe_action_delta_clip > 0.0 and "u_boiler" not in self.action_keys:
            raise ValueError("boiler_safe_action_delta_clip > 0 时动作空间必须包含 u_boiler。")
        if self.abs_safe_action_delta_clip < 0.0 or self.abs_safe_action_delta_clip > 1.0:
            raise ValueError("abs_safe_action_delta_clip 必须在 [0,1]。")
        if self.abs_safe_action_delta_clip > 0.0 and "u_abs" not in self.action_keys:
            raise ValueError("abs_safe_action_delta_clip > 0 时动作空间必须包含 u_abs。")
        if self.ech_safe_action_delta_clip < 0.0 or self.ech_safe_action_delta_clip > 1.0:
            raise ValueError("ech_safe_action_delta_clip 必须在 [0,1]。")
        if self.ech_safe_action_delta_clip > 0.0 and "u_ech" not in self.action_keys:
            raise ValueError("ech_safe_action_delta_clip > 0 时动作空间必须包含 u_ech。")
        if self.episode_days < 7 or self.episode_days > 30:
            raise ValueError("episode_days 必须在 [7,30]。")
        if self.total_env_steps <= 0:
            raise ValueError("total_env_steps 必须 > 0。")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps 必须 >= 0。")
        if self.actor_warmup_steps < 0:
            raise ValueError("actor_warmup_steps 必须 >= 0。")
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
        if self.economic_gt_distill_coef < 0.0:
            raise ValueError("economic_gt_distill_coef 必须 >= 0。")
        if self.economic_teacher_distill_coef < 0.0:
            raise ValueError("economic_teacher_distill_coef 必须 >= 0。")
        if self.economic_teacher_safe_preserve_coef < 0.0:
            raise ValueError("economic_teacher_safe_preserve_coef 必须 >= 0。")
        if self.economic_teacher_safe_preserve_low_margin_boost < 0.0:
            raise ValueError("economic_teacher_safe_preserve_low_margin_boost 必须 >= 0。")
        if self.economic_teacher_safe_preserve_high_cooling_boost < 0.0:
            raise ValueError("economic_teacher_safe_preserve_high_cooling_boost 必须 >= 0。")
        if self.economic_teacher_safe_preserve_joint_boost < 0.0:
            raise ValueError("economic_teacher_safe_preserve_joint_boost 必须 >= 0。")
        if self.economic_teacher_mismatch_focus_coef < 0.0:
            raise ValueError("economic_teacher_mismatch_focus_coef 必须 >= 0。")
        if self.economic_teacher_mismatch_focus_min_scale <= 0.0:
            raise ValueError("economic_teacher_mismatch_focus_min_scale 必须 > 0。")
        if self.economic_teacher_mismatch_focus_max_scale < self.economic_teacher_mismatch_focus_min_scale:
            raise ValueError(
                "economic_teacher_mismatch_focus_max_scale 不能小于 economic_teacher_mismatch_focus_min_scale。"
            )
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
        if self.economic_gt_full_year_warm_start_samples < 0:
            raise ValueError("economic_gt_full_year_warm_start_samples 必须 >= 0。")
        if self.economic_gt_full_year_warm_start_epochs < 0:
            raise ValueError("economic_gt_full_year_warm_start_epochs 必须 >= 0。")
        if self.economic_gt_full_year_warm_start_u_weight < 0.0:
            raise ValueError("economic_gt_full_year_warm_start_u_weight 必须 >= 0。")
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
        if (
            self.economic_bes_warm_start_economic_anchor_weight < 0.0
            or self.economic_bes_warm_start_economic_anchor_weight > 1.0
        ):
            raise ValueError(
                "economic_bes_warm_start_economic_anchor_weight 必须在 [0,1]。"
            )
        if (
            self.economic_bes_warm_start_fallback_anchor_weight < 0.0
            or self.economic_bes_warm_start_fallback_anchor_weight > 1.0
        ):
            raise ValueError(
                "economic_bes_warm_start_fallback_anchor_weight 必须在 [0,1]。"
            )
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
        if (
            self.economic_bes_anchor_max_scale < 0.0
            or self.economic_bes_anchor_max_scale > 1.0
        ):
            raise ValueError("economic_bes_anchor_max_scale 必须在 [0,1]。")
        if self.surrogate_actor_trust_coef < 0.0:
            raise ValueError("surrogate_actor_trust_coef 必须 >= 0。")
        if (
            self.surrogate_actor_trust_min_scale < 0.0
            or self.surrogate_actor_trust_min_scale > 1.0
        ):
            raise ValueError("surrogate_actor_trust_min_scale 必须在 [0,1]。")
        unknown_fallback_keys = sorted(
            set(self.actor_low_trust_raw_fallback_keys) - set(self.action_keys)
        )
        if unknown_fallback_keys:
            raise ValueError(
                f"actor_low_trust_raw_fallback_keys 包含未知动作键: {unknown_fallback_keys}"
            )
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


def _calendar_season_from_month(month: int) -> str:
    normalized_month = int(month)
    if normalized_month in {12, 1, 2}:
        return "winter"
    if normalized_month in {3, 4, 5}:
        return "spring"
    if normalized_month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _calendar_season_from_timestamp_value(value: object) -> str:
    try:
        timestamp = pd.to_datetime(value)
    except Exception:
        return "summer"
    if pd.isna(timestamp):
        return "summer"
    return _calendar_season_from_month(int(timestamp.month))


def _allocate_eval_window_season_counts(
    *,
    requested_total: int,
    available_by_season: Mapping[str, int],
) -> dict[str, int]:
    preferred_order = ("summer", "winter", "spring", "autumn")
    counts = {season: 0 for season in preferred_order}
    remaining = max(0, int(requested_total))
    active_seasons = [
        season
        for season in preferred_order
        if int(available_by_season.get(season, 0)) > 0
    ]
    if remaining <= 0 or not active_seasons:
        return counts

    for season in active_seasons:
        if remaining <= 0:
            break
        counts[season] += 1
        remaining -= 1

    while remaining > 0:
        progressed = False
        for season in active_seasons:
            if counts[season] >= int(available_by_season.get(season, 0)):
                continue
            counts[season] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
    return counts


def _select_eval_window_records_temporally(
    records: Sequence[dict[str, Any]],
    *,
    target_count: int,
) -> list[dict[str, Any]]:
    unique_by_start = {
        int(record["start_idx"]): dict(record)
        for record in records
    }
    ordered_records = sorted(unique_by_start.values(), key=lambda item: int(item["start_idx"]))
    target_total = max(0, int(target_count))
    if target_total <= 0:
        return []
    if len(ordered_records) <= target_total:
        return ordered_records
    if target_total == 1:
        return [
            max(
                ordered_records,
                key=lambda item: (
                    float(item["validation_score"]),
                    float(item["cooling_peak_mw"]),
                    int(item["start_idx"]),
                ),
            )
        ]

    selected: list[dict[str, Any]] = []
    for block in np.array_split(np.arange(len(ordered_records), dtype=int), target_total):
        if len(block) == 0:
            continue
        block_records = [ordered_records[int(index)] for index in block.tolist()]
        best_record = max(
            block_records,
            key=lambda item: (
                float(item["validation_score"]),
                float(item["cooling_peak_mw"]),
                int(item["start_idx"]),
            ),
        )
        selected.append(best_record)
    selected_unique = {
        int(record["start_idx"]): record
        for record in selected
    }
    return sorted(selected_unique.values(), key=lambda item: int(item["start_idx"]))


def _select_eval_window_records_season_balanced(
    records: Sequence[dict[str, Any]],
    *,
    target_count: int,
) -> list[dict[str, Any]]:
    unique_by_start = {
        int(record["start_idx"]): dict(record)
        for record in records
    }
    ordered_records = sorted(unique_by_start.values(), key=lambda item: int(item["start_idx"]))
    target_total = max(0, int(target_count))
    if target_total <= 0:
        return []
    if len(ordered_records) <= target_total:
        return ordered_records
    if target_total == 1:
        return _select_eval_window_records_temporally(
            ordered_records,
            target_count=1,
        )

    season_groups: dict[str, list[dict[str, Any]]] = {
        season: []
        for season in ("winter", "spring", "summer", "autumn")
    }
    for record in ordered_records:
        season_groups[str(record["season"])].append(record)
    counts = _allocate_eval_window_season_counts(
        requested_total=target_total,
        available_by_season={season: len(items) for season, items in season_groups.items()},
    )

    selected: list[dict[str, Any]] = []
    selected_start_indices: set[int] = set()
    for season in ("summer", "winter", "spring", "autumn"):
        season_selected = _select_eval_window_records_temporally(
            season_groups[season],
            target_count=int(counts.get(season, 0)),
        )
        for record in season_selected:
            start_idx = int(record["start_idx"])
            if start_idx in selected_start_indices:
                continue
            selected.append(record)
            selected_start_indices.add(start_idx)

    if len(selected) < target_total:
        remaining_records = [
            record
            for record in ordered_records
            if int(record["start_idx"]) not in selected_start_indices
        ]
        top_up = _select_eval_window_records_temporally(
            remaining_records,
            target_count=int(target_total - len(selected)),
        )
        for record in top_up:
            start_idx = int(record["start_idx"])
            if start_idx in selected_start_indices:
                continue
            selected.append(record)
            selected_start_indices.add(start_idx)

    return sorted(selected, key=lambda item: int(item["start_idx"]))


def _build_eval_window_candidate_records(
    *,
    train_df: pd.DataFrame,
    candidate_starts: Sequence[int],
    steps: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for start_idx in candidate_starts:
        end_idx = int(start_idx + steps)
        episode_df = train_df.iloc[int(start_idx):end_idx].reset_index(drop=True)
        timestamp_series = pd.to_datetime(episode_df["timestamp"])
        qc_series = pd.to_numeric(episode_df["qc_dem_mw"], errors="coerce").fillna(0.0)
        qh_series = pd.to_numeric(episode_df["qh_dem_mw"], errors="coerce").fillna(0.0)
        cooling_mean = float(qc_series.mean()) if len(qc_series) > 0 else 0.0
        cooling_peak = float(qc_series.max()) if len(qc_series) > 0 else 0.0
        heating_mean = float(qh_series.mean()) if len(qh_series) > 0 else 0.0
        validation_score = float(
            cooling_mean
            + 0.10 * cooling_peak
            + 0.05 * heating_mean
        )
        records.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_timestamp": str(timestamp_series.iloc[0].isoformat()),
                "end_timestamp": str(timestamp_series.iloc[-1].isoformat()),
                "season": _calendar_season_from_month(int(timestamp_series.iloc[0].month)),
                "cooling_mean_mw": cooling_mean,
                "cooling_peak_mw": cooling_peak,
                "heating_mean_mw": heating_mean,
                "validation_score": validation_score,
                "episode_df": episode_df,
            }
        )
    return records


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

    candidate_records = _build_eval_window_candidate_records(
        train_df=train_df,
        candidate_starts=candidate_starts,
        steps=steps,
    )
    pool_target_size = min(int(pool_size), len(candidate_starts))
    pool_records = _select_eval_window_records_season_balanced(
        candidate_records,
        target_count=pool_target_size,
    )

    window_target_count = min(int(window_count), len(pool_records))
    selected_records = _select_eval_window_records_season_balanced(
        pool_records,
        target_count=window_target_count,
    )
    selected_start_indices = {
        int(record["start_idx"])
        for record in selected_records
    }

    windows_payload: list[dict[str, Any]] = []
    episode_dfs: list[pd.DataFrame] = []
    selected_indices: list[int] = []
    season_counts = {season: 0 for season in ("winter", "spring", "summer", "autumn")}
    selected_season_counts = {season: 0 for season in ("winter", "spring", "summer", "autumn")}
    for window_index, record in enumerate(pool_records):
        selected = int(record["start_idx"]) in selected_start_indices
        season = str(record["season"])
        season_counts[season] = int(season_counts[season] + 1)
        if selected:
            episode_dfs.append(record["episode_df"])
            selected_indices.append(int(window_index))
            selected_season_counts[season] = int(selected_season_counts[season] + 1)
        windows_payload.append(
            {
                "window_index": int(window_index),
                "start_idx": int(record["start_idx"]),
                "end_idx": int(record["end_idx"]),
                "start_timestamp": str(record["start_timestamp"]),
                "end_timestamp": str(record["end_timestamp"]),
                "season": season,
                "cooling_mean_mw": float(record["cooling_mean_mw"]),
                "cooling_peak_mw": float(record["cooling_peak_mw"]),
                "heating_mean_mw": float(record["heating_mean_mw"]),
                "validation_score": float(record["validation_score"]),
                "selected_for_eval": bool(selected),
            }
        )

    return {
        "mode": "season_balanced_multi_window_pool_v2",
        "pool_size": int(len(pool_records)),
        "window_count": int(len(selected_indices)),
        "seed": int(seed),
        "episode_steps": int(steps),
        "episode_days": int(eval_episode_days),
        "selected_window_indices": selected_indices,
        "pool_season_counts": {key: int(value) for key, value in season_counts.items()},
        "selected_season_counts": {key: int(value) for key, value in selected_season_counts.items()},
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
        safe_preserve_weights = _economic_teacher_safe_preserve_weights_np(
            action_keys=self.action_keys,
        ).reshape(1, -1)
        self.economic_teacher_safe_preserve_weight_np = safe_preserve_weights.astype(
            np.float32,
            copy=True,
        )
        self.economic_teacher_safe_preserve_weight = self.torch.as_tensor(
            safe_preserve_weights,
            dtype=self.torch.float32,
            device=self.device,
        )
        self.economic_teacher_mismatch_focus_weight_np = np.ones(
            (1, len(self.action_keys)),
            dtype=np.float32,
        )
        self.economic_teacher_mismatch_focus_weight = self.torch.as_tensor(
            self.economic_teacher_mismatch_focus_weight_np,
            dtype=self.torch.float32,
            device=self.device,
        )
        self.economic_teacher_mismatch_focus_summary: dict[str, Any] = {
            "enabled": bool(float(self.config.economic_teacher_mismatch_focus_coef) > 0.0),
            "status": "not_run",
            "scale_by_action": {str(key): 1.0 for key in self.action_keys},
        }
        self.surrogate_actor_trust_weight_np = np.ones(
            (1, len(self.action_keys)),
            dtype=np.float32,
        )
        self.surrogate_actor_trust_weight = self.torch.as_tensor(
            self.surrogate_actor_trust_weight_np,
            dtype=self.torch.float32,
            device=self.device,
        )
        self.surrogate_actor_trust_summary: dict[str, Any] = {
            "enabled": bool(float(self.config.surrogate_actor_trust_coef) > 0.0),
            "status": "not_run",
            "trust_by_action": {str(key): 1.0 for key in self.action_keys},
        }
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
        self.frozen_action_keys = tuple(
            str(key) for key in self.config.frozen_action_keys if str(key) in self.action_index
        )
        self.frozen_action_enabled = bool(self.frozen_action_keys)
        self.safe_reference_action_enabled = bool(
            self.frozen_action_enabled
            or float(self.config.gt_safe_action_delta_clip) > 0.0
            or float(self.config.bes_safe_action_delta_clip) > 0.0
            or float(self.config.boiler_safe_action_delta_clip) > 0.0
            or float(self.config.tes_safe_action_delta_clip) > 0.0
            or float(self.config.abs_safe_action_delta_clip) > 0.0
            or float(self.config.ech_safe_action_delta_clip) > 0.0
        )
        self.frozen_action_safe_checkpoint_path = str(
            self.config.frozen_action_safe_checkpoint_path
        ).strip()
        self.frozen_action_safe_metadata: dict[str, Any] = {}
        self.frozen_action_safe_actor = None
        self.frozen_action_safe_obs_offset_np = np.zeros((len(self.observation_keys),), dtype=np.float32)
        self.frozen_action_safe_obs_scale_np = np.ones((len(self.observation_keys),), dtype=np.float32)
        self.frozen_action_safe_obs_offset = self.torch.zeros(
            (1, len(self.observation_keys)),
            dtype=self.torch.float32,
            device=self.device,
        )
        self.frozen_action_safe_obs_scale = self.torch.ones(
            (1, len(self.observation_keys)),
            dtype=self.torch.float32,
            device=self.device,
        )
        self.frozen_action_safe_state_feasible_action_shaping_enabled = bool(
            self.config.state_feasible_action_shaping_enabled
        )
        self.frozen_action_safe_abs_min_on_gate_th = float(self.config.abs_min_on_gate_th)
        self.frozen_action_safe_abs_min_on_u_margin = float(self.config.abs_min_on_u_margin)
        if self.safe_reference_action_enabled:
            self._load_frozen_action_safe_bundle()

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
        self.economic_teacher_mismatch_focus_summary.update(
            {
                "prefill_replay_size": int(self.replay.size),
            }
        )
        self.actor_gt_warm_start_summary: dict[str, Any] = {
            "enabled": bool(
                int(self.config.economic_gt_full_year_warm_start_samples) > 0
                and int(self.config.economic_gt_full_year_warm_start_epochs) > 0
                and "u_gt" in self.action_index
            ),
            "samples": int(self.config.economic_gt_full_year_warm_start_samples),
            "epochs": int(self.config.economic_gt_full_year_warm_start_epochs),
            "u_weight": float(self.config.economic_gt_full_year_warm_start_u_weight),
            "status": "not_run",
        }
        self.actor_teacher_gt_head_warm_start_summary: dict[str, Any] = {
            "enabled": bool(
                int(self.config.economic_teacher_full_year_warm_start_samples) > 0
                and int(self.config.economic_teacher_full_year_warm_start_epochs) > 0
                and "u_gt" in self.action_index
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

    def _observation_vector_to_dict(self, observation_vector: np.ndarray) -> dict[str, float]:
        return _observation_vector_to_dict(
            observation_vector,
            observation_keys=self.observation_keys,
        )

    def _load_frozen_action_safe_bundle(self) -> None:
        if not self.safe_reference_action_enabled:
            return
        artifact = self._resolve_prefill_checkpoint_artifact(
            checkpoint_path=self.frozen_action_safe_checkpoint_path,
        )
        if str(artifact.get("artifact_type", "")).strip().lower() != "pafc_td3_actor":
            raise ValueError(
                "frozen_action_safe_checkpoint_path 当前仅支持 PAFC-TD3 actor checkpoint。"
            )
        resolved_path = Path(artifact["resolved_path"])
        payload = load_policy(
            resolved_path,
            map_location=self.device,
        )
        metadata = dict(payload.get("metadata", {}))
        observation_keys = tuple(str(key) for key in metadata.get("observation_keys", ()))
        action_keys = tuple(str(key) for key in metadata.get("action_keys", ()))
        if observation_keys != self.observation_keys:
            raise ValueError("冻结安全策略的 observation_keys 与当前训练配置不一致。")
        if action_keys != self.action_keys:
            raise ValueError("冻结安全策略的 action_keys 与当前训练配置不一致。")
        actor = build_pafc_actor_network(
            observation_dim=int(metadata["observation_dim"]),
            action_keys=action_keys,
            hidden_dims=tuple(int(dim) for dim in metadata.get("hidden_dims", (256, 256))),
        ).to(self.device)
        actor.load_state_dict(payload["state_dict"])
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        obs_norm = dict(metadata.get("observation_norm", {}))
        offset_np = np.asarray(obs_norm.get("offset", []), dtype=np.float32).reshape(-1)
        scale_np = np.asarray(obs_norm.get("scale", []), dtype=np.float32).reshape(-1)
        if offset_np.shape[0] != len(self.observation_keys) or scale_np.shape[0] != len(
            self.observation_keys
        ):
            raise ValueError("冻结安全策略 observation_norm 维度与当前训练配置不一致。")
        scale_np = np.where(np.abs(scale_np) < _NORM_EPS, 1.0, scale_np)
        self.frozen_action_safe_actor = actor
        self.frozen_action_safe_metadata = {
            **metadata,
            "entry_path": str(Path(artifact["entry_path"]).resolve()).replace("\\", "/"),
            "resolved_path": str(resolved_path.resolve()).replace("\\", "/"),
        }
        self.frozen_action_safe_obs_offset_np = offset_np.astype(np.float32, copy=True)
        self.frozen_action_safe_obs_scale_np = scale_np.astype(np.float32, copy=True)
        self.frozen_action_safe_obs_offset = self.torch.as_tensor(
            offset_np.reshape(1, -1),
            dtype=self.torch.float32,
            device=self.device,
        )
        self.frozen_action_safe_obs_scale = self.torch.as_tensor(
            scale_np.reshape(1, -1),
            dtype=self.torch.float32,
            device=self.device,
        )
        self.frozen_action_safe_state_feasible_action_shaping_enabled = bool(
            metadata.get(
                "state_feasible_action_shaping_enabled",
                self.config.state_feasible_action_shaping_enabled,
            )
        )
        self.frozen_action_safe_abs_min_on_gate_th = float(
            metadata.get("abs_min_on_gate_th", self.config.abs_min_on_gate_th)
        )
        self.frozen_action_safe_abs_min_on_u_margin = float(
            metadata.get("abs_min_on_u_margin", self.config.abs_min_on_u_margin)
        )

    def _predict_frozen_safe_action_tensor(self, *, obs_batch):
        if not self.safe_reference_action_enabled or self.frozen_action_safe_actor is None:
            return None
        with self.torch.no_grad():
            obs_norm = (obs_batch - self.frozen_action_safe_obs_offset) / self.frozen_action_safe_obs_scale
            safe_action = self.frozen_action_safe_actor(obs_norm)
            return self._apply_abs_cooling_blend_tensor(
                obs_batch=obs_batch,
                action_batch=safe_action,
                state_feasible_action_shaping_enabled=(
                    self.frozen_action_safe_state_feasible_action_shaping_enabled
                ),
                abs_min_on_gate_th=self.frozen_action_safe_abs_min_on_gate_th,
                abs_min_on_u_margin=self.frozen_action_safe_abs_min_on_u_margin,
            )

    def _predict_frozen_safe_action_np(self, *, observation_vector: np.ndarray) -> np.ndarray | None:
        if not self.safe_reference_action_enabled or self.frozen_action_safe_actor is None:
            return None
        normalized = (
            (np.asarray(observation_vector, dtype=np.float32).reshape(-1) - self.frozen_action_safe_obs_offset_np)
            / self.frozen_action_safe_obs_scale_np
        ).astype(np.float32)
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(
                normalized.reshape(1, -1),
                dtype=self.torch.float32,
                device=self.device,
            )
            safe_action = (
                self.frozen_action_safe_actor(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
            )
        return self._apply_abs_cooling_blend_np(
            observation_vector=np.asarray(observation_vector, dtype=np.float32).reshape(-1),
            action_vector=safe_action,
            state_feasible_action_shaping_enabled=(
                self.frozen_action_safe_state_feasible_action_shaping_enabled
            ),
            abs_min_on_gate_th=self.frozen_action_safe_abs_min_on_gate_th,
            abs_min_on_u_margin=self.frozen_action_safe_abs_min_on_u_margin,
        )

    def _overwrite_frozen_action_dims_tensor(self, *, action_batch, safe_action_batch):
        if not self.frozen_action_enabled or safe_action_batch is None:
            return action_batch
        blended = action_batch.clone()
        for key in self.frozen_action_keys:
            index = int(self.action_index[key])
            blended[:, index : index + 1] = safe_action_batch[:, index : index + 1].detach()
        return blended

    def _clip_tes_near_safe_action_tensor(self, *, action_batch, safe_action_batch):
        clip_delta = float(self.config.tes_safe_action_delta_clip)
        if (
            clip_delta <= 0.0
            or safe_action_batch is None
            or "u_tes" not in self.action_index
            or "u_tes" in self.frozen_action_keys
        ):
            return action_batch
        index = int(self.action_index["u_tes"])
        safe_column = safe_action_batch[:, index : index + 1].detach()
        lower = self.torch.clamp(
            safe_column - clip_delta,
            float(self.action_low_np[index]),
            float(self.action_high_np[index]),
        )
        upper = self.torch.clamp(
            safe_column + clip_delta,
            float(self.action_low_np[index]),
            float(self.action_high_np[index]),
        )
        clipped_column = self.torch.maximum(
            self.torch.minimum(action_batch[:, index : index + 1], upper),
            lower,
        )
        mask = self.torch.zeros_like(action_batch)
        mask[:, index : index + 1] = 1.0
        return action_batch * (1.0 - mask) + clipped_column * mask

    def _clip_bes_near_safe_action_tensor(self, *, action_batch, safe_action_batch):
        clip_delta = float(self.config.bes_safe_action_delta_clip)
        if (
            clip_delta <= 0.0
            or safe_action_batch is None
            or "u_bes" not in self.action_index
            or "u_bes" in self.frozen_action_keys
        ):
            return action_batch
        index = int(self.action_index["u_bes"])
        safe_column = safe_action_batch[:, index : index + 1].detach()
        lower = self.torch.clamp(
            safe_column - clip_delta,
            float(self.action_low_np[index]),
            float(self.action_high_np[index]),
        )
        upper = self.torch.clamp(
            safe_column + clip_delta,
            float(self.action_low_np[index]),
            float(self.action_high_np[index]),
        )
        clipped_column = self.torch.maximum(
            self.torch.minimum(action_batch[:, index : index + 1], upper),
            lower,
        )
        mask = self.torch.zeros_like(action_batch)
        mask[:, index : index + 1] = 1.0
        return action_batch * (1.0 - mask) + clipped_column * mask

    def _clip_boiler_near_safe_action_tensor(self, *, action_batch, safe_action_batch):
        clip_delta = float(self.config.boiler_safe_action_delta_clip)
        if (
            clip_delta <= 0.0
            or safe_action_batch is None
            or "u_boiler" not in self.action_index
            or "u_boiler" in self.frozen_action_keys
        ):
            return action_batch
        index = int(self.action_index["u_boiler"])
        safe_column = safe_action_batch[:, index : index + 1].detach()
        lower = self.torch.clamp(
            safe_column - clip_delta,
            float(self.action_low_np[index]),
            float(self.action_high_np[index]),
        )
        upper = self.torch.clamp(
            safe_column + clip_delta,
            float(self.action_low_np[index]),
            float(self.action_high_np[index]),
        )
        clipped_column = self.torch.maximum(
            self.torch.minimum(action_batch[:, index : index + 1], upper),
            lower,
        )
        mask = self.torch.zeros_like(action_batch)
        mask[:, index : index + 1] = 1.0
        return action_batch * (1.0 - mask) + clipped_column * mask

    def _clip_abs_near_safe_action_tensor(self, *, action_batch, safe_action_batch):
        clip_delta = float(self.config.abs_safe_action_delta_clip)
        if (
            clip_delta <= 0.0
            or safe_action_batch is None
            or "u_abs" not in self.action_index
            or "u_abs" in self.frozen_action_keys
        ):
            return action_batch
        index = int(self.action_index["u_abs"])
        safe_column = safe_action_batch[:, index : index + 1].detach()
        lower = self.torch.clamp(
            safe_column - clip_delta,
            float(self.action_low_np[index]),
            float(self.action_high_np[index]),
        )
        upper = self.torch.clamp(
            safe_column + clip_delta,
            float(self.action_low_np[index]),
            float(self.action_high_np[index]),
        )
        clipped_column = self.torch.maximum(
            self.torch.minimum(action_batch[:, index : index + 1], upper),
            lower,
        )
        mask = self.torch.zeros_like(action_batch)
        mask[:, index : index + 1] = 1.0
        return action_batch * (1.0 - mask) + clipped_column * mask

    def _clip_ech_near_safe_action_tensor(self, *, action_batch, safe_action_batch):
        clip_delta = float(self.config.ech_safe_action_delta_clip)
        if (
            clip_delta <= 0.0
            or safe_action_batch is None
            or "u_ech" not in self.action_index
            or "u_ech" in self.frozen_action_keys
        ):
            return action_batch
        index = int(self.action_index["u_ech"])
        safe_column = safe_action_batch[:, index : index + 1].detach()
        lower = self.torch.clamp(
            safe_column - clip_delta,
            float(self.action_low_np[index]),
            float(self.action_high_np[index]),
        )
        upper = self.torch.clamp(
            safe_column + clip_delta,
            float(self.action_low_np[index]),
            float(self.action_high_np[index]),
        )
        clipped_column = self.torch.maximum(
            self.torch.minimum(action_batch[:, index : index + 1], upper),
            lower,
        )
        mask = self.torch.zeros_like(action_batch)
        mask[:, index : index + 1] = 1.0
        return action_batch * (1.0 - mask) + clipped_column * mask

    def _clip_gt_near_safe_action_tensor(self, *, action_batch, safe_action_batch):
        clip_delta = float(self.config.gt_safe_action_delta_clip)
        if (
            clip_delta <= 0.0
            or safe_action_batch is None
            or "u_gt" not in self.action_index
            or "u_gt" in self.frozen_action_keys
        ):
            return action_batch
        index = int(self.action_index["u_gt"])
        safe_column = safe_action_batch[:, index : index + 1].detach()
        lower = self.torch.clamp(
            safe_column - clip_delta,
            float(self.action_low_np[index]),
            float(self.action_high_np[index]),
        )
        upper = self.torch.clamp(
            safe_column + clip_delta,
            float(self.action_low_np[index]),
            float(self.action_high_np[index]),
        )
        clipped_column = self.torch.maximum(
            self.torch.minimum(action_batch[:, index : index + 1], upper),
            lower,
        )
        mask = self.torch.zeros_like(action_batch)
        mask[:, index : index + 1] = 1.0
        return action_batch * (1.0 - mask) + clipped_column * mask

    def _overwrite_frozen_action_dims_np(
        self,
        *,
        action_vector: np.ndarray,
        safe_action_vector: np.ndarray | None,
    ) -> np.ndarray:
        if not self.frozen_action_enabled or safe_action_vector is None:
            return np.asarray(action_vector, dtype=np.float32).reshape(-1)
        blended = np.asarray(action_vector, dtype=np.float32).reshape(-1).copy()
        safe_vector = np.asarray(safe_action_vector, dtype=np.float32).reshape(-1)
        for key in self.frozen_action_keys:
            blended[int(self.action_index[key])] = float(safe_vector[int(self.action_index[key])])
        return blended

    def _clip_tes_near_safe_action_np(
        self,
        *,
        action_vector: np.ndarray,
        safe_action_vector: np.ndarray | None,
    ) -> np.ndarray:
        clip_delta = float(self.config.tes_safe_action_delta_clip)
        action_np = np.asarray(action_vector, dtype=np.float32).reshape(-1).copy()
        if (
            clip_delta <= 0.0
            or safe_action_vector is None
            or "u_tes" not in self.action_index
            or "u_tes" in self.frozen_action_keys
        ):
            return action_np
        safe_vector = np.asarray(safe_action_vector, dtype=np.float32).reshape(-1)
        index = int(self.action_index["u_tes"])
        lower = max(float(self.action_low_np[index]), float(safe_vector[index]) - clip_delta)
        upper = min(float(self.action_high_np[index]), float(safe_vector[index]) + clip_delta)
        action_np[index] = float(np.clip(float(action_np[index]), lower, upper))
        return action_np

    def _clip_bes_near_safe_action_np(
        self,
        *,
        action_vector: np.ndarray,
        safe_action_vector: np.ndarray | None,
    ) -> np.ndarray:
        clip_delta = float(self.config.bes_safe_action_delta_clip)
        action_np = np.asarray(action_vector, dtype=np.float32).reshape(-1).copy()
        if (
            clip_delta <= 0.0
            or safe_action_vector is None
            or "u_bes" not in self.action_index
            or "u_bes" in self.frozen_action_keys
        ):
            return action_np
        safe_vector = np.asarray(safe_action_vector, dtype=np.float32).reshape(-1)
        index = int(self.action_index["u_bes"])
        lower = max(float(self.action_low_np[index]), float(safe_vector[index]) - clip_delta)
        upper = min(float(self.action_high_np[index]), float(safe_vector[index]) + clip_delta)
        action_np[index] = float(np.clip(float(action_np[index]), lower, upper))
        return action_np

    def _clip_boiler_near_safe_action_np(
        self,
        *,
        action_vector: np.ndarray,
        safe_action_vector: np.ndarray | None,
    ) -> np.ndarray:
        clip_delta = float(self.config.boiler_safe_action_delta_clip)
        action_np = np.asarray(action_vector, dtype=np.float32).reshape(-1).copy()
        if (
            clip_delta <= 0.0
            or safe_action_vector is None
            or "u_boiler" not in self.action_index
            or "u_boiler" in self.frozen_action_keys
        ):
            return action_np
        safe_vector = np.asarray(safe_action_vector, dtype=np.float32).reshape(-1)
        index = int(self.action_index["u_boiler"])
        lower = max(float(self.action_low_np[index]), float(safe_vector[index]) - clip_delta)
        upper = min(float(self.action_high_np[index]), float(safe_vector[index]) + clip_delta)
        action_np[index] = float(np.clip(float(action_np[index]), lower, upper))
        return action_np

    def _clip_abs_near_safe_action_np(
        self,
        *,
        action_vector: np.ndarray,
        safe_action_vector: np.ndarray | None,
    ) -> np.ndarray:
        clip_delta = float(self.config.abs_safe_action_delta_clip)
        action_np = np.asarray(action_vector, dtype=np.float32).reshape(-1).copy()
        if (
            clip_delta <= 0.0
            or safe_action_vector is None
            or "u_abs" not in self.action_index
            or "u_abs" in self.frozen_action_keys
        ):
            return action_np
        safe_vector = np.asarray(safe_action_vector, dtype=np.float32).reshape(-1)
        index = int(self.action_index["u_abs"])
        lower = max(float(self.action_low_np[index]), float(safe_vector[index]) - clip_delta)
        upper = min(float(self.action_high_np[index]), float(safe_vector[index]) + clip_delta)
        action_np[index] = float(np.clip(float(action_np[index]), lower, upper))
        return action_np

    def _clip_ech_near_safe_action_np(
        self,
        *,
        action_vector: np.ndarray,
        safe_action_vector: np.ndarray | None,
    ) -> np.ndarray:
        clip_delta = float(self.config.ech_safe_action_delta_clip)
        action_np = np.asarray(action_vector, dtype=np.float32).reshape(-1).copy()
        if (
            clip_delta <= 0.0
            or safe_action_vector is None
            or "u_ech" not in self.action_index
            or "u_ech" in self.frozen_action_keys
        ):
            return action_np
        safe_vector = np.asarray(safe_action_vector, dtype=np.float32).reshape(-1)
        index = int(self.action_index["u_ech"])
        lower = max(float(self.action_low_np[index]), float(safe_vector[index]) - clip_delta)
        upper = min(float(self.action_high_np[index]), float(safe_vector[index]) + clip_delta)
        action_np[index] = float(np.clip(float(action_np[index]), lower, upper))
        return action_np

    def _clip_gt_near_safe_action_np(
        self,
        *,
        action_vector: np.ndarray,
        safe_action_vector: np.ndarray | None,
    ) -> np.ndarray:
        clip_delta = float(self.config.gt_safe_action_delta_clip)
        action_np = np.asarray(action_vector, dtype=np.float32).reshape(-1).copy()
        if (
            clip_delta <= 0.0
            or safe_action_vector is None
            or "u_gt" not in self.action_index
            or "u_gt" in self.frozen_action_keys
        ):
            return action_np
        safe_vector = np.asarray(safe_action_vector, dtype=np.float32).reshape(-1)
        index = int(self.action_index["u_gt"])
        lower = max(float(self.action_low_np[index]), float(safe_vector[index]) - clip_delta)
        upper = min(float(self.action_high_np[index]), float(safe_vector[index]) + clip_delta)
        action_np[index] = float(np.clip(float(action_np[index]), lower, upper))
        return action_np

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
        else:
            prefill_policy = str(self.config.expert_prefill_policy).strip().lower()
            if prefill_policy in {"easy_rule_abs", "easy_rule", "rule"}:
                safe_policy, safe_info = self._build_expert_policy()
                summary["safe_teacher"] = {
                    **dict(safe_info),
                    "role": "economic_safe_compare",
                    "source": "expert_prefill_policy",
                }
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

    def _apply_abs_cooling_blend_tensor(
        self,
        *,
        obs_batch,
        action_batch,
        state_feasible_action_shaping_enabled: bool | None = None,
        abs_min_on_gate_th: float | None = None,
        abs_min_on_u_margin: float | None = None,
    ):
        shaped_columns = [
            action_batch[:, index : index + 1]
            for index in range(action_batch.shape[1])
        ]
        feasible_shaping_enabled = (
            bool(self.config.state_feasible_action_shaping_enabled)
            if state_feasible_action_shaping_enabled is None
            else bool(state_feasible_action_shaping_enabled)
        )
        resolved_abs_min_on_gate_th = (
            float(self.config.abs_min_on_gate_th)
            if abs_min_on_gate_th is None
            else float(abs_min_on_gate_th)
        )
        resolved_abs_min_on_u_margin = (
            float(self.config.abs_min_on_u_margin)
            if abs_min_on_u_margin is None
            else float(abs_min_on_u_margin)
        )

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

        if feasible_shaping_enabled:
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
        abs_effective_min_u = float(max(0.0, abs_deadzone_u_th + resolved_abs_min_on_u_margin))
        abs_min_on_gate_th = float(max(0.0, resolved_abs_min_on_gate_th))
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
        state_feasible_action_shaping_enabled: bool | None = None,
        abs_min_on_gate_th: float | None = None,
        abs_min_on_u_margin: float | None = None,
    ) -> np.ndarray:
        blended = np.asarray(action_vector, dtype=np.float32).copy()
        feasible_shaping_enabled = (
            bool(self.config.state_feasible_action_shaping_enabled)
            if state_feasible_action_shaping_enabled is None
            else bool(state_feasible_action_shaping_enabled)
        )
        resolved_abs_min_on_gate_th = (
            float(self.config.abs_min_on_gate_th)
            if abs_min_on_gate_th is None
            else float(abs_min_on_gate_th)
        )
        resolved_abs_min_on_u_margin = (
            float(self.config.abs_min_on_u_margin)
            if abs_min_on_u_margin is None
            else float(abs_min_on_u_margin)
        )

        if feasible_shaping_enabled and "u_bes" in self.action_index and "soc_bes" in self.observation_index:
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

        if feasible_shaping_enabled and "u_tes" in self.action_index and "e_tes_mwh" in self.observation_index:
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
        abs_effective_min_u = float(max(0.0, abs_deadzone_u_th + resolved_abs_min_on_u_margin))
        abs_min_on_gate_th = float(max(0.0, resolved_abs_min_on_gate_th))
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
        load_custom_objects: dict[str, Any] | None = None
        if algo in {"sac", "td3", "ddpg", "dqn"}:
            # Teacher checkpoints are only used for deterministic inference here.
            # Shrink replay-related state on load to avoid allocating the original
            # training buffer, which can otherwise dominate memory during PAFC init.
            load_custom_objects = {
                "buffer_size": 1,
                "learning_starts": 0,
            }
        model = algo_cls.load(
            str(resolved_model_path),
            env=model_env,
            device="cpu",
            custom_objects=load_custom_objects,
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
            "gt_price_prior_row_rate": 0.0,
            "gt_price_prior_on_rate": 0.0,
            "gt_price_prior_off_rate": 0.0,
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
        gt_prior_rows = 0
        gt_prior_on_rows = 0
        gt_prior_off_rows = 0
        bes_prior_rows = 0
        bes_prior_charge_rows = 0
        bes_prior_discharge_rows = 0
        u_gt_index = self.action_index.get("u_gt")
        u_bes_index = self.action_index.get("u_bes")
        u_ech_index = self.action_index.get("u_ech")
        charge_u, discharge_u = self._resolve_bes_prior_u_pair()

        for row_index in range(len(warm_targets)):
            teacher_gt_override = False
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
                    if u_gt_index is not None:
                        teacher_gt_override = bool(active_mask[int(u_gt_index)])
                    if u_bes_index is not None:
                        teacher_bes_override = bool(active_mask[int(u_bes_index)])

            if (
                u_gt_index is not None
                and not teacher_gt_override
                and {"price_e", "price_gas", "p_dem_mw", "pv_mw", "wt_mw", "t_amb_k", "heat_backup_min_needed_mw", "abs_drive_margin_k", "qc_dem_mw", "p_gt_prev_mw"}.issubset(
                    self.observation_index
                )
            ):
                gt_prior = _gt_price_prior_target_np(
                    price_e=float(observations[row_index, self.observation_index["price_e"]]),
                    price_gas=float(observations[row_index, self.observation_index["price_gas"]]),
                    p_dem_mw=float(observations[row_index, self.observation_index["p_dem_mw"]]),
                    pv_mw=float(observations[row_index, self.observation_index["pv_mw"]]),
                    wt_mw=float(observations[row_index, self.observation_index["wt_mw"]]),
                    t_amb_k=float(observations[row_index, self.observation_index["t_amb_k"]]),
                    heat_backup_min_needed_mw=float(
                        observations[row_index, self.observation_index["heat_backup_min_needed_mw"]]
                    ),
                    abs_drive_margin_k=float(
                        observations[row_index, self.observation_index["abs_drive_margin_k"]]
                    ),
                    qc_dem_mw=float(observations[row_index, self.observation_index["qc_dem_mw"]]),
                    p_gt_prev_mw=float(observations[row_index, self.observation_index["p_gt_prev_mw"]]),
                    env_config=self.env_config,
                    gt_off_deadband_ratio=float(self.config.gt_off_deadband_ratio),
                    price_low_threshold=float(self.bes_price_low_threshold),
                    price_high_threshold=float(self.bes_price_high_threshold),
                    u_bes=(
                        float(warm_targets[row_index, int(u_bes_index)])
                        if u_bes_index is not None
                        else 0.0
                    ),
                    u_ech=(
                        float(warm_targets[row_index, int(u_ech_index)])
                        if u_ech_index is not None
                        else 0.0
                    ),
                    carbon_tax=float(
                        observations[row_index, self.observation_index["carbon_tax"]]
                    )
                    if "carbon_tax" in self.observation_index
                    else 0.0,
                )
                if float(gt_prior["opportunity"]) > 0.0:
                    warm_targets[row_index, int(u_gt_index)] = float(gt_prior["target_u_gt"])
                    sample_weight_boost[row_index, 0] += float(gt_prior["opportunity"]) * float(
                        gt_prior.get("mode_weight", 1.0)
                    )
                    gt_prior_rows += 1
                    if str(gt_prior["mode"]) == "on":
                        gt_prior_on_rows += 1
                    elif str(gt_prior["mode"]) == "off":
                        gt_prior_off_rows += 1

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
                "gt_price_prior_row_rate": float(gt_prior_rows / total_rows),
                "gt_price_prior_on_rate": float(gt_prior_on_rows / total_rows),
                "gt_price_prior_off_rate": float(gt_prior_off_rows / total_rows),
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
        gt_candidate_indices: list[int] = []
        bes_candidate_indices: list[int] = []
        tes_candidate_indices: list[int] = []
        season_by_index: dict[int, str] = {}
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
            season_by_index[int(row_index)] = _calendar_season_from_timestamp_value(
                self.train_df["timestamp"].iloc[int(row_index)]
            )
            dim_weight_sum += float(row_stats["dim_weight"])
            delta_mean_sum += float(row_stats["delta_mean"])
            gt_bonus_sum += float(row_stats["gt_bonus"])
            if "u_gt" in self.action_index and active_mask[int(self.action_index["u_gt"])]:
                gt_available_count += 1
                gt_candidate_indices.append(int(row_index))
            if "u_bes" in self.action_index and active_mask[int(self.action_index["u_bes"])]:
                bes_available_count += 1
                bes_candidate_indices.append(int(row_index))
            if "u_tes" in self.action_index and active_mask[int(self.action_index["u_tes"])]:
                tes_available_count += 1
                tes_candidate_indices.append(int(row_index))

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

        requested_samples = min(
            int(self.config.economic_teacher_full_year_warm_start_samples),
            len(candidate_indices),
        )
        action_counts = _allocate_economic_teacher_action_counts(
            requested_total=requested_samples,
            gt_available=len(gt_candidate_indices),
            bes_available=len(bes_candidate_indices),
            tes_available=len(tes_candidate_indices),
        )
        selected_indices: list[int] = []
        selected_index_set: set[int] = set()
        for key, source_indices in (
            ("bes", bes_candidate_indices),
            ("tes", tes_candidate_indices),
            ("gt", gt_candidate_indices),
        ):
            group_selected = _select_temporal_priority_indices_by_season(
                indices=source_indices,
                priorities=priorities,
                season_by_index=season_by_index,
                target_count=int(action_counts.get(key, 0)),
            )
            for idx in group_selected:
                idx_int = int(idx)
                if idx_int in selected_index_set:
                    continue
                selected_indices.append(idx_int)
                selected_index_set.add(idx_int)
        if len(selected_indices) < requested_samples:
            top_up = _select_temporal_priority_indices_by_season(
                indices=[
                    int(idx)
                    for idx in candidate_indices
                    if int(idx) not in selected_index_set
                ],
                priorities=priorities,
                season_by_index=season_by_index,
                target_count=int(requested_samples - len(selected_indices)),
            )
            for idx in top_up:
                idx_int = int(idx)
                if idx_int in selected_index_set:
                    continue
                selected_indices.append(idx_int)
                selected_index_set.add(idx_int)
        selected_indices = sorted({int(idx) for idx in selected_indices})
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
        available_season_counts = {season: 0 for season in ("winter", "spring", "summer", "autumn")}
        sampled_season_counts = {season: 0 for season in ("winter", "spring", "summer", "autumn")}
        for idx in candidate_indices:
            season = str(season_by_index.get(int(idx), "summer"))
            available_season_counts[season] = int(available_season_counts[season] + 1)
        for idx in selected_indices:
            season = str(season_by_index.get(int(idx), "summer"))
            sampled_season_counts[season] = int(sampled_season_counts[season] + 1)
        summary.update(
            {
                "status": "ready",
                "samples": int(len(selected_indices)),
                "available_target_rows": int(len(candidate_indices)),
                "available_gt_count": int(gt_available_count),
                "available_bes_count": int(bes_available_count),
                "available_tes_count": int(tes_available_count),
                "requested_gt_count": int(action_counts["gt"]),
                "requested_bes_count": int(action_counts["bes"]),
                "requested_tes_count": int(action_counts["tes"]),
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
                "available_season_counts": {
                    key: int(value) for key, value in available_season_counts.items()
                },
                "sampled_season_counts": {
                    key: int(value) for key, value in sampled_season_counts.items()
                },
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
        teacher_mask_tensor_shared = teacher_mask_tensor.clone()
        self.actor_teacher_gt_head_warm_start_summary = {
            "enabled": bool("u_gt" in self.action_index),
            "samples": int(obs_tensor.shape[0]),
            "epochs": int(self.config.economic_teacher_full_year_warm_start_epochs),
            "status": "skipped_no_gt_targets",
        }
        u_gt_index = self.action_index.get("u_gt")
        gt_head_linear = None
        if (
            u_gt_index is not None
            and hasattr(self.actor, "net")
            and isinstance(getattr(self.actor, "net"), self.torch.nn.Sequential)
            and len(self.actor.net) > 0
            and isinstance(self.actor.net[-1], self.torch.nn.Linear)
        ):
            gt_head_linear = self.actor.net[-1]
        if u_gt_index is not None and gt_head_linear is not None:
            gt_row_mask = teacher_mask_tensor[:, int(u_gt_index) : int(u_gt_index) + 1] > 0.5
            gt_row_indices = self.torch.nonzero(gt_row_mask.reshape(-1), as_tuple=False).reshape(-1)
            if int(gt_row_indices.numel()) > 0:
                gt_optimizer = AdamW(
                    [gt_head_linear.weight, gt_head_linear.bias],
                    lr=float(self.config.actor_warm_start_lr),
                )
                gt_epoch_losses: list[float] = []
                gt_batch_size = min(int(self.config.actor_warm_start_batch_size), int(gt_row_indices.numel()))
                gt_indices_np = gt_row_indices.detach().cpu().numpy().astype(np.int64, copy=True)
                for _ in range(int(self.config.economic_teacher_full_year_warm_start_epochs)):
                    self.rng.shuffle(gt_indices_np)
                    batch_losses: list[float] = []
                    for start in range(0, int(len(gt_indices_np)), int(gt_batch_size)):
                        batch_indices = gt_indices_np[start : start + int(gt_batch_size)]
                        batch_obs = obs_tensor[batch_indices]
                        batch_gt_target = teacher_action_tensor[
                            batch_indices,
                            int(u_gt_index) : int(u_gt_index) + 1,
                        ]
                        batch_gt_weight = sample_weight_tensor[batch_indices]
                        prediction = self.actor(self._normalize_observation_tensor(batch_obs))
                        prediction = self._apply_abs_cooling_blend_tensor(
                            obs_batch=batch_obs,
                            action_batch=prediction,
                        )
                        gt_loss = (
                            (batch_gt_weight * (prediction[:, int(u_gt_index) : int(u_gt_index) + 1] - batch_gt_target).pow(2)).sum()
                            / batch_gt_weight.sum().clamp_min(1.0)
                        )
                        gt_optimizer.zero_grad(set_to_none=True)
                        gt_loss.backward()
                        if gt_head_linear.weight.grad is not None:
                            keep_weight = self.torch.zeros_like(gt_head_linear.weight.grad)
                            keep_weight[int(u_gt_index) : int(u_gt_index) + 1, :] = 1.0
                            gt_head_linear.weight.grad.mul_(keep_weight)
                        if gt_head_linear.bias.grad is not None:
                            keep_bias = self.torch.zeros_like(gt_head_linear.bias.grad)
                            keep_bias[int(u_gt_index) : int(u_gt_index) + 1] = 1.0
                            gt_head_linear.bias.grad.mul_(keep_bias)
                        gt_optimizer.step()
                        batch_losses.append(float(gt_loss.detach().cpu().item()))
                    gt_epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
                teacher_mask_tensor_shared[:, int(u_gt_index) : int(u_gt_index) + 1] = 0.0
                self.actor_teacher_gt_head_warm_start_summary = {
                    "enabled": True,
                    "samples": int(gt_row_indices.numel()),
                    "epochs": int(self.config.economic_teacher_full_year_warm_start_epochs),
                    "status": "applied",
                    "batch_size": int(gt_batch_size),
                    "lr": float(self.config.actor_warm_start_lr),
                    "loss_first": float(gt_epoch_losses[0]) if gt_epoch_losses else 0.0,
                    "loss_last": float(gt_epoch_losses[-1]) if gt_epoch_losses else 0.0,
                }
            else:
                self.actor_teacher_gt_head_warm_start_summary = {
                    "enabled": True,
                    "samples": 0,
                    "epochs": int(self.config.economic_teacher_full_year_warm_start_epochs),
                    "status": "skipped_no_gt_targets",
                }
        elif u_gt_index is not None:
            self.actor_teacher_gt_head_warm_start_summary = {
                "enabled": True,
                "samples": 0,
                "epochs": int(self.config.economic_teacher_full_year_warm_start_epochs),
                "status": "skipped_missing_linear_head",
            }
        safe_preserve_weight_tensor = self.torch.as_tensor(
            _economic_teacher_safe_preserve_weights_np(action_keys=self.action_keys).reshape(1, -1),
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
                batch_teacher_mask = teacher_mask_tensor_shared[batch_indices]
                batch_sample_weight = sample_weight_tensor[batch_indices]
                prediction = self.actor(self._normalize_observation_tensor(batch_obs))
                prediction = self._apply_abs_cooling_blend_tensor(
                    obs_batch=batch_obs,
                    action_batch=prediction,
                )
                batch_safe_mask = self.torch.clamp(1.0 - batch_teacher_mask, 0.0, 1.0)
                weighted_safe_mask = batch_safe_mask * safe_preserve_weight_tensor
                safe_loss = (
                    ((prediction - batch_base_target).pow(2) * weighted_safe_mask).sum(
                        dim=1,
                        keepdim=True,
                    )
                    / weighted_safe_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                ).mean()
                weighted_teacher_mask = (
                    batch_teacher_mask
                    * self.economic_teacher_action_weight
                    * self.economic_teacher_mismatch_focus_weight
                )
                teacher_sq = (
                    ((prediction - batch_teacher_target).pow(2) * weighted_teacher_mask).sum(
                        dim=1,
                        keepdim=True,
                    )
                    / weighted_teacher_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
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
                "safe_preserve_weights": {
                    str(key): float(value)
                    for key, value in zip(
                        self.action_keys,
                        _economic_teacher_safe_preserve_weights_np(action_keys=self.action_keys),
                    )
                },
                "surrogate_mismatch_focus_weights": {
                    str(key): float(value)
                    for key, value in zip(
                        self.action_keys,
                        self.economic_teacher_mismatch_focus_weight_np.reshape(-1),
                    )
                },
            }
        )
        self.actor_teacher_full_year_warm_start_summary = final_summary

    def _collect_gt_full_year_warm_start_dataset(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        empty_obs = np.zeros((0, len(self.observation_keys)), dtype=np.float32)
        empty_actions = np.zeros((0, len(self.action_keys)), dtype=np.float32)
        empty_targets = np.zeros((0, 1), dtype=np.float32)
        empty_weights = np.zeros((0, 1), dtype=np.float32)
        summary = {
            "enabled": bool(
                int(self.config.economic_gt_full_year_warm_start_samples) > 0
                and int(self.config.economic_gt_full_year_warm_start_epochs) > 0
            ),
            "requested_samples": int(self.config.economic_gt_full_year_warm_start_samples),
            "epochs": int(self.config.economic_gt_full_year_warm_start_epochs),
            "u_weight": float(self.config.economic_gt_full_year_warm_start_u_weight),
            "status": "disabled",
            "samples": 0,
        }
        if not bool(summary["enabled"]):
            return empty_obs, empty_actions, empty_targets, empty_weights, summary
        required_obs = {
            "price_e",
            "price_gas",
            "p_dem_mw",
            "pv_mw",
            "wt_mw",
            "t_amb_k",
            "heat_backup_min_needed_mw",
            "abs_drive_margin_k",
            "qc_dem_mw",
            "p_gt_prev_mw",
        }
        required_actions = {"u_gt", "u_bes", "u_ech"}
        if (
            not required_obs.issubset(self.observation_index)
            or not required_actions.issubset(self.action_index)
        ):
            summary["status"] = "missing_gt_features"
            return empty_obs, empty_actions, empty_targets, empty_weights, summary

        expert_policy, expert_policy_info = self._build_expert_policy()
        rollout = self._rollout_expert_prefill_episode(
            expert_policy=expert_policy,
            expert_policy_info=expert_policy_info,
            episode_df=self.train_df.reset_index(drop=True),
            episode_seed=int(self.config.seed) + 920_000,
            max_steps=int(len(self.train_df)),
            abs_exec_threshold=float(self.config.expert_prefill_abs_exec_threshold),
            collect_teacher_targets=bool(self.economic_teacher_policy is not None),
        )
        transitions = tuple(rollout.get("transitions", ()))
        summary.update(
            {
                "status": "no_rollout",
                "teacher": dict(expert_policy_info),
                "rollout_steps": int(len(transitions)),
                "economic_teacher_available": bool(self.economic_teacher_policy is not None),
            }
        )
        if not transitions:
            return empty_obs, empty_actions, empty_targets, empty_weights, summary

        u_gt_index = int(self.action_index["u_gt"])
        u_bes_index = int(self.action_index["u_bes"])
        u_ech_index = int(self.action_index["u_ech"])
        teacher_on_indices: list[int] = []
        teacher_off_indices: list[int] = []
        prior_on_indices: list[int] = []
        prior_off_indices: list[int] = []
        idle_indices: list[int] = []
        target_u_rows = np.zeros((len(transitions), 1), dtype=np.float32)
        priorities = np.zeros((len(transitions),), dtype=np.float32)
        sample_weights = np.ones((len(transitions), 1), dtype=np.float32)
        sampled_modes_all: list[str] = []
        sampled_sources_all: list[str] = []
        season_by_index: dict[int, str] = {}
        train_timestamps = self.train_df["timestamp"] if "timestamp" in self.train_df.columns else None
        teacher_target_count = 0
        teacher_target_on_count = 0
        teacher_target_off_count = 0
        prior_fallback_count = 0
        prior_fallback_on_count = 0
        prior_fallback_off_count = 0
        prior_fallback_idle_count = 0
        for idx, transition in enumerate(transitions):
            if train_timestamps is not None and int(idx) < len(train_timestamps):
                season_by_index[int(idx)] = _calendar_season_from_timestamp_value(
                    train_timestamps.iloc[int(idx)]
                )
            else:
                season_by_index[int(idx)] = "summer"
            obs_vector = np.asarray(transition["obs"], dtype=np.float32).reshape(-1)
            action_exec = np.asarray(transition["action_exec"], dtype=np.float32).reshape(-1)
            teacher_action_exec = np.asarray(
                transition.get("teacher_action_exec", np.zeros((len(self.action_keys),), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            teacher_action_mask = np.asarray(
                transition.get("teacher_action_mask", np.zeros((len(self.action_keys),), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            teacher_gt_available = bool(transition.get("teacher_available", False)) and bool(
                float(teacher_action_mask[u_gt_index]) > 0.5
            )
            prior = _gt_price_prior_target_np(
                price_e=float(obs_vector[self.observation_index["price_e"]]),
                price_gas=float(obs_vector[self.observation_index["price_gas"]]),
                p_dem_mw=float(obs_vector[self.observation_index["p_dem_mw"]]),
                pv_mw=float(obs_vector[self.observation_index["pv_mw"]]),
                wt_mw=float(obs_vector[self.observation_index["wt_mw"]]),
                t_amb_k=float(obs_vector[self.observation_index["t_amb_k"]]),
                heat_backup_min_needed_mw=float(
                    obs_vector[self.observation_index["heat_backup_min_needed_mw"]]
                ),
                abs_drive_margin_k=float(obs_vector[self.observation_index["abs_drive_margin_k"]]),
                qc_dem_mw=float(obs_vector[self.observation_index["qc_dem_mw"]]),
                p_gt_prev_mw=float(obs_vector[self.observation_index["p_gt_prev_mw"]]),
                env_config=self.env_config,
                gt_off_deadband_ratio=float(self.config.gt_off_deadband_ratio),
                price_low_threshold=float(self.bes_price_low_threshold),
                price_high_threshold=float(self.bes_price_high_threshold),
                u_bes=float(action_exec[u_bes_index]),
                u_ech=float(action_exec[u_ech_index]),
                carbon_tax=float(obs_vector[self.observation_index["carbon_tax"]])
                if "carbon_tax" in self.observation_index
                else 0.0,
            )
            teacher_gt_consistent = teacher_gt_available and _gt_teacher_direction_consistent_np(
                teacher_u_gt=float(teacher_action_exec[u_gt_index]),
                prior_mode=str(prior["mode"]),
            )
            if teacher_gt_consistent:
                row_stats = self._compute_economic_teacher_full_year_row_priority(
                    obs_vector=obs_vector,
                    action_exec=action_exec,
                    teacher_action_exec=teacher_action_exec,
                    teacher_action_mask=teacher_action_mask,
                )
                target_u = float(np.clip(teacher_action_exec[u_gt_index], -1.0, 1.0))
                mode = "off" if target_u <= -0.95 else "on"
                source = "teacher"
                target_u_rows[idx, 0] = target_u
                priorities[idx] = max(0.0, float(row_stats["priority"]))
                sample_weights[idx, 0] = max(1.0, float(row_stats["sample_weight"]))
                teacher_target_count += 1
                if mode == "on":
                    teacher_on_indices.append(int(idx))
                    teacher_target_on_count += 1
                else:
                    teacher_off_indices.append(int(idx))
                    teacher_target_off_count += 1
            else:
                mode = str(prior["mode"])
                source = "prior"
                target_u_rows[idx, 0] = float(prior["target_u_gt"])
                if mode == "on":
                    priorities[idx] = (
                        float(prior["opportunity"])
                        * float(prior.get("mode_weight", 1.0))
                        * (0.25 + 0.75 * float(prior.get("target_load_ratio", 0.0)))
                    )
                elif mode == "off":
                    priorities[idx] = (
                        float(prior["opportunity"])
                        * float(prior.get("mode_weight", 1.0))
                        * 1.25
                    )
                else:
                    priorities[idx] = 0.0
                sample_weights[idx, 0] = (
                    1.0
                    + priorities[idx]
                    + float(float(target_u_rows[idx, 0]) <= -0.95)
                )
                prior_fallback_count += 1
                if mode == "on":
                    prior_on_indices.append(int(idx))
                    prior_fallback_on_count += 1
                elif mode == "off":
                    prior_off_indices.append(int(idx))
                    prior_fallback_off_count += 1
                else:
                    idle_indices.append(int(idx))
                    prior_fallback_idle_count += 1
            sampled_modes_all.append(mode)
            sampled_sources_all.append(source)

        requested_samples = min(
            int(self.config.economic_gt_full_year_warm_start_samples),
            int(len(transitions)),
        )
        total_on_available = len(teacher_on_indices) + len(prior_on_indices)
        total_off_available = len(teacher_off_indices) + len(prior_off_indices)
        mode_counts = _allocate_gt_warm_start_mode_counts(
            requested_total=requested_samples,
            on_available=total_on_available,
            off_available=total_off_available,
            idle_available=len(idle_indices),
        )
        desired_on = int(mode_counts["on"])
        desired_off = int(mode_counts["off"])
        desired_idle = int(mode_counts["idle"])
        selected_indices: list[int] = []
        teacher_selected_on = _select_temporal_priority_indices_by_season(
            indices=teacher_on_indices,
            priorities=priorities,
            season_by_index=season_by_index,
            target_count=min(desired_on, len(teacher_on_indices)),
        )
        selected_indices.extend(teacher_selected_on)
        teacher_selected_off = _select_temporal_priority_indices_by_season(
            indices=teacher_off_indices,
            priorities=priorities,
            season_by_index=season_by_index,
            target_count=min(desired_off, len(teacher_off_indices)),
        )
        selected_indices.extend(teacher_selected_off)
        remaining_on = max(0, desired_on - len(teacher_selected_on))
        remaining_off = max(0, desired_off - len(teacher_selected_off))
        selected_indices.extend(
            _select_temporal_priority_indices_by_season(
                indices=prior_on_indices,
                priorities=priorities,
                season_by_index=season_by_index,
                target_count=remaining_on,
            )
        )
        selected_indices.extend(
            _select_temporal_priority_indices_by_season(
                indices=prior_off_indices,
                priorities=priorities,
                season_by_index=season_by_index,
                target_count=remaining_off,
            )
        )
        selected_indices.extend(
            _select_temporal_priority_indices_by_season(
                indices=idle_indices,
                priorities=priorities,
                season_by_index=season_by_index,
                target_count=desired_idle,
            )
        )
        selected_set = {int(idx) for idx in selected_indices}
        remaining_teacher_indices = [
            idx
            for idx in range(len(transitions))
            if int(idx) not in selected_set and sampled_sources_all[int(idx)] == "teacher"
        ]
        if len(selected_set) < requested_samples:
            fill_teacher_indices = _select_temporal_priority_indices_by_season(
                indices=remaining_teacher_indices,
                priorities=priorities,
                season_by_index=season_by_index,
                target_count=int(requested_samples - len(selected_set)),
            )
            selected_set.update(int(idx) for idx in fill_teacher_indices)
        remaining_indices = [idx for idx in range(len(transitions)) if int(idx) not in selected_set]
        if len(selected_set) < requested_samples:
            fill_indices = _select_temporal_priority_indices_by_season(
                indices=remaining_indices,
                priorities=priorities,
                season_by_index=season_by_index,
                target_count=int(requested_samples - len(selected_set)),
            )
            selected_set.update(int(idx) for idx in fill_indices)
        selected_indices = sorted(selected_set)
        if not selected_indices:
            summary["status"] = "no_selected_samples"
            return empty_obs, empty_actions, empty_targets, empty_weights, summary

        obs_rows: list[np.ndarray] = []
        action_exec_rows: list[np.ndarray] = []
        sampled_mode_counts = {"on": 0, "off": 0, "idle": 0}
        sampled_source_counts = {"teacher": 0, "prior": 0}
        replay_augmented_steps = 0
        for idx in selected_indices:
            transition = transitions[int(idx)]
            mode = str(sampled_modes_all[int(idx)])
            if mode not in sampled_mode_counts:
                mode = "idle"
            sampled_mode_counts[mode] += 1
            source = str(sampled_sources_all[int(idx)])
            if source not in sampled_source_counts:
                source = "prior"
            sampled_source_counts[source] += 1
            obs_rows.append(np.asarray(transition["obs"], dtype=np.float32).copy())
            action_exec_rows.append(np.asarray(transition["action_exec"], dtype=np.float32).copy())
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
        selected_weights = np.asarray(sample_weights[selected_indices_np], dtype=np.float32).reshape(-1, 1)
        available_season_counts = {season: 0 for season in ("winter", "spring", "summer", "autumn")}
        sampled_season_counts = {season: 0 for season in ("winter", "spring", "summer", "autumn")}
        for idx in range(len(transitions)):
            season = str(season_by_index.get(int(idx), "summer"))
            available_season_counts[season] = int(available_season_counts[season] + 1)
        for idx in selected_indices:
            season = str(season_by_index.get(int(idx), "summer"))
            sampled_season_counts[season] = int(sampled_season_counts[season] + 1)
        summary.update(
            {
                "status": "ready",
                "samples": int(len(selected_indices)),
                "teacher_target_count": int(teacher_target_count),
                "teacher_target_on_count": int(teacher_target_on_count),
                "teacher_target_off_count": int(teacher_target_off_count),
                "prior_fallback_count": int(prior_fallback_count),
                "prior_fallback_on_count": int(prior_fallback_on_count),
                "prior_fallback_off_count": int(prior_fallback_off_count),
                "prior_fallback_idle_count": int(prior_fallback_idle_count),
                "available_on_count": int(total_on_available),
                "available_off_count": int(total_off_available),
                "available_idle_count": int(len(idle_indices)),
                "sampled_on_count": int(sampled_mode_counts["on"]),
                "sampled_off_count": int(sampled_mode_counts["off"]),
                "sampled_idle_count": int(sampled_mode_counts["idle"]),
                "sampled_teacher_count": int(sampled_source_counts["teacher"]),
                "sampled_prior_count": int(sampled_source_counts["prior"]),
                "sampled_teacher_rate": float(
                    sampled_source_counts["teacher"] / max(1, len(selected_indices))
                ),
                "sampled_priority_mean": float(
                    np.asarray(priorities[selected_indices_np], dtype=np.float32).mean()
                ),
                "sampled_target_abs_mean": float(
                    np.asarray(np.abs(target_u_rows[selected_indices_np]), dtype=np.float32).mean()
                ),
                "sample_weight_mean": float(selected_weights.mean()),
                "replay_augmented_steps": int(replay_augmented_steps),
                "price_low_threshold": float(self.bes_price_low_threshold),
                "price_high_threshold": float(self.bes_price_high_threshold),
                "available_season_counts": {
                    key: int(value) for key, value in available_season_counts.items()
                },
                "sampled_season_counts": {
                    key: int(value) for key, value in sampled_season_counts.items()
                },
            }
        )
        return (
            np.asarray(obs_rows, dtype=np.float32),
            np.asarray(action_exec_rows, dtype=np.float32),
            np.asarray(target_u_rows[selected_indices_np], dtype=np.float32),
            np.asarray(selected_weights, dtype=np.float32),
            summary,
        )

    def _warm_start_actor_from_gt_full_year(self) -> None:
        (
            observations,
            action_exec_targets,
            target_u_gt,
            sample_weights,
            summary,
        ) = self._collect_gt_full_year_warm_start_dataset()
        if observations.size == 0 or action_exec_targets.size == 0 or target_u_gt.size == 0:
            self.actor_gt_warm_start_summary = dict(summary)
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
            target_u_gt,
            dtype=self.torch.float32,
            device=self.device,
        )
        sample_weight_tensor = self.torch.as_tensor(
            sample_weights,
            dtype=self.torch.float32,
            device=self.device,
        )
        preserve_scale = _economic_teacher_safe_preserve_weights_np(
            action_keys=self.action_keys,
        ).reshape(1, -1)
        u_gt_index = int(self.action_index["u_gt"])
        preserve_scale[:, u_gt_index] = 0.0
        preserve_scale_tensor = self.torch.as_tensor(
            preserve_scale,
            dtype=self.torch.float32,
            device=self.device,
        )
        batch_size = min(int(self.config.actor_warm_start_batch_size), int(obs_tensor.shape[0]))
        indices = np.arange(int(obs_tensor.shape[0]), dtype=np.int64)
        epoch_losses: list[float] = []
        epoch_gt_losses: list[float] = []

        for _ in range(int(self.config.economic_gt_full_year_warm_start_epochs)):
            self.rng.shuffle(indices)
            batch_losses: list[float] = []
            batch_gt_losses: list[float] = []
            for start in range(0, int(len(indices)), int(batch_size)):
                batch_indices = indices[start : start + int(batch_size)]
                batch_obs = obs_tensor[batch_indices]
                batch_base_target = base_action_tensor[batch_indices]
                batch_target_u = target_u_tensor[batch_indices]
                batch_weight = sample_weight_tensor[batch_indices]
                prediction = self.actor(self._normalize_observation_tensor(batch_obs))
                prediction = self._apply_abs_cooling_blend_tensor(
                    obs_batch=batch_obs,
                    action_batch=prediction,
                )
                safe_loss = (
                    ((prediction - batch_base_target).pow(2) * preserve_scale_tensor).mean(
                        dim=1,
                        keepdim=True,
                    )
                ).mean()
                gt_prediction = prediction[:, u_gt_index : u_gt_index + 1]
                gt_loss = (
                    batch_weight * (gt_prediction - batch_target_u).pow(2)
                ).sum() / batch_weight.sum().clamp_min(1.0)
                loss = safe_loss + (
                    float(self.config.economic_gt_full_year_warm_start_u_weight) * gt_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))
                batch_gt_losses.append(float(gt_loss.detach().cpu().item()))
            epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
            epoch_gt_losses.append(float(np.mean(batch_gt_losses)) if batch_gt_losses else 0.0)

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
                "gt_loss_first": float(epoch_gt_losses[0]) if epoch_gt_losses else 0.0,
                "gt_loss_last": float(epoch_gt_losses[-1]) if epoch_gt_losses else 0.0,
                "target_positive_rate": float(
                    (target_u_tensor > 0.0).float().mean().detach().cpu().item()
                ),
                "target_off_rate": float(
                    (target_u_tensor < -0.95).float().mean().detach().cpu().item()
                ),
                "target_abs_mean": float(target_u_tensor.abs().mean().detach().cpu().item()),
            }
        )
        self.actor_gt_warm_start_summary = final_summary

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
        transition_seasons: list[str] = []
        transition_local_step_indices: list[int] = []
        source_rollouts: dict[str, dict[str, Any]] = {}
        safe_action_exec_by_step: dict[int, np.ndarray] = {}
        train_timestamps = self.train_df["timestamp"] if "timestamp" in self.train_df.columns else None

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
            for local_step_index, transition in enumerate(transitions):
                all_transitions.append(dict(transition))
                transition_source_labels.append(str(source_label))
                transition_anchor_weights.append(float(source_anchor_weight))
                transition_local_step_indices.append(int(local_step_index))
                if str(source_label) == "safe":
                    safe_action_exec_by_step[int(local_step_index)] = np.asarray(
                        transition["action_exec"],
                        dtype=np.float32,
                    ).copy()
                if train_timestamps is not None and int(local_step_index) < len(train_timestamps):
                    transition_seasons.append(
                        _calendar_season_from_timestamp_value(
                            train_timestamps.iloc[int(local_step_index)]
                        )
                    )
                else:
                    transition_seasons.append("summer")
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
                source_anchor_weight=float(
                    self.config.economic_bes_warm_start_economic_anchor_weight
                ),
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
                source_anchor_weight=float(
                    self.config.economic_bes_warm_start_fallback_anchor_weight
                ),
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
        q_total_cooling_cap_mw = max(
            _NORM_EPS,
            float(self.env_config.q_abs_cool_cap_mw) + float(self.env_config.q_ech_cap_mw),
        )
        opportunity_scores = np.zeros((total_available,), dtype=np.float32)
        selection_priorities = np.zeros((total_available,), dtype=np.float32)
        target_u_rows = np.zeros((total_available, 1), dtype=np.float32)
        economic_teacher_override_flags = np.zeros((total_available,), dtype=np.float32)
        cooling_guard_flags = np.zeros((total_available,), dtype=np.float32)
        economic_teacher_override_count = 0
        economic_teacher_idle_override_count = 0
        economic_cooling_guard_count = 0
        season_by_index = {
            int(idx): str(transition_seasons[int(idx)])
            for idx in range(min(total_available, len(transition_seasons)))
        }
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
            cooling_guard_active = _bes_warm_start_cooling_guard_active_np(
                abs_drive_margin_k=float(obs_vector[self.observation_index["abs_drive_margin_k"]]),
                qc_dem_mw=float(obs_vector[self.observation_index["qc_dem_mw"]]),
                q_total_cooling_cap_mw=float(q_total_cooling_cap_mw),
                abs_margin_guard_k=float(self.config.dual_abs_margin_k),
                qc_ratio_guard=float(self.config.dual_qc_ratio_th),
            )
            cooling_guard_flags[idx] = 1.0 if cooling_guard_active else 0.0
            original_source_label = str(transition_source_labels[int(idx)])
            guarded_source_label = original_source_label
            if cooling_guard_active and original_source_label.strip().lower().startswith("economic"):
                guarded_source_label = "cooling_guard_safe"
                economic_cooling_guard_count += 1
            target_choice = _select_bes_full_year_target_np(
                source_label=guarded_source_label,
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
                if (
                    original_source_label.strip().lower().startswith("economic")
                    and not cooling_guard_active
                )
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
            if cooling_guard_active and original_source_label.strip().lower().startswith("economic"):
                base_priority *= 0.25
            opportunity_scores[idx] = float(base_priority)
            selection_priorities[idx] = _bes_full_year_selection_priority_np(
                base_priority=float(base_priority),
                source_label=guarded_source_label,
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
            teacher_selected = _select_temporal_priority_indices_by_season(
                indices=teacher_mode_indices[mode]["teacher"],
                priorities=selection_priorities,
                season_by_index=season_by_index,
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
                _select_temporal_priority_indices_by_season(
                    indices=remaining_economic_indices,
                    priorities=selection_priorities,
                    season_by_index=season_by_index,
                    target_count=int(source_targets["economic"]),
                )
            )
            selected_indices.extend(
                _select_temporal_priority_indices_by_season(
                    indices=remaining_other_indices,
                    priorities=selection_priorities,
                    season_by_index=season_by_index,
                    target_count=int(source_targets["other"]),
                )
            )
        selected_set = {int(idx) for idx in selected_indices}
        if len(selected_set) < requested_samples:
            remaining_indices = [
                idx for idx in range(total_available) if int(idx) not in selected_set
            ]
            fill_indices = _select_temporal_priority_indices_by_season(
                indices=remaining_indices,
                priorities=selection_priorities,
                season_by_index=season_by_index,
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
        safe_base_override_count = 0
        replay_augmented_steps = 0
        for idx in selected_indices:
            transition = all_transitions[int(idx)]
            obs_rows.append(np.asarray(transition["obs"], dtype=np.float32).copy())
            source_label = str(transition_source_labels[int(idx)])
            local_step_index = int(transition_local_step_indices[int(idx)])
            base_action_exec = np.asarray(
                transition["action_exec"],
                dtype=np.float32,
            ).copy()
            if source_label != "safe" and local_step_index in safe_action_exec_by_step:
                base_action_exec = np.asarray(
                    safe_action_exec_by_step[int(local_step_index)],
                    dtype=np.float32,
                ).copy()
                safe_base_override_count += 1
            action_exec_rows.append(base_action_exec)
            anchor_weight_rows.append(
                np.asarray([float(transition_anchor_weights[int(idx)])], dtype=np.float32)
            )
            sampled_modes[str(prior_modes[int(idx)])] += 1
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
        available_season_counts = {season: 0 for season in ("winter", "spring", "summer", "autumn")}
        sampled_season_counts = {season: 0 for season in ("winter", "spring", "summer", "autumn")}
        for idx in range(total_available):
            season = str(season_by_index.get(int(idx), "summer"))
            available_season_counts[season] = int(available_season_counts[season] + 1)
        for idx in selected_indices:
            season = str(season_by_index.get(int(idx), "summer"))
            sampled_season_counts[season] = int(sampled_season_counts[season] + 1)
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
                "cooling_guard_count": int(np.asarray(cooling_guard_flags, dtype=np.float32).sum()),
                "economic_cooling_guard_count": int(economic_cooling_guard_count),
                "sampled_cooling_guard_count": int(
                    np.asarray(cooling_guard_flags[selected_indices_np], dtype=np.float32).sum()
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
                "safe_base_override_count": int(safe_base_override_count),
                "replay_augmented_steps": int(replay_augmented_steps),
                "price_low_threshold": float(self.bes_price_low_threshold),
                "price_high_threshold": float(self.bes_price_high_threshold),
                "available_season_counts": {
                    key: int(value) for key, value in available_season_counts.items()
                },
                "sampled_season_counts": {
                    key: int(value) for key, value in sampled_season_counts.items()
                },
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

    def _compute_gt_price_prior_terms(self, *, obs_batch, action_exec_batch):
        required_obs = {
            "p_dem_mw",
            "pv_mw",
            "wt_mw",
            "price_e",
            "price_gas",
            "t_amb_k",
            "heat_backup_min_needed_mw",
            "abs_drive_margin_k",
            "qc_dem_mw",
            "p_gt_prev_mw",
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
        heat_backup_index = int(self.observation_index["heat_backup_min_needed_mw"])
        abs_margin_index = int(self.observation_index["abs_drive_margin_k"])
        qc_index = int(self.observation_index["qc_dem_mw"])
        p_gt_prev_index = int(self.observation_index["p_gt_prev_mw"])
        u_gt_index = int(self.action_index["u_gt"])
        u_bes_index = int(self.action_index["u_bes"])
        u_ech_index = int(self.action_index["u_ech"])
        price_e = obs_batch[:, price_e_index : price_e_index + 1]
        price_gas = obs_batch[:, price_gas_index : price_gas_index + 1]
        carbon_tax = self.torch.zeros_like(price_e)
        if "carbon_tax" in self.observation_index:
            carbon_tax_index = int(self.observation_index["carbon_tax"])
            carbon_tax = obs_batch[:, carbon_tax_index : carbon_tax_index + 1].clamp_min(0.0)
        low_price = float(self.bes_price_low_threshold)
        high_price = max(low_price + _NORM_EPS, float(self.bes_price_high_threshold))
        mid_price = 0.5 * (low_price + high_price)
        charge_span = max(_NORM_EPS, mid_price - low_price)
        discharge_span = max(_NORM_EPS, high_price - mid_price)
        grid_price_low = self.torch.clamp((mid_price - price_e) / charge_span, 0.0, 1.0)
        grid_price_high = self.torch.clamp((price_e - mid_price) / discharge_span, 0.0, 1.0)
        u_bes_exec = action_exec_batch[:, u_bes_index : u_bes_index + 1].clamp(-1.0, 1.0)
        u_ech_exec = action_exec_batch[:, u_ech_index : u_ech_index + 1].clamp(0.0, 1.0)
        p_gt_cap_mw = max(_NORM_EPS, float(self.env_config.p_gt_cap_mw))
        gt_min_output_mw = max(0.0, float(self.env_config.gt_min_output_mw))
        gt_min_ratio = float(
            np.clip(
                gt_min_output_mw / p_gt_cap_mw,
                0.0,
                1.0,
            )
        )
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
        q_ech_proxy_mw = u_ech_exec * float(self.env_config.q_ech_cap_mw)
        ech_cop = (
            float(self.env_config.cop_nominal)
            - 0.03 * (obs_batch[:, t_amb_index : t_amb_index + 1] - 298.15)
        ).clamp(
            min=float(self.env_config.cop_nominal) * float(self.env_config.ech_cop_partload_min_fraction),
            max=float(self.env_config.cop_nominal),
        )
        p_ech_proxy_mw = q_ech_proxy_mw / ech_cop.clamp_min(_NORM_EPS)
        net_grid_need_ratio = self.torch.clamp(
            (
                obs_batch[:, p_dem_index : p_dem_index + 1]
                + p_ech_proxy_mw
                + p_bes_charge_proxy_mw
                - obs_batch[:, pv_index : pv_index + 1]
                - obs_batch[:, wt_index : wt_index + 1]
                - p_bes_discharge_proxy_mw
            )
            / p_gt_cap_mw,
            0.0,
            1.0,
        )
        heat_support_need = self.torch.clamp(
            obs_batch[:, heat_backup_index : heat_backup_index + 1]
            / max(_NORM_EPS, float(self.env_config.q_boiler_cap_mw)),
            0.0,
            1.0,
        )
        qc_need_ratio = self.torch.clamp(
            obs_batch[:, qc_index : qc_index + 1]
            / max(
                _NORM_EPS,
                max(float(self.env_config.q_abs_cool_cap_mw), float(self.env_config.q_ech_cap_mw)),
            ),
            0.0,
            1.0,
        )
        abs_ready = self.torch.sigmoid(
            obs_batch[:, abs_margin_index : abs_margin_index + 1]
            / max(_NORM_EPS, float(self.env_config.abs_gate_scale_k))
        )
        cool_support_need = self.torch.clamp(
            qc_need_ratio * self.torch.clamp(abs_ready, min=0.35, max=1.0),
            0.0,
            1.0,
        )
        prev_gt_ratio = self.torch.clamp(
            obs_batch[:, p_gt_prev_index : p_gt_prev_index + 1] / p_gt_cap_mw,
            0.0,
            1.0,
        )
        eta_ref_ratio = self.torch.clamp(
            self.torch.maximum(
                prev_gt_ratio.add(net_grid_need_ratio).mul(0.5),
                self.torch.full_like(prev_gt_ratio, float(gt_min_ratio)),
            ),
            0.0,
            1.0,
        )
        eta_ref = float(self.env_config.gt_eta_min) + (
            float(self.env_config.gt_eta_max) - float(self.env_config.gt_eta_min)
        ) * eta_ref_ratio
        gt_dispatch_basis_mw = max(gt_min_output_mw, 0.25 * p_gt_cap_mw, _NORM_EPS)
        startup_basis_mwh = max(
            _NORM_EPS,
            float(getattr(self.env_config, "dt_hours", 0.25)) * gt_dispatch_basis_mw,
        )
        startup_off_factor = 1.0 - self.torch.clamp(
            obs_batch[:, p_gt_prev_index : p_gt_prev_index + 1] / gt_dispatch_basis_mw,
            0.0,
            1.0,
        )
        gt_marginal_cost = (
            price_gas / eta_ref.clamp_min(_NORM_EPS)
            + float(getattr(self.env_config, "gt_om_var_cost_per_mwh", 0.0))
            + carbon_tax
            * float(getattr(self.env_config, "gt_emission_ton_per_mwh_th", 0.0))
            / eta_ref.clamp_min(_NORM_EPS)
            + startup_off_factor
            * (
                float(getattr(self.env_config, "gt_start_cost", 0.0))
                + float(getattr(self.env_config, "gt_cycle_cost", 0.0))
            )
            / startup_basis_mwh
        )
        price_advantage = self.torch.clamp(
            (price_e - gt_marginal_cost)
            / self.torch.maximum(price_e, gt_marginal_cost).clamp_min(_NORM_EPS),
            -1.0,
            1.0,
        )
        market_commit = self.torch.maximum(grid_price_high, price_advantage.clamp_min(0.0))
        market_off = self.torch.maximum(grid_price_low, (-price_advantage).clamp_min(0.0))
        net_grid_absorb_ratio = self.torch.clamp(
            (net_grid_need_ratio - 0.75 * float(gt_min_ratio))
            / max(_NORM_EPS, 1.0 - 0.75 * float(gt_min_ratio)),
            0.0,
            1.0,
        )
        cogen_support_need = self.torch.maximum(
            heat_support_need,
            0.55 * cool_support_need,
        )
        support_commit_floor = self.torch.clamp(
            (0.55 * heat_support_need + 0.25 * cool_support_need)
            * (0.20 + 0.80 * price_advantage.clamp_min(0.0)),
            0.0,
            1.0,
        )
        commit_score = self.torch.clamp(
            self.torch.maximum(
                market_commit * self.torch.maximum(net_grid_absorb_ratio, 0.40 * cogen_support_need),
                support_commit_floor,
            ),
            0.0,
            1.0,
        )
        low_load_relief = self.torch.clamp(
            (float(gt_min_ratio) - net_grid_need_ratio) / max(_NORM_EPS, float(gt_min_ratio)),
            0.0,
            1.0,
        )
        off_score = self.torch.clamp(
            (market_off + 0.35 * low_load_relief)
            * (1.0 - self.torch.maximum(net_grid_absorb_ratio, 0.75 * cogen_support_need)).clamp_min(0.0)
            * (1.0 - 0.15 * prev_gt_ratio).clamp_min(0.0),
            0.0,
            1.0,
        )
        opportunity = self.torch.maximum(commit_score, off_score)
        base_active_mask = (opportunity > float(_GT_PRIOR_MIN_OPPORTUNITY)).to(dtype=obs_batch.dtype)
        commit_margin = (commit_score - off_score).clamp_min(0.0)
        on_margin_threshold = max(float(_GT_PRIOR_MIN_OPPORTUNITY), 0.5 * float(gt_min_ratio))
        on_mask = (
            (commit_score > off_score).to(dtype=obs_batch.dtype)
            * (commit_margin > on_margin_threshold).to(dtype=obs_batch.dtype)
            * base_active_mask
        )
        off_mask = (
            (1.0 - on_mask)
            * (off_score >= float(_GT_PRIOR_MIN_OPPORTUNITY)).to(dtype=obs_batch.dtype)
            * base_active_mask
        )
        decision_mask = (on_mask + off_mask).clamp(0.0, 1.0)
        on_signal = self.torch.maximum(
            self.torch.maximum(net_grid_absorb_ratio, cogen_support_need),
            market_commit,
        )
        dispatch_floor_ratio = self.torch.clamp(
            float(gt_min_ratio) + 0.12 * on_signal,
            min=float(gt_min_ratio),
            max=max(float(gt_min_ratio), 0.35),
        )
        dispatch_cap_ratio = self.torch.clamp(
            0.25 + 0.75 * on_signal,
            min=0.0,
            max=1.0,
        )
        dispatch_cap_ratio = self.torch.maximum(dispatch_cap_ratio, dispatch_floor_ratio)
        dispatch_strength = self.torch.clamp(commit_score * on_signal, 0.0, 1.0)
        target_load_ratio = self.torch.clamp(
            dispatch_floor_ratio
            + (dispatch_cap_ratio - dispatch_floor_ratio) * dispatch_strength,
            0.0,
            1.0,
        )
        target_u_gt = 2.0 * target_load_ratio - 1.0
        target_u_gt = (
            (-1.0) * off_mask
            + target_u_gt.clamp(-1.0, 1.0) * (1.0 - off_mask)
        ) * decision_mask
        mode_on = on_mask
        mode_off = off_mask.to(dtype=obs_batch.dtype)
        mode_weight = (
            (1.0 + 0.75 * off_score) * mode_off
            + (1.0 + 0.25 * dispatch_strength + 0.35 * commit_margin) * mode_on
        ) * decision_mask
        return {
            "target_u_gt": target_u_gt,
            "target_load_ratio": target_load_ratio * mode_on,
            "opportunity": opportunity * decision_mask,
            "commit_score": commit_score,
            "off_score": off_score,
            "net_grid_need_ratio": net_grid_need_ratio,
            "net_grid_absorb_ratio": net_grid_absorb_ratio,
            "heat_support_need": heat_support_need,
            "cool_support_need": cool_support_need,
            "cogen_support_need": cogen_support_need,
            "price_advantage": price_advantage.clamp_min(0.0),
            "mode_weight": mode_weight,
            "mode_on": mode_on,
            "mode_off": mode_off,
            "u_gt_exec": action_exec_batch[:, u_gt_index : u_gt_index + 1].clamp(-1.0, 1.0),
        }

    def _compute_gt_teacher_direction_match(
        self,
        *,
        prior_terms,
        teacher_action_exec_batch,
    ):
        if "u_gt" not in self.action_index:
            return self.torch.ones(
                (teacher_action_exec_batch.shape[0], 1),
                dtype=teacher_action_exec_batch.dtype,
                device=teacher_action_exec_batch.device,
            )
        u_gt_index = int(self.action_index["u_gt"])
        teacher_gt = teacher_action_exec_batch[:, u_gt_index : u_gt_index + 1].clamp(-1.0, 1.0)
        teacher_gt_on = (teacher_gt > -0.95).to(dtype=teacher_gt.dtype)
        off_match = self.torch.clamp(1.0 - prior_terms["mode_on"], 0.0, 1.0)
        return (
            teacher_gt_on * prior_terms["mode_on"]
            + (1.0 - teacher_gt_on) * off_match
        ).clamp(0.0, 1.0)

    def _compute_gt_prior_distill_loss(
        self,
        *,
        obs_batch,
        action_raw_batch,
        action_exec_batch,
        gap_batch,
        teacher_action_exec_batch=None,
        teacher_action_mask_batch=None,
        teacher_available_batch=None,
    ):
        if float(self.config.economic_gt_distill_coef) <= 0.0:
            zero = self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
            return zero, zero
        if "u_gt" not in self.action_index:
            zero = self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
            return zero, zero
        prior_terms = self._compute_gt_price_prior_terms(
            obs_batch=obs_batch,
            action_exec_batch=action_exec_batch,
        )
        if prior_terms is None:
            zero = self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
            return zero, zero
        projection_risk = self.torch.clamp(gap_batch[:, :1].detach(), 0.0, 1.0)
        prior_weight = (
            prior_terms["opportunity"]
            * prior_terms["mode_weight"]
            * (1.0 - projection_risk)
        )
        u_gt_index = int(self.action_index["u_gt"])
        if teacher_action_mask_batch is not None and teacher_available_batch is not None:
            teacher_gt_available = self.torch.clamp(
                teacher_available_batch,
                0.0,
                1.0,
            ) * (teacher_action_mask_batch[:, u_gt_index : u_gt_index + 1] > 0.5).to(
                dtype=obs_batch.dtype
            )
            teacher_gt_direction_match = self.torch.ones_like(teacher_gt_available)
            if teacher_action_exec_batch is not None:
                teacher_gt_direction_match = self._compute_gt_teacher_direction_match(
                    prior_terms=prior_terms,
                    teacher_action_exec_batch=teacher_action_exec_batch,
                ).to(dtype=obs_batch.dtype)
            prior_weight = prior_weight * (
                1.0 - teacher_gt_available * teacher_gt_direction_match
            )
        prior_weight_sum = prior_weight.sum()
        if float(prior_weight_sum.detach().cpu().item()) <= 0.0:
            zero = self.torch.zeros((1,), dtype=obs_batch.dtype, device=obs_batch.device).squeeze(0)
            return zero, prior_weight.mean()
        prior_sq = (
            action_raw_batch[:, u_gt_index : u_gt_index + 1].clamp(-1.0, 1.0)
            - prior_terms["target_u_gt"]
        ).pow(2)
        prior_loss = (prior_weight * prior_sq).sum() / prior_weight_sum.clamp_min(1.0)
        return prior_loss, prior_weight.mean()

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
            "qc_dem_mw",
            "p_gt_prev_mw",
        }
        required_actions = {"u_gt", "u_bes", "u_ech"}
        if (
            not required_obs.issubset(self.observation_index)
            or not required_actions.issubset(self.action_index)
        ):
            return None

        u_gt_index = int(self.action_index["u_gt"])
        p_gt_cap_mw = max(_NORM_EPS, float(self.env_config.p_gt_cap_mw))
        prior_terms = self._compute_gt_price_prior_terms(
            obs_batch=obs_batch,
            action_exec_batch=action_exec_batch,
        )
        if prior_terms is None:
            return None
        p_gt_exec = (
            (action_exec_batch[:, u_gt_index : u_gt_index + 1] + 1.0)
            * 0.5
            * p_gt_cap_mw
        )
        mode_on = prior_terms["mode_on"].clamp(0.0, 1.0)
        commit_score = prior_terms["commit_score"].clamp(0.0, 1.0)
        net_grid_need_ratio = prior_terms["net_grid_need_ratio"].clamp(0.0, 1.0)
        net_grid_absorb_ratio = prior_terms["net_grid_absorb_ratio"].clamp(0.0, 1.0)
        cogen_support_need = prior_terms["cogen_support_need"].clamp(0.0, 1.0)
        target_load_ratio = prior_terms["target_load_ratio"].clamp(0.0, 1.0)
        gt_target_proxy_mw = target_load_ratio * p_gt_cap_mw
        undercommit_ratio = (
            (gt_target_proxy_mw - p_gt_exec).clamp_min(0.0) / p_gt_cap_mw
        ) * mode_on
        support_multiplier = self.torch.clamp(
            commit_score * self.torch.maximum(net_grid_absorb_ratio, 0.75 * cogen_support_need),
            0.0,
            1.0,
        )
        return {
            "gt_target_proxy_mw": gt_target_proxy_mw,
            "p_gt_exec": p_gt_exec,
            "undercommit_ratio": undercommit_ratio,
            "price_advantage": prior_terms["price_advantage"].clamp(0.0, 1.0) * mode_on,
            "opportunity": prior_terms["opportunity"].clamp(0.0, 1.0),
            "commit_score": commit_score,
            "off_score": prior_terms["off_score"].clamp(0.0, 1.0),
            "mode_on": mode_on,
            "mode_off": prior_terms["mode_off"].clamp(0.0, 1.0),
            "heat_support_need": prior_terms["heat_support_need"].clamp(0.0, 1.0),
            "cool_support_need": prior_terms["cool_support_need"].clamp(0.0, 1.0),
            "cogen_support_need": cogen_support_need,
            "support_multiplier": support_multiplier,
            "net_grid_need_ratio": net_grid_need_ratio,
            "net_grid_absorb_ratio": net_grid_absorb_ratio,
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
        teacher_action_exec_batch=None,
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
        gt_commit_pressure = terms.get("net_grid_absorb_ratio", terms["net_grid_need_ratio"])
        gt_relax_signal = self.torch.clamp(
            2.5
            * terms["price_advantage"]
            * self.torch.maximum(gt_commit_pressure, terms["undercommit_ratio"])
            * terms["support_multiplier"]
            * (1.0 - projection_risk),
            0.0,
            1.0,
        )
        gt_prior_terms = self._compute_gt_price_prior_terms(
            obs_batch=obs_batch,
            action_exec_batch=action_exec_batch,
        )
        gt_anchor_floor = float(np.clip(self.config.exec_action_anchor_safe_floor, 0.0, 1.0))
        if "u_gt" in self.action_index:
            u_gt_index = int(self.action_index["u_gt"])
            if gt_prior_terms is not None:
                gt_prior_gap = self.torch.clamp(
                    (
                        gt_prior_terms["target_u_gt"]
                        - action_exec_batch[:, u_gt_index : u_gt_index + 1].clamp(-1.0, 1.0)
                    ).abs(),
                    0.0,
                    1.0,
                )
                gt_prior_relax_signal = self.torch.clamp(
                    2.0
                    * gt_prior_terms["opportunity"]
                    * gt_prior_gap
                    * (0.50 + 0.50 * gt_prior_terms["mode_weight"].clamp(0.0, 2.0))
                    * (1.0 - reliability_risk)
                    * (1.0 - projection_risk),
                    0.0,
                    1.0,
                )
                gt_relax_signal = self.torch.maximum(gt_relax_signal, gt_prior_relax_signal)
            if teacher_action_exec_batch is not None and teacher_available_batch is not None and teacher_action_mask_batch is not None:
                teacher_gt_available = self.torch.clamp(
                    teacher_available_batch,
                    0.0,
                    1.0,
                ) * (teacher_action_mask_batch[:, u_gt_index : u_gt_index + 1] > 0.5).to(
                    dtype=obs_batch.dtype
                )
                if float(teacher_gt_available.max().detach().cpu().item()) > 0.0:
                    teacher_gt_direction_match = self.torch.ones_like(teacher_gt_available)
                    if gt_prior_terms is not None:
                        teacher_gt_direction_match = self._compute_gt_teacher_direction_match(
                            prior_terms=gt_prior_terms,
                            teacher_action_exec_batch=teacher_action_exec_batch,
                        ).to(dtype=obs_batch.dtype)
                    teacher_gt_gap = self.torch.clamp(
                        (
                            teacher_action_exec_batch[:, u_gt_index : u_gt_index + 1].clamp(-1.0, 1.0)
                            - action_exec_batch[:, u_gt_index : u_gt_index + 1].clamp(-1.0, 1.0)
                        ).abs(),
                        0.0,
                        1.0,
                    )
                    teacher_gt_relax_signal = self.torch.clamp(
                        2.5
                        * teacher_gt_available
                        * teacher_gt_direction_match
                        * teacher_gt_gap
                        * (1.0 - reliability_risk)
                        * (1.0 - projection_risk),
                        0.0,
                        1.0,
                    )
                    gt_relax_signal = self.torch.maximum(gt_relax_signal, teacher_gt_relax_signal)
            gt_anchor_scale = self.torch.clamp(
                1.0 - (1.0 - gt_anchor_floor) * gt_relax_signal,
                gt_anchor_floor,
                1.0,
            )
            anchor_scale[:, u_gt_index : u_gt_index + 1] = gt_anchor_scale
        if "u_bes" in self.action_index:
            bes_anchor_floor = float(np.clip(gt_anchor_floor * 0.5, 0.05, 1.0))
            bes_relax_signal = self.torch.clamp(
                1.75
                * terms["price_advantage"]
                * self.torch.maximum(gt_commit_pressure, 0.5 * terms["undercommit_ratio"])
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
            bes_anchor_cap = float(
                np.clip(
                    self.config.economic_bes_anchor_max_scale,
                    bes_anchor_floor,
                    1.0,
                )
            )
            bes_anchor_scale = self.torch.clamp(
                bes_anchor_scale,
                bes_anchor_floor,
                bes_anchor_cap,
            )
            anchor_scale[:, u_bes_index : u_bes_index + 1] = bes_anchor_scale
        return anchor_scale

    def _build_actor_low_trust_fallback_action_tensor(
        self,
        *,
        action_raw_batch,
        replay_action_exec_batch,
    ):
        fallback_action = replay_action_exec_batch.detach().clone()
        for key in self.config.actor_low_trust_raw_fallback_keys:
            if key not in self.action_index:
                continue
            dim_index = int(self.action_index[key])
            fallback_action[:, dim_index : dim_index + 1] = action_raw_batch[
                :, dim_index : dim_index + 1
            ]
        return fallback_action

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
        weighted_teacher_mask = teacher_action_mask_batch * self.economic_teacher_action_weight
        active_teacher_dims = weighted_teacher_mask.sum(dim=1, keepdim=True)
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
            gt_commit_pressure = terms.get("net_grid_absorb_ratio", terms["net_grid_need_ratio"])
            opportunity_score = self.torch.clamp(
                0.5 * terms["price_advantage"]
                + 0.5 * self.torch.maximum(
                    gt_commit_pressure,
                    terms["undercommit_ratio"],
                ),
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
            (
                ((teacher_action_exec_batch - action_exec_batch).abs() * weighted_teacher_mask)
                .sum(dim=1, keepdim=True)
                .detach()
            )
            / active_teacher_dims.clamp_min(1.0),
            0.0,
            1.0,
        )
        teacher_weight = teacher_mask * self.torch.clamp(
            (0.35 + 0.65 * opportunity_score)
            * (0.50 + 0.50 * safety_margin)
            * (0.50 + 0.50 * disagreement_score),
            0.0,
            1.0,
        )
        if "u_gt" in self.action_index:
            u_gt_index = int(self.action_index["u_gt"])
            teacher_gt_available = teacher_mask * (
                teacher_action_mask_batch[:, u_gt_index : u_gt_index + 1] > 0.5
            ).to(dtype=obs_batch.dtype)
            if float(teacher_gt_available.max().detach().cpu().item()) > 0.0:
                gt_terms = self._compute_gt_price_prior_terms(
                    obs_batch=obs_batch,
                    action_exec_batch=teacher_action_exec_batch,
                )
                if gt_terms is not None:
                    teacher_gt = teacher_action_exec_batch[:, u_gt_index : u_gt_index + 1].clamp(
                        -1.0,
                        1.0,
                    )
                    current_gt = action_exec_batch[:, u_gt_index : u_gt_index + 1].clamp(
                        -1.0,
                        1.0,
                    )
                    gt_gap = self.torch.clamp(
                        0.5 * (teacher_gt - current_gt).abs().detach(),
                        0.0,
                        1.0,
                    )
                    teacher_gt_on = (teacher_gt > -0.95).to(dtype=obs_batch.dtype)
                    gt_commit_score = gt_terms["commit_score"].clamp(0.0, 1.0)
                    gt_off_score = gt_terms["off_score"].clamp(0.0, 1.0)
                    gt_direction_margin = (
                        teacher_gt_on * (gt_commit_score - gt_off_score).clamp_min(0.0)
                        + (1.0 - teacher_gt_on) * (gt_off_score - gt_commit_score).clamp_min(0.0)
                    )
                    teacher_weight = teacher_weight * (
                        1.0
                        - teacher_gt_available
                        + teacher_gt_available * (0.25 + 0.75 * gt_direction_margin)
                    )
                    gt_direction_score = (
                        teacher_gt_on
                        * self.torch.maximum(
                            gt_direction_margin,
                            gt_terms["price_advantage"].clamp(0.0, 1.0),
                        )
                        + (1.0 - teacher_gt_on) * gt_direction_margin
                    )
                    gt_teacher_weight = teacher_gt_available * self.torch.clamp(
                        (0.20 + 0.80 * gt_direction_score.clamp(0.0, 1.0))
                        * (0.35 + 0.65 * gt_gap)
                        * (0.50 + 0.50 * safety_margin),
                        0.0,
                        1.0,
                    )
                    teacher_weight = self.torch.maximum(
                        teacher_weight,
                        gt_teacher_weight,
                    )
        return teacher_weight

    def _compute_economic_teacher_safe_preserve_loss(
        self,
        *,
        obs_batch,
        action_exec_batch,
        teacher_action_exec_batch,
        teacher_action_mask_batch,
        teacher_available_batch,
    ):
        teacher_row_mask = self.torch.clamp(teacher_available_batch, 0.0, 1.0)
        active_teacher_dims = (
            teacher_action_mask_batch
            * self.economic_teacher_action_weight
            * self.economic_teacher_mismatch_focus_weight
        ).sum(dim=1, keepdim=True)
        teacher_row_mask = teacher_row_mask * (active_teacher_dims > 0.0).to(
            dtype=action_exec_batch.dtype
        )
        if float(teacher_row_mask.max().detach().cpu().item()) <= 0.0:
            return self.torch.zeros((1,), dtype=action_exec_batch.dtype, device=action_exec_batch.device).squeeze(0)
        preserve_mask = self.torch.clamp(
            1.0 - teacher_action_mask_batch,
            0.0,
            1.0,
        ) * self.economic_teacher_safe_preserve_weight
        row_weight = self.torch.ones_like(teacher_row_mask)
        low_margin_mask = None
        if "abs_drive_margin_k" in self.observation_index:
            abs_margin_index = int(self.observation_index["abs_drive_margin_k"])
            low_margin_mask = (
                obs_batch[:, abs_margin_index : abs_margin_index + 1]
                <= float(self.config.dual_abs_margin_k)
            ).to(dtype=action_exec_batch.dtype)
            row_weight = row_weight + (
                low_margin_mask
                * float(self.config.economic_teacher_safe_preserve_low_margin_boost)
            )
        high_cooling_mask = None
        if "qc_dem_mw" in self.observation_index:
            qc_index = int(self.observation_index["qc_dem_mw"])
            q_total_cooling_cap_mw = max(
                _NORM_EPS,
                float(self.env_config.q_abs_cool_cap_mw)
                + float(self.env_config.q_ech_cap_mw),
            )
            qc_ratio = obs_batch[:, qc_index : qc_index + 1] / q_total_cooling_cap_mw
            high_cooling_mask = (
                qc_ratio >= float(self.config.dual_qc_ratio_th)
            ).to(dtype=action_exec_batch.dtype)
            row_weight = row_weight + (
                high_cooling_mask
                * float(self.config.economic_teacher_safe_preserve_high_cooling_boost)
            )
        if low_margin_mask is not None and high_cooling_mask is not None:
            row_weight = row_weight + (
                low_margin_mask
                * high_cooling_mask
                * float(self.config.economic_teacher_safe_preserve_joint_boost)
            )
        preserve_sq = (
            (action_exec_batch - teacher_action_exec_batch).pow(2) * preserve_mask
        ).sum(dim=1, keepdim=True) / preserve_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        weighted_teacher_mask = teacher_row_mask * row_weight
        return (weighted_teacher_mask * preserve_sq).sum() / weighted_teacher_mask.sum().clamp_min(1.0)

    def _collect_surrogate_replay_audit(self) -> tuple[dict[str, Any], pd.DataFrame]:
        if int(self.replay._size) <= 0:
            return {}, pd.DataFrame()
        sample_count = int(self.replay._size)
        obs_batch = np.asarray(self.replay.obs[:sample_count], dtype=np.float32).copy()
        action_raw_batch = np.asarray(
            self.replay.action_raw[:sample_count], dtype=np.float32
        ).copy()
        action_exec_batch = np.asarray(
            self.replay.action_exec[:sample_count], dtype=np.float32
        ).copy()
        predicted_exec_batch = np.zeros_like(action_exec_batch)
        audit_batch_size = max(128, int(self.config.batch_size) * 4)
        for start in range(0, sample_count, audit_batch_size):
            end = min(sample_count, start + audit_batch_size)
            obs_tensor = self.torch.as_tensor(
                obs_batch[start:end],
                dtype=self.torch.float32,
                device=self.device,
            )
            action_tensor = self.torch.as_tensor(
                action_raw_batch[start:end],
                dtype=self.torch.float32,
                device=self.device,
            )
            with self.torch.no_grad():
                predicted_exec_batch[start:end] = (
                    self.surrogate.project(obs_tensor, action_tensor)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
        return _build_surrogate_audit_report(
            obs_batch=obs_batch,
            predicted_exec_batch=predicted_exec_batch,
            actual_exec_batch=action_exec_batch,
            observation_index=self.observation_index,
            action_keys=self.action_keys,
            low_abs_margin_threshold=float(self.config.dual_abs_margin_k),
            high_cooling_ratio_threshold=float(self.config.dual_qc_ratio_th),
            q_total_cooling_cap_mw=float(self.env_config.q_abs_cool_cap_mw)
            + float(self.env_config.q_ech_cap_mw),
        )

    def _refresh_economic_teacher_mismatch_focus(self) -> dict[str, Any]:
        ones = np.ones((1, len(self.action_keys)), dtype=np.float32)
        if (
            float(self.config.economic_teacher_mismatch_focus_coef) <= 0.0
            or len(self.action_keys) <= 0
        ):
            self.economic_teacher_mismatch_focus_weight_np = ones
            self.economic_teacher_mismatch_focus_weight = self.torch.as_tensor(
                ones,
                dtype=self.torch.float32,
                device=self.device,
            )
            self.economic_teacher_mismatch_focus_summary = {
                "enabled": False,
                "status": "disabled",
                "prefill_replay_size": int(self.replay.size),
                "scale_by_action": {str(key): 1.0 for key in self.action_keys},
            }
            return dict(self.economic_teacher_mismatch_focus_summary)

        audit_summary, _ = self._collect_surrogate_replay_audit()
        overall_mae = dict(audit_summary.get("overall_mae_by_action") or {})
        focused_mae = dict(audit_summary.get("focused_mae_by_action") or {})
        mae_vector = np.asarray(
            [float(overall_mae.get(str(key), 0.0)) for key in self.action_keys],
            dtype=np.float32,
        )
        if focused_mae:
            focused_vector = np.asarray(
                [float(focused_mae.get(str(key), overall_mae.get(str(key), 0.0))) for key in self.action_keys],
                dtype=np.float32,
            )
            mae_vector = 0.5 * mae_vector + 0.5 * focused_vector
        positive_mae = mae_vector[mae_vector > 0.0]
        reference_mae = float(np.mean(positive_mae)) if positive_mae.size > 0 else 1.0
        raw_scale = 1.0 + float(self.config.economic_teacher_mismatch_focus_coef) * (
            (mae_vector / max(_NORM_EPS, reference_mae)) - 1.0
        )
        clipped_scale = np.clip(
            raw_scale,
            float(self.config.economic_teacher_mismatch_focus_min_scale),
            float(self.config.economic_teacher_mismatch_focus_max_scale),
        ).astype(np.float32, copy=False)
        if not np.isfinite(clipped_scale).all():
            clipped_scale = np.ones((len(self.action_keys),), dtype=np.float32)
        scale_matrix = clipped_scale.reshape(1, -1).astype(np.float32, copy=False)
        self.economic_teacher_mismatch_focus_weight_np = scale_matrix.copy()
        self.economic_teacher_mismatch_focus_weight = self.torch.as_tensor(
            scale_matrix,
            dtype=self.torch.float32,
            device=self.device,
        )
        self.economic_teacher_mismatch_focus_summary = {
            "enabled": True,
            "status": "applied",
            "prefill_replay_size": int(self.replay.size),
            "coef": float(self.config.economic_teacher_mismatch_focus_coef),
            "min_scale": float(self.config.economic_teacher_mismatch_focus_min_scale),
            "max_scale": float(self.config.economic_teacher_mismatch_focus_max_scale),
            "reference_mae": float(reference_mae),
            "overall_worst_action_by_mae": str(audit_summary.get("overall_worst_action_by_mae", "")),
            "focused_worst_action_by_mae": str(audit_summary.get("focused_worst_action_by_mae", "")),
            "scale_by_action": {
                str(key): float(value)
                for key, value in zip(self.action_keys, clipped_scale)
            },
        }
        return dict(self.economic_teacher_mismatch_focus_summary)

    def _refresh_surrogate_actor_trust(self) -> dict[str, Any]:
        audit_summary, _ = self._collect_surrogate_replay_audit()
        scale_matrix, summary = _build_surrogate_actor_trust_scale_np(
            action_keys=self.action_keys,
            overall_mae_by_action=audit_summary.get("overall_mae_by_action"),
            focused_mae_by_action=audit_summary.get("focused_mae_by_action"),
            trust_coef=float(self.config.surrogate_actor_trust_coef),
            trust_min_scale=float(self.config.surrogate_actor_trust_min_scale),
            focused_mix=0.5,
        )
        self.surrogate_actor_trust_weight_np = np.asarray(
            scale_matrix,
            dtype=np.float32,
        ).copy()
        self.surrogate_actor_trust_weight = self.torch.as_tensor(
            self.surrogate_actor_trust_weight_np,
            dtype=self.torch.float32,
            device=self.device,
        )
        self.surrogate_actor_trust_summary = {
            **summary,
            "prefill_replay_size": int(self.replay.size),
            "overall_worst_action_by_mae": str(
                audit_summary.get("overall_worst_action_by_mae", "")
            ),
            "focused_worst_action_by_mae": str(
                audit_summary.get("focused_worst_action_by_mae", "")
            ),
        }
        return dict(self.surrogate_actor_trust_summary)

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
        action_np = self._apply_abs_cooling_blend_np(
            observation_vector=observation_vector,
            action_vector=action_np,
        )
        safe_action_np = self._predict_frozen_safe_action_np(
            observation_vector=observation_vector,
        )
        action_np = self._clip_abs_near_safe_action_np(
            action_vector=action_np,
            safe_action_vector=safe_action_np,
        )
        action_np = self._clip_ech_near_safe_action_np(
            action_vector=action_np,
            safe_action_vector=safe_action_np,
        )
        action_np = self._clip_gt_near_safe_action_np(
            action_vector=action_np,
            safe_action_vector=safe_action_np,
        )
        action_np = self._clip_bes_near_safe_action_np(
            action_vector=action_np,
            safe_action_vector=safe_action_np,
        )
        action_np = self._clip_boiler_near_safe_action_np(
            action_vector=action_np,
            safe_action_vector=safe_action_np,
        )
        action_np = self._clip_tes_near_safe_action_np(
            action_vector=action_np,
            safe_action_vector=safe_action_np,
        )
        return self._overwrite_frozen_action_dims_np(
            action_vector=action_np,
            safe_action_vector=safe_action_np,
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

    def _write_surrogate_replay_audit(self) -> dict[str, Any]:
        summary, detail_df = self._collect_surrogate_replay_audit()
        if detail_df.empty:
            return summary
        audit_path = self.run_dir / "train" / "surrogate_replay_audit.csv"
        detail_df.to_csv(audit_path, index=False)
        return {
            **summary,
            "csv_path": str(audit_path.resolve()).replace("\\", "/"),
        }

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
            "frozen_action_keys": list(self.frozen_action_keys),
            "frozen_action_safe_checkpoint_path": (
                str(Path(self.frozen_action_safe_checkpoint_path).resolve()).replace("\\", "/")
                if self.safe_reference_action_enabled and len(self.frozen_action_safe_checkpoint_path) > 0
                else ""
            ),
            "gt_safe_action_delta_clip": float(self.config.gt_safe_action_delta_clip),
            "bes_safe_action_delta_clip": float(self.config.bes_safe_action_delta_clip),
            "boiler_safe_action_delta_clip": float(self.config.boiler_safe_action_delta_clip),
            "tes_safe_action_delta_clip": float(self.config.tes_safe_action_delta_clip),
            "abs_safe_action_delta_clip": float(self.config.abs_safe_action_delta_clip),
            "ech_safe_action_delta_clip": float(self.config.ech_safe_action_delta_clip),
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
            "actor_warmup_steps": int(self.config.actor_warmup_steps),
            "exec_action_anchor_coef": float(self.config.exec_action_anchor_coef),
            "exec_action_anchor_safe_floor": float(self.config.exec_action_anchor_safe_floor),
            "gt_off_deadband_ratio": float(self.config.gt_off_deadband_ratio),
            "abs_ready_focus_coef": float(self.config.abs_ready_focus_coef),
            "invalid_abs_penalty_coef": float(self.config.invalid_abs_penalty_coef),
            "economic_boiler_proxy_coef": float(self.config.economic_boiler_proxy_coef),
            "economic_abs_tradeoff_coef": float(self.config.economic_abs_tradeoff_coef),
            "economic_gt_grid_proxy_coef": float(self.config.economic_gt_grid_proxy_coef),
            "economic_gt_distill_coef": float(self.config.economic_gt_distill_coef),
            "economic_teacher_distill_coef": float(self.config.economic_teacher_distill_coef),
            "economic_teacher_safe_preserve_coef": float(
                self.config.economic_teacher_safe_preserve_coef
            ),
            "economic_teacher_safe_preserve_low_margin_boost": float(
                self.config.economic_teacher_safe_preserve_low_margin_boost
            ),
            "economic_teacher_safe_preserve_high_cooling_boost": float(
                self.config.economic_teacher_safe_preserve_high_cooling_boost
            ),
            "economic_teacher_safe_preserve_joint_boost": float(
                self.config.economic_teacher_safe_preserve_joint_boost
            ),
            "economic_teacher_mismatch_focus_coef": float(
                self.config.economic_teacher_mismatch_focus_coef
            ),
            "economic_teacher_mismatch_focus_min_scale": float(
                self.config.economic_teacher_mismatch_focus_min_scale
            ),
            "economic_teacher_mismatch_focus_max_scale": float(
                self.config.economic_teacher_mismatch_focus_max_scale
            ),
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
            "economic_gt_full_year_warm_start_samples": int(
                self.config.economic_gt_full_year_warm_start_samples
            ),
            "economic_gt_full_year_warm_start_epochs": int(
                self.config.economic_gt_full_year_warm_start_epochs
            ),
            "economic_gt_full_year_warm_start_u_weight": float(
                self.config.economic_gt_full_year_warm_start_u_weight
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
            "economic_bes_warm_start_economic_anchor_weight": float(
                self.config.economic_bes_warm_start_economic_anchor_weight
            ),
            "economic_bes_warm_start_fallback_anchor_weight": float(
                self.config.economic_bes_warm_start_fallback_anchor_weight
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
            "economic_bes_anchor_max_scale": float(
                self.config.economic_bes_anchor_max_scale
            ),
            "surrogate_actor_trust_coef": float(
                self.config.surrogate_actor_trust_coef
            ),
            "surrogate_actor_trust_min_scale": float(
                self.config.surrogate_actor_trust_min_scale
            ),
            "actor_low_trust_raw_fallback_keys": list(
                self.config.actor_low_trust_raw_fallback_keys
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
        total_env_steps = max(1, int(self.config.total_env_steps))
        if total_env_steps <= 512:
            # Small-budget PAFC runs can peak early; keep auto cadence dense enough
            # to retain the mid-training checkpoint instead of only the final one.
            return max(32, total_env_steps // 4)
        return max(1, min(2_000, max(500, total_env_steps // 4)))

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

    def _update_networks(
        self,
        *,
        update_step: int,
        actor_update_step: int | None = None,
    ) -> dict[str, float]:
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
            next_safe_action = self._predict_frozen_safe_action_tensor(obs_batch=next_obs)
            next_action_raw = self._clip_abs_near_safe_action_tensor(
                action_batch=next_action_raw,
                safe_action_batch=next_safe_action,
            )
            next_action_raw = self._clip_ech_near_safe_action_tensor(
                action_batch=next_action_raw,
                safe_action_batch=next_safe_action,
            )
            next_action_raw = self._clip_gt_near_safe_action_tensor(
                action_batch=next_action_raw,
                safe_action_batch=next_safe_action,
            )
            next_action_raw = self._clip_bes_near_safe_action_tensor(
                action_batch=next_action_raw,
                safe_action_batch=next_safe_action,
            )
            next_action_raw = self._clip_boiler_near_safe_action_tensor(
                action_batch=next_action_raw,
                safe_action_batch=next_safe_action,
            )
            next_action_raw = self._clip_tes_near_safe_action_tensor(
                action_batch=next_action_raw,
                safe_action_batch=next_safe_action,
            )
            next_action_raw = self._overwrite_frozen_action_dims_tensor(
                action_batch=next_action_raw,
                safe_action_batch=next_safe_action,
            )
            next_action_exec_hat = self.surrogate.project(next_obs, next_action_raw)
            # For bootstrap targets we do not have a real executed next action in replay.
            # Low-trust dimensions therefore fall back to the post-shaping raw action.
            next_action_exec_for_target = _blend_surrogate_action_proxy(
                action_exec_hat=next_action_exec_hat,
                fallback_action=next_action_raw,
                trust_weight=self.surrogate_actor_trust_weight,
            )
            reward_target = reward + float(self.config.gamma) * (1.0 - done) * self.torch.minimum(
                self.q1_target(next_obs_norm, next_action_exec_for_target),
                self.q2_target(next_obs_norm, next_action_exec_for_target),
            )
            cost_targets = [
                cost[:, idx : idx + 1]
                + float(self.config.gamma)
                * (1.0 - done)
                * self.cost_target_critics[idx](
                    next_obs_norm, next_action_exec_for_target
                ).clamp_min(0.0)
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
        economic_teacher_safe_preserve_loss_value = float("nan")
        economic_gt_distill_loss_value = float("nan")
        economic_gt_distill_weight_value = float("nan")
        economic_bes_distill_loss_value = float("nan")
        economic_bes_distill_weight_value = float("nan")
        reward_actor_value = float("nan")
        mean_constraint_value = float("nan")
        surrogate_actor_trust_mean_value = float("nan")
        surrogate_actor_trust_min_value = float("nan")
        actor_step = int(update_step) if actor_update_step is None else int(actor_update_step)
        dual_scale_value = float(
            min(
                1.0,
                float(update_step) / max(1.0, float(self.config.dual_warmup_steps)),
            )
        ) if int(self.config.dual_warmup_steps) > 0 else 1.0
        if (
            actor_step >= int(self.config.actor_warmup_steps)
            and update_step % int(self.config.actor_delay) == 0
        ):
            action_raw = self.actor(obs_norm)
            action_raw = self._apply_abs_cooling_blend_tensor(
                obs_batch=obs,
                action_batch=action_raw,
            )
            safe_action_raw = self._predict_frozen_safe_action_tensor(obs_batch=obs)
            action_raw = self._clip_abs_near_safe_action_tensor(
                action_batch=action_raw,
                safe_action_batch=safe_action_raw,
            )
            action_raw = self._clip_ech_near_safe_action_tensor(
                action_batch=action_raw,
                safe_action_batch=safe_action_raw,
            )
            action_raw = self._clip_gt_near_safe_action_tensor(
                action_batch=action_raw,
                safe_action_batch=safe_action_raw,
            )
            action_raw = self._clip_bes_near_safe_action_tensor(
                action_batch=action_raw,
                safe_action_batch=safe_action_raw,
            )
            action_raw = self._clip_boiler_near_safe_action_tensor(
                action_batch=action_raw,
                safe_action_batch=safe_action_raw,
            )
            action_raw = self._clip_tes_near_safe_action_tensor(
                action_batch=action_raw,
                safe_action_batch=safe_action_raw,
            )
            action_raw = self._overwrite_frozen_action_dims_tensor(
                action_batch=action_raw,
                safe_action_batch=safe_action_raw,
            )
            action_exec_hat = self.surrogate.project(obs, action_raw)
            surrogate_actor_trust = self.surrogate_actor_trust_weight
            fallback_action_for_actor = self._build_actor_low_trust_fallback_action_tensor(
                action_raw_batch=action_raw,
                replay_action_exec_batch=action_exec,
            )
            action_exec_for_actor = _blend_surrogate_action_proxy(
                action_exec_hat=action_exec_hat,
                fallback_action=fallback_action_for_actor,
                trust_weight=surrogate_actor_trust,
            )
            reward_actor = self.torch.minimum(
                self.q1(obs_norm, action_exec_for_actor),
                self.q2(obs_norm, action_exec_for_actor),
            )
            constraint_predictions = self.torch.cat(
                [critic(obs_norm, action_exec_for_actor) for critic in self.cost_critics],
                dim=1,
            ).clamp_min(0.0)
            lambda_tensor = self.torch.as_tensor(
                (self.dual_lambdas * float(dual_scale_value)).reshape(1, -1),
                dtype=self.torch.float32,
                device=self.device,
            )
            gap_weight = surrogate_actor_trust.expand_as(action_exec_hat)
            gap_loss = (
                (gap_weight * (action_raw - action_exec_hat).pow(2)).sum(dim=1, keepdim=True)
                / gap_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            ).mean()
            support_weight = self._compute_anchor_weight(
                obs_batch=obs,
                action_exec_batch=action_exec,
                gap_batch=gap,
            )
            gt_anchor_dimension_scale = self._compute_gt_anchor_dimension_scale(
                obs_batch=obs,
                action_raw_batch=action_raw,
                action_exec_batch=action_exec_hat,
                teacher_action_exec_batch=teacher_action_exec,
                teacher_available_batch=teacher_available,
                teacher_action_mask_batch=teacher_action_mask,
            )
            trusted_anchor_scale = gt_anchor_dimension_scale * surrogate_actor_trust
            exec_anchor_loss = (
                support_weight
                * (
                    trusted_anchor_scale
                    * (action_exec_hat - action_exec).pow(2)
                ).sum(dim=1, keepdim=True)
                / trusted_anchor_scale.sum(dim=1, keepdim=True).clamp_min(1.0)
            ).mean()
            invalid_abs_penalty = self._compute_invalid_abs_request_penalty(
                obs_batch=obs,
                action_raw_batch=action_raw,
            )
            boiler_proxy_penalty = self._compute_boiler_economic_proxy_penalty(
                obs_batch=obs,
                action_exec_batch=action_exec_for_actor,
            )
            abs_tradeoff_penalty = self._compute_abs_ech_tradeoff_proxy_penalty(
                obs_batch=obs,
                action_exec_batch=action_exec_for_actor,
            )
            gt_grid_proxy_penalty = self._compute_gt_grid_economic_proxy_penalty(
                obs_batch=obs,
                action_exec_batch=action_exec_for_actor,
            )
            economic_teacher_weight = self._compute_economic_teacher_weight(
                obs_batch=obs,
                action_exec_batch=action_exec_for_actor,
                teacher_action_exec_batch=teacher_action_exec,
                teacher_action_mask_batch=teacher_action_mask,
                gap_batch=gap,
                teacher_available_batch=teacher_available,
            )
            effective_teacher_mask = (
                teacher_action_mask
                * self.economic_teacher_action_weight
                * self.economic_teacher_mismatch_focus_weight
            )
            economic_teacher_sq = (
                (action_exec_for_actor - teacher_action_exec).pow(2) * effective_teacher_mask
            ).sum(dim=1, keepdim=True) / effective_teacher_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            economic_teacher_weight_sum = economic_teacher_weight.sum()
            economic_teacher_selected_loss = (
                (economic_teacher_weight * economic_teacher_sq).sum()
                / economic_teacher_weight_sum.clamp_min(1.0)
            )
            economic_teacher_safe_preserve_loss = self._compute_economic_teacher_safe_preserve_loss(
                obs_batch=obs,
                action_exec_batch=action_exec_for_actor,
                teacher_action_exec_batch=teacher_action_exec,
                teacher_action_mask_batch=teacher_action_mask,
                teacher_available_batch=teacher_available,
            )
            economic_teacher_loss = (
                economic_teacher_selected_loss
                + float(self.config.economic_teacher_safe_preserve_coef)
                * economic_teacher_safe_preserve_loss
            )
            economic_bes_distill_loss, economic_bes_distill_weight = self._compute_bes_prior_distill_loss(
                obs_batch=obs,
                action_raw_batch=action_raw,
                action_exec_batch=action_exec_for_actor,
                gap_batch=gap,
            )
            economic_gt_distill_loss, economic_gt_distill_weight = self._compute_gt_prior_distill_loss(
                obs_batch=obs,
                action_raw_batch=action_raw,
                action_exec_batch=action_exec_for_actor,
                gap_batch=gap,
                teacher_action_exec_batch=teacher_action_exec,
                teacher_action_mask_batch=teacher_action_mask,
                teacher_available_batch=teacher_available,
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
                + float(self.config.economic_gt_distill_coef) * economic_gt_distill_loss
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
            economic_teacher_safe_preserve_loss_value = float(
                economic_teacher_safe_preserve_loss.detach().cpu().item()
            )
            economic_gt_distill_loss_value = float(economic_gt_distill_loss.detach().cpu().item())
            economic_gt_distill_weight_value = float(
                economic_gt_distill_weight.detach().cpu().item()
            )
            economic_bes_distill_loss_value = float(
                economic_bes_distill_loss.detach().cpu().item()
            )
            economic_bes_distill_weight_value = float(
                economic_bes_distill_weight.detach().cpu().item()
            )
            reward_actor_value = float(reward_actor.mean().detach().cpu().item())
            mean_constraint_value = float(constraint_predictions.mean().detach().cpu().item())
            surrogate_actor_trust_mean_value = float(
                surrogate_actor_trust.mean().detach().cpu().item()
            )
            surrogate_actor_trust_min_value = float(
                surrogate_actor_trust.min().detach().cpu().item()
            )

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
            "actor_economic_teacher_safe_preserve_loss": economic_teacher_safe_preserve_loss_value,
            "actor_economic_gt_distill_loss": economic_gt_distill_loss_value,
            "actor_economic_gt_distill_weight": economic_gt_distill_weight_value,
            "actor_economic_bes_distill_loss": economic_bes_distill_loss_value,
            "actor_economic_bes_distill_weight": economic_bes_distill_weight_value,
            "actor_surrogate_trust_mean": surrogate_actor_trust_mean_value,
            "actor_surrogate_trust_min": surrogate_actor_trust_min_value,
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
        self._warm_start_actor_from_gt_full_year()
        self._refresh_economic_teacher_mismatch_focus()
        self._refresh_surrogate_actor_trust()
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
            "actor_economic_teacher_safe_preserve_loss": float("nan"),
            "actor_economic_gt_distill_loss": float("nan"),
            "actor_economic_gt_distill_weight": float("nan"),
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
                        latest_update_metrics = self._update_networks(
                            update_step=total_env_steps,
                            actor_update_step=total_env_steps,
                        )

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
        surrogate_replay_audit = self._write_surrogate_replay_audit()
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
            "actor_warmup_steps": int(self.config.actor_warmup_steps),
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
            "economic_gt_distill_coef": float(self.config.economic_gt_distill_coef),
            "economic_teacher_distill_coef": float(self.config.economic_teacher_distill_coef),
            "economic_teacher_safe_preserve_coef": float(
                self.config.economic_teacher_safe_preserve_coef
            ),
            "economic_teacher_safe_preserve_low_margin_boost": float(
                self.config.economic_teacher_safe_preserve_low_margin_boost
            ),
            "economic_teacher_safe_preserve_high_cooling_boost": float(
                self.config.economic_teacher_safe_preserve_high_cooling_boost
            ),
            "economic_teacher_safe_preserve_joint_boost": float(
                self.config.economic_teacher_safe_preserve_joint_boost
            ),
            "economic_teacher_mismatch_focus_coef": float(
                self.config.economic_teacher_mismatch_focus_coef
            ),
            "economic_teacher_mismatch_focus_min_scale": float(
                self.config.economic_teacher_mismatch_focus_min_scale
            ),
            "economic_teacher_mismatch_focus_max_scale": float(
                self.config.economic_teacher_mismatch_focus_max_scale
            ),
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
            "economic_gt_full_year_warm_start_samples": int(
                self.config.economic_gt_full_year_warm_start_samples
            ),
            "economic_gt_full_year_warm_start_epochs": int(
                self.config.economic_gt_full_year_warm_start_epochs
            ),
            "economic_gt_full_year_warm_start_u_weight": float(
                self.config.economic_gt_full_year_warm_start_u_weight
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
            "surrogate_actor_trust_coef": float(
                self.config.surrogate_actor_trust_coef
            ),
            "surrogate_actor_trust_min_scale": float(
                self.config.surrogate_actor_trust_min_scale
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
            "economic_teacher_mismatch_focus": dict(
                self.economic_teacher_mismatch_focus_summary
            ),
            "surrogate_actor_trust": dict(self.surrogate_actor_trust_summary),
            "actor_init": dict(self.actor_init_summary),
            "actor_warm_start": dict(self.actor_warm_start_summary),
            "actor_teacher_gt_head_warm_start": dict(
                self.actor_teacher_gt_head_warm_start_summary
            ),
            "surrogate_replay_audit": dict(surrogate_replay_audit),
            "actor_teacher_full_year_warm_start": dict(
                self.actor_teacher_full_year_warm_start_summary
            ),
            "actor_gt_warm_start": dict(self.actor_gt_warm_start_summary),
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
    checkpoint_entry_path = Path(str(checkpoint_path)).expanduser()
    checkpoint_resolved_path = checkpoint_entry_path
    if checkpoint_entry_path.suffix.lower() == ".json":
        entry_payload = _json_payload_from_path(checkpoint_entry_path)
        artifact_type = str(entry_payload.get("artifact_type", "")).strip().lower()
        if artifact_type != "pafc_td3_actor":
            raise ValueError("load_pafc_td3_predictor 仅支持 pafc_td3_actor json/pt checkpoint。")
        resolved = entry_payload.get("checkpoint_path")
        if not isinstance(resolved, str) or len(resolved.strip()) == 0:
            raise ValueError("pafc_td3_actor.json 缺少 checkpoint_path。")
        checkpoint_resolved_path = Path(resolved).expanduser()
    payload = load_policy(checkpoint_resolved_path, map_location=target_device)
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
    frozen_action_keys = tuple(
        key
        for key in _normalize_action_key_tuple(metadata.get("frozen_action_keys", ()))
        if key in action_index
    )
    frozen_action_safe_checkpoint_path = str(
        metadata.get("frozen_action_safe_checkpoint_path", "")
    ).strip()
    gt_safe_action_delta_clip = float(metadata.get("gt_safe_action_delta_clip", 0.0))
    bes_safe_action_delta_clip = float(metadata.get("bes_safe_action_delta_clip", 0.0))
    boiler_safe_action_delta_clip = float(metadata.get("boiler_safe_action_delta_clip", 0.0))
    tes_safe_action_delta_clip = float(metadata.get("tes_safe_action_delta_clip", 0.0))
    abs_safe_action_delta_clip = float(metadata.get("abs_safe_action_delta_clip", 0.0))
    ech_safe_action_delta_clip = float(metadata.get("ech_safe_action_delta_clip", 0.0))
    frozen_action_safe_predictor = None
    if (
        frozen_action_keys
        or gt_safe_action_delta_clip > 0.0
        or bes_safe_action_delta_clip > 0.0
        or boiler_safe_action_delta_clip > 0.0
        or tes_safe_action_delta_clip > 0.0
        or abs_safe_action_delta_clip > 0.0
        or ech_safe_action_delta_clip > 0.0
    ):
        if len(frozen_action_safe_checkpoint_path) == 0:
            raise ValueError(
                "checkpoint metadata 缺少 frozen_action_safe_checkpoint_path，无法恢复安全参考动作。"
            )
        current_resolved = checkpoint_resolved_path.resolve()
        safe_entry_path = Path(frozen_action_safe_checkpoint_path).expanduser()
        safe_resolved = safe_entry_path.resolve()
        if safe_entry_path.suffix.lower() == ".json":
            safe_entry_payload = _json_payload_from_path(safe_entry_path)
            safe_artifact_type = str(safe_entry_payload.get("artifact_type", "")).strip().lower()
            if safe_artifact_type != "pafc_td3_actor":
                raise ValueError("冻结安全策略 metadata 仅支持 pafc_td3_actor json/pt checkpoint。")
            safe_checkpoint_payload = safe_entry_payload.get("checkpoint_path")
            if not isinstance(safe_checkpoint_payload, str) or len(safe_checkpoint_payload.strip()) == 0:
                raise ValueError("冻结安全策略 pafc_td3_actor.json 缺少 checkpoint_path。")
            safe_resolved = Path(safe_checkpoint_payload).expanduser().resolve()
        if safe_resolved == current_resolved:
            raise ValueError("冻结安全策略 checkpoint 不能指向当前 PAFC actor 本身。")
        frozen_action_safe_predictor, _ = load_pafc_td3_predictor(
            checkpoint_path=safe_entry_path,
            device=device,
            env_config=env_config,
        )
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
        if frozen_action_safe_predictor is not None:
            safe_action = dict(
                frozen_action_safe_predictor(
                    observation if isinstance(observation, Mapping) else obs_vector
                )
            )
            if (
                abs_safe_action_delta_clip > 0.0
                and "u_abs" in action_index
                and "u_abs" not in frozen_action_keys
            ):
                u_abs_index = int(action_index["u_abs"])
                safe_abs = float(safe_action["u_abs"])
                abs_low, abs_high = _ACTION_BOUNDS["u_abs"]
                action[u_abs_index] = float(
                    np.clip(
                        float(action[u_abs_index]),
                        max(abs_low, safe_abs - abs_safe_action_delta_clip),
                        min(abs_high, safe_abs + abs_safe_action_delta_clip),
                    )
                )
            if (
                ech_safe_action_delta_clip > 0.0
                and "u_ech" in action_index
                and "u_ech" not in frozen_action_keys
            ):
                u_ech_index = int(action_index["u_ech"])
                safe_ech = float(safe_action["u_ech"])
                ech_low, ech_high = _ACTION_BOUNDS["u_ech"]
                action[u_ech_index] = float(
                    np.clip(
                        float(action[u_ech_index]),
                        max(ech_low, safe_ech - ech_safe_action_delta_clip),
                        min(ech_high, safe_ech + ech_safe_action_delta_clip),
                    )
                )
            if (
                gt_safe_action_delta_clip > 0.0
                and "u_gt" in action_index
                and "u_gt" not in frozen_action_keys
            ):
                u_gt_index = int(action_index["u_gt"])
                safe_gt = float(safe_action["u_gt"])
                action[u_gt_index] = float(
                    np.clip(
                        float(action[u_gt_index]),
                        max(-1.0, safe_gt - gt_safe_action_delta_clip),
                        min(1.0, safe_gt + gt_safe_action_delta_clip),
                    )
                )
            if (
                bes_safe_action_delta_clip > 0.0
                and "u_bes" in action_index
                and "u_bes" not in frozen_action_keys
            ):
                u_bes_index = int(action_index["u_bes"])
                safe_bes = float(safe_action["u_bes"])
                action[u_bes_index] = float(
                    np.clip(
                        float(action[u_bes_index]),
                        max(-1.0, safe_bes - bes_safe_action_delta_clip),
                        min(1.0, safe_bes + bes_safe_action_delta_clip),
                    )
                )
            if (
                boiler_safe_action_delta_clip > 0.0
                and "u_boiler" in action_index
                and "u_boiler" not in frozen_action_keys
            ):
                u_boiler_index = int(action_index["u_boiler"])
                safe_boiler = float(safe_action["u_boiler"])
                boiler_low, boiler_high = _ACTION_BOUNDS["u_boiler"]
                action[u_boiler_index] = float(
                    np.clip(
                        float(action[u_boiler_index]),
                        max(boiler_low, safe_boiler - boiler_safe_action_delta_clip),
                        min(boiler_high, safe_boiler + boiler_safe_action_delta_clip),
                    )
                )
            if (
                tes_safe_action_delta_clip > 0.0
                and "u_tes" in action_index
                and "u_tes" not in frozen_action_keys
            ):
                u_tes_index = int(action_index["u_tes"])
                safe_tes = float(safe_action["u_tes"])
                action[u_tes_index] = float(
                    np.clip(
                        float(action[u_tes_index]),
                        max(-1.0, safe_tes - tes_safe_action_delta_clip),
                        min(1.0, safe_tes + tes_safe_action_delta_clip),
                    )
                )
            for key in frozen_action_keys:
                action[int(action_index[key])] = float(safe_action[key])
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
    surrogate_audit = None
    surrogate_audit_summary: dict[str, Any] = {}
    surrogate_obs_rows: list[np.ndarray] = []
    surrogate_exec_hat_rows: list[np.ndarray] = []
    surrogate_exec_rows: list[np.ndarray] = []
    surrogate_torch = None
    surrogate_checkpoint = str(metadata.get("projection_surrogate_checkpoint_path", "")).strip()
    if surrogate_checkpoint and Path(surrogate_checkpoint).exists():
        try:
            surrogate_audit = FrozenProjectionSurrogate(
                checkpoint_path=surrogate_checkpoint,
                env_config=config,
                observation_keys=tuple(str(key) for key in metadata.get("observation_keys", ())),
                action_keys=tuple(str(key) for key in metadata.get("action_keys", ())),
                device=device,
            )
            surrogate_torch, _, _, _ = _require_torch_modules()
            surrogate_audit_summary = {"enabled": True}
        except Exception as error:
            surrogate_audit = None
            surrogate_torch = None
            surrogate_audit_summary = {
                "enabled": False,
                "error": str(error),
            }
    env = CCHPPhysicalEnv(exogenous_df=eval_df, config=config, seed=seed)
    observation, _ = env.reset(seed=seed, episode_df=eval_df)
    terminated = False
    total_reward = 0.0
    step_rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}

    while not terminated:
        action = predictor(observation)
        surrogate_exec_hat = None
        obs_vector = None
        if surrogate_audit is not None and surrogate_torch is not None and isinstance(observation, Mapping):
            obs_vector = build_feature_vector(
                observation=observation,
                feature_keys=tuple(str(key) for key in metadata.get("observation_keys", ())),
            ).astype(np.float32)
            action_vector = np.asarray(
                [_safe_float(action.get(key, 0.0)) for key in metadata.get("action_keys", ())],
                dtype=np.float32,
            )
            obs_tensor = surrogate_torch.as_tensor(
                obs_vector.reshape(1, -1),
                dtype=surrogate_audit.action_low.dtype,
                device=surrogate_audit.action_low.device,
            )
            action_tensor = surrogate_torch.as_tensor(
                action_vector.reshape(1, -1),
                dtype=surrogate_audit.action_low.dtype,
                device=surrogate_audit.action_low.device,
            )
            with surrogate_torch.no_grad():
                surrogate_exec_hat = (
                    surrogate_audit.project(obs_tensor, action_tensor)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
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
        if surrogate_exec_hat is not None:
            actual_exec = _extract_action_vector_from_info(
                info,
                prefix="action_exec",
                action_keys=tuple(str(key) for key in metadata.get("action_keys", ())),
            )
            abs_error = np.abs(surrogate_exec_hat - actual_exec)
            signed_error = surrogate_exec_hat - actual_exec
            for index, key in enumerate(tuple(str(name) for name in metadata.get("action_keys", ()))):
                log_row[f"surrogate_exec_hat_{key}"] = float(surrogate_exec_hat[index])
                log_row[f"surrogate_gap_hat_exec_{key}"] = float(signed_error[index])
            log_row["surrogate_gap_hat_exec_l1"] = float(abs_error.sum())
            log_row["surrogate_gap_hat_exec_l2"] = float(
                np.sqrt(np.square(signed_error).sum())
            )
            log_row["surrogate_gap_hat_exec_max"] = float(abs_error.max())
            if obs_vector is not None:
                surrogate_obs_rows.append(obs_vector.copy())
                surrogate_exec_hat_rows.append(surrogate_exec_hat.copy())
                surrogate_exec_rows.append(actual_exec.copy())
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
    summary["surrogate_audit"] = dict(surrogate_audit_summary)
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
        "surrogate_audit_enabled": bool(surrogate_audit is not None),
    }

    step_df = pd.DataFrame(step_rows)
    if surrogate_obs_rows and surrogate_exec_hat_rows and surrogate_exec_rows:
        eval_audit_summary, eval_audit_df = _build_surrogate_audit_report(
            obs_batch=np.asarray(surrogate_obs_rows, dtype=np.float32),
            predicted_exec_batch=np.asarray(surrogate_exec_hat_rows, dtype=np.float32),
            actual_exec_batch=np.asarray(surrogate_exec_rows, dtype=np.float32),
            observation_index={
                str(key): idx for idx, key in enumerate(metadata.get("observation_keys", ()))
            },
            action_keys=tuple(str(key) for key in metadata.get("action_keys", ())),
            low_abs_margin_threshold=float(metadata.get("dual_abs_margin_k", 0.0)),
            high_cooling_ratio_threshold=float(metadata.get("dual_qc_ratio_th", 0.55)),
            q_total_cooling_cap_mw=float(config.q_abs_cool_cap_mw) + float(config.q_ech_cap_mw),
        )
        if not eval_audit_df.empty:
            eval_audit_path = output_run_dir / "eval" / "surrogate_audit.csv"
            eval_audit_df.to_csv(eval_audit_path, index=False)
            summary["surrogate_audit"] = {
                **eval_audit_summary,
                "enabled": True,
                "csv_path": str(eval_audit_path.resolve()).replace("\\", "/"),
            }
    (output_run_dir / "eval" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    step_df.to_csv(output_run_dir / "eval" / "step_log.csv", index=False)
    write_paper_eval_artifacts(
        output_run_dir / "eval",
        summary=summary,
        step_log=step_df,
        dt_h=float(config.dt_hours),
    )
    return summary
