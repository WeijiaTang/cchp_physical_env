# Ref: docs/spec/task.md
from __future__ import annotations

from dataclasses import fields
import math
from pathlib import Path
from typing import Any, Callable

from ..env.cchp_env import EnvConfig

try:
    import yaml
except ModuleNotFoundError as error:  # pragma: no cover
    raise ModuleNotFoundError(
        "缺少依赖 PyYAML，请先安装后再使用 config.yaml。"
    ) from error


DEFAULT_ENV_BLOCK_KEY = "env"
DEFAULT_TRAINING_BLOCK_KEY = "training"

TRAINING_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    # 默认对齐“论文档”配置（与 src/cchp_physical_env/config/config.yaml 保持一致）
    "policy": "sequence_rule",
    "sequence_adapter": "transformer",
    "history_steps": 32,
    "episode_days": 14,
    "episodes": 800,
    "train_steps": 409_600,
    "batch_size": 256,
    "update_epochs": 8,
    "lr": 1e-4,
    "device": "auto",
    # Task-011：可选 SB3 多算法（SAC/TD3/DDPG/PPO/DQN）
    "sb3_enabled": False,
    "sb3_algo": "ddpg",
    "sb3_backbone": "mlp",
    "sb3_history_steps": 32,
    "sb3_total_timesteps": 2_000_000,
    "sb3_n_envs": 4,
    "sb3_learning_rate": 1e-4,
    "sb3_batch_size": 256,
    "sb3_gamma": 0.99,
    "sb3_vec_norm_obs": True,
    "sb3_vec_norm_reward": True,
    "sb3_eval_freq": 50_000,
    "sb3_eval_episode_days": 14,
    "sb3_eval_window_pool_size": 12,
    "sb3_eval_window_count": 4,
    "sb3_ppo_warm_start_enabled": True,
    "sb3_residual_enabled": True,
    "sb3_residual_policy": "rule",
    "sb3_residual_scale": 0.25,
    "sb3_ppo_warm_start_samples": 16384,
    "sb3_ppo_warm_start_epochs": 4,
    "sb3_ppo_warm_start_batch_size": 256,
    "sb3_ppo_warm_start_lr": 1e-4,
    "sb3_offpolicy_prefill_enabled": True,
    "sb3_offpolicy_prefill_steps": 20_000,
    "sb3_offpolicy_prefill_policy": "rule",
    "sb3_ppo_n_steps": 2048,
    "sb3_ppo_gae_lambda": 0.95,
    "sb3_ppo_ent_coef": 0.0,
    "sb3_ppo_clip_range": 0.2,
    "sb3_dqn_action_mode": "rb_v1",
    "sb3_dqn_target_update_interval": 1_000,
    "sb3_dqn_exploration_fraction": 0.3,
    "sb3_dqn_exploration_initial_eps": 1.0,
    "sb3_dqn_exploration_final_eps": 0.05,
    "sb3_learning_starts": 20_000,
    "sb3_train_freq": 4,
    "sb3_gradient_steps": 4,
    "sb3_tau": 0.005,
    "sb3_action_noise_std": 0.05,
    # Off-policy algorithms (SAC/TD3/DDPG/DQN) use a replay buffer.
    # With window observations (K,D) and n_envs>1, the default SB3 buffer_size=1e6 can easily OOM.
    "sb3_buffer_size": 100_000,
    "sb3_optimize_memory_usage": True,
    "sb3_best_gate_enabled": True,
    "sb3_best_gate_electric_min": 1.0,
    "sb3_best_gate_heat_min": 0.99,
    "sb3_best_gate_cool_min": 0.99,
    "sb3_plateau_control_enabled": True,
    "sb3_plateau_patience_evals": 10,
    "sb3_plateau_lr_decay_factor": 0.5,
    "sb3_plateau_min_lr": 5e-5,
    "sb3_plateau_early_stop_patience_evals": 999,
    # Task-012：PAFC-TD3 通用 train/eval 路由默认项
    # 这里默认切到“强经济性”配置：更长预算、较弱保守锚定、
    # 更强经济引导、以及更稳的 checkpoint 验证。
    "pafc_projection_surrogate_checkpoint": "",
    "pafc_episode_days": 14,
    "pafc_total_env_steps": 262_144,
    "pafc_warmup_steps": 4_096,
    "pafc_actor_warmup_steps": 0,
    "pafc_replay_capacity": 100_000,
    "pafc_batch_size": 256,
    "pafc_updates_per_step": 1,
    "pafc_gamma": 0.99,
    "pafc_tau": 0.005,
    "pafc_actor_lr": 1e-4,
    "pafc_critic_lr": 3e-4,
    "pafc_dual_lr": 5e-3,
    "pafc_dual_warmup_steps": 8_192,
    "pafc_actor_delay": 2,
    "pafc_exploration_noise_std": 0.06,
    "pafc_target_policy_noise_std": 0.06,
    "pafc_target_noise_clip": 0.12,
    "pafc_gap_penalty_coef": 0.2,
    "pafc_exec_action_anchor_coef": 1.5,
    "pafc_exec_action_anchor_safe_floor": 0.05,
    "pafc_gt_off_deadband_ratio": 0.0,
    "pafc_abs_ready_focus_coef": 0.25,
    "pafc_invalid_abs_penalty_coef": 0.25,
    "pafc_economic_boiler_proxy_coef": 0.10,
    "pafc_economic_abs_tradeoff_coef": 0.05,
    "pafc_economic_gt_grid_proxy_coef": 0.50,
    "pafc_economic_gt_distill_coef": 0.10,
    "pafc_economic_teacher_distill_coef": 0.25,
    "pafc_economic_teacher_safe_preserve_coef": 1.0,
    "pafc_economic_teacher_safe_preserve_low_margin_boost": 0.75,
    "pafc_economic_teacher_safe_preserve_high_cooling_boost": 1.0,
    "pafc_economic_teacher_safe_preserve_joint_boost": 1.0,
    "pafc_economic_teacher_mismatch_focus_coef": 0.0,
    "pafc_economic_teacher_mismatch_focus_min_scale": 0.75,
    "pafc_economic_teacher_mismatch_focus_max_scale": 2.5,
    "pafc_economic_teacher_proxy_advantage_min": 0.02,
    "pafc_economic_teacher_gt_proxy_advantage_min": 0.01,
    "pafc_economic_teacher_bes_proxy_advantage_min": 0.002,
    "pafc_economic_teacher_max_safe_abs_risk_gap": 0.05,
    "pafc_economic_teacher_projection_gap_max": 0.20,
    "pafc_economic_teacher_gt_projection_gap_max": 1.0,
    "pafc_economic_teacher_bes_price_opportunity_min": 0.10,
    "pafc_economic_teacher_bes_anchor_preserve_scale": 0.85,
    "pafc_economic_teacher_warm_start_weight": 4.0,
    "pafc_economic_teacher_prefill_replay_boost": 2,
    "pafc_economic_teacher_gt_action_weight": 2.0,
    "pafc_economic_teacher_bes_action_weight": 1.5,
    "pafc_economic_teacher_tes_action_weight": 0.5,
    "pafc_economic_teacher_full_year_warm_start_samples": 4096,
    "pafc_economic_teacher_full_year_warm_start_epochs": 4,
    "pafc_economic_gt_full_year_warm_start_samples": 0,
    "pafc_economic_gt_full_year_warm_start_epochs": 0,
    "pafc_economic_gt_full_year_warm_start_u_weight": 0.0,
    "pafc_economic_bes_distill_coef": 0.15,
    "pafc_economic_bes_prior_u": 0.35,
    "pafc_economic_bes_charge_u_scale": 1.8,
    "pafc_economic_bes_discharge_u_scale": 1.0,
    "pafc_economic_bes_charge_weight": 2.0,
    "pafc_economic_bes_discharge_weight": 1.0,
    "pafc_economic_bes_charge_pressure_bonus": 1.0,
    "pafc_economic_bes_charge_soc_ceiling": 0.75,
    "pafc_economic_bes_discharge_soc_floor": 0.35,
    "pafc_economic_bes_full_year_warm_start_samples": 4096,
    "pafc_economic_bes_full_year_warm_start_epochs": 2,
    "pafc_economic_bes_full_year_warm_start_u_weight": 4.0,
    "pafc_economic_bes_warm_start_economic_anchor_weight": 0.0,
    "pafc_economic_bes_warm_start_fallback_anchor_weight": 0.0,
    "pafc_economic_bes_teacher_selection_priority_boost": 0.75,
    "pafc_economic_bes_economic_source_priority_bonus": 0.10,
    "pafc_economic_bes_economic_source_min_share": 0.75,
    "pafc_economic_bes_idle_economic_source_min_share": 0.75,
    "pafc_economic_bes_teacher_target_min_share": 0.0,
    "pafc_economic_bes_anchor_max_scale": 1.0,
    "pafc_surrogate_actor_trust_coef": 0.60,
    "pafc_surrogate_actor_trust_min_scale": 0.10,
    "pafc_actor_low_trust_raw_fallback_keys": [],
    "pafc_state_feasible_action_shaping_enabled": True,
    "pafc_abs_min_on_gate_th": 0.75,
    "pafc_abs_min_on_u_margin": 0.02,
    "pafc_expert_prefill_policy": "easy_rule_abs",
    "pafc_expert_prefill_checkpoint_path": "",
    "pafc_expert_prefill_economic_policy": "checkpoint",
    "pafc_expert_prefill_economic_checkpoint_path": "",
    "pafc_frozen_action_keys": [],
    "pafc_frozen_action_safe_checkpoint_path": "",
    "pafc_gt_safe_action_delta_clip": 0.0,
    "pafc_bes_safe_action_delta_clip": 0.0,
    "pafc_boiler_safe_action_delta_clip": 0.0,
    "pafc_abs_safe_action_delta_clip": 0.0,
    "pafc_ech_safe_action_delta_clip": 0.0,
    "pafc_tes_safe_action_delta_clip": 0.0,
    "pafc_expert_prefill_steps": 4_096,
    "pafc_expert_prefill_cooling_bias": 0.5,
    "pafc_expert_prefill_abs_replay_boost": 0,
    "pafc_expert_prefill_abs_exec_threshold": 0.05,
    "pafc_expert_prefill_abs_window_mining_candidates": 8,
    "pafc_dual_abs_margin_k": 1.25,
    "pafc_dual_qc_ratio_th": 0.55,
    "pafc_dual_heat_backup_ratio_th": 0.10,
    "pafc_dual_safe_abs_u_th": 0.60,
    "pafc_actor_warm_start_epochs": 4,
    "pafc_actor_warm_start_batch_size": 256,
    "pafc_actor_warm_start_lr": 1e-4,
    "pafc_checkpoint_interval_steps": 16_384,
    "pafc_eval_window_pool_size": 16,
    "pafc_eval_window_count": 8,
    "pafc_best_gate_enabled": True,
    "pafc_best_gate_electric_min": 1.0,
    "pafc_best_gate_heat_min": 0.99,
    "pafc_best_gate_cool_min": 0.99,
    "pafc_plateau_control_enabled": True,
    "pafc_plateau_patience_evals": 4,
    "pafc_plateau_lr_decay_factor": 0.5,
    "pafc_plateau_min_actor_lr": 2.5e-5,
    "pafc_plateau_min_critic_lr": 1e-4,
    "pafc_plateau_early_stop_patience_evals": 8,
    "pafc_hidden_dims": [256, 256, 256],
}

# env 参数校验规则表：
# - Option-C：yaml 改什么就是什么，因此 env 块要求“全量字段显式给出”，避免代码默认值污染实验口径。
# - 表驱动的目的是把“字段类型/枚举范围/数值范围”集中管理，减少散落的 if/elif 分支。
ENV_ENUM_OPTIONS: dict[str, set[str]] = {
    "constraint_mode": {"physics_in_loop", "reward_only"},
    "physics_backend": {"tespy"},
    "bes_init_strategy": {"fixed", "min", "max", "half", "random"},
    "oracle_mpc_mode": {"strict", "debug"},
}
ENV_ENUM_ERROR_MESSAGES: dict[str, str] = {
    "constraint_mode": "constraint_mode 仅支持 physics_in_loop/reward_only。",
    "physics_backend": "physics_backend 仅支持 tespy。",
    "bes_init_strategy": "bes_init_strategy 仅支持 fixed/min/max/half/random。",
    "oracle_mpc_mode": "oracle_mpc_mode 仅支持 strict/debug。",
}
ENV_STRIP_STRING_KEYS = {"pyomo_solver"}
ENV_BOOL_KEYS = {
    "bes_dod_add_calendar_age",
    "abs_gate_enabled",
    "abs_boiler_drive_enabled",
    "gt_action_smoothing_enabled",
    "gt_dynamic_om_enabled",
    "heat_backup_shield_enabled",
    "oracle_mpc_abs_enabled",
    "oracle_mpc_hard_reliability",
    "oracle_mpc_heat_backup_repair_enabled",
    "oracle_mpc_cool_backup_repair_enabled",
}
# 数值范围规则：仅对需要额外范围约束的字段登记，其余数值字段只做“类型 + finite”校验。
ENV_NUMERIC_RULES: dict[str, tuple[Callable[[float], bool], str]] = {
    "sell_price_ratio": (
        lambda value: 0.0 <= value <= 1.0,
        "sell_price_ratio 必须在 [0,1]。",
    ),
    "sell_price_cap_per_mwh": (
        lambda value: value >= 0.0,
        "sell_price_cap_per_mwh 必须 >= 0（0 表示不封顶）。",
    ),
    "penalty_export_per_mwh": (
        lambda value: value >= 0.0,
        "penalty_export_per_mwh 必须 >= 0。",
    ),
    "grid_export_soft_cap_mw": (
        lambda value: value >= 0.0,
        "grid_export_soft_cap_mw 必须 >= 0。",
    ),
    "penalty_export_over_soft_cap_per_mwh": (
        lambda value: value >= 0.0,
        "penalty_export_over_soft_cap_per_mwh 必须 >= 0。",
    ),
    "gt_cycle_cost": (
        lambda value: value >= 0.0,
        "gt_cycle_cost 必须 >= 0。",
    ),
    "gt_cycle_hours": (
        lambda value: value > 0.0,
        "gt_cycle_hours 必须 > 0。",
    ),
    "bes_self_discharge_per_hour": (
        lambda value: 0.0 <= value <= 1.0,
        "bes_self_discharge_per_hour 必须在 [0,1]。",
    ),
    "bes_aux_equip_eff": (
        lambda value: 0.0 < value <= 1.0,
        "bes_aux_equip_eff 必须在 (0,1]。",
    ),
    "bes_dod_battery_capex_per_mwh": (
        lambda value: value >= 0.0,
        "bes_dod_battery_capex_per_mwh 必须 >= 0。",
    ),
    "bes_dod_k_p": (
        lambda value: value > 0.0,
        "bes_dod_k_p 必须 > 0。",
    ),
    "bes_dod_n_fail_100": (
        lambda value: value > 0.0,
        "bes_dod_n_fail_100 必须 > 0。",
    ),
    "bes_dod_battery_life_years": (
        lambda value: value > 0.0,
        "bes_dod_battery_life_years 必须 > 0。",
    ),
    "abs_gate_scale_k": (
        lambda value: value > 0.0,
        "abs_gate_scale_k 必须 > 0。",
    ),
    "abs_t_drive_min_k": (
        lambda value: value > 273.15,
        "abs_t_drive_min_k 必须 > 273.15K。",
    ),
    "abs_t_drive_ref_k": (
        lambda value: value > 273.15,
        "abs_t_drive_ref_k 必须 > 273.15K。",
    ),
    "abs_cop_min_fraction": (
        lambda value: 0.0 <= value <= 1.0,
        "abs_cop_min_fraction 必须在 [0,1]。",
    ),
    "abs_deadzone_gate_th": (
        lambda value: 0.0 <= value <= 1.0,
        "abs_deadzone_gate_th 必须在 [0,1]。",
    ),
    "abs_deadzone_u_th": (
        lambda value: 0.0 <= value <= 1.0,
        "abs_deadzone_u_th 必须在 [0,1]。",
    ),
    "abs_invalid_req_u_th": (
        lambda value: 0.0 <= value <= 1.0,
        "abs_invalid_req_u_th 必须在 [0,1]。",
    ),
    "abs_invalid_req_gate_th": (
        lambda value: 0.0 <= value <= 1.0,
        "abs_invalid_req_gate_th 必须在 [0,1]。",
    ),
    "penalty_invalid_abs_request": (
        lambda value: value >= 0.0,
        "penalty_invalid_abs_request 必须 >= 0。",
    ),
    "ech_cop_partload_min_fraction": (
        lambda value: 0.0 < value <= 1.0,
        "ech_cop_partload_min_fraction 必须在 (0,1]。",
    ),
    "ech_cop_partload_curve_exp": (
        lambda value: value > 0.0,
        "ech_cop_partload_curve_exp 必须 > 0。",
    ),
    "abs_boiler_assist_max_mw": (
        lambda value: value >= 0.0,
        "abs_boiler_assist_max_mw 必须 >= 0。",
    ),
    "abs_boiler_assist_boiler_fraction": (
        lambda value: 0.0 <= value <= 1.0,
        "abs_boiler_assist_boiler_fraction 必须在 [0,1]。",
    ),
    "gt_min_on_steps": (
        lambda value: value >= 0.0,
        "gt_min_on_steps 必须 >= 0。",
    ),
    "gt_min_off_steps": (
        lambda value: value >= 0.0,
        "gt_min_off_steps 必须 >= 0。",
    ),
    "penalty_gt_toggle": (
        lambda value: value >= 0.0,
        "penalty_gt_toggle 必须 >= 0。",
    ),
    "penalty_gt_delta_mw": (
        lambda value: value >= 0.0,
        "penalty_gt_delta_mw 必须 >= 0。",
    ),
    "heat_unmet_th_mw": (
        lambda value: value >= 0.0,
        "heat_unmet_th_mw 必须 >= 0。",
    ),
    "cool_unmet_th_mw": (
        lambda value: value >= 0.0,
        "cool_unmet_th_mw 必须 >= 0。",
    ),
    "heat_backup_idle_th_mw": (
        lambda value: value >= 0.0,
        "heat_backup_idle_th_mw 必须 >= 0。",
    ),
    "cool_backup_idle_th_mw": (
        lambda value: value >= 0.0,
        "cool_backup_idle_th_mw 必须 >= 0。",
    ),
    "penalty_idle_heat_backup": (
        lambda value: value >= 0.0,
        "penalty_idle_heat_backup 必须 >= 0。",
    ),
    "penalty_idle_cool_backup": (
        lambda value: value >= 0.0,
        "penalty_idle_cool_backup 必须 >= 0。",
    ),
    "heat_backup_shield_margin_mw": (
        lambda value: value >= 0.0,
        "heat_backup_shield_margin_mw 必须 >= 0。",
    ),
    "oracle_mpc_max_unmet_e_mw": (
        lambda value: value >= 0.0,
        "oracle_mpc_max_unmet_e_mw 必须 >= 0。",
    ),
    "oracle_mpc_max_unmet_h_mw": (
        lambda value: value >= 0.0,
        "oracle_mpc_max_unmet_h_mw 必须 >= 0。",
    ),
    "oracle_mpc_max_unmet_c_mw": (
        lambda value: value >= 0.0,
        "oracle_mpc_max_unmet_c_mw 必须 >= 0。",
    ),
    "oracle_mpc_hard_unmet_penalty_per_mwh": (
        lambda value: value >= 0.0,
        "oracle_mpc_hard_unmet_penalty_per_mwh 必须 >= 0。",
    ),
    "oracle_mpc_tes_terminal_reserve_mwh": (
        lambda value: value >= 0.0,
        "oracle_mpc_tes_terminal_reserve_mwh 必须 >= 0。",
    ),
    "oracle_mpc_tes_terminal_reserve_penalty_per_mwh": (
        lambda value: value >= 0.0,
        "oracle_mpc_tes_terminal_reserve_penalty_per_mwh 必须 >= 0。",
    ),
    "oracle_mpc_abs_ready_cooling_threshold_mw": (
        lambda value: value >= 0.0,
        "oracle_mpc_abs_ready_cooling_threshold_mw 必须 >= 0。",
    ),
    "oracle_mpc_abs_ready_reserve_extra_mwh": (
        lambda value: value >= 0.0,
        "oracle_mpc_abs_ready_reserve_extra_mwh 必须 >= 0。",
    ),
    "oracle_mpc_abs_ready_terminal_value_per_mwh": (
        lambda value: value >= 0.0,
        "oracle_mpc_abs_ready_terminal_value_per_mwh 必须 >= 0。",
    ),
    "oracle_mpc_planning_horizon_steps": (
        lambda value: value > 0.0,
        "oracle_mpc_planning_horizon_steps 必须 > 0。",
    ),
    "oracle_mpc_replan_interval_steps": (
        lambda value: value > 0.0,
        "oracle_mpc_replan_interval_steps 必须 > 0。",
    ),
    "oracle_mpc_time_limit_seconds": (
        lambda value: value > 0.0,
        "oracle_mpc_time_limit_seconds 必须 > 0。",
    ),
    "oracle_mpc_mip_relative_gap": (
        lambda value: 0.0 <= value <= 1.0,
        "oracle_mpc_mip_relative_gap 必须在 [0,1]。",
    ),
    "oracle_ga_population_size": (
        lambda value: value >= 2.0,
        "oracle_ga_population_size 必须 >= 2。",
    ),
    "oracle_ga_generations": (
        lambda value: value >= 1.0,
        "oracle_ga_generations 必须 >= 1。",
    ),
    "oracle_ga_elite_count": (
        lambda value: value >= 1.0,
        "oracle_ga_elite_count 必须 >= 1。",
    ),
    "oracle_ga_mutation_scale": (
        lambda value: value > 0.0,
        "oracle_ga_mutation_scale 必须 > 0。",
    ),
}
ENV_LOWERCASE_STRING_KEYS = set(ENV_ENUM_OPTIONS.keys())


def _normalize_lower_text(value: Any) -> str:
    return str(value).strip().lower()


def _normalize_strip_text(value: Any) -> str:
    return str(value).strip()


def _validate_env_numeric_range(key: str, numeric_value: float) -> None:
    rule = ENV_NUMERIC_RULES.get(key)
    if rule is None:
        return
    predicate, error_message = rule
    if not predicate(numeric_value):
        raise ValueError(error_message)


def _normalize_hidden_dims_value(value: Any, *, key: str) -> tuple[int, ...]:
    if value is None:
        raise ValueError(f"{key} 不能为空。")
    if isinstance(value, str):
        tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
    elif isinstance(value, (list, tuple)):
        tokens = list(value)
    else:
        raise ValueError(f"{key} 必须是逗号分隔字符串或整数列表。")
    if not tokens:
        raise ValueError(f"{key} 不能为空。")
    dims = tuple(int(token) for token in tokens)
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"{key} 必须全部 > 0。")
    return dims


def _normalize_string_tuple_value(value: Any, *, key: str) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
    elif isinstance(value, (list, tuple, set)):
        tokens = [str(token).strip() for token in value if str(token).strip()]
    else:
        raise ValueError(f"{key} 必须是逗号分隔字符串或字符串列表。")
    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        item = str(token).strip().lower().replace("-", "_")
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return tuple(normalized)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    raw_content = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if raw_content is None:
        return {}
    if not isinstance(raw_content, dict):
        raise ValueError("配置文件顶层必须是映射（mapping）。")
    return raw_content


def load_env_overrides(path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(path)
    env_block = config.get(DEFAULT_ENV_BLOCK_KEY, {})
    if env_block is None:
        return {}
    if not isinstance(env_block, dict):
        raise ValueError(f"`{DEFAULT_ENV_BLOCK_KEY}` 配置必须是映射（mapping）。")
    return dict(env_block)


def load_training_overrides(path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(path)
    training_block = config.get(DEFAULT_TRAINING_BLOCK_KEY, {})
    if training_block is None:
        return {}
    if not isinstance(training_block, dict):
        raise ValueError(f"`{DEFAULT_TRAINING_BLOCK_KEY}` 配置必须是映射（mapping）。")
    return dict(training_block)


def validate_env_overrides(overrides: dict[str, Any]) -> None:
    allowed = {item.name for item in fields(EnvConfig)}
    unknown = sorted(set(overrides.keys()) - allowed)
    if unknown:
        raise ValueError(f"`env` 包含未知参数: {unknown}")

    # Option-C：yaml 改什么就是什么。为了避免隐式默认值回流到代码，要求 env 里显式给出所有字段。
    missing = sorted(allowed - set(overrides.keys()))
    if missing:
        raise ValueError(
            "`env` 缺少必要参数（Option-C 要求全量配置，避免使用代码默认值）: "
            f"{missing}"
        )

    # 校验流程：
    # 1) 枚举字段：先做 lower/strip 标准化，再检查是否在允许集合内。
    # 2) 特殊字符串字段：仅做 strip + 非空检查（例如 pyomo_solver）。
    # 3) 布尔字段：强制 bool 类型（避免 0/1 混入造成口径歧义）。
    # 4) 其余字段：按数值处理，校验类型、finite，并按 ENV_NUMERIC_RULES 做范围约束。
    for key, value in overrides.items():
        enum_options = ENV_ENUM_OPTIONS.get(key)
        if enum_options is not None:
            if _normalize_lower_text(value) not in enum_options:
                raise ValueError(ENV_ENUM_ERROR_MESSAGES[key])
            continue
        if key in ENV_STRIP_STRING_KEYS:
            if len(_normalize_strip_text(value)) == 0:
                raise ValueError("pyomo_solver 不能为空。")
            continue
        if key in ENV_BOOL_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"{key} 必须是布尔值（true/false）。")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} 必须是数值类型，当前为 {type(value).__name__}")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{key} 必须是有限数值，当前为 {value}")
        _validate_env_numeric_range(key, numeric_value)


def build_env_config_from_overrides(
    overrides: dict[str, Any] | None = None, *, force_constraint_mode: str | None = None
) -> EnvConfig:
    if overrides is None:
        overrides = {}

    validate_env_overrides(overrides)

    values: dict[str, Any] = {}
    for item in fields(EnvConfig):
        key = item.name
        raw_value = overrides.get(key)
        # 类型标准化与 validate_env_overrides 保持一致，确保 EnvConfig 中字段类型稳定。
        if key in ENV_LOWERCASE_STRING_KEYS:
            values[key] = _normalize_lower_text(raw_value)
        elif key in ENV_STRIP_STRING_KEYS:
            values[key] = _normalize_strip_text(raw_value)
        elif key in ENV_BOOL_KEYS:
            values[key] = bool(raw_value)
        else:
            values[key] = float(raw_value)

    if force_constraint_mode is not None:
        mode = _normalize_lower_text(force_constraint_mode)
        if mode not in ENV_ENUM_OPTIONS["constraint_mode"]:
            raise ValueError(f"不支持的 constraint_mode: {force_constraint_mode}")
        values["constraint_mode"] = mode

    if float(values["abs_t_drive_ref_k"]) <= float(values["abs_t_drive_min_k"]):
        raise ValueError("abs_t_drive_ref_k 必须大于 abs_t_drive_min_k。")

    return EnvConfig(**values)


def validate_training_overrides(overrides: dict[str, Any]) -> None:
    overrides = dict(overrides)
    # Backward compatibility: old configs may still carry sb3_eval_window_seed.
    # Under the single-entry seed protocol, we ignore this field and always derive
    # the validation-window seed from training.seed.
    overrides.pop("sb3_eval_window_seed", None)
    allowed = set(TRAINING_DEFAULTS.keys())
    unknown = sorted(set(overrides.keys()) - allowed)
    if unknown:
        raise ValueError(f"`training` 包含未知参数: {unknown}")

    bool_keys = {
        "sb3_enabled",
        "sb3_optimize_memory_usage",
        "sb3_vec_norm_obs",
        "sb3_vec_norm_reward",
        "sb3_ppo_warm_start_enabled",
        "sb3_residual_enabled",
        "sb3_offpolicy_prefill_enabled",
        "sb3_best_gate_enabled",
        "sb3_plateau_control_enabled",
        "pafc_state_feasible_action_shaping_enabled",
        "pafc_best_gate_enabled",
        "pafc_plateau_control_enabled",
    }
    int_keys = {
        "seed",
        "history_steps",
        "episode_days",
        "episodes",
        "train_steps",
        "batch_size",
        "update_epochs",
        "sb3_history_steps",
        "sb3_total_timesteps",
        "sb3_n_envs",
        "sb3_batch_size",
        "sb3_buffer_size",
        "sb3_eval_freq",
        "sb3_eval_episode_days",
        "sb3_ppo_warm_start_samples",
        "sb3_ppo_warm_start_epochs",
        "sb3_ppo_warm_start_batch_size",
        "sb3_ppo_n_steps",
        "sb3_dqn_target_update_interval",
        "sb3_train_freq",
        "sb3_gradient_steps",
        "sb3_plateau_patience_evals",
        "sb3_plateau_early_stop_patience_evals",
        "pafc_episode_days",
        "pafc_total_env_steps",
        "pafc_actor_warmup_steps",
        "pafc_replay_capacity",
        "pafc_batch_size",
        "pafc_updates_per_step",
        "pafc_dual_warmup_steps",
        "pafc_actor_delay",
        "pafc_expert_prefill_steps",
        "pafc_expert_prefill_abs_replay_boost",
        "pafc_economic_teacher_prefill_replay_boost",
        "pafc_expert_prefill_abs_window_mining_candidates",
        "pafc_economic_teacher_full_year_warm_start_samples",
        "pafc_economic_teacher_full_year_warm_start_epochs",
        "pafc_economic_gt_full_year_warm_start_samples",
        "pafc_economic_gt_full_year_warm_start_epochs",
        "pafc_economic_bes_full_year_warm_start_samples",
        "pafc_economic_bes_full_year_warm_start_epochs",
        "pafc_actor_warm_start_epochs",
        "pafc_actor_warm_start_batch_size",
        "pafc_checkpoint_interval_steps",
        "pafc_eval_window_pool_size",
        "pafc_eval_window_count",
        "pafc_plateau_patience_evals",
        "pafc_plateau_early_stop_patience_evals",
    }
    for key, value in overrides.items():
        if key in {
            "policy",
            "sequence_adapter",
            "device",
            "sb3_dqn_action_mode",
            "sb3_offpolicy_prefill_policy",
            "sb3_residual_policy",
            "pafc_expert_prefill_policy",
            "pafc_expert_prefill_economic_policy",
        }:
            if len(str(value).strip()) == 0:
                raise ValueError(f"{key} 不能为空。")
            continue
        if key in {
            "pafc_expert_prefill_checkpoint_path",
            "pafc_expert_prefill_economic_checkpoint_path",
            "pafc_frozen_action_safe_checkpoint_path",
        }:
            if not isinstance(value, str):
                raise ValueError(f"{key} 必须是字符串路径。")
            continue
        if key == "pafc_projection_surrogate_checkpoint":
            if not isinstance(value, str):
                raise ValueError("pafc_projection_surrogate_checkpoint 必须是字符串路径。")
            continue
        if key == "pafc_frozen_action_keys":
            _normalize_string_tuple_value(value, key=key)
            continue
        if key == "pafc_hidden_dims":
            _normalize_hidden_dims_value(value, key=key)
            continue
        if key in {"sb3_algo"}:
            if len(str(value).strip()) == 0:
                raise ValueError("sb3_algo 不能为空。")
            continue
        if key in {"sb3_backbone"}:
            if len(str(value).strip()) == 0:
                raise ValueError("sb3_backbone 不能为空。")
            continue
        if key in bool_keys:
            if not isinstance(value, bool):
                raise ValueError(f"{key} 必须是布尔值（true/false）。")
            continue
        if key in int_keys:
            # seed 允许逗号分隔的多seed字符串（如 "0,42,123"），由 _normalize_seed_list 解析
            # 其余 int_keys 允许字符串形式的单个整数（如 CLI 传入的 "42"）
            if isinstance(value, bool):
                raise ValueError(f"{key} 必须是整数类型。")
            if not isinstance(value, (int, float)):
                raw = str(value).strip()
                if key == "seed":
                    # 多seed字符串：每个 token 必须是合法整数
                    tokens = [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
                    if not tokens or not all(t.lstrip("-").isdigit() for t in tokens):
                        raise ValueError(f"{key} 必须是整数或逗号分隔的整数列表。")
                else:
                    try:
                        int(raw)
                    except (ValueError, TypeError):
                        raise ValueError(f"{key} 必须是整数类型。")
                    if key in {
                        "pafc_expert_prefill_abs_replay_boost",
                        "pafc_economic_teacher_prefill_replay_boost",
                        "pafc_expert_prefill_abs_window_mining_candidates",
                        "pafc_economic_teacher_full_year_warm_start_samples",
                        "pafc_economic_teacher_full_year_warm_start_epochs",
                        "pafc_economic_gt_full_year_warm_start_samples",
                        "pafc_economic_gt_full_year_warm_start_epochs",
                        "pafc_economic_bes_full_year_warm_start_samples",
                        "pafc_economic_bes_full_year_warm_start_epochs",
                        "pafc_actor_warm_start_epochs",
                        "pafc_checkpoint_interval_steps",
                        "pafc_eval_window_pool_size",
                        "pafc_eval_window_count",
                    }:
                        if int(raw) < 0:
                            raise ValueError(f"{key} 必须 >= 0。")
                    elif int(raw) <= 0:
                        raise ValueError(f"{key} 必须 > 0。")
            else:
                if key in {
                    "pafc_expert_prefill_abs_replay_boost",
                    "pafc_economic_teacher_prefill_replay_boost",
                    "pafc_expert_prefill_abs_window_mining_candidates",
                    "pafc_economic_teacher_full_year_warm_start_samples",
                    "pafc_economic_teacher_full_year_warm_start_epochs",
                    "pafc_economic_gt_full_year_warm_start_samples",
                    "pafc_economic_gt_full_year_warm_start_epochs",
                    "pafc_economic_bes_full_year_warm_start_samples",
                    "pafc_economic_bes_full_year_warm_start_epochs",
                    "pafc_actor_warmup_steps",
                    "pafc_actor_warm_start_epochs",
                    "pafc_checkpoint_interval_steps",
                    "pafc_eval_window_pool_size",
                    "pafc_eval_window_count",
                }:
                    if int(value) < 0:
                        raise ValueError(f"{key} 必须 >= 0。")
                elif int(value) <= 0 and key != "seed":
                    raise ValueError(f"{key} 必须 > 0。")
            continue
        if key in {"lr", "sb3_learning_rate", "sb3_plateau_min_lr"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是数值类型。")
            if float(value) <= 0.0:
                raise ValueError(f"{key} 必须 > 0。")
            continue
        if key in {"sb3_ppo_warm_start_lr"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是数值类型。")
            if float(value) <= 0.0:
                raise ValueError(f"{key} 必须 > 0。")
            continue
        if key in {"sb3_learning_starts"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是整数类型。")
            if int(value) < 0:
                raise ValueError(f"{key} 必须 >= 0。")
            continue
        if key in {"sb3_offpolicy_prefill_steps"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是整数类型。")
            if int(value) < 0:
                raise ValueError(f"{key} 必须 >= 0。")
            continue
        if key in {"pafc_warmup_steps", "pafc_actor_warmup_steps"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是整数类型。")
            if int(value) < 0:
                raise ValueError(f"{key} 必须 >= 0。")
            continue
        if key in {"sb3_eval_window_pool_size", "sb3_eval_window_count"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是整数类型。")
            if int(value) < 0:
                raise ValueError(f"{key} 必须 >= 0。")
            continue
        if key == "pafc_gamma":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("pafc_gamma 必须是数值类型。")
            numeric = float(value)
            if not (0.0 <= numeric <= 1.0):
                raise ValueError("pafc_gamma 必须在 [0,1]。")
            continue
        if key == "sb3_gamma":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("sb3_gamma 必须是数值类型。")
            numeric = float(value)
            if not (0.0 < numeric <= 1.0):
                raise ValueError("sb3_gamma 必须在 (0,1]。")
            continue
        if key == "pafc_tau":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("pafc_tau 必须是数值类型。")
            numeric = float(value)
            if not (0.0 < numeric <= 1.0):
                raise ValueError("pafc_tau 必须在 (0,1]。")
            continue
        if key == "sb3_ppo_gae_lambda":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("sb3_ppo_gae_lambda 必须是数值类型。")
            numeric = float(value)
            if not (0.0 < numeric <= 1.0):
                raise ValueError("sb3_ppo_gae_lambda 必须在 (0,1]。")
            continue
        if key in {"pafc_actor_lr", "pafc_critic_lr"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是数值类型。")
            if float(value) <= 0.0:
                raise ValueError(f"{key} 必须 > 0。")
            continue
        if key in {
            "pafc_dual_lr",
            "pafc_exploration_noise_std",
            "pafc_target_policy_noise_std",
            "pafc_target_noise_clip",
            "pafc_gap_penalty_coef",
            "pafc_exec_action_anchor_coef",
            "pafc_exec_action_anchor_safe_floor",
            "pafc_gt_off_deadband_ratio",
            "pafc_abs_ready_focus_coef",
            "pafc_invalid_abs_penalty_coef",
            "pafc_economic_boiler_proxy_coef",
            "pafc_economic_abs_tradeoff_coef",
            "pafc_economic_gt_grid_proxy_coef",
            "pafc_economic_gt_distill_coef",
            "pafc_economic_teacher_distill_coef",
            "pafc_economic_teacher_safe_preserve_coef",
            "pafc_economic_teacher_safe_preserve_low_margin_boost",
            "pafc_economic_teacher_safe_preserve_high_cooling_boost",
            "pafc_economic_teacher_safe_preserve_joint_boost",
            "pafc_economic_teacher_mismatch_focus_coef",
            "pafc_economic_teacher_mismatch_focus_min_scale",
            "pafc_economic_teacher_mismatch_focus_max_scale",
            "pafc_economic_teacher_proxy_advantage_min",
            "pafc_economic_teacher_gt_proxy_advantage_min",
            "pafc_economic_teacher_bes_proxy_advantage_min",
            "pafc_economic_teacher_max_safe_abs_risk_gap",
            "pafc_economic_teacher_projection_gap_max",
            "pafc_economic_teacher_gt_projection_gap_max",
            "pafc_economic_teacher_bes_price_opportunity_min",
            "pafc_economic_teacher_bes_anchor_preserve_scale",
            "pafc_economic_teacher_warm_start_weight",
            "pafc_economic_teacher_gt_action_weight",
            "pafc_economic_teacher_bes_action_weight",
            "pafc_economic_teacher_tes_action_weight",
            "pafc_economic_teacher_full_year_warm_start_samples",
            "pafc_economic_teacher_full_year_warm_start_epochs",
            "pafc_economic_gt_full_year_warm_start_samples",
            "pafc_economic_gt_full_year_warm_start_epochs",
            "pafc_economic_gt_full_year_warm_start_u_weight",
            "pafc_economic_bes_distill_coef",
            "pafc_economic_bes_prior_u",
            "pafc_economic_bes_charge_u_scale",
            "pafc_economic_bes_discharge_u_scale",
            "pafc_economic_bes_charge_weight",
            "pafc_economic_bes_discharge_weight",
            "pafc_economic_bes_charge_pressure_bonus",
            "pafc_economic_bes_charge_soc_ceiling",
            "pafc_economic_bes_discharge_soc_floor",
            "pafc_economic_bes_full_year_warm_start_u_weight",
            "pafc_economic_bes_teacher_selection_priority_boost",
            "pafc_economic_bes_economic_source_priority_bonus",
            "pafc_economic_bes_economic_source_min_share",
            "pafc_economic_bes_idle_economic_source_min_share",
            "pafc_economic_bes_teacher_target_min_share",
            "pafc_surrogate_actor_trust_coef",
            "pafc_surrogate_actor_trust_min_scale",
            "pafc_abs_min_on_gate_th",
            "pafc_abs_min_on_u_margin",
            "pafc_expert_prefill_cooling_bias",
            "pafc_expert_prefill_abs_exec_threshold",
            "pafc_dual_abs_margin_k",
            "pafc_dual_qc_ratio_th",
            "pafc_dual_heat_backup_ratio_th",
            "pafc_dual_safe_abs_u_th",
            "pafc_actor_warm_start_lr",
            "pafc_plateau_lr_decay_factor",
            "pafc_plateau_min_actor_lr",
            "pafc_plateau_min_critic_lr",
        }:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是数值类型。")
            numeric = float(value)
            if key in {
                "pafc_expert_prefill_cooling_bias",
                "pafc_expert_prefill_abs_exec_threshold",
                "pafc_abs_min_on_gate_th",
                "pafc_dual_safe_abs_u_th",
                "pafc_exec_action_anchor_safe_floor",
                "pafc_gt_off_deadband_ratio",
                "pafc_economic_teacher_proxy_advantage_min",
                "pafc_economic_teacher_gt_proxy_advantage_min",
                "pafc_economic_teacher_bes_proxy_advantage_min",
                "pafc_economic_teacher_max_safe_abs_risk_gap",
                "pafc_economic_teacher_bes_price_opportunity_min",
                "pafc_economic_teacher_bes_anchor_preserve_scale",
                "pafc_economic_bes_prior_u",
                "pafc_economic_bes_charge_soc_ceiling",
                "pafc_economic_bes_discharge_soc_floor",
                "pafc_economic_bes_economic_source_min_share",
                "pafc_economic_bes_idle_economic_source_min_share",
                "pafc_economic_bes_teacher_target_min_share",
                "pafc_surrogate_actor_trust_min_scale",
            }:
                if not (0.0 <= numeric <= 1.0):
                    raise ValueError(f"{key} 必须在 [0,1]。")
            elif numeric < 0.0:
                raise ValueError(f"{key} 必须 >= 0。")
            continue
        if key == "sb3_dqn_exploration_fraction":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("sb3_dqn_exploration_fraction 必须是数值类型。")
            numeric = float(value)
            if not (0.0 < numeric <= 1.0):
                raise ValueError("sb3_dqn_exploration_fraction 必须在 (0,1]。")
            continue
        if key in {
            "sb3_ppo_ent_coef",
            "sb3_action_noise_std",
            "sb3_residual_scale",
        }:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是数值类型。")
            if float(value) < 0.0:
                raise ValueError(f"{key} 必须 >= 0。")
            continue
        if key in {"sb3_plateau_lr_decay_factor"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是数值类型。")
            numeric = float(value)
            if not (0.0 < numeric < 1.0):
                raise ValueError(f"{key} 必须在 (0,1) 内。")
            continue
        if key in {"sb3_best_gate_electric_min", "sb3_best_gate_heat_min", "sb3_best_gate_cool_min"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是数值类型。")
            numeric = float(value)
            if not (0.0 <= numeric <= 1.0):
                raise ValueError(f"{key} 必须在 [0,1]。")
            continue
        if key in {"sb3_dqn_exploration_initial_eps", "sb3_dqn_exploration_final_eps"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是数值类型。")
            numeric = float(value)
            if not (0.0 <= numeric <= 1.0):
                raise ValueError(f"{key} 必须在 [0,1]。")
            continue
        if key in {"sb3_ppo_clip_range", "sb3_tau"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是数值类型。")
            if float(value) <= 0.0:
                raise ValueError(f"{key} 必须 > 0。")
            continue

    policy = str(overrides.get("policy", TRAINING_DEFAULTS["policy"])).strip().lower().replace("-", "_")
    sb3_enabled_flag = bool(overrides.get("sb3_enabled", TRAINING_DEFAULTS.get("sb3_enabled", False)))
    if sb3_enabled_flag:
        # sb3_enabled=true 时，policy 仅作记录，但仍需要校验以避免拼写错误污染实验口径。
        if policy not in {"rule", "easy_rule", "random", "sequence_rule", "milp_mpc", "ga_mpc", "sb3", "pafc_td3"}:
            raise ValueError("training.policy 仅支持 rule/easy_rule/random/sequence_rule/milp_mpc/ga_mpc/sb3/pafc_td3（sb3_enabled=true 时该字段仅作备注，不参与路由）。")
    else:
        if policy not in {"rule", "easy_rule", "random", "sequence_rule", "milp_mpc", "ga_mpc", "pafc_td3"}:
            raise ValueError("training.policy 仅支持 rule/easy_rule/random/sequence_rule/milp_mpc/ga_mpc/pafc_td3（sb3_enabled=false）。")
    sequence_adapter = str(
        overrides.get("sequence_adapter", TRAINING_DEFAULTS["sequence_adapter"])
    ).strip().lower()
    if sequence_adapter not in {"rule", "mlp", "transformer", "mamba"}:
        raise ValueError("training.sequence_adapter 仅支持 rule/mlp/transformer/mamba。")
    device = str(overrides.get("device", TRAINING_DEFAULTS["device"])).strip().lower()
    if device not in {"auto", "cpu", "cuda"} and not device.startswith("cuda:"):
        raise ValueError("training.device 仅支持 auto/cpu/cuda/cuda:<index>。")

    sb3_algo = str(overrides.get("sb3_algo", TRAINING_DEFAULTS["sb3_algo"])).strip().lower()
    if sb3_algo not in {"ppo", "sac", "td3", "ddpg", "dqn"}:
        raise ValueError("training.sb3_algo 仅支持 ppo/sac/td3/ddpg/dqn。")
    sb3_backbone = str(
        overrides.get("sb3_backbone", TRAINING_DEFAULTS["sb3_backbone"])
    ).strip().lower()
    if sb3_backbone not in {"mlp", "transformer", "mamba"}:
        raise ValueError("training.sb3_backbone 仅支持 mlp/transformer/mamba。")
    eval_window_pool_size = int(
        overrides.get("sb3_eval_window_pool_size", TRAINING_DEFAULTS["sb3_eval_window_pool_size"])
    )
    eval_window_count = int(
        overrides.get("sb3_eval_window_count", TRAINING_DEFAULTS["sb3_eval_window_count"])
    )
    if eval_window_pool_size == 0 and eval_window_count != 0:
        raise ValueError("sb3_eval_window_pool_size=0 时，sb3_eval_window_count 也必须为 0。")
    if eval_window_pool_size > 0 and eval_window_count > eval_window_pool_size:
        raise ValueError("sb3_eval_window_count 不能大于 sb3_eval_window_pool_size。")
    pafc_eval_window_pool_size = int(
        overrides.get("pafc_eval_window_pool_size", TRAINING_DEFAULTS["pafc_eval_window_pool_size"])
    )
    pafc_eval_window_count = int(
        overrides.get("pafc_eval_window_count", TRAINING_DEFAULTS["pafc_eval_window_count"])
    )
    if pafc_eval_window_pool_size == 0 and pafc_eval_window_count != 0:
        raise ValueError("pafc_eval_window_pool_size=0 时，pafc_eval_window_count 也必须为 0。")
    if pafc_eval_window_pool_size > 0 and pafc_eval_window_count > pafc_eval_window_pool_size:
        raise ValueError("pafc_eval_window_count 不能大于 pafc_eval_window_pool_size。")
    pafc_checkpoint_interval_steps = int(
        overrides.get("pafc_checkpoint_interval_steps", TRAINING_DEFAULTS["pafc_checkpoint_interval_steps"])
    )
    if pafc_checkpoint_interval_steps < 0:
        raise ValueError("pafc_checkpoint_interval_steps 必须 >= 0（0 表示自动）。")
    residual_policy = str(
        overrides.get("sb3_residual_policy", TRAINING_DEFAULTS["sb3_residual_policy"])
    ).strip().lower().replace("-", "_")
    if residual_policy not in {"easy_rule", "rule"}:
        raise ValueError("training.sb3_residual_policy 当前仅支持 easy_rule/rule。")
    residual_scale = float(
        overrides.get("sb3_residual_scale", TRAINING_DEFAULTS["sb3_residual_scale"])
    )
    if residual_scale < 0.0 or residual_scale > 1.0:
        raise ValueError("training.sb3_residual_scale 必须在 [0,1]。")
    dqn_action_mode = str(
        overrides.get("sb3_dqn_action_mode", TRAINING_DEFAULTS["sb3_dqn_action_mode"])
    ).strip().lower()
    if dqn_action_mode != "rb_v1":
        raise ValueError("training.sb3_dqn_action_mode 当前仅支持 rb_v1。")
    offpolicy_prefill_policy = str(
        overrides.get(
            "sb3_offpolicy_prefill_policy",
            TRAINING_DEFAULTS["sb3_offpolicy_prefill_policy"],
        )
    ).strip().lower().replace("-", "_")
    if offpolicy_prefill_policy not in {"easy_rule", "rule"}:
        raise ValueError("training.sb3_offpolicy_prefill_policy 当前仅支持 easy_rule/rule。")
    pafc_prefill_policy = str(
        overrides.get(
            "pafc_expert_prefill_policy",
            TRAINING_DEFAULTS["pafc_expert_prefill_policy"],
        )
    ).strip().lower().replace("-", "_")
    if pafc_prefill_policy not in {
        "easy_rule",
        "rule",
        "easy_rule_abs",
        "checkpoint",
        "checkpoint_dual",
    }:
        raise ValueError(
            "training.pafc_expert_prefill_policy 当前仅支持 "
            "easy_rule/rule/easy_rule_abs/checkpoint/checkpoint_dual。"
        )
    pafc_prefill_economic_policy = str(
        overrides.get(
            "pafc_expert_prefill_economic_policy",
            TRAINING_DEFAULTS["pafc_expert_prefill_economic_policy"],
        )
    ).strip().lower().replace("-", "_")
    if pafc_prefill_economic_policy not in {"checkpoint", "milp_mpc", "ga_mpc"}:
        raise ValueError(
            "training.pafc_expert_prefill_economic_policy 当前仅支持 "
            "checkpoint/milp_mpc/ga_mpc。"
        )
    pafc_prefill_checkpoint_path = str(
        overrides.get(
            "pafc_expert_prefill_checkpoint_path",
            TRAINING_DEFAULTS["pafc_expert_prefill_checkpoint_path"],
        )
    ).strip()
    if pafc_prefill_policy in {"checkpoint", "checkpoint_dual"} and len(pafc_prefill_checkpoint_path) == 0:
        raise ValueError(
            "training.pafc_expert_prefill_policy=checkpoint/checkpoint_dual 时必须提供 "
            "pafc_expert_prefill_checkpoint_path。"
        )
    pafc_prefill_economic_checkpoint_path = str(
        overrides.get(
            "pafc_expert_prefill_economic_checkpoint_path",
            TRAINING_DEFAULTS["pafc_expert_prefill_economic_checkpoint_path"],
        )
    ).strip()
    if (
        pafc_prefill_policy == "checkpoint_dual"
        and pafc_prefill_economic_policy == "checkpoint"
        and len(pafc_prefill_economic_checkpoint_path) == 0
    ):
        raise ValueError(
            "training.pafc_expert_prefill_policy=checkpoint_dual 且 "
            "training.pafc_expert_prefill_economic_policy=checkpoint 时必须提供 "
            "pafc_expert_prefill_economic_checkpoint_path。"
        )
    pafc_frozen_action_keys = _normalize_string_tuple_value(
        overrides.get("pafc_frozen_action_keys", TRAINING_DEFAULTS["pafc_frozen_action_keys"]),
        key="pafc_frozen_action_keys",
    )
    pafc_frozen_action_safe_checkpoint_path = str(
        overrides.get(
            "pafc_frozen_action_safe_checkpoint_path",
            TRAINING_DEFAULTS["pafc_frozen_action_safe_checkpoint_path"],
        )
    ).strip()
    pafc_tes_safe_action_delta_clip = float(
        overrides.get(
            "pafc_tes_safe_action_delta_clip",
            TRAINING_DEFAULTS["pafc_tes_safe_action_delta_clip"],
        )
    )
    pafc_bes_safe_action_delta_clip = float(
        overrides.get(
            "pafc_bes_safe_action_delta_clip",
            TRAINING_DEFAULTS["pafc_bes_safe_action_delta_clip"],
        )
    )
    pafc_boiler_safe_action_delta_clip = float(
        overrides.get(
            "pafc_boiler_safe_action_delta_clip",
            TRAINING_DEFAULTS["pafc_boiler_safe_action_delta_clip"],
        )
    )
    pafc_gt_safe_action_delta_clip = float(
        overrides.get(
            "pafc_gt_safe_action_delta_clip",
            TRAINING_DEFAULTS["pafc_gt_safe_action_delta_clip"],
        )
    )
    if pafc_frozen_action_keys and len(pafc_frozen_action_safe_checkpoint_path) == 0:
        raise ValueError(
            "training.pafc_frozen_action_keys 非空时必须提供 "
            "pafc_frozen_action_safe_checkpoint_path。"
        )
    if pafc_gt_safe_action_delta_clip < 0.0 or pafc_gt_safe_action_delta_clip > 1.0:
        raise ValueError("training.pafc_gt_safe_action_delta_clip 必须在 [0,1]。")
    if (
        pafc_gt_safe_action_delta_clip > 0.0
        and len(pafc_frozen_action_safe_checkpoint_path) == 0
    ):
        raise ValueError(
            "training.pafc_gt_safe_action_delta_clip > 0 时必须提供 "
            "pafc_frozen_action_safe_checkpoint_path。"
        )
    if pafc_bes_safe_action_delta_clip < 0.0 or pafc_bes_safe_action_delta_clip > 1.0:
        raise ValueError("training.pafc_bes_safe_action_delta_clip 必须在 [0,1]。")
    if (
        pafc_bes_safe_action_delta_clip > 0.0
        and len(pafc_frozen_action_safe_checkpoint_path) == 0
    ):
        raise ValueError(
            "training.pafc_bes_safe_action_delta_clip > 0 时必须提供 "
            "pafc_frozen_action_safe_checkpoint_path。"
        )
    if (
        pafc_boiler_safe_action_delta_clip < 0.0
        or pafc_boiler_safe_action_delta_clip > 1.0
    ):
        raise ValueError("training.pafc_boiler_safe_action_delta_clip 必须在 [0,1]。")
    if (
        pafc_boiler_safe_action_delta_clip > 0.0
        and len(pafc_frozen_action_safe_checkpoint_path) == 0
    ):
        raise ValueError(
            "training.pafc_boiler_safe_action_delta_clip > 0 时必须提供 "
            "pafc_frozen_action_safe_checkpoint_path。"
        )
    if pafc_tes_safe_action_delta_clip < 0.0 or pafc_tes_safe_action_delta_clip > 1.0:
        raise ValueError("training.pafc_tes_safe_action_delta_clip 必须在 [0,1]。")
    if (
        pafc_tes_safe_action_delta_clip > 0.0
        and len(pafc_frozen_action_safe_checkpoint_path) == 0
    ):
        raise ValueError(
            "training.pafc_tes_safe_action_delta_clip > 0 时必须提供 "
            "pafc_frozen_action_safe_checkpoint_path。"
        )
    dqn_initial_eps = float(
        overrides.get(
            "sb3_dqn_exploration_initial_eps",
            TRAINING_DEFAULTS["sb3_dqn_exploration_initial_eps"],
        )
    )
    dqn_final_eps = float(
        overrides.get(
            "sb3_dqn_exploration_final_eps",
            TRAINING_DEFAULTS["sb3_dqn_exploration_final_eps"],
        )
    )
    if dqn_final_eps > dqn_initial_eps:
        raise ValueError("sb3_dqn_exploration_final_eps 不能大于 sb3_dqn_exploration_initial_eps。")
    for key in ("sb3_best_gate_electric_min", "sb3_best_gate_heat_min", "sb3_best_gate_cool_min"):
        gate_value = float(overrides.get(key, TRAINING_DEFAULTS[key]))
        if gate_value < 0.0 or gate_value > 1.0:
            raise ValueError(f"{key} 必须在 [0,1]。")
    for key in ("pafc_best_gate_electric_min", "pafc_best_gate_heat_min", "pafc_best_gate_cool_min"):
        gate_value = float(overrides.get(key, TRAINING_DEFAULTS[key]))
        if gate_value < 0.0 or gate_value > 1.0:
            raise ValueError(f"{key} 必须在 [0,1]。")
    pafc_plateau_lr_decay_factor = float(
        overrides.get("pafc_plateau_lr_decay_factor", TRAINING_DEFAULTS["pafc_plateau_lr_decay_factor"])
    )
    if pafc_plateau_lr_decay_factor <= 0.0 or pafc_plateau_lr_decay_factor > 1.0:
        raise ValueError("pafc_plateau_lr_decay_factor 必须在 (0,1]。")
    pafc_plateau_min_actor_lr = float(
        overrides.get("pafc_plateau_min_actor_lr", TRAINING_DEFAULTS["pafc_plateau_min_actor_lr"])
    )
    pafc_plateau_min_critic_lr = float(
        overrides.get("pafc_plateau_min_critic_lr", TRAINING_DEFAULTS["pafc_plateau_min_critic_lr"])
    )
    if pafc_plateau_min_actor_lr <= 0.0 or pafc_plateau_min_critic_lr <= 0.0:
        raise ValueError("pafc_plateau_min_actor_lr / pafc_plateau_min_critic_lr 必须 > 0。")
    pafc_actor_lr = float(overrides.get("pafc_actor_lr", TRAINING_DEFAULTS["pafc_actor_lr"]))
    pafc_critic_lr = float(overrides.get("pafc_critic_lr", TRAINING_DEFAULTS["pafc_critic_lr"]))
    if pafc_plateau_min_actor_lr > pafc_actor_lr:
        raise ValueError("pafc_plateau_min_actor_lr 不能大于 pafc_actor_lr。")
    if pafc_plateau_min_critic_lr > pafc_critic_lr:
        raise ValueError("pafc_plateau_min_critic_lr 不能大于 pafc_critic_lr。")
    plateau_min_lr = float(overrides.get("sb3_plateau_min_lr", TRAINING_DEFAULTS["sb3_plateau_min_lr"]))
    learning_rate = float(overrides.get("sb3_learning_rate", TRAINING_DEFAULTS["sb3_learning_rate"]))
    if plateau_min_lr > learning_rate:
        raise ValueError("sb3_plateau_min_lr 不能大于 sb3_learning_rate。")
    pafc_hidden_dims = _normalize_hidden_dims_value(
        overrides.get("pafc_hidden_dims", TRAINING_DEFAULTS["pafc_hidden_dims"]),
        key="pafc_hidden_dims",
    )
    if not pafc_hidden_dims:
        raise ValueError("pafc_hidden_dims 不能为空。")


def build_training_options(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(TRAINING_DEFAULTS)
    if overrides is not None:
        incoming = dict(overrides)
        incoming.pop("sb3_eval_window_seed", None)
        merged.update(incoming)
    validate_training_overrides(merged)

    normalized = dict(merged)
    # seed 可能是逗号分隔的多seed字符串（由 _normalize_seed_list 在 __main__.py 中解析）
    # 此处保留原始字符串，不强制转 int；单个整数也保持兼容
    _seed_raw = normalized["seed"]
    if isinstance(_seed_raw, str) and ("," in _seed_raw or ";" in _seed_raw):
        normalized["seed"] = _seed_raw  # 多seed字符串，原样保留
    else:
        normalized["seed"] = int(_seed_raw)
    normalized["history_steps"] = int(normalized["history_steps"])
    normalized["episode_days"] = int(normalized["episode_days"])
    normalized["episodes"] = int(normalized["episodes"])
    normalized["train_steps"] = int(normalized["train_steps"])
    normalized["batch_size"] = int(normalized["batch_size"])
    normalized["update_epochs"] = int(normalized["update_epochs"])
    normalized["lr"] = float(normalized["lr"])
    normalized["policy"] = str(normalized["policy"]).strip().lower().replace("-", "_")
    normalized["sequence_adapter"] = str(normalized["sequence_adapter"]).strip().lower()
    normalized["device"] = str(normalized["device"]).strip().lower()
    normalized["sb3_enabled"] = bool(normalized["sb3_enabled"])
    normalized["sb3_algo"] = str(normalized["sb3_algo"]).strip().lower()
    normalized["sb3_backbone"] = str(normalized["sb3_backbone"]).strip().lower()
    normalized["sb3_history_steps"] = int(normalized["sb3_history_steps"])
    normalized["sb3_total_timesteps"] = int(normalized["sb3_total_timesteps"])
    normalized["sb3_n_envs"] = int(normalized["sb3_n_envs"])
    normalized["sb3_learning_rate"] = float(normalized["sb3_learning_rate"])
    normalized["sb3_batch_size"] = int(normalized["sb3_batch_size"])
    normalized["sb3_gamma"] = float(normalized["sb3_gamma"])
    normalized["sb3_vec_norm_obs"] = bool(normalized["sb3_vec_norm_obs"])
    normalized["sb3_vec_norm_reward"] = bool(normalized["sb3_vec_norm_reward"])
    normalized["sb3_eval_freq"] = int(normalized["sb3_eval_freq"])
    normalized["sb3_eval_episode_days"] = int(normalized["sb3_eval_episode_days"])
    normalized["sb3_eval_window_pool_size"] = int(normalized["sb3_eval_window_pool_size"])
    normalized["sb3_eval_window_count"] = int(normalized["sb3_eval_window_count"])
    normalized["sb3_ppo_warm_start_enabled"] = bool(normalized["sb3_ppo_warm_start_enabled"])
    normalized["sb3_residual_enabled"] = bool(normalized["sb3_residual_enabled"])
    normalized["sb3_residual_policy"] = (
        str(normalized["sb3_residual_policy"]).strip().lower().replace("-", "_")
    )
    normalized["sb3_residual_scale"] = float(normalized["sb3_residual_scale"])
    normalized["sb3_offpolicy_prefill_enabled"] = bool(normalized["sb3_offpolicy_prefill_enabled"])
    normalized["sb3_ppo_warm_start_samples"] = int(normalized["sb3_ppo_warm_start_samples"])
    normalized["sb3_ppo_warm_start_epochs"] = int(normalized["sb3_ppo_warm_start_epochs"])
    normalized["sb3_ppo_warm_start_batch_size"] = int(normalized["sb3_ppo_warm_start_batch_size"])
    normalized["sb3_ppo_warm_start_lr"] = float(normalized["sb3_ppo_warm_start_lr"])
    normalized["sb3_offpolicy_prefill_steps"] = int(normalized["sb3_offpolicy_prefill_steps"])
    normalized["sb3_offpolicy_prefill_policy"] = (
        str(normalized["sb3_offpolicy_prefill_policy"]).strip().lower().replace("-", "_")
    )
    normalized["sb3_ppo_n_steps"] = int(normalized["sb3_ppo_n_steps"])
    normalized["sb3_ppo_gae_lambda"] = float(normalized["sb3_ppo_gae_lambda"])
    normalized["sb3_ppo_ent_coef"] = float(normalized["sb3_ppo_ent_coef"])
    normalized["sb3_ppo_clip_range"] = float(normalized["sb3_ppo_clip_range"])
    normalized["sb3_dqn_action_mode"] = str(normalized["sb3_dqn_action_mode"]).strip().lower()
    normalized["sb3_dqn_target_update_interval"] = int(normalized["sb3_dqn_target_update_interval"])
    normalized["sb3_dqn_exploration_fraction"] = float(normalized["sb3_dqn_exploration_fraction"])
    normalized["sb3_dqn_exploration_initial_eps"] = float(normalized["sb3_dqn_exploration_initial_eps"])
    normalized["sb3_dqn_exploration_final_eps"] = float(normalized["sb3_dqn_exploration_final_eps"])
    normalized["sb3_learning_starts"] = int(normalized["sb3_learning_starts"])
    normalized["sb3_train_freq"] = int(normalized["sb3_train_freq"])
    normalized["sb3_gradient_steps"] = int(normalized["sb3_gradient_steps"])
    normalized["sb3_tau"] = float(normalized["sb3_tau"])
    normalized["sb3_action_noise_std"] = float(normalized["sb3_action_noise_std"])
    normalized["sb3_buffer_size"] = int(normalized["sb3_buffer_size"])
    normalized["sb3_optimize_memory_usage"] = bool(normalized["sb3_optimize_memory_usage"])
    normalized["sb3_best_gate_enabled"] = bool(normalized["sb3_best_gate_enabled"])
    normalized["sb3_best_gate_electric_min"] = float(normalized["sb3_best_gate_electric_min"])
    normalized["sb3_best_gate_heat_min"] = float(normalized["sb3_best_gate_heat_min"])
    normalized["sb3_best_gate_cool_min"] = float(normalized["sb3_best_gate_cool_min"])
    normalized["sb3_plateau_control_enabled"] = bool(normalized["sb3_plateau_control_enabled"])
    normalized["sb3_plateau_patience_evals"] = int(normalized["sb3_plateau_patience_evals"])
    normalized["sb3_plateau_lr_decay_factor"] = float(normalized["sb3_plateau_lr_decay_factor"])
    normalized["sb3_plateau_min_lr"] = float(normalized["sb3_plateau_min_lr"])
    normalized["sb3_plateau_early_stop_patience_evals"] = int(
        normalized["sb3_plateau_early_stop_patience_evals"]
    )
    normalized["pafc_projection_surrogate_checkpoint"] = str(
        normalized["pafc_projection_surrogate_checkpoint"]
    ).strip()
    normalized["pafc_episode_days"] = int(normalized["pafc_episode_days"])
    normalized["pafc_total_env_steps"] = int(normalized["pafc_total_env_steps"])
    normalized["pafc_warmup_steps"] = int(normalized["pafc_warmup_steps"])
    normalized["pafc_actor_warmup_steps"] = int(normalized["pafc_actor_warmup_steps"])
    normalized["pafc_replay_capacity"] = int(normalized["pafc_replay_capacity"])
    normalized["pafc_batch_size"] = int(normalized["pafc_batch_size"])
    normalized["pafc_updates_per_step"] = int(normalized["pafc_updates_per_step"])
    normalized["pafc_gamma"] = float(normalized["pafc_gamma"])
    normalized["pafc_tau"] = float(normalized["pafc_tau"])
    normalized["pafc_actor_lr"] = float(normalized["pafc_actor_lr"])
    normalized["pafc_critic_lr"] = float(normalized["pafc_critic_lr"])
    normalized["pafc_dual_lr"] = float(normalized["pafc_dual_lr"])
    normalized["pafc_dual_warmup_steps"] = int(normalized["pafc_dual_warmup_steps"])
    normalized["pafc_actor_delay"] = int(normalized["pafc_actor_delay"])
    normalized["pafc_exploration_noise_std"] = float(normalized["pafc_exploration_noise_std"])
    normalized["pafc_target_policy_noise_std"] = float(normalized["pafc_target_policy_noise_std"])
    normalized["pafc_target_noise_clip"] = float(normalized["pafc_target_noise_clip"])
    normalized["pafc_gap_penalty_coef"] = float(normalized["pafc_gap_penalty_coef"])
    normalized["pafc_exec_action_anchor_coef"] = float(normalized["pafc_exec_action_anchor_coef"])
    normalized["pafc_exec_action_anchor_safe_floor"] = float(
        normalized["pafc_exec_action_anchor_safe_floor"]
    )
    normalized["pafc_gt_off_deadband_ratio"] = float(normalized["pafc_gt_off_deadband_ratio"])
    normalized["pafc_abs_ready_focus_coef"] = float(normalized["pafc_abs_ready_focus_coef"])
    normalized["pafc_invalid_abs_penalty_coef"] = float(normalized["pafc_invalid_abs_penalty_coef"])
    normalized["pafc_economic_boiler_proxy_coef"] = float(normalized["pafc_economic_boiler_proxy_coef"])
    normalized["pafc_economic_abs_tradeoff_coef"] = float(normalized["pafc_economic_abs_tradeoff_coef"])
    normalized["pafc_economic_gt_grid_proxy_coef"] = float(
        normalized["pafc_economic_gt_grid_proxy_coef"]
    )
    normalized["pafc_economic_gt_distill_coef"] = float(
        normalized["pafc_economic_gt_distill_coef"]
    )
    normalized["pafc_economic_teacher_distill_coef"] = float(
        normalized["pafc_economic_teacher_distill_coef"]
    )
    normalized["pafc_economic_teacher_safe_preserve_coef"] = float(
        normalized["pafc_economic_teacher_safe_preserve_coef"]
    )
    normalized["pafc_economic_teacher_safe_preserve_low_margin_boost"] = float(
        normalized["pafc_economic_teacher_safe_preserve_low_margin_boost"]
    )
    normalized["pafc_economic_teacher_safe_preserve_high_cooling_boost"] = float(
        normalized["pafc_economic_teacher_safe_preserve_high_cooling_boost"]
    )
    normalized["pafc_economic_teacher_safe_preserve_joint_boost"] = float(
        normalized["pafc_economic_teacher_safe_preserve_joint_boost"]
    )
    normalized["pafc_economic_teacher_mismatch_focus_coef"] = float(
        normalized["pafc_economic_teacher_mismatch_focus_coef"]
    )
    normalized["pafc_economic_teacher_mismatch_focus_min_scale"] = float(
        normalized["pafc_economic_teacher_mismatch_focus_min_scale"]
    )
    normalized["pafc_economic_teacher_mismatch_focus_max_scale"] = float(
        normalized["pafc_economic_teacher_mismatch_focus_max_scale"]
    )
    normalized["pafc_economic_teacher_proxy_advantage_min"] = float(
        normalized["pafc_economic_teacher_proxy_advantage_min"]
    )
    normalized["pafc_economic_teacher_gt_proxy_advantage_min"] = float(
        normalized["pafc_economic_teacher_gt_proxy_advantage_min"]
    )
    normalized["pafc_economic_teacher_bes_proxy_advantage_min"] = float(
        normalized["pafc_economic_teacher_bes_proxy_advantage_min"]
    )
    normalized["pafc_economic_teacher_max_safe_abs_risk_gap"] = float(
        normalized["pafc_economic_teacher_max_safe_abs_risk_gap"]
    )
    normalized["pafc_economic_teacher_projection_gap_max"] = float(
        normalized["pafc_economic_teacher_projection_gap_max"]
    )
    normalized["pafc_economic_teacher_gt_projection_gap_max"] = float(
        normalized["pafc_economic_teacher_gt_projection_gap_max"]
    )
    normalized["pafc_economic_teacher_bes_price_opportunity_min"] = float(
        normalized["pafc_economic_teacher_bes_price_opportunity_min"]
    )
    normalized["pafc_economic_teacher_bes_anchor_preserve_scale"] = float(
        normalized["pafc_economic_teacher_bes_anchor_preserve_scale"]
    )
    normalized["pafc_economic_teacher_warm_start_weight"] = float(
        normalized["pafc_economic_teacher_warm_start_weight"]
    )
    normalized["pafc_economic_teacher_prefill_replay_boost"] = int(
        normalized["pafc_economic_teacher_prefill_replay_boost"]
    )
    normalized["pafc_economic_teacher_gt_action_weight"] = float(
        normalized["pafc_economic_teacher_gt_action_weight"]
    )
    normalized["pafc_economic_teacher_bes_action_weight"] = float(
        normalized["pafc_economic_teacher_bes_action_weight"]
    )
    normalized["pafc_economic_teacher_tes_action_weight"] = float(
        normalized["pafc_economic_teacher_tes_action_weight"]
    )
    normalized["pafc_economic_teacher_full_year_warm_start_samples"] = int(
        normalized["pafc_economic_teacher_full_year_warm_start_samples"]
    )
    normalized["pafc_economic_teacher_full_year_warm_start_epochs"] = int(
        normalized["pafc_economic_teacher_full_year_warm_start_epochs"]
    )
    normalized["pafc_economic_gt_full_year_warm_start_samples"] = int(
        normalized["pafc_economic_gt_full_year_warm_start_samples"]
    )
    normalized["pafc_economic_gt_full_year_warm_start_epochs"] = int(
        normalized["pafc_economic_gt_full_year_warm_start_epochs"]
    )
    normalized["pafc_economic_gt_full_year_warm_start_u_weight"] = float(
        normalized["pafc_economic_gt_full_year_warm_start_u_weight"]
    )
    normalized["pafc_economic_bes_distill_coef"] = float(
        normalized["pafc_economic_bes_distill_coef"]
    )
    normalized["pafc_economic_bes_prior_u"] = float(normalized["pafc_economic_bes_prior_u"])
    normalized["pafc_economic_bes_charge_u_scale"] = float(
        normalized["pafc_economic_bes_charge_u_scale"]
    )
    normalized["pafc_economic_bes_discharge_u_scale"] = float(
        normalized["pafc_economic_bes_discharge_u_scale"]
    )
    normalized["pafc_economic_bes_charge_weight"] = float(
        normalized["pafc_economic_bes_charge_weight"]
    )
    normalized["pafc_economic_bes_discharge_weight"] = float(
        normalized["pafc_economic_bes_discharge_weight"]
    )
    normalized["pafc_economic_bes_charge_pressure_bonus"] = float(
        normalized["pafc_economic_bes_charge_pressure_bonus"]
    )
    normalized["pafc_economic_bes_charge_soc_ceiling"] = float(
        normalized["pafc_economic_bes_charge_soc_ceiling"]
    )
    normalized["pafc_economic_bes_discharge_soc_floor"] = float(
        normalized["pafc_economic_bes_discharge_soc_floor"]
    )
    normalized["pafc_economic_bes_full_year_warm_start_samples"] = int(
        normalized["pafc_economic_bes_full_year_warm_start_samples"]
    )
    normalized["pafc_economic_bes_full_year_warm_start_epochs"] = int(
        normalized["pafc_economic_bes_full_year_warm_start_epochs"]
    )
    normalized["pafc_economic_bes_full_year_warm_start_u_weight"] = float(
        normalized["pafc_economic_bes_full_year_warm_start_u_weight"]
    )
    normalized["pafc_economic_bes_teacher_selection_priority_boost"] = float(
        normalized["pafc_economic_bes_teacher_selection_priority_boost"]
    )
    normalized["pafc_economic_bes_economic_source_priority_bonus"] = float(
        normalized["pafc_economic_bes_economic_source_priority_bonus"]
    )
    normalized["pafc_economic_bes_economic_source_min_share"] = float(
        normalized["pafc_economic_bes_economic_source_min_share"]
    )
    normalized["pafc_economic_bes_idle_economic_source_min_share"] = float(
        normalized["pafc_economic_bes_idle_economic_source_min_share"]
    )
    normalized["pafc_economic_bes_teacher_target_min_share"] = float(
        normalized["pafc_economic_bes_teacher_target_min_share"]
    )
    normalized["pafc_surrogate_actor_trust_coef"] = float(
        normalized["pafc_surrogate_actor_trust_coef"]
    )
    normalized["pafc_surrogate_actor_trust_min_scale"] = float(
        normalized["pafc_surrogate_actor_trust_min_scale"]
    )
    normalized["pafc_state_feasible_action_shaping_enabled"] = bool(
        normalized["pafc_state_feasible_action_shaping_enabled"]
    )
    normalized["pafc_abs_min_on_gate_th"] = float(normalized["pafc_abs_min_on_gate_th"])
    normalized["pafc_abs_min_on_u_margin"] = float(normalized["pafc_abs_min_on_u_margin"])
    normalized["pafc_expert_prefill_policy"] = (
        str(normalized["pafc_expert_prefill_policy"]).strip().lower().replace("-", "_")
    )
    normalized["pafc_expert_prefill_checkpoint_path"] = str(
        normalized["pafc_expert_prefill_checkpoint_path"]
    ).strip()
    normalized["pafc_expert_prefill_economic_policy"] = (
        str(normalized["pafc_expert_prefill_economic_policy"]).strip().lower().replace("-", "_")
    )
    normalized["pafc_expert_prefill_economic_checkpoint_path"] = str(
        normalized["pafc_expert_prefill_economic_checkpoint_path"]
    ).strip()
    normalized["pafc_frozen_action_keys"] = _normalize_string_tuple_value(
        normalized["pafc_frozen_action_keys"],
        key="pafc_frozen_action_keys",
    )
    normalized["pafc_frozen_action_safe_checkpoint_path"] = str(
        normalized["pafc_frozen_action_safe_checkpoint_path"]
    ).strip()
    normalized["pafc_tes_safe_action_delta_clip"] = float(
        normalized["pafc_tes_safe_action_delta_clip"]
    )
    normalized["pafc_bes_safe_action_delta_clip"] = float(
        normalized["pafc_bes_safe_action_delta_clip"]
    )
    normalized["pafc_boiler_safe_action_delta_clip"] = float(
        normalized["pafc_boiler_safe_action_delta_clip"]
    )
    normalized["pafc_gt_safe_action_delta_clip"] = float(
        normalized["pafc_gt_safe_action_delta_clip"]
    )
    normalized["pafc_expert_prefill_steps"] = int(normalized["pafc_expert_prefill_steps"])
    normalized["pafc_expert_prefill_cooling_bias"] = float(normalized["pafc_expert_prefill_cooling_bias"])
    normalized["pafc_expert_prefill_abs_replay_boost"] = int(normalized["pafc_expert_prefill_abs_replay_boost"])
    normalized["pafc_expert_prefill_abs_exec_threshold"] = float(normalized["pafc_expert_prefill_abs_exec_threshold"])
    normalized["pafc_expert_prefill_abs_window_mining_candidates"] = int(
        normalized["pafc_expert_prefill_abs_window_mining_candidates"]
    )
    normalized["pafc_dual_abs_margin_k"] = float(normalized["pafc_dual_abs_margin_k"])
    normalized["pafc_dual_qc_ratio_th"] = float(normalized["pafc_dual_qc_ratio_th"])
    normalized["pafc_dual_heat_backup_ratio_th"] = float(
        normalized["pafc_dual_heat_backup_ratio_th"]
    )
    normalized["pafc_dual_safe_abs_u_th"] = float(normalized["pafc_dual_safe_abs_u_th"])
    normalized["pafc_actor_warm_start_epochs"] = int(normalized["pafc_actor_warm_start_epochs"])
    normalized["pafc_actor_warm_start_batch_size"] = int(
        normalized["pafc_actor_warm_start_batch_size"]
    )
    normalized["pafc_actor_warm_start_lr"] = float(normalized["pafc_actor_warm_start_lr"])
    normalized["pafc_checkpoint_interval_steps"] = int(normalized["pafc_checkpoint_interval_steps"])
    normalized["pafc_eval_window_pool_size"] = int(normalized["pafc_eval_window_pool_size"])
    normalized["pafc_eval_window_count"] = int(normalized["pafc_eval_window_count"])
    normalized["pafc_best_gate_enabled"] = bool(normalized["pafc_best_gate_enabled"])
    normalized["pafc_best_gate_electric_min"] = float(normalized["pafc_best_gate_electric_min"])
    normalized["pafc_best_gate_heat_min"] = float(normalized["pafc_best_gate_heat_min"])
    normalized["pafc_best_gate_cool_min"] = float(normalized["pafc_best_gate_cool_min"])
    normalized["pafc_plateau_control_enabled"] = bool(normalized["pafc_plateau_control_enabled"])
    normalized["pafc_plateau_patience_evals"] = int(normalized["pafc_plateau_patience_evals"])
    normalized["pafc_plateau_lr_decay_factor"] = float(normalized["pafc_plateau_lr_decay_factor"])
    normalized["pafc_plateau_min_actor_lr"] = float(normalized["pafc_plateau_min_actor_lr"])
    normalized["pafc_plateau_min_critic_lr"] = float(normalized["pafc_plateau_min_critic_lr"])
    normalized["pafc_plateau_early_stop_patience_evals"] = int(
        normalized["pafc_plateau_early_stop_patience_evals"]
    )
    normalized["pafc_hidden_dims"] = _normalize_hidden_dims_value(
        normalized["pafc_hidden_dims"],
        key="pafc_hidden_dims",
    )
    return normalized
