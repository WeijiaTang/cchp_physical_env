# Ref: docs/spec/task.md
"""
CLI 入口：数据校验、训练、评估、标定、消融。

本模块是 `python -m cchp_physical_env` 的主入口，职责包括：
- 解析 CLI 参数与环境配置文件（config.yaml）
- 路由到不同子命令（summary/train/eval/sb3-train/sb3-eval/pafc-train/pafc-eval/calibrate/ablation）
  - 也支持 collect：扫描 runs 下的 eval 结果并汇总为论文表格 CSV
- 协调数据加载、环境构建、策略训练与评估

参数优先级（从高到低）：
1. CLI 显式参数（如 --episode-days=14）
2. config.yaml 中的 training 字段
3. 代码中的默认值（在 build_parser 中定义）

常见坑：
- 训练/评估年份硬编码为 2024/2025，不要在 CSV 路径里改年份
- SB3 训练需要安装 stable-baselines3（可选依赖）
- eval 子命令会自动识别 SB3 / PAFC checkpoint（通过 artifact_type）
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .core.data import (
    EXPECTED_STEPS_PER_YEAR,
    EVAL_YEAR,
    TRAIN_YEAR,
    compute_training_statistics,
    ensure_frozen_schema_consistency,
    load_exogenous_data,
    summarize_exogenous_data,
)
from .core.config_loader import (
    build_env_config_from_overrides,
    build_training_options,
    load_env_overrides,
    load_training_overrides,
)
from .pipeline.calibration import (
    load_calibration_config,
    run_calibration_search,
    validate_calibration_config,
)
from .pipeline.ablation import run_constraint_ablation
from .pipeline.collect import write_benchmark_tables
from .pipeline.sequence import SUPPORTED_SEQUENCE_ADAPTERS
from .pipeline.runner import evaluate_baseline, train_baseline
from .policy.checkpoint import load_policy
from .policy.pafc_td3 import PAFCTD3TrainConfig, evaluate_pafc_td3, train_pafc_td3
from .policy.sb3 import SB3TrainConfig, evaluate_sb3_policy, train_sb3_policy

# 默认路径（训练/评估数据、环境配置）
DEFAULT_TRAIN_PATH = Path("data/processed/cchp_main_15min_2024.csv")
DEFAULT_EVAL_PATH = Path("data/processed/cchp_main_15min_2025.csv")
DEFAULT_ENV_CONFIG_PATH = Path("src/cchp_physical_env/config/config.yaml")
DEFAULT_CONSTRAINT_MODES = ("physics_in_loop", "reward_only")

# 训练选项键：这些字段可以从 CLI 或 config.yaml 覆盖
TRAINING_OPTION_KEYS = (
    "seed",
    "policy",
    "sequence_adapter",
    "history_steps",
    "episode_days",
    "episodes",
    "train_steps",
    "batch_size",
    "update_epochs",
    "lr",
    "device",
    "sb3_enabled",
    "sb3_algo",
    "sb3_backbone",
    "sb3_history_steps",
    "sb3_total_timesteps",
    "sb3_n_envs",
    "sb3_learning_rate",
    "sb3_batch_size",
    "sb3_gamma",
    "sb3_vec_norm_obs",
    "sb3_vec_norm_reward",
    "sb3_eval_freq",
    "sb3_eval_episode_days",
    "sb3_eval_window_pool_size",
    "sb3_eval_window_count",
    "sb3_ppo_warm_start_enabled",
    "sb3_residual_enabled",
    "sb3_residual_policy",
    "sb3_residual_scale",
    "sb3_ppo_warm_start_samples",
    "sb3_ppo_warm_start_epochs",
    "sb3_ppo_warm_start_batch_size",
    "sb3_ppo_warm_start_lr",
    "sb3_offpolicy_prefill_enabled",
    "sb3_offpolicy_prefill_steps",
    "sb3_offpolicy_prefill_policy",
    "sb3_ppo_n_steps",
    "sb3_ppo_gae_lambda",
    "sb3_ppo_ent_coef",
    "sb3_ppo_clip_range",
    "sb3_dqn_action_mode",
    "sb3_dqn_target_update_interval",
    "sb3_dqn_exploration_fraction",
    "sb3_dqn_exploration_initial_eps",
    "sb3_dqn_exploration_final_eps",
    "sb3_learning_starts",
    "sb3_train_freq",
    "sb3_gradient_steps",
    "sb3_tau",
    "sb3_action_noise_std",
    "sb3_buffer_size",
    "sb3_optimize_memory_usage",
    "sb3_best_gate_enabled",
    "sb3_best_gate_electric_min",
    "sb3_best_gate_heat_min",
    "sb3_best_gate_cool_min",
    "sb3_plateau_control_enabled",
    "sb3_plateau_patience_evals",
    "sb3_plateau_lr_decay_factor",
    "sb3_plateau_min_lr",
    "sb3_plateau_early_stop_patience_evals",
    "pafc_projection_surrogate_checkpoint",
    "pafc_episode_days",
    "pafc_total_env_steps",
    "pafc_warmup_steps",
    "pafc_replay_capacity",
    "pafc_batch_size",
    "pafc_updates_per_step",
    "pafc_gamma",
    "pafc_tau",
    "pafc_actor_lr",
    "pafc_critic_lr",
    "pafc_dual_lr",
    "pafc_dual_warmup_steps",
    "pafc_actor_delay",
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
    "pafc_economic_teacher_prefill_replay_boost",
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
    "pafc_economic_bes_full_year_warm_start_samples",
    "pafc_economic_bes_full_year_warm_start_epochs",
    "pafc_economic_bes_full_year_warm_start_u_weight",
    "pafc_economic_bes_teacher_selection_priority_boost",
    "pafc_economic_bes_economic_source_priority_bonus",
    "pafc_economic_bes_economic_source_min_share",
    "pafc_economic_bes_idle_economic_source_min_share",
    "pafc_economic_bes_teacher_target_min_share",
    "pafc_surrogate_actor_trust_coef",
    "pafc_surrogate_actor_trust_min_scale",
    "pafc_state_feasible_action_shaping_enabled",
    "pafc_abs_min_on_gate_th",
    "pafc_abs_min_on_u_margin",
    "pafc_expert_prefill_policy",
    "pafc_expert_prefill_checkpoint_path",
    "pafc_expert_prefill_economic_policy",
    "pafc_expert_prefill_economic_checkpoint_path",
    "pafc_expert_prefill_steps",
    "pafc_expert_prefill_cooling_bias",
    "pafc_expert_prefill_abs_replay_boost",
    "pafc_expert_prefill_abs_exec_threshold",
    "pafc_expert_prefill_abs_window_mining_candidates",
    "pafc_dual_abs_margin_k",
    "pafc_dual_qc_ratio_th",
    "pafc_dual_heat_backup_ratio_th",
    "pafc_dual_safe_abs_u_th",
    "pafc_actor_warm_start_epochs",
    "pafc_actor_warm_start_batch_size",
    "pafc_actor_warm_start_lr",
    "pafc_checkpoint_interval_steps",
    "pafc_eval_window_pool_size",
    "pafc_eval_window_count",
    "pafc_best_gate_enabled",
    "pafc_best_gate_electric_min",
    "pafc_best_gate_heat_min",
    "pafc_best_gate_cool_min",
    "pafc_plateau_control_enabled",
    "pafc_plateau_patience_evals",
    "pafc_plateau_lr_decay_factor",
    "pafc_plateau_min_actor_lr",
    "pafc_plateau_min_critic_lr",
    "pafc_plateau_early_stop_patience_evals",
    "pafc_hidden_dims",
)


def _parse_seed_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    if isinstance(value, str):
        tokens = [token.strip() for token in value.replace(";", ",").split(",")]
        return [int(token) for token in tokens if token]
    return [int(value)]


def _normalize_seed_list(value: object, fallback: int = 42) -> list[int]:
    seeds = _parse_seed_list(value)
    return seeds or [int(fallback)]


def _parse_hidden_dims(value: object, fallback: tuple[int, ...] = (256, 256)) -> tuple[int, ...]:
    if value is None:
        return tuple(int(item) for item in fallback)
    if isinstance(value, (list, tuple)):
        dims = [int(item) for item in value]
    else:
        text = str(value).strip()
        if not text:
            return tuple(int(item) for item in fallback)
        dims = [int(token.strip()) for token in text.replace(";", ",").split(",") if token.strip()]
    if not dims:
        return tuple(int(item) for item in fallback)
    return tuple(dims)


def _read_json_payload(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _detect_checkpoint_artifact(
    checkpoint_path: str | Path | None,
) -> tuple[str, Path | None, dict[str, object]]:
    if checkpoint_path is None:
        return "baseline", None, {}
    path = Path(checkpoint_path)
    payload = _read_json_payload(path)
    artifact_type = str(payload.get("artifact_type", "")).strip().lower()
    if artifact_type == "sb3_policy":
        return "sb3", path, payload
    if artifact_type == "pafc_td3_actor":
        resolved = payload.get("checkpoint_path")
        if not isinstance(resolved, str) or len(resolved.strip()) == 0:
            raise ValueError("pafc_td3_actor.json 缺少 checkpoint_path。")
        return "pafc_td3", Path(resolved), payload
    if path.suffix.lower() == ".pt":
        try:
            metadata = dict(load_policy(path, map_location="cpu").get("metadata", {}))
        except Exception:
            metadata = {}
        artifact_type = str(metadata.get("artifact_type", "")).strip().lower()
        if artifact_type == "pafc_td3_actor":
            return "pafc_td3", path, metadata
    return "baseline", path, payload


def _maybe_seed_run_dir(base: Path, seed: int, multi: bool) -> Path:
    return base if not multi else base / f"seed_{seed}"


def _resolve_env_config_path(args: argparse.Namespace) -> Path:
    """
    解析环境配置文件路径。

    优先级：CLI 参数 > 默认路径（config/config.yaml）
    """
    path = getattr(args, "env_config", None)
    return DEFAULT_ENV_CONFIG_PATH if path is None else Path(path)


def _resolve_training_options(args: argparse.Namespace) -> dict:
    """
    解析训练选项，合并 config.yaml 与 CLI 参数。

    优先级：CLI 参数 > config.yaml > 默认值
    返回：完整的训练选项字典
    """
    config_path = _resolve_env_config_path(args)
    training_overrides = load_training_overrides(config_path)
    resolved = build_training_options(training_overrides)
    for key in TRAINING_OPTION_KEYS:
        if hasattr(args, key):
            resolved[key] = getattr(args, key)
    return build_training_options(resolved)


def _print_summary_block(train_path: Path, eval_path: Path) -> None:
    """
    打印训练/评估数据摘要并校验冻结 schema。

    检查项：
    - 行数是否为 EXPECTED_STEPS_PER_YEAR（35040）
    - 训练/评估集 schema 是否一致
    """
    train_df = load_exogenous_data(train_path)
    eval_df = load_exogenous_data(eval_path)
    ensure_frozen_schema_consistency(train_df, eval_df)

    train_summary = summarize_exogenous_data(train_df)
    eval_summary = summarize_exogenous_data(eval_df)
    payload = {"train": train_summary, "eval": eval_summary}

    if train_summary["n_rows"] != EXPECTED_STEPS_PER_YEAR:
        raise RuntimeError(f"训练集行数错误: {train_summary['n_rows']}")
    if eval_summary["n_rows"] != EXPECTED_STEPS_PER_YEAR:
        raise RuntimeError(f"评估集行数错误: {eval_summary['n_rows']}")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("schema_consistency: PASS")


def _command_summary(args: argparse.Namespace) -> None:
    """
    summary 子命令：打印 2024/2025 数据摘要并校验 schema。
    """
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    force_mode = getattr(args, "constraint_mode", None)
    build_env_config_from_overrides(env_overrides, force_constraint_mode=force_mode)
    _print_summary_block(train_path=args.train_path, eval_path=args.eval_path)


def _command_train(args: argparse.Namespace) -> None:
    """
    train 子命令：根据配置路由到不同训练路径。

    路由逻辑：
    1. 若 sb3_enabled=True -> 调用 SB3 训练（PPO/SAC/TD3/DDPG/DQN）
    2. 若 policy=pafc_td3 -> 调用 PAFC-TD3 训练（Task-012）
    3. 若 policy=sequence_rule 且 adapter 为 mlp/transformer/mamba -> 调用 sequence trainer
    4. 否则 -> 调用 baseline 训练（rule/easy_rule/random/sequence_rule/milp_mpc/ga_mpc）

    训练年份固定为 2024，数据来自 data/processed/cchp_main_15min_2024.csv
    """
    train_df = load_exogenous_data(args.train_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    force_mode = getattr(args, "constraint_mode", None)
    env_config = build_env_config_from_overrides(env_overrides, force_constraint_mode=force_mode)
    training_options = _resolve_training_options(args)
    seed_values = _normalize_seed_list(training_options.get("seed", 42), fallback=42)
    multi_seed = len(seed_values) > 1

    if bool(training_options.get("sb3_enabled", False)):
        warnings: list[str] = []
        if str(training_options.get("policy", "")).strip().lower() in {
            "rule",
            "easy_rule",
            "random",
            "sequence_rule",
            "milp_mpc",
            "milp-mpc",
            "ga_mpc",
            "ga-mpc",
        }:
            warnings.append("sb3_enabled=true 时 training.policy 仅作记录，不参与路由。")
        sb3_backbone = str(training_options.get("sb3_backbone", "mlp")).strip().lower()
        sb3_steps = int(training_options.get("sb3_total_timesteps", 0) or 0)
        if sb3_backbone in {"transformer", "mamba"} and sb3_steps > 0 and sb3_steps < 300_000:
            warnings.append("SB3 序列骨干建议更大训练预算：sb3_total_timesteps<300k 可能欠训练。")
        ignored_keys = sorted(
            set(TRAINING_OPTION_KEYS)
            - {
                "seed",
                "episode_days",
                "device",
                "sb3_enabled",
                "sb3_algo",
                "sb3_backbone",
                "sb3_history_steps",
                "sb3_total_timesteps",
                "sb3_n_envs",
                "sb3_learning_rate",
                "sb3_batch_size",
                "sb3_gamma",
                "sb3_vec_norm_obs",
                "sb3_vec_norm_reward",
                "sb3_eval_freq",
                "sb3_eval_episode_days",
                "sb3_eval_window_pool_size",
                "sb3_eval_window_count",
                "sb3_ppo_warm_start_enabled",
                "sb3_residual_enabled",
                "sb3_residual_policy",
                "sb3_residual_scale",
                "sb3_ppo_warm_start_samples",
                "sb3_ppo_warm_start_epochs",
                "sb3_ppo_warm_start_batch_size",
                "sb3_ppo_warm_start_lr",
                "sb3_offpolicy_prefill_enabled",
                "sb3_offpolicy_prefill_steps",
                "sb3_ppo_n_steps",
                "sb3_ppo_gae_lambda",
                "sb3_ppo_ent_coef",
                "sb3_ppo_clip_range",
                "sb3_dqn_action_mode",
                "sb3_dqn_target_update_interval",
                "sb3_dqn_exploration_fraction",
                "sb3_dqn_exploration_initial_eps",
                "sb3_dqn_exploration_final_eps",
                "sb3_learning_starts",
                "sb3_train_freq",
                "sb3_gradient_steps",
                "sb3_tau",
                "sb3_action_noise_std",
                "sb3_buffer_size",
                "sb3_optimize_memory_usage",
                "sb3_best_gate_enabled",
                "sb3_best_gate_electric_min",
                "sb3_best_gate_heat_min",
                "sb3_best_gate_cool_min",
                "sb3_plateau_control_enabled",
                "sb3_plateau_patience_evals",
                "sb3_plateau_lr_decay_factor",
                "sb3_plateau_min_lr",
                "sb3_plateau_early_stop_patience_evals",
            }
        )
        eval_df_cache: pd.DataFrame | None = None
        outputs: list[dict[str, object]] = []
        for seed in seed_values:
            current_options = dict(training_options)
            current_options["seed"] = seed
            config = SB3TrainConfig(
                algo=current_options["sb3_algo"],
                backbone=current_options["sb3_backbone"],
                history_steps=current_options["sb3_history_steps"],
                total_timesteps=current_options["sb3_total_timesteps"],
                episode_days=current_options["episode_days"],
                n_envs=current_options["sb3_n_envs"],
                learning_rate=current_options["sb3_learning_rate"],
                batch_size=current_options["sb3_batch_size"],
                gamma=current_options["sb3_gamma"],
                vec_norm_obs=bool(current_options["sb3_vec_norm_obs"]),
                vec_norm_reward=bool(current_options["sb3_vec_norm_reward"]),
                eval_freq=current_options["sb3_eval_freq"],
                eval_episode_days=current_options["sb3_eval_episode_days"],
                eval_window_pool_size=current_options["sb3_eval_window_pool_size"],
                eval_window_count=current_options["sb3_eval_window_count"],
                eval_window_seed=seed,
                ppo_warm_start_enabled=bool(current_options["sb3_ppo_warm_start_enabled"]),
                residual_enabled=bool(current_options["sb3_residual_enabled"]),
                residual_policy=current_options["sb3_residual_policy"],
                residual_scale=current_options["sb3_residual_scale"],
                ppo_warm_start_samples=current_options["sb3_ppo_warm_start_samples"],
                ppo_warm_start_epochs=current_options["sb3_ppo_warm_start_epochs"],
                ppo_warm_start_batch_size=current_options["sb3_ppo_warm_start_batch_size"],
                ppo_warm_start_lr=current_options["sb3_ppo_warm_start_lr"],
                offpolicy_prefill_enabled=bool(current_options["sb3_offpolicy_prefill_enabled"]),
                offpolicy_prefill_steps=current_options["sb3_offpolicy_prefill_steps"],
                offpolicy_prefill_policy=current_options["sb3_offpolicy_prefill_policy"],
                ppo_n_steps=current_options["sb3_ppo_n_steps"],
                ppo_gae_lambda=current_options["sb3_ppo_gae_lambda"],
                ppo_ent_coef=current_options["sb3_ppo_ent_coef"],
                ppo_clip_range=current_options["sb3_ppo_clip_range"],
                dqn_action_mode=current_options["sb3_dqn_action_mode"],
                dqn_target_update_interval=current_options["sb3_dqn_target_update_interval"],
                dqn_exploration_fraction=current_options["sb3_dqn_exploration_fraction"],
                dqn_exploration_initial_eps=current_options["sb3_dqn_exploration_initial_eps"],
                dqn_exploration_final_eps=current_options["sb3_dqn_exploration_final_eps"],
                learning_starts=current_options["sb3_learning_starts"],
                train_freq=current_options["sb3_train_freq"],
                gradient_steps=current_options["sb3_gradient_steps"],
                tau=current_options["sb3_tau"],
                action_noise_std=current_options["sb3_action_noise_std"],
                buffer_size=current_options["sb3_buffer_size"],
                optimize_memory_usage=bool(current_options["sb3_optimize_memory_usage"]),
                best_gate_enabled=bool(current_options["sb3_best_gate_enabled"]),
                best_gate_electric_min=current_options["sb3_best_gate_electric_min"],
                best_gate_heat_min=current_options["sb3_best_gate_heat_min"],
                best_gate_cool_min=current_options["sb3_best_gate_cool_min"],
                plateau_control_enabled=bool(current_options["sb3_plateau_control_enabled"]),
                plateau_patience_evals=current_options["sb3_plateau_patience_evals"],
                plateau_lr_decay_factor=current_options["sb3_plateau_lr_decay_factor"],
                plateau_min_lr=current_options["sb3_plateau_min_lr"],
                plateau_early_stop_patience_evals=current_options["sb3_plateau_early_stop_patience_evals"],
                seed=seed,
                device=current_options["device"],
            )
            result = train_sb3_policy(
                train_df=train_df,
                env_config=env_config,
                config=config,
                run_root=args.run_root,
            )
            payload: dict[str, object] = {
                "mode": "train",
                "train_year": TRAIN_YEAR,
                "policy": "sb3",
                "seed": seed,
                "ignored_training_keys": ignored_keys,
                "warnings": warnings,
                **result,
            }
            if bool(getattr(args, "eval_after_train", False)):
                if eval_df_cache is None:
                    eval_df_cache = load_exogenous_data(args.eval_path)
                run_dir = Path(str(result.get("run_dir", "") or "")).resolve()
                checkpoint_json = run_dir / "checkpoints" / "baseline_policy.json"
                payload["eval_summary"] = evaluate_sb3_policy(
                    eval_df=eval_df_cache,
                    env_config=env_config,
                    checkpoint_json=checkpoint_json,
                    run_dir=run_dir,
                    seed=seed,
                    deterministic=True,
                    model_source="best",
                    device=current_options["device"],
                )
            outputs.append(payload)
        print(json.dumps(outputs[0] if not multi_seed else outputs, indent=2, ensure_ascii=False))
        return

    if training_options["policy"] == "pafc_td3":
        ignored_keys = sorted(
            set(TRAINING_OPTION_KEYS)
            - {
                "seed",
                "policy",
                "device",
                "pafc_projection_surrogate_checkpoint",
                "pafc_episode_days",
                "pafc_total_env_steps",
                "pafc_warmup_steps",
                "pafc_replay_capacity",
                "pafc_batch_size",
                "pafc_updates_per_step",
                "pafc_gamma",
                "pafc_tau",
                "pafc_actor_lr",
                "pafc_critic_lr",
                "pafc_dual_lr",
                "pafc_dual_warmup_steps",
                "pafc_actor_delay",
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
                "pafc_economic_teacher_prefill_replay_boost",
                "pafc_economic_teacher_gt_action_weight",
                "pafc_economic_teacher_bes_action_weight",
                "pafc_economic_teacher_tes_action_weight",
                "pafc_economic_teacher_full_year_warm_start_samples",
                "pafc_economic_teacher_full_year_warm_start_epochs",
                "pafc_economic_bes_distill_coef",
                "pafc_economic_bes_prior_u",
                "pafc_economic_bes_charge_u_scale",
                "pafc_economic_bes_discharge_u_scale",
                "pafc_economic_bes_charge_weight",
                "pafc_economic_bes_discharge_weight",
                "pafc_economic_bes_charge_pressure_bonus",
                "pafc_economic_bes_charge_soc_ceiling",
                "pafc_economic_bes_discharge_soc_floor",
                "pafc_economic_gt_full_year_warm_start_samples",
                "pafc_economic_gt_full_year_warm_start_epochs",
                "pafc_economic_gt_full_year_warm_start_u_weight",
                "pafc_economic_bes_full_year_warm_start_samples",
                "pafc_economic_bes_full_year_warm_start_epochs",
                "pafc_economic_bes_full_year_warm_start_u_weight",
                "pafc_economic_bes_teacher_selection_priority_boost",
                "pafc_economic_bes_economic_source_priority_bonus",
                "pafc_economic_bes_economic_source_min_share",
                "pafc_economic_bes_idle_economic_source_min_share",
                "pafc_economic_bes_teacher_target_min_share",
                "pafc_surrogate_actor_trust_coef",
                "pafc_surrogate_actor_trust_min_scale",
                "pafc_state_feasible_action_shaping_enabled",
                "pafc_abs_min_on_gate_th",
                "pafc_abs_min_on_u_margin",
                "pafc_expert_prefill_policy",
                "pafc_expert_prefill_checkpoint_path",
                "pafc_expert_prefill_economic_policy",
                "pafc_expert_prefill_economic_checkpoint_path",
                "pafc_expert_prefill_steps",
                "pafc_expert_prefill_cooling_bias",
                "pafc_expert_prefill_abs_replay_boost",
                "pafc_expert_prefill_abs_exec_threshold",
                "pafc_expert_prefill_abs_window_mining_candidates",
                "pafc_dual_abs_margin_k",
                "pafc_dual_qc_ratio_th",
                "pafc_dual_heat_backup_ratio_th",
                "pafc_dual_safe_abs_u_th",
                "pafc_actor_warm_start_epochs",
                "pafc_actor_warm_start_batch_size",
                "pafc_actor_warm_start_lr",
                "pafc_checkpoint_interval_steps",
                "pafc_eval_window_pool_size",
                "pafc_eval_window_count",
                "pafc_best_gate_enabled",
                "pafc_best_gate_electric_min",
                "pafc_best_gate_heat_min",
                "pafc_best_gate_cool_min",
                "pafc_plateau_control_enabled",
                "pafc_plateau_patience_evals",
                "pafc_plateau_lr_decay_factor",
                "pafc_plateau_min_actor_lr",
                "pafc_plateau_min_critic_lr",
                "pafc_plateau_early_stop_patience_evals",
                "pafc_hidden_dims",
            }
        )
        projection_surrogate_checkpoint = str(
            training_options.get("pafc_projection_surrogate_checkpoint", "")
        ).strip()
        if len(projection_surrogate_checkpoint) == 0:
            raise ValueError(
                "policy=pafc_td3 时必须提供 pafc_projection_surrogate_checkpoint；"
                "可在 config.yaml 的 training.pafc_projection_surrogate_checkpoint 中配置，"
                "或通过 --pafc-projection-surrogate-checkpoint 传入。"
            )
        train_statistics = compute_training_statistics(train_df)
        eval_df_cache: pd.DataFrame | None = None
        outputs: list[dict[str, object]] = []
        for seed in seed_values:
            current_options = dict(training_options)
            current_options["seed"] = seed
            config = PAFCTD3TrainConfig(
                projection_surrogate_checkpoint_path=projection_surrogate_checkpoint,
                episode_days=int(current_options["pafc_episode_days"]),
                total_env_steps=int(current_options["pafc_total_env_steps"]),
                warmup_steps=int(current_options["pafc_warmup_steps"]),
                replay_capacity=int(current_options["pafc_replay_capacity"]),
                batch_size=int(current_options["pafc_batch_size"]),
                updates_per_step=int(current_options["pafc_updates_per_step"]),
                gamma=float(current_options["pafc_gamma"]),
                tau=float(current_options["pafc_tau"]),
                actor_lr=float(current_options["pafc_actor_lr"]),
                critic_lr=float(current_options["pafc_critic_lr"]),
                dual_lr=float(current_options["pafc_dual_lr"]),
                dual_warmup_steps=int(current_options["pafc_dual_warmup_steps"]),
                actor_delay=int(current_options["pafc_actor_delay"]),
                exploration_noise_std=float(current_options["pafc_exploration_noise_std"]),
                target_policy_noise_std=float(current_options["pafc_target_policy_noise_std"]),
                target_noise_clip=float(current_options["pafc_target_noise_clip"]),
                gap_penalty_coef=float(current_options["pafc_gap_penalty_coef"]),
                exec_action_anchor_coef=float(current_options["pafc_exec_action_anchor_coef"]),
                exec_action_anchor_safe_floor=float(
                    current_options["pafc_exec_action_anchor_safe_floor"]
                ),
                gt_off_deadband_ratio=float(current_options["pafc_gt_off_deadband_ratio"]),
                abs_ready_focus_coef=float(current_options["pafc_abs_ready_focus_coef"]),
                invalid_abs_penalty_coef=float(current_options["pafc_invalid_abs_penalty_coef"]),
                economic_boiler_proxy_coef=float(current_options["pafc_economic_boiler_proxy_coef"]),
                economic_abs_tradeoff_coef=float(current_options["pafc_economic_abs_tradeoff_coef"]),
                economic_gt_grid_proxy_coef=float(
                    current_options["pafc_economic_gt_grid_proxy_coef"]
                ),
                economic_gt_distill_coef=float(
                    current_options["pafc_economic_gt_distill_coef"]
                ),
                economic_teacher_distill_coef=float(
                    current_options["pafc_economic_teacher_distill_coef"]
                ),
                economic_teacher_safe_preserve_coef=float(
                    current_options["pafc_economic_teacher_safe_preserve_coef"]
                ),
                economic_teacher_safe_preserve_low_margin_boost=float(
                    current_options["pafc_economic_teacher_safe_preserve_low_margin_boost"]
                ),
                economic_teacher_safe_preserve_high_cooling_boost=float(
                    current_options["pafc_economic_teacher_safe_preserve_high_cooling_boost"]
                ),
                economic_teacher_safe_preserve_joint_boost=float(
                    current_options["pafc_economic_teacher_safe_preserve_joint_boost"]
                ),
                economic_teacher_mismatch_focus_coef=float(
                    current_options["pafc_economic_teacher_mismatch_focus_coef"]
                ),
                economic_teacher_mismatch_focus_min_scale=float(
                    current_options["pafc_economic_teacher_mismatch_focus_min_scale"]
                ),
                economic_teacher_mismatch_focus_max_scale=float(
                    current_options["pafc_economic_teacher_mismatch_focus_max_scale"]
                ),
                economic_teacher_proxy_advantage_min=float(
                    current_options["pafc_economic_teacher_proxy_advantage_min"]
                ),
                economic_teacher_gt_proxy_advantage_min=float(
                    current_options["pafc_economic_teacher_gt_proxy_advantage_min"]
                ),
                economic_teacher_bes_proxy_advantage_min=float(
                    current_options["pafc_economic_teacher_bes_proxy_advantage_min"]
                ),
                economic_teacher_max_safe_abs_risk_gap=float(
                    current_options["pafc_economic_teacher_max_safe_abs_risk_gap"]
                ),
                economic_teacher_projection_gap_max=float(
                    current_options["pafc_economic_teacher_projection_gap_max"]
                ),
                economic_teacher_gt_projection_gap_max=float(
                    current_options["pafc_economic_teacher_gt_projection_gap_max"]
                ),
                economic_teacher_bes_price_opportunity_min=float(
                    current_options["pafc_economic_teacher_bes_price_opportunity_min"]
                ),
                economic_teacher_bes_anchor_preserve_scale=float(
                    current_options["pafc_economic_teacher_bes_anchor_preserve_scale"]
                ),
                economic_teacher_warm_start_weight=float(
                    current_options["pafc_economic_teacher_warm_start_weight"]
                ),
                economic_teacher_prefill_replay_boost=int(
                    current_options["pafc_economic_teacher_prefill_replay_boost"]
                ),
                economic_teacher_gt_action_weight=float(
                    current_options["pafc_economic_teacher_gt_action_weight"]
                ),
                economic_teacher_bes_action_weight=float(
                    current_options["pafc_economic_teacher_bes_action_weight"]
                ),
                economic_teacher_tes_action_weight=float(
                    current_options["pafc_economic_teacher_tes_action_weight"]
                ),
                economic_teacher_full_year_warm_start_samples=int(
                    current_options["pafc_economic_teacher_full_year_warm_start_samples"]
                ),
                economic_teacher_full_year_warm_start_epochs=int(
                    current_options["pafc_economic_teacher_full_year_warm_start_epochs"]
                ),
                economic_gt_full_year_warm_start_samples=int(
                    current_options["pafc_economic_gt_full_year_warm_start_samples"]
                ),
                economic_gt_full_year_warm_start_epochs=int(
                    current_options["pafc_economic_gt_full_year_warm_start_epochs"]
                ),
                economic_gt_full_year_warm_start_u_weight=float(
                    current_options["pafc_economic_gt_full_year_warm_start_u_weight"]
                ),
                economic_bes_distill_coef=float(
                    current_options["pafc_economic_bes_distill_coef"]
                ),
                economic_bes_prior_u=float(current_options["pafc_economic_bes_prior_u"]),
                economic_bes_charge_u_scale=float(
                    current_options["pafc_economic_bes_charge_u_scale"]
                ),
                economic_bes_discharge_u_scale=float(
                    current_options["pafc_economic_bes_discharge_u_scale"]
                ),
                economic_bes_charge_weight=float(
                    current_options["pafc_economic_bes_charge_weight"]
                ),
                economic_bes_discharge_weight=float(
                    current_options["pafc_economic_bes_discharge_weight"]
                ),
                economic_bes_charge_pressure_bonus=float(
                    current_options["pafc_economic_bes_charge_pressure_bonus"]
                ),
                economic_bes_charge_soc_ceiling=float(
                    current_options["pafc_economic_bes_charge_soc_ceiling"]
                ),
                economic_bes_discharge_soc_floor=float(
                    current_options["pafc_economic_bes_discharge_soc_floor"]
                ),
                economic_bes_full_year_warm_start_samples=int(
                    current_options["pafc_economic_bes_full_year_warm_start_samples"]
                ),
                economic_bes_full_year_warm_start_epochs=int(
                    current_options["pafc_economic_bes_full_year_warm_start_epochs"]
                ),
                economic_bes_full_year_warm_start_u_weight=float(
                    current_options["pafc_economic_bes_full_year_warm_start_u_weight"]
                ),
                economic_bes_teacher_selection_priority_boost=float(
                    current_options["pafc_economic_bes_teacher_selection_priority_boost"]
                ),
                economic_bes_economic_source_priority_bonus=float(
                    current_options["pafc_economic_bes_economic_source_priority_bonus"]
                ),
                economic_bes_economic_source_min_share=float(
                    current_options["pafc_economic_bes_economic_source_min_share"]
                ),
                economic_bes_idle_economic_source_min_share=float(
                    current_options["pafc_economic_bes_idle_economic_source_min_share"]
                ),
                economic_bes_teacher_target_min_share=float(
                    current_options["pafc_economic_bes_teacher_target_min_share"]
                ),
                surrogate_actor_trust_coef=float(
                    current_options["pafc_surrogate_actor_trust_coef"]
                ),
                surrogate_actor_trust_min_scale=float(
                    current_options["pafc_surrogate_actor_trust_min_scale"]
                ),
                state_feasible_action_shaping_enabled=bool(
                    current_options["pafc_state_feasible_action_shaping_enabled"]
                ),
                abs_min_on_gate_th=float(current_options["pafc_abs_min_on_gate_th"]),
                abs_min_on_u_margin=float(current_options["pafc_abs_min_on_u_margin"]),
                expert_prefill_policy=str(current_options["pafc_expert_prefill_policy"]),
                expert_prefill_checkpoint_path=str(current_options["pafc_expert_prefill_checkpoint_path"]),
                expert_prefill_economic_policy=str(
                    current_options["pafc_expert_prefill_economic_policy"]
                ),
                expert_prefill_economic_checkpoint_path=str(
                    current_options["pafc_expert_prefill_economic_checkpoint_path"]
                ),
                expert_prefill_steps=int(current_options["pafc_expert_prefill_steps"]),
                expert_prefill_cooling_bias=float(current_options["pafc_expert_prefill_cooling_bias"]),
                expert_prefill_abs_replay_boost=int(current_options["pafc_expert_prefill_abs_replay_boost"]),
                expert_prefill_abs_exec_threshold=float(current_options["pafc_expert_prefill_abs_exec_threshold"]),
                expert_prefill_abs_window_mining_candidates=int(current_options["pafc_expert_prefill_abs_window_mining_candidates"]),
                dual_abs_margin_k=float(current_options["pafc_dual_abs_margin_k"]),
                dual_qc_ratio_th=float(current_options["pafc_dual_qc_ratio_th"]),
                dual_heat_backup_ratio_th=float(current_options["pafc_dual_heat_backup_ratio_th"]),
                dual_safe_abs_u_th=float(current_options["pafc_dual_safe_abs_u_th"]),
                actor_warm_start_epochs=int(current_options["pafc_actor_warm_start_epochs"]),
                actor_warm_start_batch_size=int(current_options["pafc_actor_warm_start_batch_size"]),
                actor_warm_start_lr=float(current_options["pafc_actor_warm_start_lr"]),
                checkpoint_interval_steps=int(current_options["pafc_checkpoint_interval_steps"]),
                eval_window_pool_size=int(current_options["pafc_eval_window_pool_size"]),
                eval_window_count=int(current_options["pafc_eval_window_count"]),
                best_gate_enabled=bool(current_options["pafc_best_gate_enabled"]),
                best_gate_electric_min=float(current_options["pafc_best_gate_electric_min"]),
                best_gate_heat_min=float(current_options["pafc_best_gate_heat_min"]),
                best_gate_cool_min=float(current_options["pafc_best_gate_cool_min"]),
                plateau_control_enabled=bool(current_options["pafc_plateau_control_enabled"]),
                plateau_patience_evals=int(current_options["pafc_plateau_patience_evals"]),
                plateau_lr_decay_factor=float(current_options["pafc_plateau_lr_decay_factor"]),
                plateau_min_actor_lr=float(current_options["pafc_plateau_min_actor_lr"]),
                plateau_min_critic_lr=float(current_options["pafc_plateau_min_critic_lr"]),
                plateau_early_stop_patience_evals=int(
                    current_options["pafc_plateau_early_stop_patience_evals"]
                ),
                hidden_dims=_parse_hidden_dims(current_options["pafc_hidden_dims"]),
                seed=int(seed),
                device=str(current_options["device"]),
            )
            result = train_pafc_td3(
                train_df=train_df,
                train_statistics=train_statistics,
                env_config=env_config,
                trainer_config=config,
                run_root=args.run_root,
            )
            payload: dict[str, object] = {
                "mode": "train",
                "train_year": TRAIN_YEAR,
                "policy": "pafc_td3",
                "seed": int(seed),
                "ignored_training_keys": ignored_keys,
                **result,
            }
            if bool(getattr(args, "eval_after_train", False)):
                if eval_df_cache is None:
                    eval_df_cache = load_exogenous_data(args.eval_path)
                run_dir = Path(str(result.get("run_dir", "") or "")).resolve()
                payload["eval_summary"] = evaluate_pafc_td3(
                    eval_df=eval_df_cache,
                    config=env_config,
                    checkpoint_path=Path(str(result["actor_checkpoint_path"])).resolve(),
                    run_dir=run_dir,
                    seed=int(seed),
                    device=str(current_options["device"]),
                )
            outputs.append(payload)
        print(json.dumps(outputs[0] if not multi_seed else outputs, indent=2, ensure_ascii=False))
        return

    sequence_enabled = (
        training_options["policy"] == "sequence_rule"
        and training_options["sequence_adapter"] in {"mlp", "transformer", "mamba"}
    )
    if sequence_enabled:
        warnings: list[str] = []
        episodes_config = int(training_options.get("episodes", 0) or 0)
        if episodes_config > 0:
            warnings.append("sequence_rule 深度训练使用 train_steps 作为预算；training.episodes 在该路径不生效。")
        adapter_name = str(training_options.get("sequence_adapter", "rule")).strip().lower()
        train_steps = int(training_options.get("train_steps", 0) or 0)
        update_epochs = int(training_options.get("update_epochs", 0) or 0)
        if adapter_name in {"transformer", "mamba"} and train_steps > 0 and train_steps < 100_000:
            warnings.append("transformer/mamba 建议 train_steps>=100k；更小预算通常不公平/不稳定。")
        if update_epochs >= 20 and train_steps > 0 and train_steps < 200_000:
            warnings.append("update_epochs 很大但 train_steps 很小：可能出现过拟合/不稳定（建议降低 update_epochs 或提高 train_steps）。")
        ignored_keys = sorted(
            set(TRAINING_OPTION_KEYS)
            - {
                "seed",
                "policy",
                "sequence_adapter",
                "history_steps",
                "episode_days",
                "train_steps",
                "batch_size",
                "update_epochs",
                "lr",
                "device",
            }
        )
        from .policy.trainer import SequenceTrainerConfig, train_sequence_policy

        train_statistics = compute_training_statistics(train_df)
        outputs: list[dict[str, object]] = []
        for seed in seed_values:
            current_options = dict(training_options)
            current_options["seed"] = seed
            trainer_config = SequenceTrainerConfig(
                policy_backbone=current_options["sequence_adapter"],
                history_steps=current_options["history_steps"],
                episode_days=current_options["episode_days"],
                train_steps=current_options["train_steps"],
                batch_size=current_options["batch_size"],
                update_epochs=current_options["update_epochs"],
                lr=current_options["lr"],
                seed=seed,
                device=current_options["device"],
            )
            result = train_sequence_policy(
                train_df=train_df,
                train_statistics=train_statistics,
                env_config=env_config,
                trainer_config=trainer_config,
                run_root=args.run_root,
            )
            outputs.append(
                {
                    "mode": "train",
                    "train_year": TRAIN_YEAR,
                    "policy": current_options["policy"],
                    "sequence_adapter": current_options["sequence_adapter"],
                    "seed": seed,
                    "ignored_training_keys": ignored_keys,
                    "warnings": warnings,
                    **result,
                }
            )
        print(json.dumps(outputs[0] if not multi_seed else outputs, indent=2, ensure_ascii=False))
        return

    ignored_keys = sorted(
        set(TRAINING_OPTION_KEYS)
        - {
            "seed",
            "policy",
            "sequence_adapter",
            "history_steps",
            "episode_days",
            "episodes",
        }
    )
    outputs: list[dict[str, object]] = []
    for seed in seed_values:
        current_options = dict(training_options)
        current_options["seed"] = seed
        run_dir = train_baseline(
            train_df=train_df,
            episode_days=current_options["episode_days"],
            episodes=current_options["episodes"],
            policy_name=current_options["policy"],
            history_steps=current_options["history_steps"],
            sequence_adapter=current_options["sequence_adapter"],
            seed=seed,
            run_root=args.run_root,
            config=env_config,
        )
        outputs.append(
            {
                "mode": "train",
                "train_year": TRAIN_YEAR,
                "run_dir": str(run_dir),
                "policy": current_options["policy"],
                "history_steps": current_options["history_steps"],
                "sequence_adapter": current_options["sequence_adapter"],
                "episodes": current_options["episodes"],
                "episode_days": current_options["episode_days"],
                "seed": seed,
                "ignored_training_keys": ignored_keys,
            }
        )
    print(json.dumps(outputs[0] if not multi_seed else outputs, indent=2, ensure_ascii=False))


def _command_eval(args: argparse.Namespace) -> None:
    """
    eval 子命令：运行 2025 年评估，自动识别 checkpoint 类型。

    路由逻辑：
    1. 若 checkpoint 包含 artifact_type=sb3_policy -> 调用 SB3 评估
    2. 若 checkpoint 包含 artifact_type=pafc_td3_actor -> 调用 PAFC-TD3 评估
    3. 否则 -> 调用 baseline 评估（rule/easy_rule/random/sequence_rule/milp_mpc/ga_mpc）

    评估年份固定为 2025，数据来自 data/processed/cchp_main_15min_2025.csv
    """
    eval_df = load_exogenous_data(args.eval_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    force_mode = getattr(args, "constraint_mode", None)
    env_config = build_env_config_from_overrides(env_overrides, force_constraint_mode=force_mode)
    training_options = _resolve_training_options(args)
    seed_values = _normalize_seed_list(training_options.get("seed", 42), fallback=42)
    multi_seed = len(seed_values) > 1

    checkpoint_kind, resolved_checkpoint_path, checkpoint_payload = _detect_checkpoint_artifact(
        args.checkpoint
    )
    requested_policy = str(training_options.get("policy", "")).strip().lower().replace("-", "_")
    if checkpoint_kind == "baseline" and requested_policy == "pafc_td3":
        if args.checkpoint is None:
            raise ValueError("policy=pafc_td3 的 generic eval 必须显式提供 --checkpoint。")
        checkpoint_kind, resolved_checkpoint_path, checkpoint_payload = _detect_checkpoint_artifact(
            args.checkpoint
        )
        if checkpoint_kind != "pafc_td3":
            raise ValueError(
                "当前 checkpoint 未识别为 pafc_td3_actor；"
                "请传入 pafc_td3_actor.json，或传入带有 PAFC metadata 的 actor .pt。"
            )

    if args.run_dir is not None:
        base_run_dir = Path(args.run_dir)
    elif args.checkpoint is not None:
        base_run_dir = Path(args.checkpoint).resolve().parent.parent
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_run_dir = Path("runs") / f"{stamp}_eval_only"

    if checkpoint_kind == "sb3" and multi_seed:
        raise ValueError(
            "SB3 `eval` 当前不允许 deterministic 多-seed sweep。"
            "固定 2025 全年 + deterministic policy replay 在当前配置下通常不会因 eval seed 改变 KPI，"
            "只会生成误导性的 `seed_*` 目录。"
            "如果要比较随机种子，请使用多训练 seed；如果确实要做 replay 随机性检查，请显式使用 `sb3-eval --stochastic`。"
        )

    outputs: list[dict[str, object]] = []
    for seed in seed_values:
        current_options = dict(training_options)
        current_options["seed"] = seed
        target_run_dir = _maybe_seed_run_dir(base_run_dir, seed, multi_seed)
        if checkpoint_kind == "sb3":
            summary = evaluate_sb3_policy(
                eval_df=eval_df,
                env_config=env_config,
                checkpoint_json=args.checkpoint,
                run_dir=target_run_dir,
                seed=seed,
                deterministic=True,
                model_source=args.model_source,
                device=current_options["device"],
            )
        elif checkpoint_kind == "pafc_td3":
            if resolved_checkpoint_path is None:
                raise ValueError("PAFC-TD3 eval 缺少 actor checkpoint。")
            summary = evaluate_pafc_td3(
                eval_df=eval_df,
                config=env_config,
                checkpoint_path=resolved_checkpoint_path,
                run_dir=target_run_dir,
                seed=int(seed),
                device=current_options["device"],
            )
        else:
            summary = evaluate_baseline(
                eval_df=eval_df,
                run_dir=target_run_dir,
                policy_name=current_options["policy"],
                history_steps=current_options["history_steps"],
                sequence_adapter=current_options["sequence_adapter"],
                seed=seed,
                checkpoint_path=resolved_checkpoint_path,
                device=current_options["device"],
                config=env_config,
            )
        outputs.append(
            {
                "mode": "eval",
                "eval_year": EVAL_YEAR,
                "run_dir": str(target_run_dir),
                "seed": seed,
                "summary": summary,
            }
        )

    if multi_seed:
        _write_multi_seed_eval_summary(base_run_dir, outputs, preferred_seed=seed_values[0])
    print(json.dumps(outputs[0] if not multi_seed else outputs, indent=2, ensure_ascii=False))


def _command_sb3_train(args: argparse.Namespace) -> None:
    """
    sb3-train 子命令：使用 Stable-Baselines3 训练 PPO/SAC/TD3/DDPG/DQN。

    与 train 子命令的区别：
    - 显式指定算法（--algo）
    - 不依赖 sb3_enabled 标志
    - 产出 baseline_policy.json（artifact_type=sb3_policy）
    """
    train_df = load_exogenous_data(args.train_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    force_mode = getattr(args, "constraint_mode", None)
    env_config = build_env_config_from_overrides(env_overrides, force_constraint_mode=force_mode)
    training_defaults = build_training_options(load_training_overrides(_resolve_env_config_path(args)))

    def _arg_or_training_default(arg_name: str, training_key: str):
        if hasattr(args, arg_name):
            return getattr(args, arg_name)
        return training_defaults[training_key]

    seed_values = _normalize_seed_list(getattr(args, "seed", None), fallback=int(training_defaults["seed"]))
    multi_seed = len(seed_values) > 1
    eval_df_cache: pd.DataFrame | None = None
    outputs: list[dict[str, object]] = []

    for seed in seed_values:
        config = SB3TrainConfig(
            algo=str(_arg_or_training_default("algo", "sb3_algo")),
            backbone=str(_arg_or_training_default("backbone", "sb3_backbone")),
            history_steps=int(_arg_or_training_default("history_steps", "sb3_history_steps")),
            total_timesteps=int(_arg_or_training_default("total_timesteps", "sb3_total_timesteps")),
            episode_days=int(_arg_or_training_default("episode_days", "episode_days")),
            n_envs=int(_arg_or_training_default("n_envs", "sb3_n_envs")),
            learning_rate=float(_arg_or_training_default("learning_rate", "sb3_learning_rate")),
            batch_size=int(_arg_or_training_default("batch_size", "sb3_batch_size")),
            gamma=float(_arg_or_training_default("gamma", "sb3_gamma")),
            vec_norm_obs=bool(_arg_or_training_default("vec_norm_obs", "sb3_vec_norm_obs")),
            vec_norm_reward=bool(_arg_or_training_default("vec_norm_reward", "sb3_vec_norm_reward")),
            eval_freq=int(_arg_or_training_default("eval_freq", "sb3_eval_freq")),
            eval_episode_days=int(_arg_or_training_default("eval_episode_days", "sb3_eval_episode_days")),
            eval_window_pool_size=int(_arg_or_training_default("eval_window_pool_size", "sb3_eval_window_pool_size")),
            eval_window_count=int(_arg_or_training_default("eval_window_count", "sb3_eval_window_count")),
            eval_window_seed=seed,
            ppo_warm_start_enabled=bool(_arg_or_training_default("ppo_warm_start_enabled", "sb3_ppo_warm_start_enabled")),
            residual_enabled=bool(_arg_or_training_default("residual_enabled", "sb3_residual_enabled")),
            residual_policy=str(_arg_or_training_default("residual_policy", "sb3_residual_policy")),
            residual_scale=float(_arg_or_training_default("residual_scale", "sb3_residual_scale")),
            ppo_warm_start_samples=int(_arg_or_training_default("ppo_warm_start_samples", "sb3_ppo_warm_start_samples")),
            ppo_warm_start_epochs=int(_arg_or_training_default("ppo_warm_start_epochs", "sb3_ppo_warm_start_epochs")),
            ppo_warm_start_batch_size=int(_arg_or_training_default("ppo_warm_start_batch_size", "sb3_ppo_warm_start_batch_size")),
            ppo_warm_start_lr=float(_arg_or_training_default("ppo_warm_start_lr", "sb3_ppo_warm_start_lr")),
            offpolicy_prefill_enabled=bool(_arg_or_training_default("offpolicy_prefill_enabled", "sb3_offpolicy_prefill_enabled")),
            offpolicy_prefill_steps=int(_arg_or_training_default("offpolicy_prefill_steps", "sb3_offpolicy_prefill_steps")),
            offpolicy_prefill_policy=str(_arg_or_training_default("offpolicy_prefill_policy", "sb3_offpolicy_prefill_policy")),
            ppo_n_steps=int(_arg_or_training_default("ppo_n_steps", "sb3_ppo_n_steps")),
            ppo_gae_lambda=float(_arg_or_training_default("ppo_gae_lambda", "sb3_ppo_gae_lambda")),
            ppo_ent_coef=float(_arg_or_training_default("ppo_ent_coef", "sb3_ppo_ent_coef")),
            ppo_clip_range=float(_arg_or_training_default("ppo_clip_range", "sb3_ppo_clip_range")),
            dqn_action_mode=str(_arg_or_training_default("dqn_action_mode", "sb3_dqn_action_mode")),
            dqn_target_update_interval=int(_arg_or_training_default("dqn_target_update_interval", "sb3_dqn_target_update_interval")),
            dqn_exploration_fraction=float(_arg_or_training_default("dqn_exploration_fraction", "sb3_dqn_exploration_fraction")),
            dqn_exploration_initial_eps=float(_arg_or_training_default("dqn_exploration_initial_eps", "sb3_dqn_exploration_initial_eps")),
            dqn_exploration_final_eps=float(_arg_or_training_default("dqn_exploration_final_eps", "sb3_dqn_exploration_final_eps")),
            learning_starts=int(_arg_or_training_default("learning_starts", "sb3_learning_starts")),
            train_freq=int(_arg_or_training_default("train_freq", "sb3_train_freq")),
            gradient_steps=int(_arg_or_training_default("gradient_steps", "sb3_gradient_steps")),
            tau=float(_arg_or_training_default("tau", "sb3_tau")),
            action_noise_std=float(_arg_or_training_default("action_noise_std", "sb3_action_noise_std")),
            buffer_size=int(_arg_or_training_default("buffer_size", "sb3_buffer_size")),
            optimize_memory_usage=bool(_arg_or_training_default("optimize_memory_usage", "sb3_optimize_memory_usage")),
            best_gate_enabled=bool(_arg_or_training_default("best_gate_enabled", "sb3_best_gate_enabled")),
            best_gate_electric_min=float(_arg_or_training_default("best_gate_electric_min", "sb3_best_gate_electric_min")),
            best_gate_heat_min=float(_arg_or_training_default("best_gate_heat_min", "sb3_best_gate_heat_min")),
            best_gate_cool_min=float(_arg_or_training_default("best_gate_cool_min", "sb3_best_gate_cool_min")),
            plateau_control_enabled=bool(_arg_or_training_default("plateau_control_enabled", "sb3_plateau_control_enabled")),
            plateau_patience_evals=int(_arg_or_training_default("plateau_patience_evals", "sb3_plateau_patience_evals")),
            plateau_lr_decay_factor=float(_arg_or_training_default("plateau_lr_decay_factor", "sb3_plateau_lr_decay_factor")),
            plateau_min_lr=float(_arg_or_training_default("plateau_min_lr", "sb3_plateau_min_lr")),
            plateau_early_stop_patience_evals=int(
                _arg_or_training_default(
                    "plateau_early_stop_patience_evals",
                    "sb3_plateau_early_stop_patience_evals",
                )
            ),
            seed=seed,
            device=str(_arg_or_training_default("device", "device")),
        )
        result = train_sb3_policy(
            train_df=train_df,
            env_config=env_config,
            config=config,
            run_root=args.run_root,
        )
        payload: dict[str, object] = {"mode": "sb3_train", "seed": seed, **result}
        if bool(getattr(args, "eval_after_train", False)):
            if eval_df_cache is None:
                eval_df_cache = load_exogenous_data(args.eval_path)
            run_dir = Path(str(result.get("run_dir", "") or "")).resolve()
            checkpoint_json = run_dir / "checkpoints" / "baseline_policy.json"
            payload["eval_summary"] = evaluate_sb3_policy(
                eval_df=eval_df_cache,
                env_config=env_config,
                checkpoint_json=checkpoint_json,
                run_dir=run_dir,
                seed=seed,
                deterministic=True,
                model_source="best",
                device=args.device,
            )
        outputs.append(payload)
    print(json.dumps(outputs[0] if not multi_seed else outputs, indent=2, ensure_ascii=False))


def _command_sb3_eval(args: argparse.Namespace) -> None:
    """
    sb3-eval 子命令：使用 SB3 checkpoint 运行 2025 年评估。

    与 eval 子命令的区别：
    - 必须显式指定 --checkpoint（baseline_policy.json）
    - 支持 --stochastic 标志（随机动作采样）
    """
    eval_df = load_exogenous_data(args.eval_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    force_mode = getattr(args, "constraint_mode", None)
    env_config = build_env_config_from_overrides(env_overrides, force_constraint_mode=force_mode)
    seed_values = _normalize_seed_list(args.seed, fallback=42)
    multi_seed = len(seed_values) > 1
    base_run_dir = Path(args.run_dir)
    if multi_seed and not args.stochastic:
        raise ValueError(
            "SB3 `sb3-eval` 当前不允许 deterministic 多-seed sweep。"
            "固定 2025 全年 replay 在当前配置下通常不会因 eval seed 改变 KPI。"
            "如果目标是种子敏感性，请比较多训练 seed；如果目标是策略采样随机性，请显式传 `--stochastic`。"
        )
    outputs: list[dict[str, object]] = []
    for seed in seed_values:
        target_run_dir = _maybe_seed_run_dir(base_run_dir, seed, multi_seed)
        summary = evaluate_sb3_policy(
            eval_df=eval_df,
            env_config=env_config,
            checkpoint_json=args.checkpoint,
            run_dir=target_run_dir,
            seed=seed,
            deterministic=not args.stochastic,
            model_source=args.model_source,
            device=args.device,
        )
        outputs.append(
            {
                "mode": "sb3_eval",
                "run_dir": str(target_run_dir),
                "seed": seed,
                "summary": summary,
            }
        )

    if multi_seed:
        _write_multi_seed_eval_summary(base_run_dir, outputs, preferred_seed=seed_values[0])
    print(json.dumps(outputs[0] if not multi_seed else outputs, indent=2, ensure_ascii=False))


def _command_pafc_train(args: argparse.Namespace) -> None:
    train_df = load_exogenous_data(args.train_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    force_mode = getattr(args, "constraint_mode", None)
    env_config = build_env_config_from_overrides(env_overrides, force_constraint_mode=force_mode)
    training_defaults = build_training_options(load_training_overrides(_resolve_env_config_path(args)))
    train_statistics = compute_training_statistics(train_df)

    def _arg_or_training_default(arg_name: str, training_key: str | None, fallback):
        if hasattr(args, arg_name):
            return getattr(args, arg_name)
        if training_key is not None and training_key in training_defaults:
            return training_defaults[training_key]
        return fallback

    seed_values = _normalize_seed_list(
        getattr(args, "seed", None),
        fallback=int(training_defaults["seed"]),
    )
    multi_seed = len(seed_values) > 1
    eval_df_cache: pd.DataFrame | None = None
    outputs: list[dict[str, object]] = []

    for seed in seed_values:
        config = PAFCTD3TrainConfig(
            projection_surrogate_checkpoint_path=args.projection_surrogate_checkpoint,
            episode_days=int(_arg_or_training_default("episode_days", "pafc_episode_days", 14)),
            total_env_steps=int(_arg_or_training_default("total_env_steps", "pafc_total_env_steps", 262_144)),
            warmup_steps=int(_arg_or_training_default("warmup_steps", "pafc_warmup_steps", 4_096)),
            replay_capacity=int(_arg_or_training_default("replay_capacity", "pafc_replay_capacity", 100_000)),
            batch_size=int(_arg_or_training_default("batch_size", "pafc_batch_size", 256)),
            updates_per_step=int(_arg_or_training_default("updates_per_step", "pafc_updates_per_step", 1)),
            gamma=float(_arg_or_training_default("gamma", "pafc_gamma", 0.99)),
            tau=float(_arg_or_training_default("tau", "pafc_tau", 0.005)),
            actor_lr=float(_arg_or_training_default("actor_lr", "pafc_actor_lr", 1e-4)),
            critic_lr=float(_arg_or_training_default("critic_lr", "pafc_critic_lr", 3e-4)),
            dual_lr=float(_arg_or_training_default("dual_lr", "pafc_dual_lr", 5e-3)),
            dual_warmup_steps=int(
                _arg_or_training_default("dual_warmup_steps", "pafc_dual_warmup_steps", 8_192)
            ),
            actor_delay=int(_arg_or_training_default("actor_delay", "pafc_actor_delay", 2)),
            exploration_noise_std=float(_arg_or_training_default("exploration_noise_std", "pafc_exploration_noise_std", 0.06)),
            target_policy_noise_std=float(_arg_or_training_default("target_policy_noise_std", "pafc_target_policy_noise_std", 0.06)),
            target_noise_clip=float(_arg_or_training_default("target_noise_clip", "pafc_target_noise_clip", 0.12)),
            gap_penalty_coef=float(_arg_or_training_default("gap_penalty_coef", "pafc_gap_penalty_coef", 0.2)),
            exec_action_anchor_coef=float(
                _arg_or_training_default("exec_action_anchor_coef", "pafc_exec_action_anchor_coef", 1.5)
            ),
            exec_action_anchor_safe_floor=float(
                _arg_or_training_default(
                    "exec_action_anchor_safe_floor",
                    "pafc_exec_action_anchor_safe_floor",
                    0.05,
                )
            ),
            gt_off_deadband_ratio=float(
                _arg_or_training_default(
                    "gt_off_deadband_ratio",
                    "pafc_gt_off_deadband_ratio",
                    0.0,
                )
            ),
            abs_ready_focus_coef=float(
                _arg_or_training_default("abs_ready_focus_coef", "pafc_abs_ready_focus_coef", 0.25)
            ),
            invalid_abs_penalty_coef=float(
                _arg_or_training_default("invalid_abs_penalty_coef", "pafc_invalid_abs_penalty_coef", 0.25)
            ),
            economic_boiler_proxy_coef=float(
                _arg_or_training_default(
                    "economic_boiler_proxy_coef",
                    "pafc_economic_boiler_proxy_coef",
                    0.10,
                )
            ),
            economic_abs_tradeoff_coef=float(
                _arg_or_training_default(
                    "economic_abs_tradeoff_coef",
                    "pafc_economic_abs_tradeoff_coef",
                    0.05,
                )
            ),
            economic_gt_grid_proxy_coef=float(
                _arg_or_training_default(
                    "economic_gt_grid_proxy_coef",
                    "pafc_economic_gt_grid_proxy_coef",
                    0.50,
                )
            ),
            economic_gt_distill_coef=float(
                _arg_or_training_default(
                    "economic_gt_distill_coef",
                    "pafc_economic_gt_distill_coef",
                    0.10,
                )
            ),
            economic_teacher_distill_coef=float(
                _arg_or_training_default(
                    "economic_teacher_distill_coef",
                    "pafc_economic_teacher_distill_coef",
                    0.25,
                )
            ),
            economic_teacher_safe_preserve_coef=float(
                _arg_or_training_default(
                    "economic_teacher_safe_preserve_coef",
                    "pafc_economic_teacher_safe_preserve_coef",
                    1.0,
                )
            ),
            economic_teacher_safe_preserve_low_margin_boost=float(
                _arg_or_training_default(
                    "economic_teacher_safe_preserve_low_margin_boost",
                    "pafc_economic_teacher_safe_preserve_low_margin_boost",
                    0.75,
                )
            ),
            economic_teacher_safe_preserve_high_cooling_boost=float(
                _arg_or_training_default(
                    "economic_teacher_safe_preserve_high_cooling_boost",
                    "pafc_economic_teacher_safe_preserve_high_cooling_boost",
                    1.0,
                )
            ),
            economic_teacher_safe_preserve_joint_boost=float(
                _arg_or_training_default(
                    "economic_teacher_safe_preserve_joint_boost",
                    "pafc_economic_teacher_safe_preserve_joint_boost",
                    1.0,
                )
            ),
            economic_teacher_mismatch_focus_coef=float(
                _arg_or_training_default(
                    "economic_teacher_mismatch_focus_coef",
                    "pafc_economic_teacher_mismatch_focus_coef",
                    0.0,
                )
            ),
            economic_teacher_mismatch_focus_min_scale=float(
                _arg_or_training_default(
                    "economic_teacher_mismatch_focus_min_scale",
                    "pafc_economic_teacher_mismatch_focus_min_scale",
                    0.75,
                )
            ),
            economic_teacher_mismatch_focus_max_scale=float(
                _arg_or_training_default(
                    "economic_teacher_mismatch_focus_max_scale",
                    "pafc_economic_teacher_mismatch_focus_max_scale",
                    2.5,
                )
            ),
            economic_teacher_proxy_advantage_min=float(
                _arg_or_training_default(
                    "economic_teacher_proxy_advantage_min",
                    "pafc_economic_teacher_proxy_advantage_min",
                    0.02,
                )
            ),
            economic_teacher_gt_proxy_advantage_min=float(
                _arg_or_training_default(
                    "economic_teacher_gt_proxy_advantage_min",
                    "pafc_economic_teacher_gt_proxy_advantage_min",
                    0.01,
                )
            ),
            economic_teacher_bes_proxy_advantage_min=float(
                _arg_or_training_default(
                    "economic_teacher_bes_proxy_advantage_min",
                    "pafc_economic_teacher_bes_proxy_advantage_min",
                    0.002,
                )
            ),
            economic_teacher_max_safe_abs_risk_gap=float(
                _arg_or_training_default(
                    "economic_teacher_max_safe_abs_risk_gap",
                    "pafc_economic_teacher_max_safe_abs_risk_gap",
                    0.05,
                )
            ),
            economic_teacher_projection_gap_max=float(
                _arg_or_training_default(
                    "economic_teacher_projection_gap_max",
                    "pafc_economic_teacher_projection_gap_max",
                    0.20,
                )
            ),
            economic_teacher_gt_projection_gap_max=float(
                _arg_or_training_default(
                    "economic_teacher_gt_projection_gap_max",
                    "pafc_economic_teacher_gt_projection_gap_max",
                    1.0,
                )
            ),
            economic_teacher_bes_price_opportunity_min=float(
                _arg_or_training_default(
                    "economic_teacher_bes_price_opportunity_min",
                    "pafc_economic_teacher_bes_price_opportunity_min",
                    0.10,
                )
            ),
            economic_teacher_bes_anchor_preserve_scale=float(
                _arg_or_training_default(
                    "economic_teacher_bes_anchor_preserve_scale",
                    "pafc_economic_teacher_bes_anchor_preserve_scale",
                    0.85,
                )
            ),
            economic_teacher_warm_start_weight=float(
                _arg_or_training_default(
                    "economic_teacher_warm_start_weight",
                    "pafc_economic_teacher_warm_start_weight",
                    4.0,
                )
            ),
            economic_teacher_prefill_replay_boost=int(
                _arg_or_training_default(
                    "economic_teacher_prefill_replay_boost",
                    "pafc_economic_teacher_prefill_replay_boost",
                    2,
                )
            ),
            economic_teacher_gt_action_weight=float(
                _arg_or_training_default(
                    "economic_teacher_gt_action_weight",
                    "pafc_economic_teacher_gt_action_weight",
                    2.0,
                )
            ),
            economic_teacher_bes_action_weight=float(
                _arg_or_training_default(
                    "economic_teacher_bes_action_weight",
                    "pafc_economic_teacher_bes_action_weight",
                    1.5,
                )
            ),
            economic_teacher_tes_action_weight=float(
                _arg_or_training_default(
                    "economic_teacher_tes_action_weight",
                    "pafc_economic_teacher_tes_action_weight",
                    0.5,
                )
            ),
            economic_teacher_full_year_warm_start_samples=int(
                _arg_or_training_default(
                    "economic_teacher_full_year_warm_start_samples",
                    "pafc_economic_teacher_full_year_warm_start_samples",
                    4096,
                )
            ),
            economic_teacher_full_year_warm_start_epochs=int(
                _arg_or_training_default(
                    "economic_teacher_full_year_warm_start_epochs",
                    "pafc_economic_teacher_full_year_warm_start_epochs",
                    4,
                )
            ),
            economic_gt_full_year_warm_start_samples=int(
                _arg_or_training_default(
                    "economic_gt_full_year_warm_start_samples",
                    "pafc_economic_gt_full_year_warm_start_samples",
                    0,
                )
            ),
            economic_gt_full_year_warm_start_epochs=int(
                _arg_or_training_default(
                    "economic_gt_full_year_warm_start_epochs",
                    "pafc_economic_gt_full_year_warm_start_epochs",
                    0,
                )
            ),
            economic_gt_full_year_warm_start_u_weight=float(
                _arg_or_training_default(
                    "economic_gt_full_year_warm_start_u_weight",
                    "pafc_economic_gt_full_year_warm_start_u_weight",
                    0.0,
                )
            ),
            economic_bes_distill_coef=float(
                _arg_or_training_default(
                    "economic_bes_distill_coef",
                    "pafc_economic_bes_distill_coef",
                    0.15,
                )
            ),
            economic_bes_prior_u=float(
                _arg_or_training_default("economic_bes_prior_u", "pafc_economic_bes_prior_u", 0.35)
            ),
            economic_bes_charge_u_scale=float(
                _arg_or_training_default(
                    "economic_bes_charge_u_scale",
                    "pafc_economic_bes_charge_u_scale",
                    1.8,
                )
            ),
            economic_bes_discharge_u_scale=float(
                _arg_or_training_default(
                    "economic_bes_discharge_u_scale",
                    "pafc_economic_bes_discharge_u_scale",
                    1.0,
                )
            ),
            economic_bes_charge_weight=float(
                _arg_or_training_default(
                    "economic_bes_charge_weight",
                    "pafc_economic_bes_charge_weight",
                    2.0,
                )
            ),
            economic_bes_discharge_weight=float(
                _arg_or_training_default(
                    "economic_bes_discharge_weight",
                    "pafc_economic_bes_discharge_weight",
                    1.0,
                )
            ),
            economic_bes_charge_pressure_bonus=float(
                _arg_or_training_default(
                    "economic_bes_charge_pressure_bonus",
                    "pafc_economic_bes_charge_pressure_bonus",
                    1.0,
                )
            ),
            economic_bes_charge_soc_ceiling=float(
                _arg_or_training_default(
                    "economic_bes_charge_soc_ceiling",
                    "pafc_economic_bes_charge_soc_ceiling",
                    0.75,
                )
            ),
            economic_bes_discharge_soc_floor=float(
                _arg_or_training_default(
                    "economic_bes_discharge_soc_floor",
                    "pafc_economic_bes_discharge_soc_floor",
                    0.35,
                )
            ),
            economic_bes_full_year_warm_start_samples=int(
                _arg_or_training_default(
                    "economic_bes_full_year_warm_start_samples",
                    "pafc_economic_bes_full_year_warm_start_samples",
                    4096,
                )
            ),
            economic_bes_full_year_warm_start_epochs=int(
                _arg_or_training_default(
                    "economic_bes_full_year_warm_start_epochs",
                    "pafc_economic_bes_full_year_warm_start_epochs",
                    2,
                )
            ),
            economic_bes_full_year_warm_start_u_weight=float(
                _arg_or_training_default(
                    "economic_bes_full_year_warm_start_u_weight",
                    "pafc_economic_bes_full_year_warm_start_u_weight",
                    4.0,
                )
            ),
            economic_bes_teacher_selection_priority_boost=float(
                _arg_or_training_default(
                    "economic_bes_teacher_selection_priority_boost",
                    "pafc_economic_bes_teacher_selection_priority_boost",
                    0.75,
                )
            ),
            economic_bes_economic_source_priority_bonus=float(
                _arg_or_training_default(
                    "economic_bes_economic_source_priority_bonus",
                    "pafc_economic_bes_economic_source_priority_bonus",
                    0.10,
                )
            ),
            economic_bes_economic_source_min_share=float(
                _arg_or_training_default(
                    "economic_bes_economic_source_min_share",
                    "pafc_economic_bes_economic_source_min_share",
                    0.75,
                )
            ),
            economic_bes_idle_economic_source_min_share=float(
                _arg_or_training_default(
                    "economic_bes_idle_economic_source_min_share",
                    "pafc_economic_bes_idle_economic_source_min_share",
                    0.75,
                )
            ),
            economic_bes_teacher_target_min_share=float(
                _arg_or_training_default(
                    "economic_bes_teacher_target_min_share",
                    "pafc_economic_bes_teacher_target_min_share",
                    0.0,
                )
            ),
            surrogate_actor_trust_coef=float(
                _arg_or_training_default(
                    "surrogate_actor_trust_coef",
                    "pafc_surrogate_actor_trust_coef",
                    0.60,
                )
            ),
            surrogate_actor_trust_min_scale=float(
                _arg_or_training_default(
                    "surrogate_actor_trust_min_scale",
                    "pafc_surrogate_actor_trust_min_scale",
                    0.10,
                )
            ),
            state_feasible_action_shaping_enabled=bool(
                _arg_or_training_default(
                    "state_feasible_action_shaping_enabled",
                    "pafc_state_feasible_action_shaping_enabled",
                    True,
                )
            ),
            abs_min_on_gate_th=float(
                _arg_or_training_default("abs_min_on_gate_th", "pafc_abs_min_on_gate_th", 0.75)
            ),
            abs_min_on_u_margin=float(
                _arg_or_training_default("abs_min_on_u_margin", "pafc_abs_min_on_u_margin", 0.02)
            ),
            expert_prefill_policy=str(
                _arg_or_training_default("expert_prefill_policy", "pafc_expert_prefill_policy", "easy_rule_abs")
            ),
            expert_prefill_checkpoint_path=str(
                _arg_or_training_default(
                    "expert_prefill_checkpoint_path",
                    "pafc_expert_prefill_checkpoint_path",
                    "",
                )
            ),
            expert_prefill_economic_policy=str(
                _arg_or_training_default(
                    "expert_prefill_economic_policy",
                    "pafc_expert_prefill_economic_policy",
                    "checkpoint",
                )
            ),
            expert_prefill_economic_checkpoint_path=str(
                _arg_or_training_default(
                    "expert_prefill_economic_checkpoint_path",
                    "pafc_expert_prefill_economic_checkpoint_path",
                    "",
                )
            ),
            expert_prefill_steps=int(
                _arg_or_training_default("expert_prefill_steps", "pafc_expert_prefill_steps", 4_096)
            ),
            expert_prefill_cooling_bias=float(
                _arg_or_training_default("expert_prefill_cooling_bias", "pafc_expert_prefill_cooling_bias", 0.5)
            ),
            expert_prefill_abs_replay_boost=int(
                _arg_or_training_default("expert_prefill_abs_replay_boost", "pafc_expert_prefill_abs_replay_boost", 0)
            ),
            expert_prefill_abs_exec_threshold=float(
                _arg_or_training_default("expert_prefill_abs_exec_threshold", "pafc_expert_prefill_abs_exec_threshold", 0.05)
            ),
            expert_prefill_abs_window_mining_candidates=int(
                _arg_or_training_default(
                    "expert_prefill_abs_window_mining_candidates",
                    "pafc_expert_prefill_abs_window_mining_candidates",
                    8,
                )
            ),
            dual_abs_margin_k=float(
                _arg_or_training_default("dual_abs_margin_k", "pafc_dual_abs_margin_k", 1.25)
            ),
            dual_qc_ratio_th=float(
                _arg_or_training_default("dual_qc_ratio_th", "pafc_dual_qc_ratio_th", 0.55)
            ),
            dual_heat_backup_ratio_th=float(
                _arg_or_training_default(
                    "dual_heat_backup_ratio_th",
                    "pafc_dual_heat_backup_ratio_th",
                    0.10,
                )
            ),
            dual_safe_abs_u_th=float(
                _arg_or_training_default("dual_safe_abs_u_th", "pafc_dual_safe_abs_u_th", 0.60)
            ),
            actor_warm_start_epochs=int(
                _arg_or_training_default("actor_warm_start_epochs", "pafc_actor_warm_start_epochs", 4)
            ),
            actor_warm_start_batch_size=int(
                _arg_or_training_default("actor_warm_start_batch_size", "pafc_actor_warm_start_batch_size", 256)
            ),
            actor_warm_start_lr=float(
                _arg_or_training_default("actor_warm_start_lr", "pafc_actor_warm_start_lr", 1e-4)
            ),
            checkpoint_interval_steps=int(
                _arg_or_training_default("checkpoint_interval_steps", "pafc_checkpoint_interval_steps", 16_384)
            ),
            eval_window_pool_size=int(
                _arg_or_training_default("eval_window_pool_size", "pafc_eval_window_pool_size", 16)
            ),
            eval_window_count=int(
                _arg_or_training_default("eval_window_count", "pafc_eval_window_count", 8)
            ),
            best_gate_enabled=bool(
                _arg_or_training_default("best_gate_enabled", "pafc_best_gate_enabled", True)
            ),
            best_gate_electric_min=float(
                _arg_or_training_default("best_gate_electric_min", "pafc_best_gate_electric_min", 1.0)
            ),
            best_gate_heat_min=float(
                _arg_or_training_default("best_gate_heat_min", "pafc_best_gate_heat_min", 0.99)
            ),
            best_gate_cool_min=float(
                _arg_or_training_default("best_gate_cool_min", "pafc_best_gate_cool_min", 0.99)
            ),
            plateau_control_enabled=bool(
                _arg_or_training_default("plateau_control_enabled", "pafc_plateau_control_enabled", True)
            ),
            plateau_patience_evals=int(
                _arg_or_training_default("plateau_patience_evals", "pafc_plateau_patience_evals", 4)
            ),
            plateau_lr_decay_factor=float(
                _arg_or_training_default("plateau_lr_decay_factor", "pafc_plateau_lr_decay_factor", 0.5)
            ),
            plateau_min_actor_lr=float(
                _arg_or_training_default("plateau_min_actor_lr", "pafc_plateau_min_actor_lr", 2.5e-5)
            ),
            plateau_min_critic_lr=float(
                _arg_or_training_default("plateau_min_critic_lr", "pafc_plateau_min_critic_lr", 1e-4)
            ),
            plateau_early_stop_patience_evals=int(
                _arg_or_training_default(
                    "plateau_early_stop_patience_evals",
                    "pafc_plateau_early_stop_patience_evals",
                    8,
                )
            ),
            hidden_dims=_parse_hidden_dims(
                _arg_or_training_default("hidden_dims", "pafc_hidden_dims", (256, 256, 256))
            ),
            seed=int(seed),
            device=str(_arg_or_training_default("device", "device", "auto")),
        )
        result = train_pafc_td3(
            train_df=train_df,
            train_statistics=train_statistics,
            env_config=env_config,
            trainer_config=config,
            run_root=args.run_root,
        )
        payload: dict[str, object] = {"mode": "pafc_train", "seed": int(seed), **result}
        if bool(getattr(args, "eval_after_train", False)):
            if eval_df_cache is None:
                eval_df_cache = load_exogenous_data(args.eval_path)
            run_dir = Path(str(result.get("run_dir", "") or "")).resolve()
            payload["eval_summary"] = evaluate_pafc_td3(
                eval_df=eval_df_cache,
                config=env_config,
                checkpoint_path=Path(str(result["actor_checkpoint_path"])).resolve(),
                run_dir=run_dir,
                seed=int(seed),
                device=str(config.device),
            )
        outputs.append(payload)

    print(json.dumps(outputs[0] if not multi_seed else outputs, indent=2, ensure_ascii=False))


def _command_pafc_eval(args: argparse.Namespace) -> None:
    eval_df = load_exogenous_data(args.eval_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    force_mode = getattr(args, "constraint_mode", None)
    env_config = build_env_config_from_overrides(env_overrides, force_constraint_mode=force_mode)
    seed_values = _normalize_seed_list(args.seed, fallback=42)
    multi_seed = len(seed_values) > 1

    if args.run_dir is not None:
        base_run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_run_dir = Path("runs") / f"{stamp}_eval_pafc_td3"

    outputs: list[dict[str, object]] = []
    checkpoint_kind, resolved_checkpoint_path, _ = _detect_checkpoint_artifact(args.checkpoint)
    if checkpoint_kind != "pafc_td3" or resolved_checkpoint_path is None:
        raise ValueError("pafc-eval 只接受 PAFC-TD3 actor checkpoint（.pt 或 pafc_td3_actor.json）。")
    for seed in seed_values:
        target_run_dir = _maybe_seed_run_dir(base_run_dir, seed, multi_seed)
        summary = evaluate_pafc_td3(
            eval_df=eval_df,
            config=env_config,
            checkpoint_path=resolved_checkpoint_path,
            run_dir=target_run_dir,
            seed=int(seed),
            device=args.device,
        )
        outputs.append(
            {
                "mode": "pafc_eval",
                "run_dir": str(target_run_dir),
                "seed": int(seed),
                "summary": summary,
            }
        )

    if multi_seed:
        _write_multi_seed_eval_summary(base_run_dir, outputs, preferred_seed=seed_values[0])
    print(json.dumps(outputs[0] if not multi_seed else outputs, indent=2, ensure_ascii=False))


def _write_multi_seed_eval_summary(
    base_run_dir: Path,
    outputs: list[dict[str, object]],
    *,
    preferred_seed: int = 42,
) -> None:
    """
    多 seed 评估时，额外写一份 base_run_dir/eval/summary.json，方便外部导出工具读取。

    约定：
    - 每个 seed 的详细 summary 仍写在 seed_x/eval/summary.json（原行为）
    - base summary 默认选择“本次传入 seeds 中的第一个 seed”（若存在），否则选择排序后的第一个 seed
    - 同时写入 eval/summary_seeds.json，记录所有 seed 的摘要
    """
    if not outputs:
        return
    base_run_dir = Path(base_run_dir)
    (base_run_dir / "eval").mkdir(parents=True, exist_ok=True)

    seed_to_summary: dict[int, dict[str, object]] = {}
    seed_entries: list[dict[str, object]] = []
    for item in outputs:
        try:
            seed = int(item.get("seed", 0))
        except Exception:
            continue
        summary = item.get("summary")
        if isinstance(summary, dict):
            seed_to_summary[seed] = dict(summary)
        seed_entries.append(
            {
                "seed": seed,
                "run_dir": str(item.get("run_dir", "")),
            }
        )

    # 写入 seed 列表索引。
    (base_run_dir / "eval" / "summary_seeds.json").write_text(
        json.dumps(seed_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 选择 canonical seed summary。
    canonical_seed = preferred_seed if preferred_seed in seed_to_summary else None
    if canonical_seed is None:
        canonical_seed = sorted(seed_to_summary.keys())[0] if seed_to_summary else None
    if canonical_seed is None:
        return
    canonical = dict(seed_to_summary[canonical_seed])
    canonical["multi_seed"] = True
    canonical["canonical_seed"] = int(canonical_seed)
    canonical["seed_summaries_path"] = str(Path("eval") / "summary_seeds.json")
    (base_run_dir / "eval" / "summary.json").write_text(
        json.dumps(canonical, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _command_calibrate(args: argparse.Namespace) -> None:
    """
    calibrate 子命令：运行物理参数标定搜索（Task-002）。

    功能：
    - 从配置文件读取搜索空间
    - 在训练集上采样参数组合
    - 在评估集上验证效果
    - 输出最优参数配置
    """
    train_df = load_exogenous_data(args.train_path)
    eval_df = load_exogenous_data(args.eval_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    training_options = _resolve_training_options(args)
    config = load_calibration_config(args.config)
    force_mode = getattr(args, "constraint_mode", None)
    if force_mode is not None:
        # calibrate 的 env_config 仍从 base_env_overrides 构建；这里仅把模式强制覆盖透传下去。
        env_overrides = dict(env_overrides)
        env_overrides["constraint_mode"] = str(force_mode)
    search_block = dict(config.get("search", {}))
    search_block["history_steps"] = int(training_options["history_steps"])
    search_block["sequence_adapter"] = str(training_options["sequence_adapter"]).strip().lower()
    config["search"] = search_block
    validate_calibration_config(config)
    result = run_calibration_search(
        train_df=train_df,
        eval_df=eval_df,
        config=config,
        n_samples=args.n_samples,
        seed=training_options["seed"],
        run_root=args.run_root,
        base_env_overrides=env_overrides,
    )
    output = {
        "mode": "calibrate",
        "train_year": TRAIN_YEAR,
        "eval_year": EVAL_YEAR,
        "config": str(args.config),
        **result,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def _command_ablation(args: argparse.Namespace) -> None:
    """
    ablation 子命令：运行约束方式消融实验（Task-003）。

    功能：
    - 对比不同约束处理方式（physics_in_loop vs reward_only）
    - 在训练集上训练策略
    - 在评估集上对比性能指标
    - 输出消融结果汇总
    """
    train_df = load_exogenous_data(args.train_path)
    eval_df = load_exogenous_data(args.eval_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    training_options = _resolve_training_options(args)
    force_mode = getattr(args, "constraint_mode", None)
    if force_mode is not None:
        env_overrides = dict(env_overrides)
        env_overrides["constraint_mode"] = str(force_mode)
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]
    result = run_constraint_ablation(
        train_df=train_df,
        eval_df=eval_df,
        modes=modes,
        policy_name=training_options["policy"],
        history_steps=training_options["history_steps"],
        sequence_adapter=training_options["sequence_adapter"],
        seed=training_options["seed"],
        run_root=args.run_root,
        params_path=args.params,
        base_env_overrides=env_overrides,
    )
    output = {
        "mode": "ablation",
        "train_year": TRAIN_YEAR,
        "eval_year": EVAL_YEAR,
        "sequence_adapter": training_options["sequence_adapter"],
        **result,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def _command_collect(args: argparse.Namespace) -> None:
    """
    collect 子命令：扫描 runs/ 下的 eval/summary.json，汇总为论文表格 CSV（Task-011）。
    """
    result = write_benchmark_tables(
        runs_root=args.runs_root,
        output_csv=args.output,
        full_output_csv=args.full_output,
    )
    print(json.dumps({"mode": "collect", **result}, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    """
    构建 CLI 参数解析器。

    子命令：
    - summary: 打印数据摘要并校验 schema
    - train: 运行训练（支持 baseline/sequence/SB3）
    - eval: 运行评估（自动识别 checkpoint 类型）
    - sb3-train: 显式调用 SB3 训练
    - sb3-eval: 显式调用 SB3 评估
    - pafc-train: 显式调用 Task-012 的 PAFC-TD3 训练
    - pafc-eval: 显式调用 Task-012 的 PAFC-TD3 评估
    - calibrate: 物理参数标定搜索
    - ablation: 约束方式消融实验
    - collect: 汇总 runs 下的 eval 结果到论文表格 CSV
    """
    parser = argparse.ArgumentParser(
        prog="python -m cchp_physical_env",
        description="CCHP Python-only 数据/训练/评估入口",
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    parser.add_argument(
        "--constraint-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(DEFAULT_CONSTRAINT_MODES),
        help="覆盖 env.constraint_mode（其余 env 参数仍从 --env-config 读取）。",
    )

    subparsers = parser.add_subparsers(dest="command")

    summary_parser = subparsers.add_parser("summary", help="打印 2024/2025 摘要并校验冻结 schema。")
    summary_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )
    summary_parser.add_argument(
        "--constraint-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(DEFAULT_CONSTRAINT_MODES),
        help="覆盖 env.constraint_mode（子命令级覆盖）。",
    )

    train_parser = subparsers.add_parser("train", help="运行通用训练骨架（2024，支持 baseline/SB3/PAFC）。")
    train_parser.add_argument("--run-root", type=Path, default=Path("runs"))
    train_parser.add_argument(
        "--constraint-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(DEFAULT_CONSTRAINT_MODES),
        help="覆盖 env.constraint_mode（子命令级覆盖）。",
    )
    train_parser.add_argument("--episode-days", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--episodes", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--policy",
        type=str,
        default=argparse.SUPPRESS,
        choices=["rule", "easy_rule", "random", "sequence_rule", "milp_mpc", "milp-mpc", "ga_mpc", "ga-mpc", "sb3", "pafc_td3", "pafc-td3"],
    )
    train_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )
    train_parser.add_argument("--history-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--sequence-adapter",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(SUPPORTED_SEQUENCE_ADAPTERS),
        help="当 policy=sequence_rule 时选择序列后端。",
    )
    train_parser.add_argument("--train-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--update-epochs", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    train_parser.add_argument("--seed", type=str, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--eval-after-train",
        action="store_true",
        default=False,
        help="训练结束后立即跑一次 2025 评估，并将结果写入同一 run_dir/eval/（可选，较耗时）。",
    )
    train_parser.add_argument(
        "--sb3-enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 SB3 多算法训练（否则走 baseline/sequence trainer）。",
    )
    train_parser.add_argument(
        "--no-sb3",
        dest="sb3_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="禁用 SB3（覆盖 config.yaml 里的 sb3_enabled=true）。",
    )
    train_parser.add_argument("--sb3-algo", type=str, default=argparse.SUPPRESS, choices=["ppo", "sac", "td3", "ddpg", "dqn"])
    train_parser.add_argument(
        "--sb3-backbone",
        type=str,
        default=argparse.SUPPRESS,
        choices=["mlp", "transformer", "mamba"],
        help="SB3 policy backbone（用于 SAC+Transformer 等组合）。",
    )
    train_parser.add_argument("--sb3-history-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-total-timesteps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-n-envs", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-learning-rate", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-batch-size", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-gamma", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--sb3-vec-norm-obs",
        dest="sb3_vec_norm_obs",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 SB3 VecNormalize 观测归一化。",
    )
    train_parser.add_argument(
        "--no-sb3-vec-norm-obs",
        dest="sb3_vec_norm_obs",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 SB3 VecNormalize 观测归一化。",
    )
    train_parser.add_argument(
        "--sb3-vec-norm-reward",
        dest="sb3_vec_norm_reward",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 SB3 VecNormalize 奖励归一化。",
    )
    train_parser.add_argument(
        "--no-sb3-vec-norm-reward",
        dest="sb3_vec_norm_reward",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 SB3 VecNormalize 奖励归一化。",
    )
    train_parser.add_argument("--sb3-eval-freq", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-eval-episode-days", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-eval-window-pool-size", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-eval-window-count", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--sb3-ppo-warm-start-enabled",
        dest="sb3_ppo_warm_start_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 easy_rule 行为克隆预热（当前仅 PPO 支持）。",
    )
    train_parser.add_argument(
        "--no-sb3-ppo-warm-start",
        dest="sb3_ppo_warm_start_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 PPO easy_rule 预热。",
    )
    train_parser.add_argument(
        "--sb3-residual-enabled",
        dest="sb3_residual_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用连续动作算法的 rule residual 模式：动作解释为相对基线策略的残差。",
    )
    train_parser.add_argument(
        "--no-sb3-residual",
        dest="sb3_residual_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭连续动作算法的 residual 模式。",
    )
    train_parser.add_argument(
        "--sb3-residual-policy",
        type=str,
        default=argparse.SUPPRESS,
        choices=["easy_rule", "rule"],
        help="residual 基线策略来源：easy_rule 或 rule。",
    )
    train_parser.add_argument("--sb3-residual-scale", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-ppo-warm-start-samples", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-ppo-warm-start-epochs", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-ppo-warm-start-batch-size", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-ppo-warm-start-lr", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--sb3-offpolicy-prefill-enabled",
        dest="sb3_offpolicy_prefill_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="为 SAC/TD3/DDPG/DQN 启用规则 replay buffer 预填充。",
    )
    train_parser.add_argument(
        "--no-sb3-offpolicy-prefill",
        dest="sb3_offpolicy_prefill_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 off-policy replay buffer 预填充。",
    )
    train_parser.add_argument("--sb3-offpolicy-prefill-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--sb3-offpolicy-prefill-policy",
        type=str,
        default=argparse.SUPPRESS,
        choices=["easy_rule", "rule"],
        help="off-policy 预填充的专家策略来源：easy_rule 或 rule。",
    )
    train_parser.add_argument("--sb3-ppo-n-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-ppo-gae-lambda", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-ppo-ent-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-ppo-clip-range", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-dqn-action-mode", type=str, default=argparse.SUPPRESS, choices=["rb_v1"])
    train_parser.add_argument("--sb3-dqn-target-update-interval", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-dqn-exploration-fraction", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-dqn-exploration-initial-eps", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-dqn-exploration-final-eps", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-learning-starts", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-train-freq", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-gradient-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-tau", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-action-noise-std", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-buffer-size", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--sb3-optimize-memory-usage",
        action="store_true",
        default=argparse.SUPPRESS,
        help="对 off-policy replay buffer 启用内存优化（减少 next_obs 存储；推荐）。",
    )
    train_parser.add_argument(
        "--no-sb3-optimize-memory-usage",
        dest="sb3_optimize_memory_usage",
        action="store_false",
        default=argparse.SUPPRESS,
        help="禁用 replay buffer 的内存优化（不推荐，可能 OOM）。",
    )
    train_parser.add_argument(
        "--sb3-best-gate-enabled",
        dest="sb3_best_gate_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="best checkpoint 选择时启用可靠性门槛。",
    )
    train_parser.add_argument(
        "--no-sb3-best-gate",
        dest="sb3_best_gate_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭可靠性门槛，回退到纯 reward best。",
    )
    train_parser.add_argument("--sb3-best-gate-electric-min", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-best-gate-heat-min", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-best-gate-cool-min", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--sb3-plateau-control-enabled",
        dest="sb3_plateau_control_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 plateau 检测：先降 LR fine-tune，再在持续停滞时提前停止训练。",
    )
    train_parser.add_argument(
        "--no-sb3-plateau-control",
        dest="sb3_plateau_control_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 plateau 低 LR / early-stop 机制。",
    )
    train_parser.add_argument("--sb3-plateau-patience-evals", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-plateau-lr-decay-factor", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-plateau-min-lr", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-plateau-early-stop-patience-evals", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--pafc-projection-surrogate-checkpoint",
        type=str,
        default=argparse.SUPPRESS,
        help="PAFC-TD3 训练使用的 projection surrogate checkpoint（.pt）。",
    )
    train_parser.add_argument("--pafc-episode-days", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-total-env-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-warmup-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-replay-capacity", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-batch-size", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-updates-per-step", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-gamma", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-tau", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-actor-lr", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-critic-lr", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-dual-lr", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-dual-warmup-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-actor-delay", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-exploration-noise-std", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-target-policy-noise-std", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-target-noise-clip", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-gap-penalty-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-exec-action-anchor-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-exec-action-anchor-safe-floor", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-gt-off-deadband-ratio", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-abs-ready-focus-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-invalid-abs-penalty-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-boiler-proxy-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-abs-tradeoff-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-gt-grid-proxy-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-gt-distill-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-distill-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-safe-preserve-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--pafc-economic-teacher-safe-preserve-low-margin-boost",
        type=float,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-teacher-safe-preserve-high-cooling-boost",
        type=float,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-teacher-safe-preserve-joint-boost",
        type=float,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument("--pafc-economic-teacher-proxy-advantage-min", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-gt-proxy-advantage-min", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-bes-proxy-advantage-min", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-max-safe-abs-risk-gap", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-projection-gap-max", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-gt-projection-gap-max", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-bes-price-opportunity-min", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-bes-anchor-preserve-scale", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-warm-start-weight", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-prefill-replay-boost", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-gt-action-weight", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-bes-action-weight", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-teacher-tes-action-weight", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--pafc-economic-teacher-full-year-warm-start-samples",
        type=int,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-teacher-full-year-warm-start-epochs",
        type=int,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-gt-full-year-warm-start-samples",
        type=int,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-gt-full-year-warm-start-epochs",
        type=int,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-gt-full-year-warm-start-u-weight",
        type=float,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument("--pafc-economic-bes-distill-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-bes-prior-u", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-bes-charge-u-scale", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-bes-discharge-u-scale", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-bes-charge-weight", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-bes-discharge-weight", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-economic-bes-charge-pressure-bonus", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--pafc-economic-bes-charge-soc-ceiling", type=float, default=argparse.SUPPRESS
    )
    train_parser.add_argument(
        "--pafc-economic-bes-discharge-soc-floor", type=float, default=argparse.SUPPRESS
    )
    train_parser.add_argument(
        "--pafc-economic-bes-full-year-warm-start-samples",
        type=int,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-bes-full-year-warm-start-epochs",
        type=int,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-bes-full-year-warm-start-u-weight",
        type=float,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-bes-teacher-selection-priority-boost",
        type=float,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-bes-economic-source-priority-bonus",
        type=float,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-bes-economic-source-min-share",
        type=float,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-bes-idle-economic-source-min-share",
        type=float,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--pafc-economic-bes-teacher-target-min-share",
        type=float,
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument("--pafc-surrogate-actor-trust-coef", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-surrogate-actor-trust-min-scale", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--pafc-state-feasible-action-shaping-enabled",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    train_parser.add_argument("--pafc-abs-min-on-gate-th", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-abs-min-on-u-margin", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-expert-prefill-policy", type=str, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-expert-prefill-checkpoint-path", type=str, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-expert-prefill-economic-policy", type=str, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-expert-prefill-economic-checkpoint-path", type=str, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-expert-prefill-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-expert-prefill-cooling-bias", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-expert-prefill-abs-replay-boost", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-expert-prefill-abs-exec-threshold", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-expert-prefill-abs-window-mining-candidates", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-dual-abs-margin-k", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-dual-qc-ratio-th", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-dual-heat-backup-ratio-th", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-dual-safe-abs-u-th", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-actor-warm-start-epochs", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-actor-warm-start-batch-size", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-actor-warm-start-lr", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-checkpoint-interval-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-eval-window-pool-size", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-eval-window-count", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--pafc-best-gate-enabled",
        dest="pafc_best_gate_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="PAFC-TD3 best checkpoint 选择时启用可靠性门槛。",
    )
    train_parser.add_argument(
        "--no-pafc-best-gate",
        dest="pafc_best_gate_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 PAFC-TD3 best checkpoint 的可靠性门槛，回退到成本/奖励排序。",
    )
    train_parser.add_argument("--pafc-best-gate-electric-min", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-best-gate-heat-min", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-best-gate-cool-min", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--pafc-plateau-control-enabled",
        dest="pafc_plateau_control_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 PAFC-TD3 plateau 检测：先降学习率，再在持续停滞时提前停止。",
    )
    train_parser.add_argument(
        "--no-pafc-plateau-control",
        dest="pafc_plateau_control_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 PAFC-TD3 plateau 检测。",
    )
    train_parser.add_argument("--pafc-plateau-patience-evals", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-plateau-lr-decay-factor", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-plateau-min-actor-lr", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-plateau-min-critic-lr", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--pafc-plateau-early-stop-patience-evals", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--pafc-hidden-dims",
        type=str,
        default=argparse.SUPPRESS,
        help="PAFC actor/critic 隐层宽度，逗号分隔，例如 256,256。",
    )

    eval_parser = subparsers.add_parser("eval", help="运行通用评估（固定 2025，自动识别 baseline/SB3/PAFC）。")
    eval_parser.add_argument("--run-dir", type=Path, default=None)
    eval_parser.add_argument("--checkpoint", type=Path, default=None)
    eval_parser.add_argument(
        "--constraint-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(DEFAULT_CONSTRAINT_MODES),
        help="覆盖 env.constraint_mode（子命令级覆盖）。",
    )
    eval_parser.add_argument(
        "--policy",
        type=str,
        default=argparse.SUPPRESS,
        choices=["rule", "easy_rule", "random", "sequence_rule", "milp_mpc", "milp-mpc", "ga_mpc", "ga-mpc", "sb3", "pafc_td3", "pafc-td3"],
    )
    eval_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )
    eval_parser.add_argument("--history-steps", type=int, default=argparse.SUPPRESS)
    eval_parser.add_argument(
        "--sequence-adapter",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(SUPPORTED_SEQUENCE_ADAPTERS),
        help="当 policy=sequence_rule 时选择序列后端。",
    )
    eval_parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    eval_parser.add_argument("--seed", type=str, default=argparse.SUPPRESS)
    eval_parser.add_argument(
        "--model-source",
        type=str,
        default="best",
        choices=["best", "last"],
        help="SB3 checkpoint 评估时选择 best 或 last 模型（仅 SB3 生效）。",
    )

    sb3_train_parser = subparsers.add_parser("sb3-train", help="用 SB3 训练 PPO/SAC/TD3/DDPG/DQN（Task-011，可选依赖）。")
    sb3_train_parser.add_argument("--run-root", type=Path, default=Path("runs"))
    sb3_train_parser.add_argument(
        "--constraint-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(DEFAULT_CONSTRAINT_MODES),
        help="覆盖 env.constraint_mode（子命令级覆盖）。",
    )
    sb3_train_parser.add_argument("--algo", type=str, choices=["ppo", "sac", "td3", "ddpg", "dqn"], default=argparse.SUPPRESS)
    sb3_train_parser.add_argument(
        "--backbone",
        type=str,
        choices=["mlp", "transformer", "mamba"],
        default=argparse.SUPPRESS,
        help="SB3 特征提取骨干：mlp/transformer/mamba（用于 SAC+Transformer 等对比）。",
    )
    sb3_train_parser.add_argument(
        "--history-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="SB3 序列窗口长度（步）。",
    )
    sb3_train_parser.add_argument("--total-timesteps", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--episode-days", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--n-envs", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--gamma", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument(
        "--vec-norm-obs",
        dest="vec_norm_obs",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 VecNormalize 观测归一化（默认开启）。",
    )
    sb3_train_parser.add_argument(
        "--no-vec-norm-obs",
        dest="vec_norm_obs",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 VecNormalize 观测归一化。",
    )
    sb3_train_parser.add_argument(
        "--vec-norm-reward",
        dest="vec_norm_reward",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 VecNormalize 奖励归一化（默认开启）。",
    )
    sb3_train_parser.add_argument(
        "--no-vec-norm-reward",
        dest="vec_norm_reward",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 VecNormalize 奖励归一化。",
    )
    sb3_train_parser.add_argument("--eval-freq", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--eval-episode-days", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--eval-window-pool-size", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--eval-window-count", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument(
        "--ppo-warm-start-enabled",
        dest="ppo_warm_start_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 easy_rule 行为克隆预热（当前仅 PPO 支持）。",
    )
    sb3_train_parser.add_argument(
        "--no-ppo-warm-start",
        dest="ppo_warm_start_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 PPO easy_rule 预热。",
    )
    sb3_train_parser.add_argument(
        "--residual-enabled",
        dest="residual_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用连续动作算法的 rule residual 模式：动作解释为相对基线策略的残差。",
    )
    sb3_train_parser.add_argument(
        "--no-residual",
        dest="residual_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭连续动作算法的 residual 模式。",
    )
    sb3_train_parser.add_argument(
        "--residual-policy",
        type=str,
        default=argparse.SUPPRESS,
        choices=["easy_rule", "rule"],
        help="residual 基线策略来源：easy_rule 或 rule。",
    )
    sb3_train_parser.add_argument("--residual-scale", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--ppo-warm-start-samples", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--ppo-warm-start-epochs", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--ppo-warm-start-batch-size", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--ppo-warm-start-lr", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument(
        "--offpolicy-prefill-enabled",
        dest="offpolicy_prefill_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="为 SAC/TD3/DDPG/DQN 启用规则 replay buffer 预填充。",
    )
    sb3_train_parser.add_argument(
        "--no-offpolicy-prefill",
        dest="offpolicy_prefill_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 off-policy replay buffer 预填充。",
    )
    sb3_train_parser.add_argument(
        "--offpolicy-prefill-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="预填充步数；0 表示自动使用 learning_starts。",
    )
    sb3_train_parser.add_argument(
        "--offpolicy-prefill-policy",
        type=str,
        default=argparse.SUPPRESS,
        choices=["easy_rule", "rule"],
        help="off-policy 预填充的专家策略来源：easy_rule 或 rule。",
    )
    sb3_train_parser.add_argument("--ppo-n-steps", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--ppo-gae-lambda", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--ppo-ent-coef", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--ppo-clip-range", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--dqn-action-mode", type=str, choices=["rb_v1"], default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--dqn-target-update-interval", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--dqn-exploration-fraction", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--dqn-exploration-initial-eps", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--dqn-exploration-final-eps", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--learning-starts", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--train-freq", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--gradient-steps", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--tau", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--action-noise-std", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--buffer-size", type=int, default=argparse.SUPPRESS, help="off-policy replay buffer 大小。")
    sb3_train_parser.add_argument(
        "--optimize-memory-usage",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 replay buffer 内存优化（默认开启；可显著降低内存占用）。",
    )
    sb3_train_parser.add_argument(
        "--no-optimize-memory-usage",
        dest="optimize_memory_usage",
        action="store_false",
        default=argparse.SUPPRESS,
        help="禁用 replay buffer 内存优化（不推荐）。",
    )
    sb3_train_parser.add_argument(
        "--best-gate-enabled",
        dest="best_gate_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="best checkpoint 选择时启用可靠性门槛（默认开启）。",
    )
    sb3_train_parser.add_argument(
        "--no-best-gate",
        dest="best_gate_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭可靠性门槛，回退到纯 reward best。",
    )
    sb3_train_parser.add_argument("--best-gate-electric-min", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--best-gate-heat-min", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--best-gate-cool-min", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument(
        "--plateau-control-enabled",
        dest="plateau_control_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 plateau 检测：先降 LR fine-tune，再在持续停滞时提前停止训练。",
    )
    sb3_train_parser.add_argument(
        "--no-plateau-control",
        dest="plateau_control_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 plateau 低 LR / early-stop 机制。",
    )
    sb3_train_parser.add_argument("--plateau-patience-evals", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--plateau-lr-decay-factor", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--plateau-min-lr", type=float, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--plateau-early-stop-patience-evals", type=int, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument("--seed", type=str, default=argparse.SUPPRESS)
    sb3_train_parser.add_argument(
        "--eval-after-train",
        action="store_true",
        default=False,
        help="训练结束后立即跑一次 2025 评估，并将结果写入训练 run_dir/eval/（可选，较耗时）。",
    )
    sb3_train_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )

    sb3_eval_parser = subparsers.add_parser("sb3-eval", help="用 SB3 checkpoint 跑 2025 年评估（Task-011）。")
    sb3_eval_parser.add_argument("--run-dir", type=Path, required=True)
    sb3_eval_parser.add_argument("--checkpoint", type=Path, required=True, help="sb3-train 产出的 baseline_policy.json")
    sb3_eval_parser.add_argument(
        "--constraint-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(DEFAULT_CONSTRAINT_MODES),
        help="覆盖 env.constraint_mode（子命令级覆盖）。",
    )
    sb3_eval_parser.add_argument("--device", type=str, default="auto")
    sb3_eval_parser.add_argument("--seed", type=str, default="42")
    sb3_eval_parser.add_argument("--stochastic", action="store_true", help="使用随机动作采样（默认 deterministic）。")
    sb3_eval_parser.add_argument(
        "--model-source",
        type=str,
        default="best",
        choices=["best", "last"],
        help="选择评估 best 或 last checkpoint（默认 best）。",
    )
    sb3_eval_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )

    pafc_train_parser = subparsers.add_parser("pafc-train", help="训练 Task-012 的 PAFC-TD3（2024）。")
    pafc_train_parser.add_argument("--run-root", type=Path, default=Path("runs"))
    pafc_train_parser.add_argument(
        "--projection-surrogate-checkpoint",
        type=Path,
        required=True,
        help="projection surrogate 的 .pt checkpoint。",
    )
    pafc_train_parser.add_argument(
        "--constraint-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(DEFAULT_CONSTRAINT_MODES),
        help="覆盖 env.constraint_mode（子命令级覆盖）。",
    )
    pafc_train_parser.add_argument("--episode-days", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--total-env-steps", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--warmup-steps", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--replay-capacity", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--updates-per-step", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--gamma", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--tau", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--actor-lr", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--critic-lr", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--dual-lr", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--dual-warmup-steps", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--actor-delay", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--exploration-noise-std", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--target-policy-noise-std", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--target-noise-clip", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--gap-penalty-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--exec-action-anchor-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--exec-action-anchor-safe-floor", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--gt-off-deadband-ratio", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--abs-ready-focus-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--invalid-abs-penalty-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-boiler-proxy-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-abs-tradeoff-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-gt-grid-proxy-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-gt-distill-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-distill-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-safe-preserve-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument(
        "--economic-teacher-safe-preserve-low-margin-boost",
        type=float,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument(
        "--economic-teacher-safe-preserve-high-cooling-boost",
        type=float,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument(
        "--economic-teacher-safe-preserve-joint-boost",
        type=float,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument("--economic-teacher-mismatch-focus-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument(
        "--economic-teacher-mismatch-focus-min-scale",
        type=float,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument(
        "--economic-teacher-mismatch-focus-max-scale",
        type=float,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument("--economic-teacher-proxy-advantage-min", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-gt-proxy-advantage-min", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-bes-proxy-advantage-min", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-max-safe-abs-risk-gap", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-projection-gap-max", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-gt-projection-gap-max", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-bes-price-opportunity-min", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-bes-anchor-preserve-scale", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-warm-start-weight", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-prefill-replay-boost", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-gt-action-weight", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-bes-action-weight", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-tes-action-weight", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-full-year-warm-start-samples", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-teacher-full-year-warm-start-epochs", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-gt-full-year-warm-start-samples", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-gt-full-year-warm-start-epochs", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-gt-full-year-warm-start-u-weight", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-bes-distill-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument(
        "--state-feasible-action-shaping-enabled",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument("--abs-min-on-gate-th", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--abs-min-on-u-margin", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--expert-prefill-policy", type=str, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--expert-prefill-checkpoint-path", type=str, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--expert-prefill-economic-policy", type=str, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--expert-prefill-economic-checkpoint-path", type=str, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--expert-prefill-steps", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--expert-prefill-cooling-bias", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--expert-prefill-abs-replay-boost", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--expert-prefill-abs-exec-threshold", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--expert-prefill-abs-window-mining-candidates", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--dual-abs-margin-k", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--dual-qc-ratio-th", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--dual-heat-backup-ratio-th", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--dual-safe-abs-u-th", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-bes-prior-u", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-bes-charge-u-scale", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-bes-discharge-u-scale", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-bes-charge-weight", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-bes-discharge-weight", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-bes-charge-pressure-bonus", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-bes-charge-soc-ceiling", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--economic-bes-discharge-soc-floor", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument(
        "--economic-bes-full-year-warm-start-samples",
        type=int,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument(
        "--economic-bes-full-year-warm-start-epochs",
        type=int,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument(
        "--economic-bes-full-year-warm-start-u-weight",
        type=float,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument(
        "--economic-bes-teacher-selection-priority-boost",
        type=float,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument(
        "--economic-bes-economic-source-priority-bonus",
        type=float,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument(
        "--economic-bes-economic-source-min-share",
        type=float,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument(
        "--economic-bes-idle-economic-source-min-share",
        type=float,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument(
        "--economic-bes-teacher-target-min-share",
        type=float,
        default=argparse.SUPPRESS,
    )
    pafc_train_parser.add_argument("--surrogate-actor-trust-coef", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--surrogate-actor-trust-min-scale", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--actor-warm-start-epochs", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--actor-warm-start-batch-size", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--actor-warm-start-lr", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--checkpoint-interval-steps", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--eval-window-pool-size", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--eval-window-count", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument(
        "--best-gate-enabled",
        dest="best_gate_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="PAFC-TD3 best checkpoint 选择时启用可靠性门槛。",
    )
    pafc_train_parser.add_argument(
        "--no-best-gate",
        dest="best_gate_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 PAFC-TD3 best checkpoint 的可靠性门槛。",
    )
    pafc_train_parser.add_argument("--best-gate-electric-min", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--best-gate-heat-min", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--best-gate-cool-min", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument(
        "--plateau-control-enabled",
        dest="plateau_control_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 PAFC-TD3 plateau 检测。",
    )
    pafc_train_parser.add_argument(
        "--no-plateau-control",
        dest="plateau_control_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="关闭 PAFC-TD3 plateau 检测。",
    )
    pafc_train_parser.add_argument("--plateau-patience-evals", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--plateau-lr-decay-factor", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--plateau-min-actor-lr", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--plateau-min-critic-lr", type=float, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--plateau-early-stop-patience-evals", type=int, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument(
        "--hidden-dims",
        type=str,
        default=argparse.SUPPRESS,
        help="逗号分隔，例如 256,256。",
    )
    pafc_train_parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument("--seed", type=str, default=argparse.SUPPRESS)
    pafc_train_parser.add_argument(
        "--eval-after-train",
        action="store_true",
        default=False,
        help="训练结束后立即在 2025 上评估，并写入同一 run_dir/eval/。",
    )
    pafc_train_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )

    pafc_eval_parser = subparsers.add_parser("pafc-eval", help="评估 Task-012 的 PAFC-TD3（固定 2025）。")
    pafc_eval_parser.add_argument("--checkpoint", type=Path, required=True, help="PAFC-TD3 actor checkpoint（.pt 或 pafc_td3_actor.json）")
    pafc_eval_parser.add_argument("--run-dir", type=Path, default=None)
    pafc_eval_parser.add_argument(
        "--constraint-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(DEFAULT_CONSTRAINT_MODES),
        help="覆盖 env.constraint_mode（子命令级覆盖）。",
    )
    pafc_eval_parser.add_argument("--device", type=str, default="auto")
    pafc_eval_parser.add_argument("--seed", type=str, default="42")
    pafc_eval_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )

    calibrate_parser = subparsers.add_parser("calibrate", help="运行物理参数标定搜索（Task-002）。")
    calibrate_parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/spec/calibration_config.json"),
        help="标定配置 JSON",
    )
    calibrate_parser.add_argument(
        "--constraint-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(DEFAULT_CONSTRAINT_MODES),
        help="覆盖 env.constraint_mode（子命令级覆盖）。",
    )
    calibrate_parser.add_argument("--n-samples", type=int, default=6)
    calibrate_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )
    calibrate_parser.add_argument("--history-steps", type=int, default=argparse.SUPPRESS)
    calibrate_parser.add_argument(
        "--sequence-adapter",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(SUPPORTED_SEQUENCE_ADAPTERS),
        help="当 search.policy=sequence_rule 时选择序列后端。",
    )
    calibrate_parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    calibrate_parser.add_argument("--run-root", type=Path, default=Path("runs"))

    ablation_parser = subparsers.add_parser("ablation", help="运行约束方式消融（Task-003）。")
    ablation_parser.add_argument(
        "--modes",
        type=str,
        default="physics_in_loop,reward_only",
        help="逗号分隔，例如 physics_in_loop,reward_only",
    )
    ablation_parser.add_argument(
        "--constraint-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(DEFAULT_CONSTRAINT_MODES),
        help="覆盖 env.constraint_mode（子命令级覆盖；通常不需要，ablation 会按 --modes 生成）。",
    )
    ablation_parser.add_argument(
        "--policy", type=str, default=argparse.SUPPRESS, choices=["rule", "easy_rule", "random", "sequence_rule", "milp_mpc", "milp-mpc", "ga_mpc", "ga-mpc"]
    )
    ablation_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )
    ablation_parser.add_argument("--history-steps", type=int, default=argparse.SUPPRESS)
    ablation_parser.add_argument(
        "--sequence-adapter",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(SUPPORTED_SEQUENCE_ADAPTERS),
        help="当 policy=sequence_rule 时选择序列后端。",
    )
    ablation_parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    ablation_parser.add_argument("--run-root", type=Path, default=Path("runs"))
    ablation_parser.add_argument("--params", type=Path, default=None, help="可选参数覆盖 JSON")

    collect_parser = subparsers.add_parser("collect", help="汇总 runs 下的 eval 结果到论文表格 CSV（Task-011）。")
    collect_parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    collect_parser.add_argument(
        "--constraint-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(DEFAULT_CONSTRAINT_MODES),
        help="保留字段：collect 不使用 constraint_mode，仅用于兼容统一脚本参数。",
    )
    collect_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/paper/benchmark_summary.csv"),
        help="精选列输出 CSV（论文表格友好）。",
    )
    collect_parser.add_argument(
        "--full-output",
        type=Path,
        default=Path("runs/paper/benchmark_summary_full.csv"),
        help="全量列输出 CSV（便于二次筛选）。",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    command = args.command or "summary"
    if command == "summary":
        _command_summary(args)
        return
    if command == "train":
        _command_train(args)
        return
    if command == "eval":
        _command_eval(args)
        return
    if command == "sb3-train":
        _command_sb3_train(args)
        return
    if command == "sb3-eval":
        _command_sb3_eval(args)
        return
    if command == "pafc-train":
        _command_pafc_train(args)
        return
    if command == "pafc-eval":
        _command_pafc_eval(args)
        return
    if command == "calibrate":
        _command_calibrate(args)
        return
    if command == "ablation":
        _command_ablation(args)
        return
    if command == "collect":
        _command_collect(args)
        return
    raise ValueError(f"未知命令: {command}")


if __name__ == "__main__":
    main()
