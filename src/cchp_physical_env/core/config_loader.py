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
    # Task-011：可选 SB3 多算法（SAC/TD3/DDPG/PPO）
    "sb3_enabled": False,
    "sb3_algo": "sac",
    "sb3_backbone": "transformer",
    "sb3_history_steps": 32,
    "sb3_total_timesteps": 2_000_000,
    "sb3_n_envs": 4,
    "sb3_learning_rate": 3e-4,
    "sb3_batch_size": 512,
    "sb3_gamma": 0.99,
    "sb3_vec_norm_obs": True,
    "sb3_vec_norm_reward": True,
    "sb3_eval_freq": 50_000,
    "sb3_eval_episode_days": 14,
    "sb3_eval_window_pool_size": 12,
    "sb3_eval_window_count": 4,
    "sb3_eval_window_seed": 42,
    "sb3_ppo_n_steps": 2048,
    "sb3_ppo_gae_lambda": 0.95,
    "sb3_ppo_ent_coef": 0.0,
    "sb3_ppo_clip_range": 0.2,
    "sb3_learning_starts": 5_000,
    "sb3_train_freq": 1,
    "sb3_gradient_steps": 1,
    "sb3_tau": 0.005,
    "sb3_action_noise_std": 0.1,
    # Off-policy algorithms (SAC/TD3/DDPG) use a replay buffer.
    # With window observations (K,D) and n_envs>1, the default SB3 buffer_size=1e6 can easily OOM.
    "sb3_buffer_size": 50_000,
    "sb3_optimize_memory_usage": True,
}

# env 参数校验规则表：
# - Option-C：yaml 改什么就是什么，因此 env 块要求“全量字段显式给出”，避免代码默认值污染实验口径。
# - 表驱动的目的是把“字段类型/枚举范围/数值范围”集中管理，减少散落的 if/elif 分支。
ENV_ENUM_OPTIONS: dict[str, set[str]] = {
    "constraint_mode": {"physics_in_loop", "reward_only"},
    "physics_backend": {"tespy"},
    "bes_init_strategy": {"fixed", "min", "max", "half", "random"},
}
ENV_ENUM_ERROR_MESSAGES: dict[str, str] = {
    "constraint_mode": "constraint_mode 仅支持 physics_in_loop/reward_only。",
    "physics_backend": "physics_backend 仅支持 tespy。",
    "bes_init_strategy": "bes_init_strategy 仅支持 fixed/min/max/half/random。",
}
ENV_STRIP_STRING_KEYS = {"pyomo_solver"}
ENV_BOOL_KEYS = {
    "bes_dod_add_calendar_age",
    "abs_gate_enabled",
    "gt_action_smoothing_enabled",
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
    "penalty_invalid_abs_request": (
        lambda value: value >= 0.0,
        "penalty_invalid_abs_request 必须 >= 0。",
    ),
    "gt_min_on_steps": (
        lambda value: value >= 0.0,
        "gt_min_on_steps 必须 >= 0。",
    ),
    "gt_min_off_steps": (
        lambda value: value >= 0.0,
        "gt_min_off_steps 必须 >= 0。",
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

    return EnvConfig(**values)


def validate_training_overrides(overrides: dict[str, Any]) -> None:
    allowed = set(TRAINING_DEFAULTS.keys())
    unknown = sorted(set(overrides.keys()) - allowed)
    if unknown:
        raise ValueError(f"`training` 包含未知参数: {unknown}")

    bool_keys = {
        "sb3_enabled",
        "sb3_optimize_memory_usage",
        "sb3_vec_norm_obs",
        "sb3_vec_norm_reward",
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
        "sb3_ppo_n_steps",
        "sb3_train_freq",
        "sb3_gradient_steps",
    }
    for key, value in overrides.items():
        if key in {"policy", "sequence_adapter", "device"}:
            if len(str(value).strip()) == 0:
                raise ValueError(f"{key} 不能为空。")
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
                    if int(raw) <= 0:
                        raise ValueError(f"{key} 必须 > 0。")
            else:
                if int(value) <= 0 and key != "seed":
                    raise ValueError(f"{key} 必须 > 0。")
            continue
        if key in {"lr", "sb3_learning_rate"}:
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
        if key in {"sb3_eval_window_seed"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是整数类型。")
            continue
        if key in {"sb3_eval_window_pool_size", "sb3_eval_window_count"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是整数类型。")
            if int(value) < 0:
                raise ValueError(f"{key} 必须 >= 0。")
            continue
        if key == "sb3_gamma":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("sb3_gamma 必须是数值类型。")
            numeric = float(value)
            if not (0.0 < numeric <= 1.0):
                raise ValueError("sb3_gamma 必须在 (0,1]。")
            continue
        if key == "sb3_ppo_gae_lambda":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("sb3_ppo_gae_lambda 必须是数值类型。")
            numeric = float(value)
            if not (0.0 < numeric <= 1.0):
                raise ValueError("sb3_ppo_gae_lambda 必须在 (0,1]。")
            continue
        if key in {"sb3_ppo_ent_coef", "sb3_action_noise_std"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是数值类型。")
            if float(value) < 0.0:
                raise ValueError(f"{key} 必须 >= 0。")
            continue
        if key in {"sb3_ppo_clip_range", "sb3_tau"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是数值类型。")
            if float(value) <= 0.0:
                raise ValueError(f"{key} 必须 > 0。")
            continue

    policy = str(overrides.get("policy", TRAINING_DEFAULTS["policy"])).strip().lower()
    sb3_enabled_flag = bool(overrides.get("sb3_enabled", TRAINING_DEFAULTS.get("sb3_enabled", False)))
    if sb3_enabled_flag:
        # sb3_enabled=true 时，policy 仅作记录，但仍需要校验以避免拼写错误污染实验口径。
        if policy not in {"rule", "easy_rule", "random", "sequence_rule", "sb3"}:
            raise ValueError("training.policy 仅支持 rule/easy_rule/random/sequence_rule/sb3（sb3_enabled=true 时该字段仅作备注，不参与路由）。")
    else:
        if policy not in {"rule", "easy_rule", "random", "sequence_rule"}:
            raise ValueError("training.policy 仅支持 rule/easy_rule/random/sequence_rule（sb3_enabled=false）。")
    sequence_adapter = str(
        overrides.get("sequence_adapter", TRAINING_DEFAULTS["sequence_adapter"])
    ).strip().lower()
    if sequence_adapter not in {"rule", "mlp", "transformer", "mamba"}:
        raise ValueError("training.sequence_adapter 仅支持 rule/mlp/transformer/mamba。")
    device = str(overrides.get("device", TRAINING_DEFAULTS["device"])).strip().lower()
    if device not in {"auto", "cpu", "cuda"} and not device.startswith("cuda:"):
        raise ValueError("training.device 仅支持 auto/cpu/cuda/cuda:<index>。")

    sb3_algo = str(overrides.get("sb3_algo", TRAINING_DEFAULTS["sb3_algo"])).strip().lower()
    if sb3_algo not in {"ppo", "sac", "td3", "ddpg"}:
        raise ValueError("training.sb3_algo 仅支持 ppo/sac/td3/ddpg。")
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


def build_training_options(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(TRAINING_DEFAULTS)
    if overrides is not None:
        merged.update(dict(overrides))
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
    normalized["policy"] = str(normalized["policy"]).strip().lower()
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
    normalized["sb3_eval_window_seed"] = int(normalized["sb3_eval_window_seed"])
    normalized["sb3_ppo_n_steps"] = int(normalized["sb3_ppo_n_steps"])
    normalized["sb3_ppo_gae_lambda"] = float(normalized["sb3_ppo_gae_lambda"])
    normalized["sb3_ppo_ent_coef"] = float(normalized["sb3_ppo_ent_coef"])
    normalized["sb3_ppo_clip_range"] = float(normalized["sb3_ppo_clip_range"])
    normalized["sb3_learning_starts"] = int(normalized["sb3_learning_starts"])
    normalized["sb3_train_freq"] = int(normalized["sb3_train_freq"])
    normalized["sb3_gradient_steps"] = int(normalized["sb3_gradient_steps"])
    normalized["sb3_tau"] = float(normalized["sb3_tau"])
    normalized["sb3_action_noise_std"] = float(normalized["sb3_action_noise_std"])
    normalized["sb3_buffer_size"] = int(normalized["sb3_buffer_size"])
    normalized["sb3_optimize_memory_usage"] = bool(normalized["sb3_optimize_memory_usage"])
    return normalized
