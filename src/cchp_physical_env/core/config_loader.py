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
    "policy": "rule",
    "sequence_adapter": "rule",
    "history_steps": 16,
    "episode_days": 14,
    "episodes": 8,
    "train_steps": 4096,
    "batch_size": 128,
    "update_epochs": 4,
    "lr": 3e-4,
    "device": "auto",
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
ENV_BOOL_KEYS = {"bes_dod_add_calendar_age"}
# 数值范围规则：仅对需要额外范围约束的字段登记，其余数值字段只做“类型 + finite”校验。
ENV_NUMERIC_RULES: dict[str, tuple[Callable[[float], bool], str]] = {
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
                raise ValueError("bes_dod_add_calendar_age 必须是布尔值（true/false）。")
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

    int_keys = {
        "seed",
        "history_steps",
        "episode_days",
        "episodes",
        "train_steps",
        "batch_size",
        "update_epochs",
    }
    for key, value in overrides.items():
        if key in {"policy", "sequence_adapter", "device"}:
            if len(str(value).strip()) == 0:
                raise ValueError(f"{key} 不能为空。")
            continue
        if key in int_keys:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是整数类型。")
            if int(value) <= 0 and key != "seed":
                raise ValueError(f"{key} 必须 > 0。")
            continue
        if key == "lr":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("lr 必须是数值类型。")
            if float(value) <= 0.0:
                raise ValueError("lr 必须 > 0。")
            continue

    policy = str(overrides.get("policy", TRAINING_DEFAULTS["policy"])).strip().lower()
    if policy not in {"rule", "random", "sequence_rule"}:
        raise ValueError("training.policy 仅支持 rule/random/sequence_rule。")
    sequence_adapter = str(
        overrides.get("sequence_adapter", TRAINING_DEFAULTS["sequence_adapter"])
    ).strip().lower()
    if sequence_adapter not in {"rule", "transformer", "mamba"}:
        raise ValueError("training.sequence_adapter 仅支持 rule/transformer/mamba。")
    device = str(overrides.get("device", TRAINING_DEFAULTS["device"])).strip().lower()
    if device not in {"auto", "cpu", "cuda"} and not device.startswith("cuda:"):
        raise ValueError("training.device 仅支持 auto/cpu/cuda/cuda:<index>。")


def build_training_options(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(TRAINING_DEFAULTS)
    if overrides is not None:
        merged.update(dict(overrides))
    validate_training_overrides(merged)

    normalized = dict(merged)
    normalized["seed"] = int(normalized["seed"])
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
    return normalized
