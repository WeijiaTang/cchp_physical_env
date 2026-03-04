# Ref: docs/spec/task.md
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config_loader import build_env_config_from_overrides
from ..core.data import EVAL_YEAR, TRAIN_YEAR, compute_training_statistics, make_episode_sampler
from ..env.cchp_env import CCHPPhysicalEnv, EnvConfig
from .runner import RandomPolicy, RulePolicy
from .sequence import SUPPORTED_SEQUENCE_ADAPTERS, SequenceRulePolicy

REQUIRED_MANDATORY_PARAMS = {
    "ua_mw_per_k",
    "sigma_per_hour",
    "cop_nominal",
    "m_exh_per_fuel_ratio",
    "t_exh_offset_k",
    "t_exh_slope_k_per_mw",
}
REQUIRED_SECONDARY_PARAMS = {
    "p_gt_cap_mw",
    "q_boiler_cap_mw",
    "q_ech_cap_mw",
    "p_bes_cap_mw",
    "e_bes_cap_mwh",
    "e_tes_cap_mwh",
}
REQUIRED_FROZEN_PARAMS = {
    "penalty_unmet_e_per_mwh",
    "penalty_unmet_h_per_mwh",
    "penalty_unmet_c_per_mwh",
    "penalty_violation_per_flag",
    "sell_price_ratio",
    "penalty_curtail_per_mwh",
}
ALIASES = {"a_m": "m_exh_per_fuel_ratio", "a0": "t_exh_offset_k", "a1": "t_exh_slope_k_per_mw"}


def _extract_year(df: pd.DataFrame) -> int:
    years = sorted({int(value.year) for value in pd.to_datetime(df["timestamp"])})
    if len(years) != 1:
        raise ValueError(f"仅支持单年数据，当前年份集合: {years}")
    return years[0]


def _normalize_param_name(name: str) -> str:
    return ALIASES.get(name, name)


def _normalize_parameter_block(parameters: dict) -> dict:
    normalized: dict[str, dict] = {}
    for key, value in parameters.items():
        canonical = _normalize_param_name(key)
        if canonical in normalized:
            raise ValueError(f"参数重复定义（含别名冲突）: {key} -> {canonical}")
        normalized[canonical] = value
    return normalized


def load_calibration_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"标定配置不存在: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config


def _validate_range_spec(name: str, spec: dict) -> None:
    if "min" in spec and "max" in spec:
        if float(spec["min"]) >= float(spec["max"]):
            raise ValueError(f"{name} 的 min 必须小于 max。")
        return
    if "base" in spec and "delta_pct" in spec:
        if float(spec["delta_pct"]) < 0.0:
            raise ValueError(f"{name} 的 delta_pct 必须非负。")
        return
    raise ValueError(f"{name} 的范围定义无效，需使用 (min,max) 或 (base,delta_pct)。")


def validate_calibration_config(config: dict) -> None:
    for required_key in ("mandatory_parameters", "secondary_parameters", "frozen_parameters", "search"):
        if required_key not in config:
            raise ValueError(f"标定配置缺少字段: {required_key}")

    mandatory = _normalize_parameter_block(dict(config["mandatory_parameters"]))
    secondary = _normalize_parameter_block(dict(config["secondary_parameters"]))
    frozen = {_normalize_param_name(item) for item in config["frozen_parameters"]}
    search = dict(config["search"])

    missing_mandatory = sorted(REQUIRED_MANDATORY_PARAMS - set(mandatory.keys()))
    if missing_mandatory:
        raise ValueError(f"必标定参数缺失: {missing_mandatory}")

    missing_secondary = sorted(REQUIRED_SECONDARY_PARAMS - set(secondary.keys()))
    if missing_secondary:
        raise ValueError(f"次标定参数缺失: {missing_secondary}")

    missing_frozen = sorted(REQUIRED_FROZEN_PARAMS - frozen)
    if missing_frozen:
        raise ValueError(f"冻结参数清单缺失: {missing_frozen}")

    overlapped = (set(mandatory.keys()) | set(secondary.keys())) & frozen
    if overlapped:
        raise ValueError(f"冻结参数不得进入优化变量列表: {sorted(overlapped)}")

    for name, spec in mandatory.items():
        _validate_range_spec(name, spec)
    for name, spec in secondary.items():
        if "base" not in spec:
            raise ValueError(f"次标定参数 {name} 必须包含 base。")
        if float(spec.get("delta_pct", 0.0)) != 0.20:
            raise ValueError(f"次标定参数 {name} 必须使用 ±20%（delta_pct=0.20）。")
        _validate_range_spec(name, spec)

    policy = str(search.get("policy", "rule")).lower()
    if policy not in {"rule", "random", "sequence_rule"}:
        raise ValueError("search.policy 仅支持 rule/random/sequence_rule。")
    episode_days = int(search.get("train_episode_days", 14))
    if episode_days < 7 or episode_days > 30:
        raise ValueError("search.train_episode_days 必须在 [7,30]。")
    episodes = int(search.get("train_episodes", 2))
    if episodes <= 0:
        raise ValueError("search.train_episodes 必须 > 0。")
    history_steps = int(search.get("history_steps", 16))
    if history_steps <= 0:
        raise ValueError("search.history_steps 必须 > 0。")
    sequence_adapter = str(search.get("sequence_adapter", "rule")).strip().lower()
    if sequence_adapter not in SUPPORTED_SEQUENCE_ADAPTERS:
        raise ValueError(
            f"search.sequence_adapter 仅支持 {SUPPORTED_SEQUENCE_ADAPTERS}。"
        )
    search["sequence_adapter"] = sequence_adapter

    # 用标准化结果回写，确保后续流程统一使用 canonical 名称。
    config["mandatory_parameters"] = mandatory
    config["secondary_parameters"] = secondary
    config["frozen_parameters"] = sorted(frozen)
    config["search"] = search


def _sample_single_param(rng: np.random.Generator, spec: dict) -> float:
    if "min" in spec and "max" in spec:
        return float(rng.uniform(float(spec["min"]), float(spec["max"])))
    base = float(spec["base"])
    delta_pct = float(spec["delta_pct"])
    low = base * (1.0 - delta_pct)
    high = base * (1.0 + delta_pct)
    return float(rng.uniform(low, high))


def sample_physical_params(config: dict, seed: int, n_samples: int) -> list[dict]:
    validate_calibration_config(config)
    if n_samples <= 0:
        raise ValueError("n_samples 必须 > 0。")

    rng = np.random.default_rng(seed)
    mandatory = config["mandatory_parameters"]
    secondary = config["secondary_parameters"]

    candidates: list[dict] = []
    for trial_id in range(n_samples):
        item: dict[str, float | int] = {"trial_id": trial_id}
        for name, spec in mandatory.items():
            item[name] = _sample_single_param(rng, spec)
        for name, spec in secondary.items():
            item[name] = _sample_single_param(rng, spec)
        candidates.append(item)
    return candidates


def _build_policy(
    policy_name: str,
    seed: int,
    train_statistics: dict,
    history_steps: int,
    sequence_adapter: str = "rule",
):
    if policy_name == "rule":
        return RulePolicy(train_statistics=train_statistics)
    if policy_name == "random":
        return RandomPolicy(seed=seed)
    if policy_name == "sequence_rule":
        if sequence_adapter != "rule":
            raise ValueError(
                "calibration 当前仅支持 sequence_adapter=rule；"
                "Transformer/Mamba 请先独立训练并在 eval 阶段加载 checkpoint。"
            )
        return SequenceRulePolicy(
            train_statistics=train_statistics,
            history_steps=history_steps,
            sequence_adapter=sequence_adapter,
        )
    raise ValueError(f"不支持的策略: {policy_name}")


def _build_env_config_from_params(
    params: dict, base_env_overrides: dict | None = None
) -> EnvConfig:
    merged_overrides: dict[str, float | str] = {}
    for key, value in (base_env_overrides or {}).items():
        merged_overrides[key] = value
    for key, value in params.items():
        if key == "trial_id":
            continue
        merged_overrides[key] = float(value)

    config = build_env_config_from_overrides(merged_overrides)
    if config.e_tes_init_mwh > config.e_tes_cap_mwh:
        config.e_tes_init_mwh = config.e_tes_cap_mwh
    if config.e_tes_init_mwh <= 0.0:
        config.e_tes_init_mwh = 0.5 * config.e_tes_cap_mwh
    return config


def _simulate_episode(
    env: CCHPPhysicalEnv,
    policy,
    *,
    seed: int,
    episode_df: pd.DataFrame,
) -> tuple[float, dict]:
    observation, _ = env.reset(seed=seed, episode_df=episode_df)
    reset_episode_fn = getattr(policy, "reset_episode", None)
    if callable(reset_episode_fn):
        reset_episode_fn(observation)
    total_reward = 0.0
    terminated = False
    final_info = {}

    while not terminated:
        action = policy.act(observation)
        observation, reward, terminated, _, info = env.step(action)
        total_reward += reward
        final_info = info

    summary = final_info.get("episode_summary", env.kpi.summary())
    summary["total_reward_from_loop"] = float(total_reward)
    return float(total_reward), summary


def run_calibration_trial(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    params: dict,
    *,
    search_options: dict | None = None,
    base_env_overrides: dict | None = None,
) -> dict:
    train_year = _extract_year(train_df)
    eval_year = _extract_year(eval_df)
    if train_year != TRAIN_YEAR:
        raise ValueError(f"trial 训练数据必须为 {TRAIN_YEAR}，当前 {train_year}")
    if eval_year != EVAL_YEAR:
        raise ValueError(f"trial 评估数据必须为 {EVAL_YEAR}，当前 {eval_year}")

    options = search_options or {}
    seed = int(options.get("seed", 42))
    policy_name = str(options.get("policy", "rule")).lower()
    history_steps = int(options.get("history_steps", 16))
    sequence_adapter = str(options.get("sequence_adapter", "rule")).strip().lower()
    episode_days = int(options.get("train_episode_days", 14))
    train_episodes = int(options.get("train_episodes", 2))

    trial_params = {key: float(value) for key, value in params.items() if key != "trial_id"}
    env_config = _build_env_config_from_params(
        trial_params, base_env_overrides=base_env_overrides
    )
    train_statistics = compute_training_statistics(train_df)
    policy = _build_policy(
        policy_name=policy_name,
        seed=seed,
        train_statistics=train_statistics,
        history_steps=history_steps,
        sequence_adapter=sequence_adapter,
    )
    sampler = make_episode_sampler(train_df, episode_days=episode_days, seed=seed)

    train_episode_costs: list[float] = []
    train_episode_violations: list[float] = []
    train_episode_unmet_h: list[float] = []
    train_env = CCHPPhysicalEnv(exogenous_df=train_df, config=env_config, seed=seed)
    for episode_id in range(train_episodes):
        _, episode_df = next(sampler)
        _, summary = _simulate_episode(train_env, policy, seed=seed + episode_id, episode_df=episode_df)
        train_episode_costs.append(float(summary["total_cost"]))
        train_episode_violations.append(float(summary["violation_rate"]))
        train_episode_unmet_h.append(float(summary["unmet_energy_mwh"]["heat"]))

    eval_policy = _build_policy(
        policy_name=policy_name,
        seed=seed + 10_000,
        train_statistics=train_statistics,
        history_steps=history_steps,
        sequence_adapter=sequence_adapter,
    )
    eval_env = CCHPPhysicalEnv(exogenous_df=eval_df, config=env_config, seed=seed + 10_000)
    _, eval_summary = _simulate_episode(eval_env, eval_policy, seed=seed + 10_001, episode_df=eval_df)

    return {
        "trial_id": int(params.get("trial_id", -1)),
        "params": trial_params,
        "train": {
            "mean_total_cost": float(np.mean(train_episode_costs)),
            "mean_violation_rate": float(np.mean(train_episode_violations)),
            "mean_unmet_h_mwh": float(np.mean(train_episode_unmet_h)),
            "episodes": int(train_episodes),
            "episode_days": int(episode_days),
            "policy": policy_name,
            "sequence_adapter": sequence_adapter,
        },
        "eval": eval_summary,
    }


def _trial_to_row(trial_result: dict) -> dict:
    eval_summary = trial_result["eval"]
    unmet = eval_summary["unmet_energy_mwh"]
    row = {
        "trial_id": trial_result["trial_id"],
        "policy": str(trial_result["train"]["policy"]),
        "sequence_adapter": str(trial_result["train"]["sequence_adapter"]),
        "eval_total_cost": float(eval_summary["total_cost"]),
        "eval_violation_rate": float(eval_summary["violation_rate"]),
        "eval_unmet_e_mwh": float(unmet["electric"]),
        "eval_unmet_h_mwh": float(unmet["heat"]),
        "eval_unmet_c_mwh": float(unmet["cooling"]),
        "train_mean_total_cost": float(trial_result["train"]["mean_total_cost"]),
        "train_mean_violation_rate": float(trial_result["train"]["mean_violation_rate"]),
    }
    row.update({f"param_{k}": v for k, v in trial_result["params"].items()})
    return row


def run_calibration_search(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    config: dict,
    n_samples: int,
    seed: int,
    run_root: str | Path,
    base_env_overrides: dict | None = None,
) -> dict:
    validate_calibration_config(config)
    run_dir = Path(run_root) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_calibration"
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates = sample_physical_params(config=config, seed=seed, n_samples=n_samples)
    search_options = dict(config.get("search", {}))
    search_options["seed"] = seed

    trial_results = []
    for params in candidates:
        result = run_calibration_trial(
            train_df=train_df,
            eval_df=eval_df,
            params=params,
            search_options=search_options,
            base_env_overrides=base_env_overrides,
        )
        trial_results.append(result)

    rows = [_trial_to_row(item) for item in trial_results]
    trials_df = pd.DataFrame(rows).sort_values(
        by=["eval_total_cost", "eval_violation_rate", "eval_unmet_h_mwh"], ascending=[True, True, True]
    )
    trials_df.to_csv(run_dir / "trials.csv", index=False)

    best_trial_id = int(trials_df.iloc[0]["trial_id"])
    best_trial = next(item for item in trial_results if item["trial_id"] == best_trial_id)
    (run_dir / "best_params.json").write_text(
        json.dumps(best_trial["params"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "best_eval_summary.json").write_text(
        json.dumps(best_trial["eval"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "run_dir": str(run_dir),
        "n_trials": int(len(trial_results)),
        "policy": str(search_options.get("policy", "rule")),
        "sequence_adapter": str(search_options.get("sequence_adapter", "rule")),
        "best_trial_id": best_trial_id,
        "best_eval_total_cost": float(best_trial["eval"]["total_cost"]),
        "best_eval_violation_rate": float(best_trial["eval"]["violation_rate"]),
    }
