# Ref: docs/spec/task.md
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..core.config_loader import build_env_config_from_overrides
from ..core.data import EVAL_YEAR, TRAIN_YEAR, compute_training_statistics
from ..env.cchp_env import CCHPPhysicalEnv, EnvConfig
from .runner import RandomPolicy, RulePolicy
from .sequence import SUPPORTED_SEQUENCE_ADAPTERS, SequenceRulePolicy

VALID_CONSTRAINT_MODES = ("physics_in_loop", "reward_only")


def _extract_year(df: pd.DataFrame) -> int:
    years = sorted({int(value.year) for value in pd.to_datetime(df["timestamp"])})
    if len(years) != 1:
        raise ValueError(f"仅支持单年数据，当前年份集合: {years}")
    return years[0]


def _load_param_overrides(params_path: str | Path | None) -> dict:
    if params_path is None:
        return {}
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"参数文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_env_config(
    constraint_mode: str, param_overrides: dict | None, base_env_overrides: dict | None
) -> EnvConfig:
    mode = constraint_mode.strip().lower()
    if mode not in VALID_CONSTRAINT_MODES:
        raise ValueError(f"不支持的约束模式: {constraint_mode}")

    merged_overrides: dict[str, float | str] = {}
    for key, value in (base_env_overrides or {}).items():
        merged_overrides[key] = value
    for key, value in (param_overrides or {}).items():
        if key != "constraint_mode":
            merged_overrides[key] = float(value)
    return build_env_config_from_overrides(merged_overrides, force_constraint_mode=mode)


def _build_policy(
    policy_name: str,
    seed: int,
    train_statistics: dict,
    env_config: EnvConfig,
    history_steps: int,
    sequence_adapter: str = "rule",
):
    policy = policy_name.strip().lower()
    if policy == "rule":
        return RulePolicy(
            train_statistics=train_statistics,
            p_gt_cap_mw=env_config.p_gt_cap_mw,
            q_ech_cap_mw=env_config.q_ech_cap_mw,
        )
    if policy == "random":
        return RandomPolicy(seed=seed)
    if policy == "sequence_rule":
        if sequence_adapter != "rule":
            raise ValueError(
                "ablation 当前仅支持 sequence_adapter=rule；"
                "Transformer/Mamba 请先独立训练并在 eval 阶段加载 checkpoint。"
            )
        return SequenceRulePolicy(
            train_statistics=train_statistics,
            history_steps=history_steps,
            p_gt_cap_mw=env_config.p_gt_cap_mw,
            q_ech_cap_mw=env_config.q_ech_cap_mw,
            sequence_adapter=sequence_adapter,
        )
    raise ValueError(f"不支持的策略: {policy_name}")


def _simulate_eval_episode(
    eval_df: pd.DataFrame,
    *,
    env_config: EnvConfig,
    policy,
    seed: int,
) -> dict:
    env = CCHPPhysicalEnv(exogenous_df=eval_df, config=env_config, seed=seed)
    observation, _ = env.reset(seed=seed, episode_df=eval_df)
    reset_episode_fn = getattr(policy, "reset_episode", None)
    if callable(reset_episode_fn):
        reset_episode_fn(observation)
    terminated = False
    final_info = {}
    total_reward = 0.0
    while not terminated:
        action = policy.act(observation)
        observation, reward, terminated, _, info = env.step(action)
        total_reward += reward
        final_info = info

    summary = final_info.get("episode_summary", env.kpi.summary())
    summary["total_reward_from_loop"] = float(total_reward)
    summary["constraint_mode"] = env_config.constraint_mode
    return summary


def run_constraint_ablation(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    modes: list[str] | tuple[str, ...] = VALID_CONSTRAINT_MODES,
    policy_name: str = "rule",
    history_steps: int = 16,
    sequence_adapter: str = "rule",
    seed: int = 42,
    run_root: str | Path = "runs",
    params_path: str | Path | None = None,
    base_env_overrides: dict | None = None,
) -> dict:
    train_year = _extract_year(train_df)
    eval_year = _extract_year(eval_df)
    if train_year != TRAIN_YEAR:
        raise ValueError(f"训练数据必须是 {TRAIN_YEAR}，当前 {train_year}")
    if eval_year != EVAL_YEAR:
        raise ValueError(f"评估数据必须是 {EVAL_YEAR}，当前 {eval_year}")

    selected_modes = [mode.strip().lower() for mode in modes]
    if not selected_modes:
        raise ValueError("modes 不能为空。")
    for mode in selected_modes:
        if mode not in VALID_CONSTRAINT_MODES:
            raise ValueError(f"未知模式: {mode}")
    sequence_adapter_name = sequence_adapter.strip().lower()
    if sequence_adapter_name not in SUPPORTED_SEQUENCE_ADAPTERS:
        raise ValueError(
            f"不支持的 sequence_adapter: {sequence_adapter}，支持 {SUPPORTED_SEQUENCE_ADAPTERS}"
        )

    run_dir = Path(run_root) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ablation_constraints"
    run_dir.mkdir(parents=True, exist_ok=True)

    param_overrides = _load_param_overrides(params_path=params_path)
    train_statistics = compute_training_statistics(train_df)

    result_rows = []
    mode_summaries: dict[str, dict] = {}
    for index, mode in enumerate(selected_modes):
        env_config = _build_env_config(
            mode, param_overrides, base_env_overrides=base_env_overrides
        )
        policy = _build_policy(
            policy_name=policy_name,
            seed=seed + index,
            train_statistics=train_statistics,
            env_config=env_config,
            history_steps=history_steps,
            sequence_adapter=sequence_adapter_name,
        )
        summary = _simulate_eval_episode(
            eval_df=eval_df,
            env_config=env_config,
            policy=policy,
            seed=seed + 10_000 + index,
        )
        mode_summaries[mode] = summary
        (run_dir / f"{mode}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        unmet = summary["unmet_energy_mwh"]
        result_rows.append(
            {
                "constraint_mode": mode,
                "policy": policy_name,
                "sequence_adapter": sequence_adapter_name,
                "history_steps": int(history_steps),
                "total_cost": float(summary["total_cost"]),
                "violation_rate": float(summary["violation_rate"]),
                "unmet_e_mwh": float(unmet["electric"]),
                "unmet_h_mwh": float(unmet["heat"]),
                "unmet_c_mwh": float(unmet["cooling"]),
                "gt_starts": int(summary["starts"]["gt"]),
                "boiler_starts": int(summary["starts"]["boiler"]),
                "ech_starts": int(summary["starts"]["ech"]),
            }
        )

    summary_df = pd.DataFrame(result_rows).sort_values(
        by=["total_cost", "violation_rate", "unmet_h_mwh"], ascending=[True, True, True]
    )
    summary_df.to_csv(run_dir / "summary.csv", index=False)

    return {
        "run_dir": str(run_dir),
        "modes": selected_modes,
        "policy": policy_name,
        "sequence_adapter": sequence_adapter_name,
        "history_steps": int(history_steps),
        "seed": seed,
        "summary_csv": str(run_dir / "summary.csv"),
        "best_mode": str(summary_df.iloc[0]["constraint_mode"]),
        "best_total_cost": float(summary_df.iloc[0]["total_cost"]),
        "param_overrides_path": str(params_path) if params_path is not None else None,
    }
