# Ref: docs/spec/task.md
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ..core.data import (
    EVAL_YEAR,
    TRAIN_YEAR,
    compute_training_statistics,
    dump_statistics_json,
    make_episode_sampler,
)
from ..core.reporting import write_paper_eval_artifacts
from ..env.cchp_env import CCHPPhysicalEnv, EnvConfig
from .sequence import SUPPORTED_SEQUENCE_ADAPTERS, SequenceRulePolicy


def _extract_year(df: pd.DataFrame) -> int:
    years = sorted({int(value.year) for value in pd.to_datetime(df["timestamp"])})
    if len(years) != 1:
        raise ValueError(f"仅支持单年数据，当前年份集合: {years}")
    return years[0]


class Policy(Protocol):
    def act(self, observation: dict[str, float]) -> dict[str, float]:
        ...


@dataclass(slots=True)
class RandomPolicy:
    seed: int
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, observation: dict[str, float]) -> dict[str, float]:
        return {
            "u_gt": float(self.rng.uniform(-1.0, 1.0)),
            "u_bes": float(self.rng.uniform(-1.0, 1.0)),
            "u_boiler": float(self.rng.uniform(0.0, 1.0)),
            "u_abs": float(self.rng.uniform(0.0, 1.0)),
            "u_ech": float(self.rng.uniform(0.0, 1.0)),
            "u_tes": float(self.rng.uniform(-1.0, 1.0)),
        }


@dataclass(slots=True)
class RulePolicy:
    train_statistics: dict
    p_gt_cap_mw: float = 12.0
    q_ech_cap_mw: float = 6.0
    price_low: float = 0.0
    price_high: float = 1.0
    load_med: float = 0.0
    heat_med: float = 0.0
    cool_med: float = 0.0

    def __post_init__(self) -> None:
        stats = self.train_statistics.get("stats", {})
        price_stats = stats.get("price_e", {})
        load_stats = stats.get("p_dem_mw", {})
        heat_stats = stats.get("qh_dem_mw", {})
        cool_stats = stats.get("qc_dem_mw", {})

        self.price_low = float(price_stats.get("p05", 0.0))
        self.price_high = float(price_stats.get("p95", 1.0))
        self.load_med = float(load_stats.get("p50", 0.0))
        self.heat_med = float(heat_stats.get("p50", 0.0))
        self.cool_med = float(cool_stats.get("p50", 0.0))

    def act(self, observation: dict[str, float]) -> dict[str, float]:
        p_dem = observation["p_dem_mw"]
        p_re = observation["pv_mw"] + observation["wt_mw"]
        qh_dem = observation["qh_dem_mw"]
        qc_dem = observation["qc_dem_mw"]
        soc_bes = observation["soc_bes"]
        price_e = observation["price_e"]
        t_hot_k = observation["t_tes_hot_k"]

        net_load = max(0.0, p_dem - p_re)
        gt_ratio = min(1.0, net_load / max(1e-6, self.p_gt_cap_mw))
        u_gt = gt_ratio * 2.0 - 1.0

        if price_e >= self.price_high and soc_bes > 0.25:
            u_bes = 0.8
        elif price_e <= self.price_low and soc_bes < 0.85:
            u_bes = -0.8
        else:
            u_bes = 0.0

        u_boiler = min(1.0, max(0.0, (qh_dem - self.heat_med * 0.6) / max(1e-6, self.heat_med)))
        u_abs = 0.9 if (qc_dem > self.cool_med * 0.5 and t_hot_k >= 358.15) else 0.0
        u_ech = min(1.0, max(0.0, qc_dem / max(1e-6, self.q_ech_cap_mw)))

        if qh_dem > self.heat_med and observation["e_tes_mwh"] > 2.0:
            u_tes = 0.6
        elif qh_dem < self.heat_med * 0.5 and observation["e_tes_mwh"] < 16.0:
            u_tes = -0.5
        else:
            u_tes = 0.0

        return {
            "u_gt": float(u_gt),
            "u_bes": float(u_bes),
            "u_boiler": float(u_boiler),
            "u_abs": float(u_abs),
            "u_ech": float(u_ech),
            "u_tes": float(u_tes),
        }


def _build_policy(
    policy_name: str,
    seed: int,
    train_statistics: dict,
    history_steps: int,
    sequence_adapter: str = "rule",
    sequence_predictor=None,
) -> Policy:
    normalized = policy_name.lower().strip()
    if normalized == "random":
        return RandomPolicy(seed=seed)
    if normalized == "rule":
        return RulePolicy(train_statistics=train_statistics)
    if normalized == "sequence_rule":
        adapter_name = sequence_adapter.lower().strip()
        if adapter_name not in SUPPORTED_SEQUENCE_ADAPTERS:
            raise ValueError(
                f"不支持的 sequence_adapter: {sequence_adapter}，支持 {SUPPORTED_SEQUENCE_ADAPTERS}"
            )
        return SequenceRulePolicy(
            train_statistics=train_statistics,
            history_steps=history_steps,
            sequence_adapter=adapter_name,
            sequence_predictor=sequence_predictor,
        )
    raise ValueError(f"不支持的策略名称: {policy_name}")


def _run_single_episode(
    env: CCHPPhysicalEnv,
    policy: Policy,
    collect_step_log: bool,
    *,
    seed: int,
    episode_df: pd.DataFrame | None = None,
) -> tuple[float, dict, list[dict]]:
    observation, _ = env.reset(seed=seed, episode_df=episode_df)
    reset_episode_fn = getattr(policy, "reset_episode", None)
    if callable(reset_episode_fn):
        reset_episode_fn(observation)
    terminated = False
    total_reward = 0.0
    step_rows: list[dict] = []
    final_info: dict = {}

    while not terminated:
        action = policy.act(observation)
        observation, reward, terminated, _, info = env.step(action)
        total_reward += reward
        final_info = info
        if collect_step_log:
            log_row = {
                key: value
                for key, value in info.items()
                if key not in {"violation_flags", "diagnostic_flags"}
            }
            log_row["violation_flags_json"] = json.dumps(
                info.get("violation_flags", {}), ensure_ascii=False
            )
            log_row["diagnostic_flags_json"] = json.dumps(
                info.get("diagnostic_flags", {}), ensure_ascii=False
            )
            step_rows.append(log_row)

    summary = final_info.get("episode_summary", env.kpi.summary())
    summary["total_reward_from_loop"] = float(total_reward)
    return total_reward, summary, step_rows


def _create_run_directory(run_root: str | Path, policy_name: str, mode: str) -> Path:
    root = Path(run_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{timestamp}_{mode}_{policy_name}"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def train_baseline(
    train_df: pd.DataFrame,
    *,
    episode_days: int,
    episodes: int,
    policy_name: str,
    history_steps: int = 16,
    sequence_adapter: str = "rule",
    seed: int,
    run_root: str | Path,
    config: EnvConfig,
) -> Path:
    year = _extract_year(train_df)
    if year != TRAIN_YEAR:
        raise ValueError(f"训练必须使用 {TRAIN_YEAR}，当前年份 {year}")

    if config is None:
        raise ValueError("config 不能为空：当前采用 Option-C 全量 yaml 配置模式，请先构建 EnvConfig 并传入。")

    run_dir = _create_run_directory(run_root=run_root, policy_name=policy_name, mode="train")
    train_statistics = compute_training_statistics(train_df)
    dump_statistics_json(train_statistics, run_dir / "train" / "train_statistics.json")

    policy = _build_policy(
        policy_name=policy_name,
        seed=seed,
        train_statistics=train_statistics,
        history_steps=history_steps,
        sequence_adapter=sequence_adapter,
    )
    env = CCHPPhysicalEnv(exogenous_df=train_df, config=config, seed=seed)
    sampler = make_episode_sampler(df=train_df, episode_days=episode_days, seed=seed)

    episode_rows = []
    for episode_id in range(episodes):
        window, episode_df = next(sampler)
        total_reward, summary, _ = _run_single_episode(
            env=env,
            policy=policy,
            collect_step_log=False,
            seed=seed + episode_id,
            episode_df=episode_df,
        )
        episode_rows.append(
            {
                "episode_id": int(episode_id),
                "start_idx": int(window.start_idx),
                "end_idx": int(window.end_idx),
                "start_timestamp": window.start_timestamp.isoformat(),
                "end_timestamp": window.end_timestamp.isoformat(),
                "total_reward": float(total_reward),
                "total_cost": float(summary["total_cost"]),
                "violation_rate": float(summary["violation_rate"]),
                "unmet_e_mwh": float(summary["unmet_energy_mwh"]["electric"]),
                "unmet_h_mwh": float(summary["unmet_energy_mwh"]["heat"]),
                "unmet_c_mwh": float(summary["unmet_energy_mwh"]["cooling"]),
            }
        )

    episodes_df = pd.DataFrame(episode_rows)
    episodes_df.to_csv(run_dir / "train" / "episodes.csv", index=False)

    train_summary = {
        "mode": "train",
        "year": TRAIN_YEAR,
        "policy": policy_name,
        "sequence_adapter": sequence_adapter,
        "history_steps": history_steps,
        "seed": seed,
        "episode_days": episode_days,
        "episodes": episodes,
        "mean_total_cost": float(episodes_df["total_cost"].mean()),
        "mean_violation_rate": float(episodes_df["violation_rate"].mean()),
        "mean_unmet_e_mwh": float(episodes_df["unmet_e_mwh"].mean()),
        "mean_unmet_h_mwh": float(episodes_df["unmet_h_mwh"].mean()),
        "mean_unmet_c_mwh": float(episodes_df["unmet_c_mwh"].mean()),
    }
    (run_dir / "train" / "summary.json").write_text(
        json.dumps(train_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    checkpoint = {
        "policy_name": policy_name,
        "sequence_adapter": sequence_adapter,
        "history_steps": history_steps,
        "seed": seed,
        "train_year": TRAIN_YEAR,
        "episode_days": episode_days,
        "episodes": episodes,
        "train_statistics_path": str(run_dir / "train" / "train_statistics.json"),
    }
    (run_dir / "checkpoints" / "baseline_policy.json").write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def evaluate_baseline(
    eval_df: pd.DataFrame,
    *,
    run_dir: str | Path,
    policy_name: str = "rule",
    history_steps: int = 16,
    sequence_adapter: str = "rule",
    seed: int = 42,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
    config: EnvConfig,
) -> dict:
    year = _extract_year(eval_df)
    if year != EVAL_YEAR:
        raise ValueError(f"评估必须使用 {EVAL_YEAR}，当前年份 {year}")

    if config is None:
        raise ValueError("config 不能为空：当前采用 Option-C 全量 yaml 配置模式，请先构建 EnvConfig 并传入。")

    output_run_dir = Path(run_dir)
    (output_run_dir / "eval").mkdir(parents=True, exist_ok=True)

    train_statistics: dict = {"stats": {}}
    selected_policy = policy_name
    sequence_predictor = None
    if checkpoint_path is not None:
        checkpoint_data = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        selected_policy = checkpoint_data.get("policy_name", policy_name)
        history_steps = int(checkpoint_data.get("history_steps", history_steps))
        sequence_adapter = str(
            checkpoint_data.get("sequence_adapter", sequence_adapter)
        )
        model_checkpoint_path = checkpoint_data.get("model_checkpoint_path")
        if model_checkpoint_path:
            from ..policy.checkpoint import load_policy_predictor

            sequence_predictor, model_metadata = load_policy_predictor(
                checkpoint_path=model_checkpoint_path, device=device
            )
            sequence_adapter = str(
                model_metadata.get("sequence_adapter", sequence_adapter)
            )
            history_steps = int(model_metadata.get("history_steps", history_steps))
        train_stats_path = checkpoint_data.get("train_statistics_path")
        if train_stats_path:
            train_statistics = json.loads(Path(train_stats_path).read_text(encoding="utf-8"))

    policy = _build_policy(
        policy_name=selected_policy,
        seed=seed,
        train_statistics=train_statistics,
        history_steps=history_steps,
        sequence_adapter=sequence_adapter,
        sequence_predictor=sequence_predictor,
    )
    env = CCHPPhysicalEnv(exogenous_df=eval_df, config=config, seed=seed)
    total_reward, summary, step_rows = _run_single_episode(
        env=env,
        policy=policy,
        collect_step_log=True,
        seed=seed,
        episode_df=eval_df,
    )

    summary["mode"] = "eval"
    summary["year"] = EVAL_YEAR
    summary["policy"] = selected_policy
    summary["sequence_adapter"] = sequence_adapter
    summary["history_steps"] = int(history_steps)
    summary["seed"] = seed
    summary["total_reward"] = float(total_reward)

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
