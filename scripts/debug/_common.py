"""Shared helpers for quick diagnostics."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cchp_physical_env.core.config_loader import (  # noqa: E402
    build_env_config_from_overrides,
    load_env_overrides,
)
from cchp_physical_env.core.data import (  # noqa: E402
    compute_training_statistics,
    load_exogenous_data,
)
from cchp_physical_env.env.cchp_env import CCHPPhysicalEnv, EnvConfig  # noqa: E402
from cchp_physical_env.pipeline.runner import RulePolicy  # noqa: E402


__all__ = ["DebugContext", "EpisodeResult", "collect_episode", "load_context"]


@dataclass(slots=True)
class DebugContext:
    env_config: EnvConfig
    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    train_stats: dict


@dataclass(slots=True)
class EpisodeResult:
    steps: pd.DataFrame
    terminated: bool
    step_count: int
    final_info: dict


def load_context(
    env_config_path: str | Path,
    train_path: str | Path,
    eval_path: str | Path,
) -> DebugContext:
    env_overrides = load_env_overrides(str(env_config_path))
    env_config = build_env_config_from_overrides(env_overrides)
    train_df = load_exogenous_data(str(train_path))
    eval_df = load_exogenous_data(str(eval_path))
    train_stats = compute_training_statistics(train_df)
    return DebugContext(
        env_config=env_config,
        train_df=train_df,
        eval_df=eval_df,
        train_stats=train_stats,
    )


def collect_episode(
    ctx: DebugContext,
    *,
    dataset: Literal["train", "eval"] = "eval",
    seed: int = 42,
    max_steps: int | None = None,
) -> EpisodeResult:
    df = ctx.eval_df if dataset == "eval" else ctx.train_df
    policy = RulePolicy(train_statistics=ctx.train_stats)
    env = CCHPPhysicalEnv(exogenous_df=df, config=ctx.env_config, seed=seed)
    obs, _ = env.reset(seed=seed, episode_df=df)

    rows: list[dict] = []
    done = False
    step_idx = 0
    final_info: dict = {}

    while not done:
        action = policy.act(obs)
        obs, reward, done, _, info = env.step(action)
        final_info = info
        row_data = df.iloc[step_idx % len(df)]
        rows.append(
            {
                "timestamp": info.get("timestamp", ""),
                "p_gt_mw": info["p_gt_mw"],
                "q_hrsg_rec_mw": info["q_hrsg_rec_mw"],
                "q_boiler_mw": info["q_boiler_mw"],
                "q_abs_cool_mw": info["q_abs_cool_mw"],
                "q_ech_cool_mw": info["q_ech_cool_mw"],
                "q_tes_charge_mw": info["q_tes_charge_mw"],
                "q_tes_discharge_mw": info["q_tes_discharge_mw"],
                "t_tes_hot_k": info["t_tes_hot_k"],
                "e_tes_mwh": info["e_tes_mwh"],
                "qh_dem_mw": float(row_data["qh_dem_mw"]),
                "qc_dem_mw": float(row_data["qc_dem_mw"]),
                "cost_total": info["cost_total"],
                "cost_unmet_h": info["cost_unmet_h"],
                "cost_unmet_c": info["cost_unmet_c"],
                "cost_unmet_e": info["cost_unmet_e"],
                "cost_viol": info["cost_viol"],
                "cost_gt_fuel": info["cost_gt_fuel"],
                "cost_grid_import": info["cost_grid_import"],
                "energy_unmet_h_mwh": info["energy_unmet_h_mwh"],
                "energy_unmet_c_mwh": info["energy_unmet_c_mwh"],
                "energy_demand_h_mwh": info["energy_demand_h_mwh"],
                "energy_demand_c_mwh": info["energy_demand_c_mwh"],
                "gt_started": info.get("gt_started", 0),
                "hrsg_cap_inv": info["violation_flags"].get("hrsg_capacity_invalid", False),
                "gt_min_enforced": info["violation_flags"].get("safety_gt_min_output_enforced", False),
                "gt_ramp_limited": info["violation_flags"].get("safety_gt_ramp_limited", False),
                "abs_temp_low": info["diagnostic_flags"].get("abs_drive_temp_low_state", False),
            }
        )
        step_idx += 1
        if max_steps is not None and step_idx >= max_steps:
            break

    steps_df = pd.DataFrame(rows)
    return EpisodeResult(
        steps=steps_df,
        terminated=bool(done),
        step_count=step_idx,
        final_info=final_info,
    )
