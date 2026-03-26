# Ref: docs/spec/task.md (Task-ID: 011)
# Ref: docs/spec/architecture.md (Pattern: Policy / Optional Dependency Integration)
from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.data import (
    EVAL_YEAR,
    TRAIN_YEAR,
    compute_training_statistics,
    load_exogenous_data,
    make_episode_sampler,
)
from ..core.reporting import (
    flatten_mapping,
    write_learning_curve_artifacts,
    write_one_row_csv,
    write_paper_eval_artifacts,
)
from ..env.cchp_env import CCHPPhysicalEnv, EnvConfig

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = Any
    Tensor = Any
    nn = Any

try:
    from transformers import MambaConfig, MambaModel
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    MambaConfig = None
    MambaModel = None

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ModuleNotFoundError:  # pragma: no cover
    BaseFeaturesExtractor = object


OBS_KEYS: tuple[str, ...] = (
    "p_dem_mw",
    "qh_dem_mw",
    "qc_dem_mw",
    "pv_mw",
    "wt_mw",
    "price_e",
    "price_gas",
    "carbon_tax",
    "t_amb_k",
    "sp_pa",
    "rh_pct",
    "wind_speed",
    "wind_direction",
    "ghi_wm2",
    "dni_wm2",
    "dhi_wm2",
    "soc_bes",
    "gt_on",
    "gt_state",
    "gt_on_steps",
    "gt_off_steps",
    "gt_min_on_remaining_steps",
    "gt_min_off_remaining_steps",
    "p_gt_prev_mw",
    "gt_ramp_headroom_up_mw",
    "gt_ramp_headroom_down_mw",
    "e_tes_mwh",
    "t_tes_hot_k",
    "abs_drive_margin_k",
    "q_hrsg_est_now_mw",
    "q_tes_discharge_feasible_mw",
    "heat_deficit_if_boiler_off_mw",
    "heat_backup_min_needed_mw",
    "sin_t",
    "cos_t",
    "sin_week",
    "cos_week",
)


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 torch。PPO/SAC/TD3/DDPG/DQN 训练需要 PyTorch。\n"
            "建议先安装 PyTorch（按本机 CUDA/CPU 版本选择）。"
        )


def _require_mamba() -> None:
    if MambaConfig is None or MambaModel is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 Transformers 中的 Mamba 实现。"
            "请安装/升级 transformers，并确保 torch 可用。"
        )


def _build_mamba_config(*, d_model: int, n_layer: int, d_state: int, d_conv: int, expand: int):
    _require_mamba()
    return MambaConfig(
        vocab_size=1,
        hidden_size=int(d_model),
        state_size=int(d_state),
        num_hidden_layers=int(n_layer),
        conv_kernel=int(d_conv),
        expand=int(expand),
        use_cache=False,
        use_mambapy=False,
    )


def _build_sinusoidal_position_encoding(*, seq_len: int, d_model: int) -> Tensor:
    _require_torch()
    if int(seq_len) <= 0:
        raise ValueError("seq_len 必须 > 0。")
    if int(d_model) <= 0:
        raise ValueError("d_model 必须 > 0。")
    position = torch.arange(int(seq_len), dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, int(d_model), 2, dtype=torch.float32) * (-np.log(10_000.0) / float(d_model))
    )
    encoding = torch.zeros(int(seq_len), int(d_model), dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


def _require_sb3_modules():
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ModuleNotFoundError as error:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 sb3 相关依赖。请先安装（并确保 torch 可用）：\n"
            "  uv pip install -e '.[sb3]'\n"
            "或：\n"
            "  uv pip install stable-baselines3 gymnasium\n"
            "然后再运行 sb3-train/sb3-eval。"
        ) from error
    return gym, spaces, PPO, SAC, TD3, DDPG, DQN, DummyVecEnv, VecNormalize


def _timestamped_run_dir(
    run_root: str | Path, *, mode: str, algo: str, backbone: str, history_steps: int
) -> Path:
    root = Path(run_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"{stamp}_{mode}_sb3_{algo}_{backbone}_k{int(history_steps)}"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _extract_year(df: pd.DataFrame) -> int:
    years = sorted({int(value.year) for value in pd.to_datetime(df["timestamp"])})
    if len(years) != 1:
        raise ValueError(f"仅支持单年数据，当前年份集合: {years}")
    return years[0]


def _observation_dict_to_vector(observation: dict[str, float], *, keys: tuple[str, ...]) -> np.ndarray:
    return np.asarray([float(observation[key]) for key in keys], dtype=np.float32)


@dataclass(slots=True)
class ObservationAffineNormalizer:
    offset: np.ndarray
    scale: np.ndarray
    clip_value: float = 10.0

    def apply(self, vector: np.ndarray) -> np.ndarray:
        raw = np.asarray(vector, dtype=np.float32).reshape(-1)
        normalized = (raw - self.offset) / self.scale
        clipped = np.clip(normalized, -float(self.clip_value), float(self.clip_value))
        return clipped.astype(np.float32, copy=False)


def _build_observation_normalizer(
    *, train_statistics: dict, env_config: EnvConfig, keys: tuple[str, ...], eps: float = 1e-6
) -> ObservationAffineNormalizer:
    stats = dict(train_statistics.get("stats", {}) or {})
    offsets: list[float] = []
    scales: list[float] = []
    for key in keys:
        if key in stats:
            mean = float(stats[key].get("mean", 0.0))
            std = float(stats[key].get("std", 1.0))
            offsets.append(mean)
            scales.append(max(float(eps), float(std)))
            continue

        if key == "soc_bes":
            offsets.append(0.5)
            scales.append(0.5)
        elif key == "gt_on":
            offsets.append(0.5)
            scales.append(0.5)
        elif key == "gt_state":
            offsets.append(1.0)
            scales.append(1.0)
        elif key == "p_gt_prev_mw":
            cap = max(float(eps), float(getattr(env_config, "p_gt_cap_mw", 0.0)))
            offsets.append(0.5 * cap)
            scales.append(0.5 * cap)
        elif key in {"gt_ramp_headroom_up_mw", "gt_ramp_headroom_down_mw"}:
            ramp = max(float(eps), float(getattr(env_config, "gt_ramp_mw_per_step", 0.0)))
            offsets.append(0.5 * ramp)
            scales.append(0.5 * ramp)
        elif key == "e_tes_mwh":
            cap = max(float(eps), float(getattr(env_config, "e_tes_cap_mwh", 0.0)))
            offsets.append(0.5 * cap)
            scales.append(0.5 * cap)
        elif key == "t_tes_hot_k":
            offsets.append(360.0)
            scales.append(20.0)
        elif key == "abs_drive_margin_k":
            offsets.append(0.0)
            scales.append(max(2.0, float(getattr(env_config, "abs_gate_scale_k", 2.0))))
        elif key == "q_hrsg_est_now_mw":
            cap = max(
                float(eps),
                float(getattr(env_config, "q_boiler_cap_mw", 0.0))
                + float(getattr(env_config, "q_tes_discharge_cap_mw", 0.0)),
            )
            offsets.append(0.5 * cap)
            scales.append(0.5 * cap)
        elif key == "q_tes_discharge_feasible_mw":
            cap = max(float(eps), float(getattr(env_config, "q_tes_discharge_cap_mw", 0.0)))
            offsets.append(0.5 * cap)
            scales.append(0.5 * cap)
        elif key in {"heat_deficit_if_boiler_off_mw", "heat_backup_min_needed_mw"}:
            cap = max(float(eps), float(getattr(env_config, "q_boiler_cap_mw", 0.0)))
            offsets.append(0.5 * cap)
            scales.append(0.5 * cap)
        elif key.startswith("sin_") or key.startswith("cos_"):
            offsets.append(0.0)
            scales.append(1.0)
        else:
            offsets.append(0.0)
            scales.append(1.0)

    offset_arr = np.asarray(offsets, dtype=np.float32)
    scale_arr = np.asarray(scales, dtype=np.float32)
    return ObservationAffineNormalizer(offset=offset_arr, scale=scale_arr, clip_value=10.0)


class WindowBuffer:
    def __init__(self, *, history_steps: int, obs_dim: int) -> None:
        if history_steps <= 0:
            raise ValueError("history_steps 必须 > 0。")
        if obs_dim <= 0:
            raise ValueError("obs_dim 必须 > 0。")
        self.history_steps = int(history_steps)
        self.obs_dim = int(obs_dim)
        self.window = np.zeros((self.history_steps, self.obs_dim), dtype=np.float32)

    def reset(self, first_obs: np.ndarray) -> np.ndarray:
        vector = np.asarray(first_obs, dtype=np.float32).reshape(self.obs_dim)
        self.window[:] = vector[None, :]
        return self.window

    def push(self, obs: np.ndarray) -> np.ndarray:
        vector = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        self.window[:-1, :] = self.window[1:, :]
        self.window[-1, :] = vector
        return self.window


def _action_vector_to_env_action(action_vector: np.ndarray) -> dict[str, float]:
    vector = np.asarray(action_vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != 6:
        raise ValueError(f"动作维度不匹配：期望 6，当前 {int(vector.shape[0])}")
    u_gt, u_bes, u_boiler, u_abs, u_ech, u_tes = (float(item) for item in vector.tolist())
    return {
        "u_gt": float(np.clip(u_gt, -1.0, 1.0)),
        "u_bes": float(np.clip(u_bes, -1.0, 1.0)),
        "u_boiler": float(np.clip(u_boiler, 0.0, 1.0)),
        "u_abs": float(np.clip(u_abs, 0.0, 1.0)),
        "u_ech": float(np.clip(u_ech, 0.0, 1.0)),
        "u_tes": float(np.clip(u_tes, -1.0, 1.0)),
    }


def _action_vector_to_residual_delta(action_vector: np.ndarray) -> dict[str, float]:
    vector = np.asarray(action_vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != 6:
        raise ValueError(f"残差动作维度不匹配：期望 6，当前 {int(vector.shape[0])}")
    u_gt, u_bes, u_boiler, u_abs, u_ech, u_tes = (float(item) for item in vector.tolist())
    return {
        "u_gt": float(np.clip(u_gt, -1.0, 1.0)),
        "u_bes": float(np.clip(u_bes, -1.0, 1.0)),
        "u_boiler": float(np.clip(u_boiler, -1.0, 1.0)),
        "u_abs": float(np.clip(u_abs, -1.0, 1.0)),
        "u_ech": float(np.clip(u_ech, -1.0, 1.0)),
        "u_tes": float(np.clip(u_tes, -1.0, 1.0)),
    }


def _action_dict_to_vector(action_dict: Mapping[str, float]) -> np.ndarray:
    return np.asarray(
        [
            float(np.clip(action_dict.get("u_gt", 0.0), -1.0, 1.0)),
            float(np.clip(action_dict.get("u_bes", 0.0), -1.0, 1.0)),
            float(np.clip(action_dict.get("u_boiler", 0.0), 0.0, 1.0)),
            float(np.clip(action_dict.get("u_abs", 0.0), 0.0, 1.0)),
            float(np.clip(action_dict.get("u_ech", 0.0), 0.0, 1.0)),
            float(np.clip(action_dict.get("u_tes", 0.0), -1.0, 1.0)),
        ],
        dtype=np.float32,
    )


def _clip_env_action_value(action_key: str, value: float) -> float:
    if action_key in {"u_gt", "u_bes", "u_tes"}:
        return float(np.clip(value, -1.0, 1.0))
    return float(np.clip(value, 0.0, 1.0))


def _compose_residual_action(
    *,
    base_action: Mapping[str, float],
    delta_action: Mapping[str, float],
    residual_scale: float,
) -> tuple[dict[str, float], dict[str, float | bool | str]]:
    scale = float(max(0.0, residual_scale))
    composed: dict[str, float] = {}
    delta_l1 = 0.0
    base_l1 = 0.0
    for key in ("u_gt", "u_bes", "u_boiler", "u_abs", "u_ech", "u_tes"):
        base_value = _clip_env_action_value(key, float(base_action.get(key, 0.0)))
        delta_value = float(np.clip(delta_action.get(key, 0.0), -1.0, 1.0)) * scale
        composed[key] = _clip_env_action_value(key, base_value + delta_value)
        delta_l1 += abs(delta_value)
        base_l1 += abs(base_value)
    return composed, {
        "policy_residual_enabled": True,
        "policy_residual_scale": float(scale),
        "policy_residual_delta_l1": float(delta_l1),
        "policy_base_action_l1": float(base_l1),
    }


def _paper_model_label(
    *,
    algo: str,
    dqn_action_mode: str = "",
    residual_enabled: bool = False,
    residual_policy: str = "rule",
) -> str:
    normalized_algo = str(algo).strip().lower()
    if normalized_algo == "dqn" and str(dqn_action_mode).strip().lower() == "rb_v1":
        return "rbDQN"
    base_label = normalized_algo.upper()
    if bool(residual_enabled) and normalized_algo in {"ppo", "sac", "td3", "ddpg"}:
        residual_prefix = str(residual_policy).strip().lower().replace("-", "_")
        return f"{base_label}+{residual_prefix}_residual"
    return base_label


def _build_residual_expert_policy(
    *,
    policy_name: str,
    env_config: EnvConfig,
    train_statistics: dict[str, Any] | None,
) -> Any:
    from ..pipeline.runner import EasyRulePolicy, RulePolicy

    normalized = str(policy_name).strip().lower().replace("-", "_")
    if normalized == "rule":
        return RulePolicy(
            train_statistics={} if train_statistics is None else train_statistics,
            p_gt_cap_mw=float(env_config.p_gt_cap_mw),
            q_ech_cap_mw=float(env_config.q_ech_cap_mw),
        )
    if normalized == "easy_rule":
        return EasyRulePolicy(
            p_gt_cap_mw=float(env_config.p_gt_cap_mw),
            q_boiler_cap_mw=float(env_config.q_boiler_cap_mw),
            q_ech_cap_mw=float(env_config.q_ech_cap_mw),
        )
    raise ValueError("residual_policy 当前仅支持 rule/easy_rule。")


@dataclass(slots=True)
class RuleBasedDiscreteActionMapper:
    env_config: EnvConfig
    train_statistics: dict
    action_mode: str = "rb_v1"
    _rule_policy: Any = field(init=False, repr=False)
    _easy_rule_policy: Any = field(init=False, repr=False)
    _action_labels: tuple[str, ...] = field(
        init=False,
        repr=False,
        default=(
            "easy_rule",
            "easy_gt_off",
            "easy_gt_mid",
            "easy_gt_high",
            "easy_bes_charge",
            "easy_bes_discharge",
            "rule",
            "rule_gt_off",
            "rule_gt_mid",
            "rule_bes_charge",
            "rule_bes_discharge",
            "rule_boiler_boost",
            "rule_ech_only",
            "rule_abs_prefer",
            "safety_backup",
            "tes_discharge",
        ),
    )

    def __post_init__(self) -> None:
        normalized_mode = str(self.action_mode).strip().lower()
        if normalized_mode != "rb_v1":
            raise ValueError("DQN 离散动作模式当前仅支持 rb_v1。")
        self.action_mode = normalized_mode
        from ..pipeline.runner import EasyRulePolicy, RulePolicy

        self._rule_policy = RulePolicy(
            train_statistics=self.train_statistics,
            p_gt_cap_mw=float(self.env_config.p_gt_cap_mw),
            q_ech_cap_mw=float(self.env_config.q_ech_cap_mw),
        )
        self._easy_rule_policy = EasyRulePolicy(
            p_gt_cap_mw=float(self.env_config.p_gt_cap_mw),
            q_boiler_cap_mw=float(self.env_config.q_boiler_cap_mw),
            q_ech_cap_mw=float(self.env_config.q_ech_cap_mw),
        )

    @property
    def action_labels(self) -> tuple[str, ...]:
        return self._action_labels

    @property
    def action_count(self) -> int:
        return int(len(self._action_labels))

    def expert_prefill_action(
        self,
        observation: Mapping[str, float],
        *,
        expert_policy: str = "easy_rule",
    ) -> int:
        del observation
        normalized = str(expert_policy).strip().lower().replace("-", "_")
        if normalized == "easy_rule":
            return int(self._action_labels.index("easy_rule"))
        if normalized == "rule":
            return int(self._action_labels.index("rule"))
        raise ValueError(f"不支持的 DQN prefill expert_policy: {expert_policy}")

    def _gt_action_from_mw(self, p_gt_target_mw: float) -> float:
        cap = max(1e-6, float(self.env_config.p_gt_cap_mw))
        normalized = 2.0 * (float(p_gt_target_mw) / cap) - 1.0
        return float(np.clip(normalized, -1.0, 1.0))

    def _boiler_follow(self, observation: Mapping[str, float]) -> float:
        q_boiler_need = float(
            observation.get(
                "heat_backup_min_needed_mw",
                observation.get("qh_dem_mw", 0.0),
            )
        )
        return float(
            np.clip(
                q_boiler_need / max(1e-6, float(self.env_config.q_boiler_cap_mw)),
                0.0,
                1.0,
            )
        )

    def _ech_follow(self, observation: Mapping[str, float]) -> float:
        qc_dem = float(observation.get("qc_dem_mw", 0.0))
        return float(np.clip(qc_dem / max(1e-6, float(self.env_config.q_ech_cap_mw)), 0.0, 1.0))

    def _abs_prefer(self, action: dict[str, float], observation: Mapping[str, float]) -> None:
        drive_margin_k = float(observation.get("abs_drive_margin_k", 0.0))
        if drive_margin_k <= 0.0:
            action["u_abs"] = 0.0
            action["u_ech"] = self._ech_follow(observation)
            return
        qc_dem = float(observation.get("qc_dem_mw", 0.0))
        q_abs_cool_cap = max(1e-6, float(self.env_config.q_abs_cool_cap_mw))
        action["u_abs"] = float(np.clip(max(0.35, qc_dem / q_abs_cool_cap), 0.0, 1.0))
        action["u_ech"] = float(min(float(action.get("u_ech", 0.0)), 0.25))

    def decode(self, action: Any, observation: Mapping[str, float]) -> dict[str, float]:
        action_index = int(np.asarray(action).reshape(-1)[0])
        if action_index < 0 or action_index >= self.action_count:
            raise ValueError(f"DQN 离散动作索引越界: {action_index}")
        label = self._action_labels[action_index]

        if label.startswith("easy_"):
            decoded = dict(self._easy_rule_policy.act(dict(observation)))
        elif label == "safety_backup":
            decoded = {
                "u_gt": -1.0,
                "u_bes": 0.0,
                "u_boiler": self._boiler_follow(observation),
                "u_abs": 0.0,
                "u_ech": self._ech_follow(observation),
                "u_tes": 0.0,
            }
        elif label == "tes_discharge":
            decoded = dict(self._rule_policy.act(dict(observation)))
        else:
            decoded = dict(self._rule_policy.act(dict(observation)))

        if label == "easy_gt_off" or label == "rule_gt_off":
            decoded["u_gt"] = -1.0
        elif label == "easy_gt_mid" or label == "rule_gt_mid":
            decoded["u_gt"] = self._gt_action_from_mw(0.5 * float(self.env_config.p_gt_cap_mw))
        elif label == "easy_gt_high":
            decoded["u_gt"] = self._gt_action_from_mw(0.85 * float(self.env_config.p_gt_cap_mw))

        if label == "easy_bes_charge" or label == "rule_bes_charge":
            decoded["u_bes"] = -0.8
        elif label == "easy_bes_discharge" or label == "rule_bes_discharge":
            decoded["u_bes"] = 0.8

        if label == "rule_boiler_boost":
            decoded["u_boiler"] = max(float(decoded.get("u_boiler", 0.0)), max(0.35, self._boiler_follow(observation)))
        elif label == "rule_ech_only":
            decoded["u_abs"] = 0.0
            decoded["u_ech"] = self._ech_follow(observation)
        elif label == "rule_abs_prefer":
            self._abs_prefer(decoded, observation)
        elif label == "tes_discharge":
            e_tes = float(observation.get("e_tes_mwh", 0.0))
            decoded["u_tes"] = 0.6 if e_tes > 0.25 * float(self.env_config.e_tes_cap_mwh) else 0.0

        decoded["u_gt"] = float(np.clip(decoded.get("u_gt", 0.0), -1.0, 1.0))
        decoded["u_bes"] = float(np.clip(decoded.get("u_bes", 0.0), -1.0, 1.0))
        decoded["u_boiler"] = float(np.clip(decoded.get("u_boiler", 0.0), 0.0, 1.0))
        decoded["u_abs"] = float(np.clip(decoded.get("u_abs", 0.0), 0.0, 1.0))
        decoded["u_ech"] = float(np.clip(decoded.get("u_ech", 0.0), 0.0, 1.0))
        decoded["u_tes"] = float(np.clip(decoded.get("u_tes", 0.0), -1.0, 1.0))
        return decoded


@dataclass(slots=True)
class SB3TrainConfig:
    algo: str
    backbone: str = "mlp"
    history_steps: int = 16
    total_timesteps: int = 200_000
    episode_days: int = 14
    n_envs: int = 1
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    vec_norm_obs: bool = True
    vec_norm_reward: bool = True
    eval_freq: int = 50_000
    eval_episode_days: int = 14
    eval_window_pool_size: int = 0
    eval_window_count: int = 0
    eval_window_seed: int | None = None
    ppo_warm_start_enabled: bool = True
    residual_enabled: bool = False
    residual_policy: str = "rule"
    residual_scale: float = 0.35
    ppo_warm_start_samples: int = 16_384
    ppo_warm_start_epochs: int = 4
    ppo_warm_start_batch_size: int = 256
    ppo_warm_start_lr: float = 1e-4
    offpolicy_prefill_enabled: bool = False
    offpolicy_prefill_steps: int = 0
    offpolicy_prefill_policy: str = "easy_rule"
    ppo_n_steps: int = 2048
    ppo_gae_lambda: float = 0.95
    ppo_ent_coef: float = 0.0
    ppo_clip_range: float = 0.2
    dqn_action_mode: str = "rb_v1"
    dqn_target_update_interval: int = 1_000
    dqn_exploration_fraction: float = 0.3
    dqn_exploration_initial_eps: float = 1.0
    dqn_exploration_final_eps: float = 0.05
    learning_starts: int = 5_000
    train_freq: int = 1
    gradient_steps: int = 1
    tau: float = 0.005
    action_noise_std: float = 0.1
    buffer_size: int = 50_000
    optimize_memory_usage: bool = True
    best_gate_enabled: bool = True
    best_gate_electric_min: float = 1.0
    best_gate_heat_min: float = 0.99
    best_gate_cool_min: float = 0.99
    plateau_control_enabled: bool = True
    plateau_patience_evals: int = 10
    plateau_lr_decay_factor: float = 0.5
    plateau_min_lr: float = 5e-5
    plateau_early_stop_patience_evals: int = 999
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        self.algo = str(self.algo).strip().lower()
        if self.algo not in {"ppo", "sac", "td3", "ddpg", "dqn"}:
            raise ValueError("sb3 algo 仅支持 ppo/sac/td3/ddpg/dqn。")
        self.backbone = str(self.backbone).strip().lower()
        if self.backbone not in {"mlp", "transformer", "mamba"}:
            raise ValueError("sb3 backbone 仅支持 mlp/transformer/mamba。")
        self.history_steps = int(self.history_steps)
        if self.history_steps <= 0:
            raise ValueError("history_steps 必须 > 0。")
        self.total_timesteps = int(self.total_timesteps)
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps 必须 > 0。")
        self.episode_days = int(self.episode_days)
        if self.episode_days < 7 or self.episode_days > 30:
            raise ValueError("episode_days 必须在 [7,30]。")
        self.n_envs = int(self.n_envs)
        if self.n_envs <= 0:
            raise ValueError("n_envs 必须 > 0。")
        self.learning_rate = float(self.learning_rate)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须 > 0。")
        self.batch_size = int(self.batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须 > 0。")
        self.gamma = float(self.gamma)
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma 必须在 (0,1]。")
        self.vec_norm_obs = bool(self.vec_norm_obs)
        self.vec_norm_reward = bool(self.vec_norm_reward)
        self.eval_freq = int(self.eval_freq)
        if self.eval_freq <= 0:
            raise ValueError("eval_freq 必须 > 0。")
        self.eval_episode_days = int(self.eval_episode_days)
        if self.eval_episode_days < 7 or self.eval_episode_days > 30:
            raise ValueError("eval_episode_days 必须在 [7,30]。")
        self.eval_window_pool_size = int(self.eval_window_pool_size)
        self.eval_window_count = int(self.eval_window_count)
        if self.eval_window_pool_size < 0:
            raise ValueError("eval_window_pool_size 必须 >= 0。")
        if self.eval_window_count < 0:
            raise ValueError("eval_window_count 必须 >= 0。")
        if self.eval_window_pool_size == 0:
            self.eval_window_count = 0
        if self.eval_window_count > self.eval_window_pool_size > 0:
            raise ValueError("eval_window_count 不能大于 eval_window_pool_size。")
        if self.eval_window_seed is None:
            self.eval_window_seed = int(self.seed)
        else:
            self.eval_window_seed = int(self.eval_window_seed)
        self.ppo_warm_start_enabled = bool(self.ppo_warm_start_enabled)
        self.residual_enabled = bool(self.residual_enabled)
        self.residual_policy = str(self.residual_policy).strip().lower().replace("-", "_")
        if self.residual_policy not in {"rule", "easy_rule"}:
            raise ValueError("residual_policy 当前仅支持 rule/easy_rule。")
        self.residual_scale = float(self.residual_scale)
        if self.residual_scale < 0.0 or self.residual_scale > 1.0:
            raise ValueError("residual_scale 必须在 [0,1]。")
        self.ppo_warm_start_samples = int(self.ppo_warm_start_samples)
        if self.ppo_warm_start_samples <= 0:
            raise ValueError("ppo_warm_start_samples 必须 > 0。")
        self.ppo_warm_start_epochs = int(self.ppo_warm_start_epochs)
        if self.ppo_warm_start_epochs <= 0:
            raise ValueError("ppo_warm_start_epochs 必须 > 0。")
        self.ppo_warm_start_batch_size = int(self.ppo_warm_start_batch_size)
        if self.ppo_warm_start_batch_size <= 0:
            raise ValueError("ppo_warm_start_batch_size 必须 > 0。")
        self.ppo_warm_start_lr = float(self.ppo_warm_start_lr)
        if self.ppo_warm_start_lr <= 0.0:
            raise ValueError("ppo_warm_start_lr 必须 > 0。")
        self.offpolicy_prefill_enabled = bool(self.offpolicy_prefill_enabled)
        self.offpolicy_prefill_steps = int(self.offpolicy_prefill_steps)
        if self.offpolicy_prefill_steps < 0:
            raise ValueError("offpolicy_prefill_steps 必须 >= 0。")
        self.offpolicy_prefill_policy = (
            str(self.offpolicy_prefill_policy).strip().lower().replace("-", "_")
        )
        if self.offpolicy_prefill_policy not in {"easy_rule", "rule"}:
            raise ValueError("offpolicy_prefill_policy 当前仅支持 easy_rule/rule。")
        self.ppo_n_steps = int(self.ppo_n_steps)
        if self.ppo_n_steps <= 0:
            raise ValueError("ppo_n_steps 必须 > 0。")
        self.ppo_gae_lambda = float(self.ppo_gae_lambda)
        if not (0.0 < self.ppo_gae_lambda <= 1.0):
            raise ValueError("ppo_gae_lambda 必须在 (0,1]。")
        self.ppo_ent_coef = float(self.ppo_ent_coef)
        if self.ppo_ent_coef < 0.0:
            raise ValueError("ppo_ent_coef 必须 >= 0。")
        self.ppo_clip_range = float(self.ppo_clip_range)
        if self.ppo_clip_range <= 0.0:
            raise ValueError("ppo_clip_range 必须 > 0。")
        self.dqn_action_mode = str(self.dqn_action_mode).strip().lower()
        if self.dqn_action_mode != "rb_v1":
            raise ValueError("dqn_action_mode 当前仅支持 rb_v1。")
        self.dqn_target_update_interval = int(self.dqn_target_update_interval)
        if self.dqn_target_update_interval <= 0:
            raise ValueError("dqn_target_update_interval 必须 > 0。")
        self.dqn_exploration_fraction = float(self.dqn_exploration_fraction)
        if not (0.0 < self.dqn_exploration_fraction <= 1.0):
            raise ValueError("dqn_exploration_fraction 必须在 (0,1]。")
        self.dqn_exploration_initial_eps = float(self.dqn_exploration_initial_eps)
        self.dqn_exploration_final_eps = float(self.dqn_exploration_final_eps)
        if not (0.0 <= self.dqn_exploration_final_eps <= self.dqn_exploration_initial_eps <= 1.0):
            raise ValueError("DQN epsilon 参数必须满足 0 <= final <= initial <= 1。")
        self.learning_starts = int(self.learning_starts)
        if self.learning_starts < 0:
            raise ValueError("learning_starts 必须 >= 0。")
        self.train_freq = int(self.train_freq)
        if self.train_freq <= 0:
            raise ValueError("train_freq 必须 > 0。")
        self.gradient_steps = int(self.gradient_steps)
        if self.gradient_steps <= 0:
            raise ValueError("gradient_steps 必须 > 0。")
        self.tau = float(self.tau)
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau 必须在 (0,1]。")
        self.action_noise_std = float(self.action_noise_std)
        if self.action_noise_std < 0.0:
            raise ValueError("action_noise_std 必须 >= 0。")
        self.buffer_size = int(self.buffer_size)
        if self.buffer_size <= 0:
            raise ValueError("buffer_size 必须 > 0。")
        if self.buffer_size < self.batch_size:
            raise ValueError("buffer_size 必须 >= batch_size（否则 replay buffer 采样无意义）。")
        self.optimize_memory_usage = bool(self.optimize_memory_usage)
        self.best_gate_enabled = bool(self.best_gate_enabled)
        self.best_gate_electric_min = float(self.best_gate_electric_min)
        self.best_gate_heat_min = float(self.best_gate_heat_min)
        self.best_gate_cool_min = float(self.best_gate_cool_min)
        for field_name, value in (
            ("best_gate_electric_min", self.best_gate_electric_min),
            ("best_gate_heat_min", self.best_gate_heat_min),
            ("best_gate_cool_min", self.best_gate_cool_min),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{field_name} 必须在 [0,1]。")
        self.plateau_control_enabled = bool(self.plateau_control_enabled)
        self.plateau_patience_evals = int(self.plateau_patience_evals)
        if self.plateau_patience_evals <= 0:
            raise ValueError("plateau_patience_evals 必须 > 0。")
        self.plateau_lr_decay_factor = float(self.plateau_lr_decay_factor)
        if not (0.0 < self.plateau_lr_decay_factor < 1.0):
            raise ValueError("plateau_lr_decay_factor 必须在 (0,1) 内。")
        self.plateau_min_lr = float(self.plateau_min_lr)
        if self.plateau_min_lr <= 0.0:
            raise ValueError("plateau_min_lr 必须 > 0。")
        if self.plateau_min_lr > self.learning_rate:
            raise ValueError("plateau_min_lr 不能大于 learning_rate。")
        self.plateau_early_stop_patience_evals = int(self.plateau_early_stop_patience_evals)
        if self.plateau_early_stop_patience_evals <= 0:
            raise ValueError("plateau_early_stop_patience_evals 必须 > 0。")
        self.seed = int(self.seed)
        self.device = str(self.device).strip().lower()


def _build_spaces(
    *,
    history_steps: int,
    obs_dim: int,
    algo: str,
    discrete_action_count: int = 0,
    residual_enabled: bool = False,
):
    _, spaces, *_ = _require_sb3_modules()
    if str(algo).strip().lower() == "dqn":
        if int(discrete_action_count) <= 1:
            raise ValueError("DQN 离散动作数量必须 > 1。")
        action_space = spaces.Discrete(int(discrete_action_count))
    else:
        if bool(residual_enabled):
            action_space = spaces.Box(
                low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                shape=(6,),
                dtype=np.float32,
            )
        else:
            # action: [u_gt,u_bes,u_boiler,u_abs,u_ech,u_tes]
            action_space = spaces.Box(
                low=np.asarray([-1.0, -1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
                high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                shape=(6,),
                dtype=np.float32,
            )
    history_steps_int = int(history_steps)
    if history_steps_int <= 0:
        raise ValueError("history_steps 必须 > 0。")
    obs_dim = int(obs_dim)
    if obs_dim <= 0:
        raise ValueError("obs_dim 必须 > 0。")
    # observation: windowed floats (K,D)
    # 这里采用保守的大范围有限边界，避免依赖外部统计文件；论文口径建议再加归一化消融。
    obs_low = np.full((history_steps_int, obs_dim), -1e9, dtype=np.float32)
    obs_high = np.full((history_steps_int, obs_dim), 1e9, dtype=np.float32)
    observation_space = spaces.Box(
        low=obs_low,
        high=obs_high,
        shape=(history_steps_int, obs_dim),
        dtype=np.float32,
    )
    return observation_space, action_space


def make_train_env_factory(
    *,
    train_df: pd.DataFrame,
    env_config: EnvConfig,
    seed: int,
    episode_days: int,
    history_steps: int,
    observation_keys: tuple[str, ...],
    normalizer: ObservationAffineNormalizer | None,
    algo: str = "ppo",
    discrete_action_mapper: RuleBasedDiscreteActionMapper | None = None,
    residual_policy_name: str | None = None,
    residual_scale: float = 0.0,
    train_statistics: dict[str, Any] | None = None,
    fixed_episode_df: pd.DataFrame | None = None,
    fixed_episode_dfs: tuple[pd.DataFrame, ...] | None = None,
) -> Callable[[], Any]:
    gym, *_ = _require_sb3_modules()
    residual_active = bool(residual_policy_name) and discrete_action_mapper is None
    observation_space, action_space = _build_spaces(
        history_steps=history_steps,
        obs_dim=len(observation_keys),
        algo=algo,
        discrete_action_count=0 if discrete_action_mapper is None else discrete_action_mapper.action_count,
        residual_enabled=residual_active,
    )

    class _CCHPSB3TrainEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = observation_space
            self.action_space = action_space
            self.discrete_action_mapper = discrete_action_mapper
            self.residual_policy_name = None if residual_policy_name is None else str(residual_policy_name)
            self.residual_scale = float(max(0.0, residual_scale))
            self.residual_policy = None
            if residual_active:
                self.residual_policy = _build_residual_expert_policy(
                    policy_name=str(residual_policy_name),
                    env_config=env_config,
                    train_statistics=train_statistics,
                )
            self.rng = np.random.default_rng(seed)
            self.fixed_episode_dfs = None
            self.fixed_episode_cursor = 0
            if fixed_episode_dfs:
                self.fixed_episode_dfs = tuple(
                    item.reset_index(drop=True) for item in fixed_episode_dfs
                )
            self.fixed_episode_df = (
                None if fixed_episode_df is None else fixed_episode_df.reset_index(drop=True)
            )
            self.sampler = None
            if self.fixed_episode_df is None and self.fixed_episode_dfs is None:
                self.sampler = make_episode_sampler(
                    train_df, episode_days=episode_days, seed=int(seed)
                )
            self.env = CCHPPhysicalEnv(exogenous_df=train_df, config=env_config, seed=int(seed))
            self.observation: dict[str, float] | None = None
            self.buffer = WindowBuffer(history_steps=int(history_steps), obs_dim=len(observation_keys))

        def reset_fixed_episode_cursor(self) -> None:
            self.fixed_episode_cursor = 0

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            del options
            if seed is not None:
                self.rng = np.random.default_rng(int(seed))
            if self.fixed_episode_dfs is not None:
                index = int(self.fixed_episode_cursor % len(self.fixed_episode_dfs))
                episode_df = self.fixed_episode_dfs[index]
                self.fixed_episode_cursor = int(self.fixed_episode_cursor + 1)
            elif self.fixed_episode_df is not None:
                episode_df = self.fixed_episode_df
            else:
                if self.sampler is None:
                    raise RuntimeError("训练采样器未初始化。")
                _, episode_df = next(self.sampler)
            observation, _ = self.env.reset(seed=int(seed or 0), episode_df=episode_df)
            self.observation = observation
            vector = _observation_dict_to_vector(observation, keys=observation_keys)
            if normalizer is not None:
                vector = normalizer.apply(vector)
            window = self.buffer.reset(vector)
            return window.copy(), {}

        def step(self, action):
            if self.observation is None:
                raise RuntimeError("环境未 reset。")
            residual_debug: dict[str, float | bool | str] = {}
            if self.discrete_action_mapper is None:
                if self.residual_policy is None:
                    action_dict = _action_vector_to_env_action(action)
                else:
                    base_action = self.residual_policy.act(self.observation)
                    delta_action = _action_vector_to_residual_delta(action)
                    action_dict, residual_debug = _compose_residual_action(
                        base_action=base_action,
                        delta_action=delta_action,
                        residual_scale=self.residual_scale,
                    )
                    residual_debug["policy_residual_policy"] = str(self.residual_policy_name)
            else:
                action_dict = self.discrete_action_mapper.decode(action, self.observation)
            next_obs, reward, terminated, truncated, info = self.env.step(action_dict)
            if residual_debug:
                info = dict(info)
                info.update(residual_debug)
            self.observation = next_obs
            vector = _observation_dict_to_vector(next_obs, keys=observation_keys)
            if normalizer is not None:
                vector = normalizer.apply(vector)
            window = self.buffer.push(vector)
            return window.copy(), float(reward), bool(terminated), bool(truncated), dict(info)

    return _CCHPSB3TrainEnv


def make_eval_env_factory(
    *,
    eval_df: pd.DataFrame | None,
    env_config: EnvConfig,
    seed: int,
    history_steps: int,
    observation_keys: tuple[str, ...],
    normalizer: ObservationAffineNormalizer | None,
    algo: str = "ppo",
    discrete_action_mapper: RuleBasedDiscreteActionMapper | None = None,
    residual_policy_name: str | None = None,
    residual_scale: float = 0.0,
    train_statistics: dict[str, Any] | None = None,
    eval_episode_dfs: tuple[pd.DataFrame, ...] | None = None,
) -> Callable[[], Any]:
    if eval_episode_dfs:
        base_df = eval_episode_dfs[0]
    elif eval_df is not None:
        base_df = eval_df
    else:
        raise ValueError("make_eval_env_factory 需要 eval_df 或 eval_episode_dfs。")
    episode_days = max(7, int(np.ceil(len(base_df) * max(1e-6, float(env_config.dt_hours)) / 24.0)))
    return make_train_env_factory(
        train_df=base_df,
        env_config=env_config,
        seed=seed,
        episode_days=episode_days,
        history_steps=history_steps,
        observation_keys=observation_keys,
        normalizer=normalizer,
        algo=algo,
        discrete_action_mapper=discrete_action_mapper,
        residual_policy_name=residual_policy_name,
        residual_scale=residual_scale,
        train_statistics=train_statistics,
        fixed_episode_df=None if eval_df is None else eval_df.reset_index(drop=True),
        fixed_episode_dfs=eval_episode_dfs,
    )


def _steps_per_day(*, env_config: EnvConfig) -> int:
    return max(1, int(round(24.0 / max(1e-6, float(env_config.dt_hours)))))


def _build_fixed_eval_episode_df(
    *, train_df: pd.DataFrame, env_config: EnvConfig, eval_episode_days: int
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


def _iter_unwrapped_envs(vec_env: Any) -> list[Any]:
    current = vec_env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        envs = getattr(current, "envs", None)
        if envs is not None:
            return list(envs)
        current = getattr(current, "venv", None)
    return []


def _reset_eval_window_cursor(vec_env: Any) -> None:
    for env in _iter_unwrapped_envs(vec_env):
        target = getattr(env, "env", env)
        reset_cursor = getattr(target, "reset_fixed_episode_cursor", None)
        if callable(reset_cursor):
            reset_cursor()


if torch is not None and BaseFeaturesExtractor is not object:

    def _sanitize_window_observations(observations: Tensor, *, clip_value: float = 1e6) -> Tensor:
        safe = torch.nan_to_num(
            observations,
            nan=0.0,
            posinf=float(clip_value),
            neginf=-float(clip_value),
        )
        return torch.clamp(safe, min=-float(clip_value), max=float(clip_value))

class SB3TransformerWindowExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: Any,
        *,
        d_model: int = 128,
        n_head: int = 4,
        n_layer: int = 3,
        dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",
    ) -> None:
        shape = getattr(observation_space, "shape", None)
        if not shape or len(shape) != 2:
            raise ValueError("TransformerWindowExtractor 仅支持 Box(K,D) observation_space。")
        seq_len, obs_dim = int(shape[0]), int(shape[1])
        super().__init__(observation_space, features_dim=int(d_model))
        self.pos_encoding = str(pos_encoding).strip().lower()
        if self.pos_encoding != "sinusoidal":
            raise ValueError("TransformerWindowExtractor 当前仅支持 sinusoidal positional encoding。")
        self.input_norm = nn.LayerNorm(obs_dim)
        self.input_proj = nn.Linear(obs_dim, int(d_model))
        self.input_dropout = nn.Dropout(float(dropout))
        self.register_buffer(
            "positional_encoding",
            _build_sinusoidal_position_encoding(seq_len=seq_len, d_model=int(d_model)),
            persistent=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_head),
            dim_feedforward=int(d_model) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=int(n_layer))
        self.output_norm = nn.LayerNorm(int(d_model))

    def forward(self, observations: Tensor) -> Tensor:
        if observations.dim() != 3:
            raise ValueError("SB3 Transformer extractor 输入必须是 (B,K,D)。")
        safe_observations = _sanitize_window_observations(observations)
        normalized_observations = self.input_norm(safe_observations)
        hidden = self.input_proj(normalized_observations)
        hidden = hidden + self.positional_encoding[:, : hidden.shape[1], :].to(hidden.dtype)
        hidden = self.input_dropout(hidden)
        hidden = self.transformer(hidden)
        hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e4, neginf=-1e4)
        hidden_last = self.output_norm(hidden[:, -1, :])
        return hidden_last


    class SB3MambaWindowExtractor(BaseFeaturesExtractor):
        def __init__(
            self,
            observation_space: Any,
            *,
            d_model: int = 128,
            n_layer: int = 4,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            dropout: float = 0.1,
        ) -> None:
            _require_mamba()
            shape = getattr(observation_space, "shape", None)
            if not shape or len(shape) != 2:
                raise ValueError("MambaWindowExtractor 仅支持 Box(K,D) observation_space。")
            _, obs_dim = int(shape[0]), int(shape[1])
            super().__init__(observation_space, features_dim=int(d_model))
            self.input_norm = nn.LayerNorm(obs_dim)
            self.input_proj = nn.Linear(obs_dim, int(d_model))
            self.mamba = MambaModel(
                _build_mamba_config(
                    d_model=int(d_model),
                    n_layer=int(n_layer),
                    d_state=int(d_state),
                    d_conv=int(d_conv),
                    expand=int(expand),
                )
            )
            self.dropout = nn.Dropout(float(dropout))
            self.output_norm = nn.LayerNorm(int(d_model))

        def forward(self, observations: Tensor) -> Tensor:
            if observations.dim() != 3:
                raise ValueError("SB3 Mamba extractor 输入必须是 (B,K,D)。")
            safe_observations = _sanitize_window_observations(observations)
            normalized_observations = self.input_norm(safe_observations)
            hidden = self.input_proj(normalized_observations)
            outputs = self.mamba(inputs_embeds=hidden, use_cache=False, return_dict=True)
            hidden = torch.nan_to_num(outputs.last_hidden_state, nan=0.0, posinf=1e4, neginf=-1e4)
            hidden_last = self.dropout(self.output_norm(hidden[:, -1, :]))
            return hidden_last


def _sb3_policy_kwargs_for_backbone(*, backbone: str) -> dict[str, Any]:
    normalized = str(backbone).strip().lower()
    if normalized == "mlp":
        return {}
    _require_torch()
    if normalized == "transformer":
        if BaseFeaturesExtractor is object:  # pragma: no cover
            raise ModuleNotFoundError("未检测到 stable-baselines3。")
        return {
            "features_extractor_class": SB3TransformerWindowExtractor,
            "features_extractor_kwargs": {
                "d_model": 128,
                "n_head": 4,
                "n_layer": 3,
                "dropout": 0.1,
                "pos_encoding": "sinusoidal",
            },
        }
    if normalized == "mamba":
        if BaseFeaturesExtractor is object:  # pragma: no cover
            raise ModuleNotFoundError("未检测到 stable-baselines3。")
        return {
            "features_extractor_class": SB3MambaWindowExtractor,
            "features_extractor_kwargs": {
                "d_model": 128,
                "n_layer": 4,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
            },
        }
    raise ValueError("sb3 backbone 仅支持 mlp/transformer/mamba。")


def _get_single_unwrapped_env(vec_env: Any) -> Any:
    envs = _iter_unwrapped_envs(vec_env)
    if not envs:
        raise RuntimeError("无法从 VecEnv 中提取底层环境。")
    current = envs[0]
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "observation"):
            return current
        next_env = getattr(current, "env", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    return envs[0]


def _copy_running_mean_std(source_rms: Any, target_rms: Any) -> None:
    if source_rms is None or target_rms is None:
        return
    for field in ("mean", "var", "count"):
        if not hasattr(source_rms, field) or not hasattr(target_rms, field):
            continue
        source_value = getattr(source_rms, field)
        if field == "count":
            setattr(target_rms, field, float(source_value))
        else:
            setattr(target_rms, field, np.asarray(source_value, dtype=np.float64).copy())


def _copy_vecnormalize_stats(source_env: Any, target_env: Any) -> None:
    if source_env is None or target_env is None:
        return
    source_obs_rms = getattr(source_env, "obs_rms", None)
    target_obs_rms = getattr(target_env, "obs_rms", None)
    if isinstance(source_obs_rms, dict) and isinstance(target_obs_rms, dict):
        for key, source_item in source_obs_rms.items():
            _copy_running_mean_std(source_item, target_obs_rms.get(key))
    else:
        _copy_running_mean_std(source_obs_rms, target_obs_rms)
    _copy_running_mean_std(getattr(source_env, "ret_rms", None), getattr(target_env, "ret_rms", None))


def _constant_lr_schedule(progress_remaining: float, *, learning_rate: float) -> float:
    del progress_remaining
    return float(learning_rate)


def _iter_model_optimizers(model: Any) -> list[Any]:
    candidates: list[Any] = []
    for owner in (
        model,
        getattr(model, "policy", None),
        getattr(model, "actor", None),
        getattr(model, "critic", None),
    ):
        optimizer = getattr(owner, "optimizer", None) if owner is not None else None
        if optimizer is not None:
            candidates.append(optimizer)
    ent_coef_optimizer = getattr(model, "ent_coef_optimizer", None)
    if ent_coef_optimizer is not None:
        candidates.append(ent_coef_optimizer)
    optimizers_attr = getattr(getattr(model, "policy", None), "optimizers", None)
    if isinstance(optimizers_attr, (list, tuple)):
        for optimizer in optimizers_attr:
            if optimizer is not None:
                candidates.append(optimizer)
    unique: list[Any] = []
    seen: set[int] = set()
    for optimizer in candidates:
        if optimizer is None or not hasattr(optimizer, "param_groups"):
            continue
        optimizer_id = id(optimizer)
        if optimizer_id in seen:
            continue
        seen.add(optimizer_id)
        unique.append(optimizer)
    return unique


def _get_model_learning_rate(model: Any, *, fallback: float) -> float:
    for optimizer in _iter_model_optimizers(model):
        param_groups = getattr(optimizer, "param_groups", None)
        if not param_groups:
            continue
        lr_value = param_groups[0].get("lr")
        if lr_value is not None:
            return float(lr_value)
    learning_rate = getattr(model, "learning_rate", None)
    if learning_rate is not None:
        try:
            return float(learning_rate)
        except (TypeError, ValueError):
            pass
    return float(fallback)


def _set_model_learning_rate(model: Any, *, learning_rate: float) -> None:
    lr_value = float(learning_rate)
    if lr_value <= 0.0:
        raise ValueError("learning_rate 必须 > 0。")
    if hasattr(model, "learning_rate"):
        model.learning_rate = lr_value
    if hasattr(model, "lr_schedule"):
        model.lr_schedule = functools.partial(_constant_lr_schedule, learning_rate=lr_value)
    for optimizer in _iter_model_optimizers(model):
        for param_group in getattr(optimizer, "param_groups", []):
            param_group["lr"] = lr_value


def _extract_eval_episode_summary(*, info: Mapping[str, Any] | None, vec_env: Any) -> dict[str, Any]:
    if isinstance(info, Mapping):
        raw_summary = info.get("episode_summary")
        if isinstance(raw_summary, Mapping):
            return dict(raw_summary)
    base_env = _get_single_unwrapped_env(vec_env)
    carrier = getattr(base_env, "env", base_env)
    kpi = getattr(carrier, "kpi", None)
    if kpi is None:
        raise RuntimeError("评估环境缺少 KPI tracker，无法提取 episode_summary。")
    return dict(kpi.summary())


def _aggregate_eval_metrics(
    *,
    episode_rewards: list[float],
    episode_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    reward_array = np.asarray(episode_rewards, dtype=np.float64)
    total_costs = np.asarray(
        [float(summary.get("total_cost", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    heat_reliability = np.asarray(
        [float((summary.get("reliability") or {}).get("heat", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    cool_reliability = np.asarray(
        [float((summary.get("reliability") or {}).get("cooling", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    electric_reliability = np.asarray(
        [float((summary.get("reliability") or {}).get("electric", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    violation_rates = np.asarray(
        [float(summary.get("violation_rate", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    unmet_heat = np.asarray(
        [float((summary.get("unmet_energy_mwh") or {}).get("heat", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    unmet_cool = np.asarray(
        [float((summary.get("unmet_energy_mwh") or {}).get("cooling", 0.0)) for summary in episode_summaries],
        dtype=np.float64,
    )
    return {
        "episodes": int(len(episode_rewards)),
        "mean_reward": float(reward_array.mean()) if len(reward_array) else float("-inf"),
        "mean_total_cost": float(total_costs.mean()) if len(total_costs) else float("inf"),
        "mean_violation_rate": float(violation_rates.mean()) if len(violation_rates) else 0.0,
        "mean_unmet_heat_mwh": float(unmet_heat.mean()) if len(unmet_heat) else 0.0,
        "mean_unmet_cool_mwh": float(unmet_cool.mean()) if len(unmet_cool) else 0.0,
        "reliability_mean": {
            "electric": float(electric_reliability.mean()) if len(electric_reliability) else 0.0,
            "heat": float(heat_reliability.mean()) if len(heat_reliability) else 0.0,
            "cooling": float(cool_reliability.mean()) if len(cool_reliability) else 0.0,
        },
        "reliability_min": {
            "electric": float(electric_reliability.min()) if len(electric_reliability) else 0.0,
            "heat": float(heat_reliability.min()) if len(heat_reliability) else 0.0,
            "cooling": float(cool_reliability.min()) if len(cool_reliability) else 0.0,
        },
        "episode_rewards": [float(value) for value in reward_array.tolist()],
        "episode_total_costs": [float(value) for value in total_costs.tolist()],
        "episode_reliability_heat": [float(value) for value in heat_reliability.tolist()],
        "episode_reliability_cooling": [float(value) for value in cool_reliability.tolist()],
        "episode_reliability_electric": [float(value) for value in electric_reliability.tolist()],
    }


def _evaluate_current_policy(
    *,
    model: Any,
    eval_env: Any,
    n_eval_episodes: int,
    deterministic: bool = True,
) -> dict[str, Any]:
    _reset_eval_window_cursor(eval_env)
    episode_rewards: list[float] = []
    episode_summaries: list[dict[str, Any]] = []
    for _ in range(max(1, int(n_eval_episodes))):
        observation = eval_env.reset()
        terminated = False
        total_reward = 0.0
        final_info: dict[str, Any] = {}
        while not terminated:
            action_vec, _ = model.predict(observation, deterministic=bool(deterministic))
            observation, rewards, dones, infos = eval_env.step(action_vec)
            total_reward += float(np.asarray(rewards, dtype=np.float64).reshape(-1)[0])
            terminated = bool(np.asarray(dones).reshape(-1)[0])
            final_info = dict(infos[0] if infos else {})
        episode_rewards.append(float(total_reward))
        episode_summaries.append(_extract_eval_episode_summary(info=final_info, vec_env=eval_env))
    aggregated = _aggregate_eval_metrics(
        episode_rewards=episode_rewards,
        episode_summaries=episode_summaries,
    )
    aggregated["episode_summaries"] = episode_summaries
    return aggregated


def _load_eval_history_rows(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        text = str(line).strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _build_reliability_gate_result(
    *,
    metrics: Mapping[str, Any],
    config: SB3TrainConfig,
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
            key: float(reliability_min.get(key, 0.0)) for key in ("electric", "heat", "cooling")
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


def _extract_actor_mean_actions(model: Any, obs_tensor: Tensor) -> Tensor:
    distribution = model.policy.get_distribution(obs_tensor)
    inner_distribution = getattr(distribution, "distribution", None)
    mean_actions = getattr(inner_distribution, "mean", None)
    if mean_actions is not None:
        return mean_actions
    return distribution.get_actions(deterministic=True)


def _collect_easy_rule_warm_start_dataset(
    *,
    train_df: pd.DataFrame,
    env_config: EnvConfig,
    config: SB3TrainConfig,
    observation_keys: tuple[str, ...],
    DummyVecEnv: Any,
    VecNormalize: Any,
    use_vecnormalize: bool,
) -> tuple[np.ndarray, np.ndarray, Any | None]:
    from ..pipeline.runner import EasyRulePolicy

    warm_factory = make_train_env_factory(
        train_df=train_df,
        env_config=env_config,
        seed=int(config.seed) + 10_000,
        episode_days=config.episode_days,
        history_steps=config.history_steps,
        observation_keys=observation_keys,
        normalizer=None,
    )
    warm_vec_env = DummyVecEnv([warm_factory])
    if use_vecnormalize:
        warm_vec_env = VecNormalize(
            warm_vec_env,
            training=True,
            norm_obs=bool(config.vec_norm_obs),
            norm_reward=bool(config.vec_norm_reward),
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=float(config.gamma),
        )

    easy_rule = EasyRulePolicy(
        p_gt_cap_mw=float(env_config.p_gt_cap_mw),
        q_boiler_cap_mw=float(env_config.q_boiler_cap_mw),
        q_ech_cap_mw=float(env_config.q_ech_cap_mw),
    )
    base_env = _get_single_unwrapped_env(warm_vec_env)
    observation = warm_vec_env.reset()
    obs_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    sample_target = int(config.ppo_warm_start_samples)

    try:
        while len(obs_batches) < sample_target:
            raw_observation = getattr(base_env, "observation", None)
            if raw_observation is None:
                raise RuntimeError("warm-start 数据采样失败：底层环境缺少 observation。")
            action_dict = easy_rule.act(raw_observation)
            obs_batches.append(np.asarray(observation[0], dtype=np.float32).copy())
            target_batches.append(
                np.asarray(
                    [
                        float(np.clip(action_dict["u_boiler"], 0.0, 1.0)),
                        float(np.clip(action_dict["u_ech"], 0.0, 1.0)),
                    ],
                    dtype=np.float32,
                )
            )
            action_vec = _action_dict_to_vector(action_dict).reshape(1, -1)
            observation, _, dones, _ = warm_vec_env.step(action_vec)
            if bool(np.asarray(dones).reshape(-1)[0]):
                observation = warm_vec_env.reset()
    except Exception:
        warm_vec_env.close()
        raise

    return (
        np.asarray(obs_batches, dtype=np.float32),
        np.asarray(target_batches, dtype=np.float32),
        warm_vec_env,
    )


def _collect_residual_zero_warm_start_dataset(
    *,
    train_df: pd.DataFrame,
    env_config: EnvConfig,
    config: SB3TrainConfig,
    observation_keys: tuple[str, ...],
    train_statistics: Mapping[str, Any] | None,
    DummyVecEnv: Any,
    VecNormalize: Any,
    use_vecnormalize: bool,
) -> tuple[np.ndarray, np.ndarray, Any | None]:
    residual_base_policy = _build_residual_expert_policy(
        policy_name=str(config.residual_policy),
        env_config=env_config,
        train_statistics=None if train_statistics is None else dict(train_statistics),
    )

    warm_factory = make_train_env_factory(
        train_df=train_df,
        env_config=env_config,
        seed=int(config.seed) + 20_000,
        episode_days=config.episode_days,
        history_steps=config.history_steps,
        observation_keys=observation_keys,
        normalizer=None,
    )
    warm_vec_env = DummyVecEnv([warm_factory])
    if use_vecnormalize:
        warm_vec_env = VecNormalize(
            warm_vec_env,
            training=True,
            norm_obs=bool(config.vec_norm_obs),
            norm_reward=bool(config.vec_norm_reward),
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=float(config.gamma),
        )

    base_env = _get_single_unwrapped_env(warm_vec_env)
    observation = warm_vec_env.reset()
    obs_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    sample_target = int(config.ppo_warm_start_samples)

    try:
        while len(obs_batches) < sample_target:
            raw_observation = getattr(base_env, "observation", None)
            if raw_observation is None:
                raise RuntimeError("residual warm-start 数据采样失败：底层环境缺少 observation。")
            action_dict = residual_base_policy.act(raw_observation)
            obs_batches.append(np.asarray(observation[0], dtype=np.float32).copy())
            target_batches.append(np.zeros(6, dtype=np.float32))
            action_vec = _action_dict_to_vector(action_dict).reshape(1, -1)
            observation, _, dones, _ = warm_vec_env.step(action_vec)
            if bool(np.asarray(dones).reshape(-1)[0]):
                observation = warm_vec_env.reset()
    except Exception:
        warm_vec_env.close()
        raise

    return (
        np.asarray(obs_batches, dtype=np.float32),
        np.asarray(target_batches, dtype=np.float32),
        warm_vec_env,
    )


def _warm_start_ppo_from_easy_rule(
    *,
    model: Any,
    observations: np.ndarray,
    targets: np.ndarray,
    config: SB3TrainConfig,
    target_indices: Sequence[int] = (2, 4),
    target_action_keys: Sequence[str] = ("u_boiler", "u_ech"),
    mode: str = "easy_rule_bc_v1",
) -> dict[str, Any]:
    _require_torch()
    if torch is None or F is Any:  # pragma: no cover
        raise ModuleNotFoundError("PPO warm-start 需要 PyTorch。")

    device = next(model.policy.parameters()).device
    batch_size = min(int(config.ppo_warm_start_batch_size), int(len(observations)))
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=float(config.ppo_warm_start_lr))
    dataset_size = int(len(observations))
    last_loss = float("nan")
    loss_history: list[float] = []
    target_index_list = [int(index) for index in target_indices]
    target_action_key_list = [str(item) for item in target_action_keys]

    model.policy.train()
    for _ in range(int(config.ppo_warm_start_epochs)):
        permutation = np.random.permutation(dataset_size)
        for start in range(0, dataset_size, batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch_obs = torch.as_tensor(observations[batch_indices], dtype=torch.float32, device=device)
            batch_targets = torch.as_tensor(targets[batch_indices], dtype=torch.float32, device=device)
            predicted_actions = _extract_actor_mean_actions(model, batch_obs)
            loss = F.mse_loss(predicted_actions[:, target_index_list], batch_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=1.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            loss_history.append(last_loss)

    return {
        "enabled": True,
        "mode": str(mode),
        "target_action_keys": target_action_key_list,
        "target_action_indices": target_index_list,
        "samples": int(dataset_size),
        "epochs": int(config.ppo_warm_start_epochs),
        "batch_size": int(batch_size),
        "lr": float(config.ppo_warm_start_lr),
        "mean_bc_loss": float(np.mean(loss_history)) if loss_history else None,
        "final_bc_loss": None if not np.isfinite(last_loss) else float(last_loss),
    }


def _resolve_observation_carrier_envs(vec_env: Any) -> list[Any]:
    carriers: list[Any] = []
    for env in _iter_unwrapped_envs(vec_env):
        current = env
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "observation"):
                carriers.append(current)
                break
            next_env = getattr(current, "env", None)
            if next_env is None or next_env is current:
                break
            current = next_env
    return carriers


def _with_terminal_observations(
    *,
    next_obs: np.ndarray,
    dones: np.ndarray,
    infos: list[dict[str, Any]],
) -> np.ndarray:
    stored_next_obs = np.asarray(next_obs, dtype=np.float32).copy()
    done_flags = np.asarray(dones).reshape(-1)
    for index, done in enumerate(done_flags.tolist()):
        if not bool(done):
            continue
        terminal_observation = infos[index].get("terminal_observation")
        if terminal_observation is None:
            continue
        stored_next_obs[index] = np.asarray(terminal_observation, dtype=np.float32)
    return stored_next_obs


def _prefill_offpolicy_replay_buffer(
    *,
    model: Any,
    train_df: pd.DataFrame,
    env_config: EnvConfig,
    config: SB3TrainConfig,
    train_statistics: dict[str, Any],
    observation_keys: tuple[str, ...],
    DummyVecEnv: Any,
    discrete_action_mapper: RuleBasedDiscreteActionMapper | None = None,
) -> dict[str, Any]:
    from ..pipeline.runner import EasyRulePolicy, RulePolicy

    target_steps = int(config.offpolicy_prefill_steps)
    if target_steps <= 0:
        target_steps = int(config.learning_starts)
    target_steps = max(0, target_steps)
    if target_steps == 0:
        return {
            "enabled": bool(config.offpolicy_prefill_enabled),
            "applied": False,
            "status": "skipped_zero_steps",
            "steps": 0,
        }

    env_fns = [
        make_train_env_factory(
            train_df=train_df,
            env_config=env_config,
            seed=int(config.seed) + 20_000 + idx,
            episode_days=config.episode_days,
            history_steps=config.history_steps,
            observation_keys=observation_keys,
            normalizer=None,
            algo=config.algo,
            discrete_action_mapper=discrete_action_mapper,
            residual_policy_name=(
                str(config.residual_policy)
                if bool(config.residual_enabled and config.algo != "dqn")
                else None
            ),
            residual_scale=float(config.residual_scale),
            train_statistics=train_statistics,
        )
        for idx in range(int(config.n_envs))
    ]
    prefill_env = DummyVecEnv(env_fns)
    observation = np.asarray(prefill_env.reset(), dtype=np.float32)
    carrier_envs = _resolve_observation_carrier_envs(prefill_env)
    if not carrier_envs:
        prefill_env.close()
        raise RuntimeError("off-policy prefill 失败：未找到底层 observation carrier。")

    if config.offpolicy_prefill_policy == "rule":
        expert_policy = RulePolicy(
            train_statistics=train_statistics,
            p_gt_cap_mw=float(env_config.p_gt_cap_mw),
            q_ech_cap_mw=float(env_config.q_ech_cap_mw),
        )
    else:
        expert_policy = EasyRulePolicy(
            p_gt_cap_mw=float(env_config.p_gt_cap_mw),
            q_boiler_cap_mw=float(env_config.q_boiler_cap_mw),
            q_ech_cap_mw=float(env_config.q_ech_cap_mw),
        )

    collected_steps = 0
    episode_resets = 0
    try:
        while collected_steps < target_steps:
            if config.algo == "dqn":
                action_batch = np.asarray(
                    [
                        int(
                            getattr(carrier_env, "discrete_action_mapper").expert_prefill_action(
                                getattr(carrier_env, "observation"),
                                expert_policy=config.offpolicy_prefill_policy,
                            )
                        )
                        for carrier_env in carrier_envs
                    ],
                    dtype=np.int64,
                )
                replay_actions = action_batch.reshape(-1, 1)
            else:
                if bool(config.residual_enabled):
                    action_batch = np.zeros((len(carrier_envs), 6), dtype=np.float32)
                else:
                    action_batch = np.asarray(
                        [
                            _action_dict_to_vector(
                                expert_policy.act(getattr(carrier_env, "observation"))
                            )
                            for carrier_env in carrier_envs
                        ],
                        dtype=np.float32,
                    )
                replay_actions = action_batch
            next_observation, rewards, dones, infos = prefill_env.step(action_batch)
            next_observation = np.asarray(next_observation, dtype=np.float32)
            rewards_array = np.asarray(rewards, dtype=np.float32).reshape(-1)
            dones_array = np.asarray(dones, dtype=bool).reshape(-1)
            stored_next_obs = _with_terminal_observations(
                next_obs=next_observation,
                dones=dones_array,
                infos=list(infos),
            )
            model.replay_buffer.add(
                observation,
                stored_next_obs,
                replay_actions,
                rewards_array,
                dones_array,
                list(infos),
            )
            observation = next_observation
            collected_steps += int(len(action_batch))
            episode_resets += int(dones_array.sum())
    finally:
        prefill_env.close()

    original_learning_starts = int(getattr(model, "learning_starts", config.learning_starts))
    effective_learning_starts = max(0, original_learning_starts - int(collected_steps))
    model.learning_starts = int(effective_learning_starts)
    return {
        "enabled": True,
        "applied": True,
        "mode": f"{config.offpolicy_prefill_policy}_replay_prefill_v1",
        "policy_name": str(config.offpolicy_prefill_policy),
        "residual_enabled": bool(config.residual_enabled and config.algo != "dqn"),
        "replay_action_mode": (
            "zero_delta_follow_residual_base_v1"
            if bool(config.residual_enabled and config.algo != "dqn")
            else "expert_action_v1"
        ),
        "steps": int(collected_steps),
        "target_steps": int(target_steps),
        "effective_learning_starts": int(effective_learning_starts),
        "original_learning_starts": int(original_learning_starts),
        "episode_resets": int(episode_resets),
    }


def _safe_numeric_series(
    dataframe: pd.DataFrame,
    column: str,
    *,
    default: float,
) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(float(default), index=dataframe.index, dtype=np.float64)
    numeric = pd.to_numeric(dataframe[column], errors="coerce").astype(np.float64)
    return numeric.fillna(float(default))


def _safe_bool_series(
    dataframe: pd.DataFrame,
    column: str,
    *,
    default: bool = False,
) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(bool(default), index=dataframe.index, dtype=bool)
    series = dataframe[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(bool(default)).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n", "nan", "none", ""}
    mapped = normalized.map(lambda value: True if value in truthy else False if value in falsy else bool(default))
    return mapped.astype(bool)


def _write_eval_diagnostics(
    *,
    eval_dir: Path,
    step_df: pd.DataFrame,
    env_config: EnvConfig,
) -> dict[str, str]:
    if step_df.empty or "timestamp" not in step_df.columns:
        return {}

    timestamps = pd.to_datetime(step_df["timestamp"], errors="coerce")
    valid_mask = timestamps.notna()
    if not bool(valid_mask.any()):
        return {}

    diagnostics_df = step_df.loc[valid_mask].copy()
    timestamps = timestamps.loc[valid_mask]
    diagnostics_df["month"] = timestamps.dt.month.astype(int)
    diagnostics_df["hour"] = timestamps.dt.hour.astype(int)

    abs_gate_th = float(max(0.0, getattr(env_config, "abs_deadzone_gate_th", 0.1)))
    diagnostics_df["total_cost"] = _safe_numeric_series(diagnostics_df, "cost_total", default=0.0)
    diagnostics_df["export_penalty"] = _safe_numeric_series(
        diagnostics_df, "cost_grid_export_penalty", default=0.0
    )
    diagnostics_df["unmet_h_mwh"] = _safe_numeric_series(
        diagnostics_df, "energy_unmet_h_mwh", default=0.0
    )
    diagnostics_df["unmet_c_mwh"] = _safe_numeric_series(
        diagnostics_df, "energy_unmet_c_mwh", default=0.0
    )
    diagnostics_df["starts_gt"] = _safe_numeric_series(diagnostics_df, "gt_started", default=0.0)
    diagnostics_df["heat_unmet_step"] = _safe_numeric_series(
        diagnostics_df, "qh_unmet_mw", default=0.0
    ) > float(max(0.0, getattr(env_config, "heat_unmet_th_mw", 0.0)))
    diagnostics_df["cool_unmet_step"] = _safe_numeric_series(
        diagnostics_df, "qc_unmet_mw", default=0.0
    ) > float(max(0.0, getattr(env_config, "cool_unmet_th_mw", 0.0)))
    diagnostics_df["idle_heat_backup_step"] = (
        _safe_numeric_series(diagnostics_df, "cost_idle_heat_backup", default=0.0) > 0.0
    )
    diagnostics_df["idle_cool_backup_step"] = (
        _safe_numeric_series(diagnostics_df, "cost_idle_cool_backup", default=0.0) > 0.0
    )
    diagnostics_df["gt_toggle_step"] = (
        _safe_numeric_series(diagnostics_df, "cost_gt_toggle", default=0.0) > 0.0
    )
    diagnostics_df["export_over_soft_cap_step"] = (
        _safe_numeric_series(diagnostics_df, "p_grid_export_over_soft_cap_mw", default=0.0) > 1e-9
    )
    diagnostics_df["abs_blocked_step"] = _safe_bool_series(
        diagnostics_df, "u_abs_deadzone_applied", default=False
    ) | (
        _safe_numeric_series(diagnostics_df, "u_abs_gate", default=1.0) < abs_gate_th
    )

    group_columns = {
        "steps": ("timestamp", "count"),
        "total_cost": ("total_cost", "sum"),
        "unmet_h_mwh": ("unmet_h_mwh", "sum"),
        "unmet_c_mwh": ("unmet_c_mwh", "sum"),
        "export_penalty": ("export_penalty", "sum"),
        "starts_gt": ("starts_gt", "sum"),
        "abs_blocked_rate": ("abs_blocked_step", "mean"),
        "heat_unmet_steps": ("heat_unmet_step", "sum"),
        "cool_unmet_steps": ("cool_unmet_step", "sum"),
        "idle_heat_backup_steps": ("idle_heat_backup_step", "sum"),
        "idle_cool_backup_steps": ("idle_cool_backup_step", "sum"),
        "gt_toggle_steps": ("gt_toggle_step", "sum"),
        "export_over_soft_cap_steps": ("export_over_soft_cap_step", "sum"),
    }

    month_df = diagnostics_df.groupby("month", sort=True).agg(**group_columns).reset_index()
    hour_df = diagnostics_df.groupby("hour", sort=True).agg(**group_columns).reset_index()

    month_json_path = eval_dir / "kpi_by_month.json"
    hour_json_path = eval_dir / "kpi_by_hour.json"
    behavior_json_path = eval_dir / "behavior_metrics.json"

    month_df.to_csv(eval_dir / "kpi_by_month.csv", index=False)
    hour_df.to_csv(eval_dir / "kpi_by_hour.csv", index=False)
    month_json_path.write_text(
        json.dumps(month_df.to_dict("records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    hour_json_path.write_text(
        json.dumps(hour_df.to_dict("records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    behavior_metrics = {
        "abs_deadzone_gate_threshold": abs_gate_th,
        "steps": int(len(diagnostics_df)),
        "abs_blocked_rate": float(diagnostics_df["abs_blocked_step"].mean()),
        "heat_unmet_steps": int(diagnostics_df["heat_unmet_step"].sum()),
        "cool_unmet_steps": int(diagnostics_df["cool_unmet_step"].sum()),
        "idle_heat_backup_steps": int(diagnostics_df["idle_heat_backup_step"].sum()),
        "idle_cool_backup_steps": int(diagnostics_df["idle_cool_backup_step"].sum()),
        "gt_toggle_steps": int(diagnostics_df["gt_toggle_step"].sum()),
        "export_over_soft_cap_steps": int(diagnostics_df["export_over_soft_cap_step"].sum()),
        "worst_heat_months": month_df.sort_values("unmet_h_mwh", ascending=False).head(3).to_dict("records"),
        "worst_cool_hours": hour_df.sort_values("unmet_c_mwh", ascending=False).head(3).to_dict("records"),
        "worst_export_months": month_df.sort_values("export_penalty", ascending=False).head(3).to_dict("records"),
    }
    behavior_json_path.write_text(
        json.dumps(behavior_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "kpi_by_month": str(month_json_path).replace("\\", "/"),
        "kpi_by_hour": str(hour_json_path).replace("\\", "/"),
        "behavior_metrics": str(behavior_json_path).replace("\\", "/"),
    }


def train_sb3_policy(
    *,
    train_df: pd.DataFrame,
    env_config: EnvConfig,
    config: SB3TrainConfig,
    run_root: str | Path = "runs",
) -> dict[str, Any]:
    train_start_time = time.perf_counter()
    gym, _, PPO, SAC, TD3, DDPG, DQN, DummyVecEnv, VecNormalize = _require_sb3_modules()
    del gym

    try:
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
        from stable_baselines3.common.logger import configure as sb3_configure_logger
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.noise import NormalActionNoise
    except ModuleNotFoundError as error:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 stable-baselines3 训练依赖。请确认已安装：uv pip install -e '.[sb3]'"
        ) from error

    year = _extract_year(train_df)
    if year != TRAIN_YEAR:
        raise ValueError(f"训练必须使用 {TRAIN_YEAR}，当前年份 {year}")

    run_dir = _timestamped_run_dir(
        run_root=run_root,
        mode="train",
        algo=config.algo,
        backbone=config.backbone,
        history_steps=config.history_steps,
    )
    train_summary_path = run_dir / "train" / "summary.json"
    train_statistics_path = run_dir / "train" / "train_statistics.json"
    checkpoint_path = run_dir / "checkpoints" / "baseline_policy.json"
    model_path = run_dir / "checkpoints" / "sb3_model.zip"
    best_model_path = run_dir / "checkpoints" / "best_model.zip"
    best_reward_model_path = run_dir / "checkpoints" / "best_reward_model.zip"
    vecnormalize_path = run_dir / "checkpoints" / "vecnormalize.pkl"
    best_vecnormalize_path = run_dir / "checkpoints" / "best_vecnormalize.pkl"
    best_reward_vecnormalize_path = run_dir / "checkpoints" / "best_reward_vecnormalize.pkl"
    eval_windows_path = run_dir / "checkpoints" / "eval_windows.json"
    eval_history_path = run_dir / "train" / "eval_callback" / "reliability_eval_history.jsonl"

    train_statistics = compute_training_statistics(train_df)
    train_statistics_path.write_text(
        json.dumps(train_statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    observation_keys = tuple(OBS_KEYS)
    discrete_action_mapper = None
    if config.algo == "dqn":
        discrete_action_mapper = RuleBasedDiscreteActionMapper(
            env_config=env_config,
            train_statistics=train_statistics,
            action_mode=config.dqn_action_mode,
        )
    residual_active = bool(config.residual_enabled and config.algo != "dqn")
    paper_model_label = _paper_model_label(
        algo=config.algo,
        dqn_action_mode=config.dqn_action_mode,
        residual_enabled=residual_active,
        residual_policy=config.residual_policy,
    )
    eval_window_pool = _build_eval_window_pool(
        train_df=train_df,
        env_config=env_config,
        eval_episode_days=config.eval_episode_days,
        pool_size=config.eval_window_pool_size,
        window_count=config.eval_window_count,
        seed=int(config.eval_window_seed),
    )
    eval_episode_df: pd.DataFrame | None = None
    eval_episode_dfs: tuple[pd.DataFrame, ...] | None = None
    eval_n_episodes = 1
    eval_protocol: dict[str, Any]
    if eval_window_pool is None:
        eval_episode_df = _build_fixed_eval_episode_df(
            train_df=train_df,
            env_config=env_config,
            eval_episode_days=config.eval_episode_days,
        )
        eval_protocol = {
            "mode": "fixed_tail_v1",
            "eval_freq": int(config.eval_freq),
            "eval_episode_days": int(config.eval_episode_days),
            "eval_window_start": str(pd.to_datetime(eval_episode_df["timestamp"].iloc[0]).isoformat()),
            "eval_window_end": str(pd.to_datetime(eval_episode_df["timestamp"].iloc[-1]).isoformat()),
            "model_source_default": "best",
        }
    else:
        eval_episode_dfs = tuple(eval_window_pool["episode_dfs"])
        eval_n_episodes = max(1, len(eval_episode_dfs))
        eval_windows_payload = {
            key: value
            for key, value in eval_window_pool.items()
            if key != "episode_dfs"
        }
        eval_windows_path.write_text(
            json.dumps(eval_windows_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        eval_protocol = {
            "mode": str(eval_window_pool["mode"]),
            "eval_freq": int(config.eval_freq),
            "eval_episode_days": int(config.eval_episode_days),
            "pool_size": int(eval_window_pool["pool_size"]),
            "window_count": int(eval_window_pool["window_count"]),
            "seed": int(eval_window_pool["seed"]),
            "selected_window_indices": list(eval_window_pool["selected_window_indices"]),
            "eval_windows_path": str(eval_windows_path).replace("\\", "/"),
            "model_source_default": "best",
        }

    env_fns = [
        make_train_env_factory(
            train_df=train_df,
            env_config=env_config,
            seed=config.seed + idx,
            episode_days=config.episode_days,
            history_steps=config.history_steps,
            observation_keys=observation_keys,
            normalizer=None,
            algo=config.algo,
            discrete_action_mapper=discrete_action_mapper,
            residual_policy_name=str(config.residual_policy) if residual_active else None,
            residual_scale=float(config.residual_scale),
            train_statistics=train_statistics,
        )
        for idx in range(config.n_envs)
    ]
    # 记录每个并行环境的 episode 回报/长度，便于论文复现实验曲线。
    wrapped_env_fns = []
    for idx, factory in enumerate(env_fns):
        monitor_path = run_dir / "train" / f"monitor_env{idx}.csv"

        def _make(mon_path=monitor_path, base_factory=factory):
            env = base_factory()
            return Monitor(env, filename=str(mon_path))

        wrapped_env_fns.append(_make)

    train_env = DummyVecEnv(wrapped_env_fns)
    use_vecnormalize = bool(config.vec_norm_obs or config.vec_norm_reward)
    if use_vecnormalize:
        train_env = VecNormalize(
            train_env,
            training=True,
            norm_obs=bool(config.vec_norm_obs),
            norm_reward=bool(config.vec_norm_reward),
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=float(config.gamma),
        )

    algo_cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG, "dqn": DQN}[config.algo]
    policy_kwargs = _sb3_policy_kwargs_for_backbone(backbone=config.backbone)
    policy_kwargs_serializable: dict[str, Any] = {}
    if policy_kwargs:
        extractor_cls = policy_kwargs.get("features_extractor_class")
        extractor_name = (
            str(getattr(extractor_cls, "__name__", "")) if extractor_cls is not None else ""
        )
        policy_kwargs_serializable = {
            "features_extractor_class": extractor_name,
            "features_extractor_kwargs": dict(policy_kwargs.get("features_extractor_kwargs", {})),
        }
    algo_kwargs: dict[str, Any] = {}
    if config.algo == "ppo":
        algo_kwargs.update(
            {
                "n_steps": int(config.ppo_n_steps),
                "gae_lambda": float(config.ppo_gae_lambda),
                "ent_coef": float(config.ppo_ent_coef),
                "clip_range": float(config.ppo_clip_range),
            }
        )
    if config.algo == "dqn":
        algo_kwargs["buffer_size"] = int(config.buffer_size)
        algo_kwargs["learning_starts"] = int(config.learning_starts)
        algo_kwargs["train_freq"] = (int(config.train_freq), "step")
        algo_kwargs["gradient_steps"] = int(config.gradient_steps)
        algo_kwargs["target_update_interval"] = int(config.dqn_target_update_interval)
        algo_kwargs["exploration_fraction"] = float(config.dqn_exploration_fraction)
        algo_kwargs["exploration_initial_eps"] = float(config.dqn_exploration_initial_eps)
        algo_kwargs["exploration_final_eps"] = float(config.dqn_exploration_final_eps)
    if config.algo in {"sac", "td3", "ddpg"}:
        algo_kwargs["buffer_size"] = int(config.buffer_size)
        algo_kwargs["learning_starts"] = int(config.learning_starts)
        algo_kwargs["train_freq"] = (int(config.train_freq), "step")
        algo_kwargs["gradient_steps"] = int(config.gradient_steps)
        algo_kwargs["tau"] = float(config.tau)
        # SB3 off-policy algorithms accept optimize_memory_usage (ReplayBuffer) in recent versions.
        # If an older SB3 is installed, we silently skip it for compatibility.
        try:
            import inspect

            algo_sig_params = inspect.signature(algo_cls).parameters
            if "optimize_memory_usage" in algo_sig_params:
                algo_kwargs["optimize_memory_usage"] = bool(config.optimize_memory_usage)
            if bool(algo_kwargs.get("optimize_memory_usage", False)):
                if "replay_buffer_kwargs" in algo_sig_params:
                    algo_kwargs["replay_buffer_kwargs"] = {"handle_timeout_termination": False}
                else:
                    algo_kwargs["optimize_memory_usage"] = False
        except Exception:
            pass
        if config.algo in {"td3", "ddpg"} and float(config.action_noise_std) > 0.0:
            algo_kwargs["action_noise"] = NormalActionNoise(
                mean=np.zeros(6, dtype=np.float32),
                sigma=np.full(6, float(config.action_noise_std), dtype=np.float32),
            )

    model = algo_cls(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        gamma=config.gamma,
        seed=config.seed,
        device=config.device,
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        verbose=1,
        **algo_kwargs,
    )
    model.set_logger(sb3_configure_logger(folder=str(run_dir / "train"), format_strings=["stdout", "csv"]))

    warm_start_summary: dict[str, Any] = {
        "enabled": bool(config.ppo_warm_start_enabled),
        "applied": False,
    }
    offpolicy_prefill_summary: dict[str, Any] = {
        "enabled": bool(config.offpolicy_prefill_enabled),
        "applied": False,
    }
    if bool(config.ppo_warm_start_enabled):
        if config.algo != "ppo":
            warm_start_summary["status"] = "skipped_non_ppo"
        else:
            if residual_active:
                warm_obs, warm_targets, warm_vec_env = _collect_residual_zero_warm_start_dataset(
                    train_df=train_df,
                    env_config=env_config,
                    config=config,
                    observation_keys=observation_keys,
                    train_statistics=train_statistics,
                    DummyVecEnv=DummyVecEnv,
                    VecNormalize=VecNormalize,
                    use_vecnormalize=use_vecnormalize,
                )
                warm_start_kwargs = {
                    "target_indices": (0, 1, 2, 3, 4, 5),
                    "target_action_keys": ("du_gt", "du_bes", "du_boiler", "du_abs", "du_ech", "du_tes"),
                    "mode": "residual_zero_delta_bc_v1",
                }
            else:
                warm_obs, warm_targets, warm_vec_env = _collect_easy_rule_warm_start_dataset(
                    train_df=train_df,
                    env_config=env_config,
                    config=config,
                    observation_keys=observation_keys,
                    DummyVecEnv=DummyVecEnv,
                    VecNormalize=VecNormalize,
                    use_vecnormalize=use_vecnormalize,
                )
                warm_start_kwargs = {
                    "target_indices": (2, 4),
                    "target_action_keys": ("u_boiler", "u_ech"),
                    "mode": "easy_rule_bc_v1",
                }
            try:
                if use_vecnormalize and warm_vec_env is not None:
                    _copy_vecnormalize_stats(warm_vec_env, train_env)
                warm_start_summary = _warm_start_ppo_from_easy_rule(
                    model=model,
                    observations=warm_obs,
                    targets=warm_targets,
                    config=config,
                    **warm_start_kwargs,
                )
                warm_start_summary["applied"] = True
            finally:
                if warm_vec_env is not None:
                    warm_vec_env.close()
    if bool(config.offpolicy_prefill_enabled):
        if config.algo not in {"sac", "td3", "ddpg", "dqn"}:
            offpolicy_prefill_summary["status"] = "skipped_non_offpolicy"
        else:
            offpolicy_prefill_summary = _prefill_offpolicy_replay_buffer(
                model=model,
                train_df=train_df,
                env_config=env_config,
                config=config,
                train_statistics=train_statistics,
                observation_keys=observation_keys,
                DummyVecEnv=DummyVecEnv,
                discrete_action_mapper=discrete_action_mapper,
            )

    eval_factory = make_eval_env_factory(
        eval_df=eval_episode_df,
        env_config=env_config,
        seed=config.seed,
        history_steps=config.history_steps,
        observation_keys=observation_keys,
        normalizer=None,
        algo=config.algo,
        discrete_action_mapper=discrete_action_mapper,
        residual_policy_name=str(config.residual_policy) if residual_active else None,
        residual_scale=float(config.residual_scale),
        train_statistics=train_statistics,
        eval_episode_dfs=eval_episode_dfs,
    )

    def _make_eval() -> Any:
        env = eval_factory()
        return Monitor(env, filename=str(run_dir / "train" / "eval_monitor.csv"))

    eval_env = DummyVecEnv([_make_eval])
    if use_vecnormalize:
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=bool(config.vec_norm_obs),
            norm_reward=False,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=float(config.gamma),
        )
        _copy_vecnormalize_stats(train_env, eval_env)

    payload = {
        "artifact_type": "sb3_policy",
        "algo": config.algo,
        "paper_model_label": paper_model_label,
        "policy": "MlpPolicy",
        "backbone": str(config.backbone),
        "history_steps": int(config.history_steps),
        "seed": int(config.seed),
        "train_year": int(TRAIN_YEAR),
        "episode_days": int(config.episode_days),
        "n_envs": int(config.n_envs),
        "total_timesteps": int(config.total_timesteps),
        "training_complete": False,
        "learning_rate": float(config.learning_rate),
        "batch_size": int(config.batch_size),
        "gamma": float(config.gamma),
        "vec_normalize": {
            "enabled": bool(use_vecnormalize),
            "norm_obs": bool(config.vec_norm_obs),
            "norm_reward": bool(config.vec_norm_reward),
            "clip_obs": 10.0,
            "clip_reward": 10.0,
        },
        "eval_protocol": eval_protocol,
        "warm_start": warm_start_summary,
        "offpolicy_prefill": offpolicy_prefill_summary,
        "residual": {
            "enabled": bool(residual_active),
            "requested": bool(config.residual_enabled),
            "policy": str(config.residual_policy),
            "scale": float(config.residual_scale),
        },
        "best_model_selection": {
            "mode": "reliability_shortfall_then_reward_v2" if bool(config.best_gate_enabled) else "reward_best_v1",
            "gate_enabled": bool(config.best_gate_enabled),
            "thresholds": {
                "electric": float(config.best_gate_electric_min),
                "heat": float(config.best_gate_heat_min),
                "cooling": float(config.best_gate_cool_min),
            },
            "history_path": str(eval_history_path).replace("\\", "/"),
        },
        "plateau_control": {
            "enabled": bool(config.plateau_control_enabled),
            "patience_evals": int(config.plateau_patience_evals),
            "lr_decay_factor": float(config.plateau_lr_decay_factor),
            "min_lr": float(config.plateau_min_lr),
            "early_stop_patience_evals": int(config.plateau_early_stop_patience_evals),
        },
        "ppo_hyperparameters": {
            "warm_start_enabled": bool(config.ppo_warm_start_enabled),
            "warm_start_samples": int(config.ppo_warm_start_samples),
            "warm_start_epochs": int(config.ppo_warm_start_epochs),
            "warm_start_batch_size": int(config.ppo_warm_start_batch_size),
            "warm_start_lr": float(config.ppo_warm_start_lr),
            "n_steps": int(config.ppo_n_steps),
            "gae_lambda": float(config.ppo_gae_lambda),
            "ent_coef": float(config.ppo_ent_coef),
            "clip_range": float(config.ppo_clip_range),
        },
        "offpolicy_hyperparameters": {
            "prefill_enabled": bool(config.offpolicy_prefill_enabled),
            "prefill_steps": int(config.offpolicy_prefill_steps),
            "prefill_policy": str(config.offpolicy_prefill_policy),
            "learning_starts": int(config.learning_starts),
            "train_freq": int(config.train_freq),
            "gradient_steps": int(config.gradient_steps),
            "tau": float(config.tau),
            "action_noise_std": float(config.action_noise_std),
            "buffer_size": int(config.buffer_size),
            "optimize_memory_usage": bool(config.optimize_memory_usage),
        },
        "dqn_hyperparameters": {
            "action_mode": str(config.dqn_action_mode),
            "target_update_interval": int(config.dqn_target_update_interval),
            "exploration_fraction": float(config.dqn_exploration_fraction),
            "exploration_initial_eps": float(config.dqn_exploration_initial_eps),
            "exploration_final_eps": float(config.dqn_exploration_final_eps),
        },
        "dqn_action_labels": []
        if discrete_action_mapper is None
        else list(discrete_action_mapper.action_labels),
        "buffer_size": int(config.buffer_size),
        "optimize_memory_usage": bool(config.optimize_memory_usage),
        "device": str(config.device),
        "observation_keys": list(observation_keys),
        "obs_norm": {
            "mode": "vecnormalize_v1" if use_vecnormalize else "raw_v1",
            "clip_value": 10.0,
        },
        "policy_kwargs": policy_kwargs_serializable,
        "model_path": str(model_path).replace("\\", "/"),
        "best_model_path": str(best_model_path).replace("\\", "/"),
        "best_reward_model_path": str(best_reward_model_path).replace("\\", "/"),
        "vecnormalize_path": (
            str(vecnormalize_path).replace("\\", "/") if use_vecnormalize else None
        ),
        "best_vecnormalize_path": (
            str(best_vecnormalize_path).replace("\\", "/") if use_vecnormalize else None
        ),
        "best_reward_vecnormalize_path": (
            str(best_reward_vecnormalize_path).replace("\\", "/") if use_vecnormalize else None
        ),
        "train_statistics_path": str(train_statistics_path).replace("\\", "/"),
        "run_dir": str(run_dir).replace("\\", "/"),
        "eval_windows_path": (
            str(eval_windows_path).replace("\\", "/")
            if eval_window_pool is not None
            else None
        ),
    }

    def _write_payload() -> None:
        checkpoint_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        train_summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        write_one_row_csv(run_dir / "train" / "summary_flat.csv", flatten_mapping(payload))

    def _save_vecnormalize_bundle(path: Path | None) -> None:
        if path is None or not isinstance(train_env, VecNormalize):
            return
        train_env.save(str(path))

    # 先保存一次初始模型与 checkpoint_json，便于 Kaggle 等环境中断时仍可评估/继续分析。
    model.save(str(model_path))
    _save_vecnormalize_bundle(vecnormalize_path if use_vecnormalize else None)
    _write_payload()

    class _LatestModelCallback(BaseCallback):
        def __init__(self, *, save_every_steps: int) -> None:
            super().__init__()
            self.save_every_steps = int(save_every_steps)
            self._last_saved = 0

        def _save_bundle(self) -> None:
            self.model.save(str(model_path))
            _save_vecnormalize_bundle(vecnormalize_path if use_vecnormalize else None)

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_saved < self.save_every_steps:
                return True
            self._save_bundle()
            self._last_saved = int(self.num_timesteps)
            return True

        def _on_training_end(self) -> None:
            self._save_bundle()

    class _ReliabilityAwareEvalCallback(BaseCallback):
        def __init__(
            self,
            *,
            eval_env: Any,
            n_eval_episodes: int,
            eval_freq: int,
            deterministic: bool,
            history_path: Path,
        ) -> None:
            super().__init__()
            self.eval_env = eval_env
            self.n_eval_episodes = int(max(1, n_eval_episodes))
            self.eval_freq = int(max(1, eval_freq))
            self.deterministic = bool(deterministic)
            self.history_path = history_path
            self.best_mean_reward = float("-inf")
            self.best_reward_mean = float("-inf")
            self.best_gate_passed = False
            self.best_gate_shortfall_total = float("inf")
            self.best_gate_shortfall_max = float("inf")
            self.best_snapshot: dict[str, Any] | None = None
            self.best_reward_snapshot: dict[str, Any] | None = None
            self.no_improve_evals = 0
            self.lr_decay_count = 0
            self.fine_tune_applied = False
            self.stop_requested = False
            self.stop_reason = ""
            self.current_learning_rate = float(config.learning_rate)
            self.plateau_events: list[dict[str, Any]] = []

        def _save_selection_bundle(self, *, path: Path, vecnorm_path: Path | None) -> None:
            self.model.save(str(path))
            _save_vecnormalize_bundle(vecnorm_path if use_vecnormalize else None)

        def _append_history(self, payload_item: dict[str, Any]) -> None:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload_item, ensure_ascii=False) + "\n")

        def _record_plateau_event(self, event: dict[str, Any]) -> dict[str, Any]:
            normalized = dict(event)
            self.plateau_events.append(normalized)
            if len(self.plateau_events) > 16:
                self.plateau_events = self.plateau_events[-16:]
            return normalized

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq != 0:
                return True
            if use_vecnormalize:
                _copy_vecnormalize_stats(train_env, self.eval_env)
            metrics = _evaluate_current_policy(
                model=self.model,
                eval_env=self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
            )
            gate_result = _build_reliability_gate_result(metrics=metrics, config=config)

            if float(metrics["mean_reward"]) > float(self.best_reward_mean):
                self.best_reward_mean = float(metrics["mean_reward"])
                self.best_reward_snapshot = {
                    "timesteps": int(self.num_timesteps),
                    "metrics": dict(metrics),
                    "gate": dict(gate_result),
                }
                self._save_selection_bundle(
                    path=best_reward_model_path,
                    vecnorm_path=best_reward_vecnormalize_path,
                )

            shortfall = dict(gate_result.get("shortfall") or {})
            current_shortfall_total = float(shortfall.get("total", 0.0))
            current_shortfall_max = float(shortfall.get("max", 0.0))
            current_rank = (
                1 if bool(gate_result["passed"]) else 0,
                -current_shortfall_total,
                -current_shortfall_max,
                float(metrics["mean_reward"]),
            )
            best_rank = (
                1 if bool(self.best_gate_passed) else 0,
                -float(self.best_gate_shortfall_total),
                -float(self.best_gate_shortfall_max),
                float(self.best_mean_reward),
            )
            improved = bool(current_rank > best_rank)
            if current_rank > best_rank:
                self.best_gate_passed = bool(gate_result["passed"])
                self.best_gate_shortfall_total = current_shortfall_total
                self.best_gate_shortfall_max = current_shortfall_max
                self.best_mean_reward = float(metrics["mean_reward"])
                self.best_snapshot = {
                    "timesteps": int(self.num_timesteps),
                    "metrics": dict(metrics),
                    "gate": dict(gate_result),
                }
                self._save_selection_bundle(
                    path=best_model_path,
                    vecnorm_path=best_vecnormalize_path,
                )
            if improved:
                self.no_improve_evals = 0
            else:
                self.no_improve_evals = int(self.no_improve_evals + 1)

            plateau_event: dict[str, Any] | None = None
            current_lr = _get_model_learning_rate(self.model, fallback=self.current_learning_rate)
            if bool(config.plateau_control_enabled):
                if (not improved) and (not self.fine_tune_applied) and self.no_improve_evals >= int(config.plateau_patience_evals):
                    new_lr = max(float(config.plateau_min_lr), current_lr * float(config.plateau_lr_decay_factor))
                    if new_lr < current_lr - 1e-12:
                        _set_model_learning_rate(self.model, learning_rate=new_lr)
                        self.current_learning_rate = float(new_lr)
                        self.lr_decay_count = int(self.lr_decay_count + 1)
                        self.fine_tune_applied = True
                        plateau_event = self._record_plateau_event(
                            {
                                "timesteps": int(self.num_timesteps),
                                "action": "low_lr_fine_tune",
                                "stale_evals": int(self.no_improve_evals),
                                "old_lr": float(current_lr),
                                "new_lr": float(new_lr),
                            }
                        )
                        self.no_improve_evals = 0
                        current_lr = float(new_lr)
                    else:
                        self.fine_tune_applied = True
                        self.current_learning_rate = float(current_lr)
                elif self.fine_tune_applied and self.no_improve_evals >= int(config.plateau_early_stop_patience_evals):
                    self.stop_requested = True
                    self.stop_reason = "plateau_after_low_lr_fine_tune"
                    plateau_event = self._record_plateau_event(
                        {
                            "timesteps": int(self.num_timesteps),
                            "action": "early_stop",
                            "stale_evals": int(self.no_improve_evals),
                            "learning_rate": float(current_lr),
                        }
                    )
                else:
                    self.current_learning_rate = float(current_lr)
            else:
                self.current_learning_rate = float(current_lr)

            history_item = {
                "timesteps": int(self.num_timesteps),
                "mean_reward": float(metrics["mean_reward"]),
                "mean_total_cost": float(metrics["mean_total_cost"]),
                "mean_violation_rate": float(metrics["mean_violation_rate"]),
                "mean_unmet_heat_mwh": float(metrics["mean_unmet_heat_mwh"]),
                "mean_unmet_cool_mwh": float(metrics["mean_unmet_cool_mwh"]),
                "reliability_mean": dict(metrics["reliability_mean"]),
                "reliability_min": dict(metrics["reliability_min"]),
                "gate": gate_result,
                "learning_rate": float(self.current_learning_rate),
                "improved": bool(improved),
                "plateau": {
                    "enabled": bool(config.plateau_control_enabled),
                    "fine_tune_applied": bool(self.fine_tune_applied),
                    "lr_decay_count": int(self.lr_decay_count),
                    "no_improve_evals": int(self.no_improve_evals),
                    "stop_requested": bool(self.stop_requested),
                    "stop_reason": str(self.stop_reason),
                },
            }
            if plateau_event is not None:
                history_item["plateau"]["event"] = plateau_event
            self._append_history(history_item)
            self.logger.record("eval/mean_reward", float(metrics["mean_reward"]))
            self.logger.record("eval/mean_total_cost", float(metrics["mean_total_cost"]))
            self.logger.record("eval/reliability_heat_min", float(metrics["reliability_min"]["heat"]))
            self.logger.record("eval/reliability_cooling_min", float(metrics["reliability_min"]["cooling"]))
            self.logger.record("eval/reliability_electric_min", float(metrics["reliability_min"]["electric"]))
            self.logger.record("eval/gate_passed", float(1.0 if gate_result["passed"] else 0.0))
            self.logger.record("eval/gate_shortfall_total", float((gate_result.get("shortfall") or {}).get("total", 0.0)))
            self.logger.record("eval/gate_shortfall_max", float((gate_result.get("shortfall") or {}).get("max", 0.0)))
            self.logger.record("train/current_learning_rate", float(self.current_learning_rate))
            self.logger.record("train/plateau_no_improve_evals", float(self.no_improve_evals))
            self.logger.record("train/plateau_fine_tune_applied", float(1.0 if self.fine_tune_applied else 0.0))
            self.logger.record("train/plateau_lr_decay_count", float(self.lr_decay_count))
            self.logger.record("train/early_stop_requested", float(1.0 if self.stop_requested else 0.0))
            return not self.stop_requested

    eval_callback = _ReliabilityAwareEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=int(eval_n_episodes),
        eval_freq=max(1, int(config.eval_freq) // max(1, int(config.n_envs))),
        deterministic=True,
        history_path=eval_history_path,
    )
    latest_callback = _LatestModelCallback(
        save_every_steps=max(10_000, int(config.eval_freq)),
    )
    callback = CallbackList([latest_callback, eval_callback])

    model.learn(total_timesteps=int(config.total_timesteps), callback=callback)
    model.save(str(model_path))
    _save_vecnormalize_bundle(vecnormalize_path if use_vecnormalize else None)
    if not best_model_path.exists():
        model.save(str(best_model_path))
    if use_vecnormalize and not best_vecnormalize_path.exists():
        _save_vecnormalize_bundle(best_vecnormalize_path)
    if not best_reward_model_path.exists():
        model.save(str(best_reward_model_path))
    if use_vecnormalize and not best_reward_vecnormalize_path.exists():
        _save_vecnormalize_bundle(best_reward_vecnormalize_path)
    payload["training_complete"] = True
    payload["training_timesteps_completed"] = int(getattr(model, "num_timesteps", config.total_timesteps))
    payload["training_wall_time_s"] = float(time.perf_counter() - train_start_time)
    payload["training_steps_per_second"] = float(
        float(payload["training_timesteps_completed"]) / max(1e-9, float(payload["training_wall_time_s"]))
    )
    payload["training_early_stopped"] = bool(getattr(eval_callback, "stop_requested", False))
    payload["training_stop_reason"] = str(getattr(eval_callback, "stop_reason", ""))
    payload["final_learning_rate"] = float(getattr(eval_callback, "current_learning_rate", config.learning_rate))
    best_reward = float(getattr(eval_callback, "best_mean_reward", float("-inf")))
    reward_leader = float(getattr(eval_callback, "best_reward_mean", float("-inf")))
    payload["best_mean_reward"] = None if not np.isfinite(best_reward) else best_reward
    payload["best_reward_mean_reward"] = None if not np.isfinite(reward_leader) else reward_leader
    if getattr(eval_callback, "best_snapshot", None) is not None:
        payload["best_model_selection"]["selected"] = eval_callback.best_snapshot
        payload["best_model_selection"]["gate_passed"] = bool(eval_callback.best_gate_passed)
    else:
        payload["best_model_selection"]["selected"] = None
        payload["best_model_selection"]["gate_passed"] = False
        payload["best_model_selection"]["fallback_reason"] = "no_eval_triggered"
    if getattr(eval_callback, "best_reward_snapshot", None) is not None:
        payload["best_model_selection"]["reward_leader"] = eval_callback.best_reward_snapshot
    payload["plateau_control"].update(
        {
            "fine_tune_applied": bool(getattr(eval_callback, "fine_tune_applied", False)),
            "lr_decay_count": int(getattr(eval_callback, "lr_decay_count", 0)),
            "current_learning_rate": float(getattr(eval_callback, "current_learning_rate", config.learning_rate)),
            "stopped_early": bool(getattr(eval_callback, "stop_requested", False)),
            "stop_reason": str(getattr(eval_callback, "stop_reason", "")),
            "events": list(getattr(eval_callback, "plateau_events", [])),
        }
    )
    progress_path = run_dir / "train" / "progress.csv"
    progress_df = None
    if progress_path.exists():
        try:
            progress_df = pd.read_csv(progress_path)
        except Exception:
            progress_df = None
    convergence_artifacts = write_learning_curve_artifacts(
        run_dir / "train",
        eval_history_rows=_load_eval_history_rows(eval_history_path),
        progress_df=progress_df,
        selected_snapshot=payload["best_model_selection"].get("selected"),
        reward_leader_snapshot=payload["best_model_selection"].get("reward_leader"),
        total_timesteps=int(payload["training_timesteps_completed"]),
    )
    payload["learning_curve_artifacts"] = convergence_artifacts.get("files", {})
    payload["convergence_summary"] = convergence_artifacts.get("summary", {})
    _write_payload()
    eval_env.close()
    train_env.close()
    return payload


def _resolve_sb3_model_path(
    *,
    checkpoint_json: Path,
    model_path_value: str,
    run_dir_value: str | None,
    default_filename: str = "sb3_model.zip",
) -> Path:
    candidates: list[Path] = []
    raw = Path(str(model_path_value))
    candidates.append(raw)

    checkpoint_dir = checkpoint_json.resolve().parent
    if not raw.is_absolute():
        candidates.append(checkpoint_dir / raw)

    if run_dir_value:
        run_dir_path = Path(str(run_dir_value))
        candidates.append(run_dir_path / "checkpoints" / default_filename)
        # 常见场景：checkpoint_json 位于 <run_dir>/checkpoints/baseline_policy.json，直接回退到该 run_dir。
        candidates.append(checkpoint_dir.parent / "checkpoints" / default_filename)

    candidates.append(checkpoint_dir / default_filename)

    for item in candidates:
        try:
            if item.exists():
                return item
        except OSError:
            continue

    searched = "\n".join(f"- {item.as_posix()}" for item in candidates)
    raise FileNotFoundError(
        f"无法定位 SB3 模型文件（{default_filename}）。\n"
        f"checkpoint_json={checkpoint_json.as_posix()}\n"
        f"model_path={model_path_value}\n"
        "已尝试路径：\n"
        f"{searched}"
    )


def _resolve_checkpoint_sidecar_path(
    *,
    checkpoint_json: Path,
    path_value: str,
    artifact_label: str,
) -> Path:
    """
    解析 checkpoint sidecar 文件路径（例如 train_statistics.json）。

    背景：
    - baseline_policy.json 中记录的路径通常是相对项目根目录的相对路径；
    - 导出到 Kaggle / 归档目录后，checkpoint 可能位于更深层级；
    - 直接 `Path(path_value)` 读取会依赖当前工作目录，迁移后容易失效。

    策略：
    1. 先尝试原始路径（兼容原始 run 目录）；
    2. 若是相对路径，则依次尝试以 checkpoint 所在目录及其上级目录为锚点重建路径。
    """
    raw = Path(str(path_value))
    checkpoint_resolved = checkpoint_json.resolve()
    candidates: list[Path] = [raw]

    if not raw.is_absolute():
        for parent in checkpoint_resolved.parents[:5]:
            candidates.append(parent / raw)

    deduped: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        normalized = item.as_posix()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)

    for item in deduped:
        try:
            if item.exists():
                return item
        except OSError:
            continue

    searched = "\n".join(f"- {item.as_posix()}" for item in deduped)
    raise FileNotFoundError(
        f"无法定位 {artifact_label}。\n"
        f"checkpoint_json={checkpoint_json.as_posix()}\n"
        f"path_value={path_value}\n"
        "已尝试路径：\n"
        f"{searched}"
    )


def _resolve_optional_checkpoint_sidecar_path(
    *,
    checkpoint_json: Path,
    path_value: str | None,
    artifact_label: str,
) -> Path | None:
    if not path_value:
        return None
    try:
        return _resolve_checkpoint_sidecar_path(
            checkpoint_json=checkpoint_json,
            path_value=str(path_value),
            artifact_label=artifact_label,
        )
    except FileNotFoundError:
        return None


def _resolve_sb3_eval_artifacts(
    *,
    checkpoint_json: Path,
    checkpoint_payload: dict[str, Any],
    model_source: str,
) -> tuple[Path, Path | None, str]:
    normalized_source = str(model_source).strip().lower()
    if normalized_source not in {"best", "last"}:
        raise ValueError("model_source 仅支持 best/last。")

    run_dir_value = checkpoint_payload.get("run_dir")
    if normalized_source == "best":
        model_candidates = [
            ("best", checkpoint_payload.get("best_model_path"), "best_model.zip"),
            ("last", checkpoint_payload.get("model_path"), "sb3_model.zip"),
        ]
        vecnorm_candidates = [
            checkpoint_payload.get("best_vecnormalize_path"),
            checkpoint_payload.get("vecnormalize_path"),
        ]
    else:
        model_candidates = [
            ("last", checkpoint_payload.get("model_path"), "sb3_model.zip"),
            ("best", checkpoint_payload.get("best_model_path"), "best_model.zip"),
        ]
        vecnorm_candidates = [
            checkpoint_payload.get("vecnormalize_path"),
            checkpoint_payload.get("best_vecnormalize_path"),
        ]

    resolved_model_path: Path | None = None
    resolved_source = normalized_source
    for candidate_source, candidate_path, default_filename in model_candidates:
        if not candidate_path:
            continue
        try:
            resolved_model_path = _resolve_sb3_model_path(
                checkpoint_json=checkpoint_json,
                model_path_value=str(candidate_path),
                run_dir_value=run_dir_value,
                default_filename=default_filename,
            )
            resolved_source = str(candidate_source)
            break
        except FileNotFoundError:
            continue

    if resolved_model_path is None:
        raise FileNotFoundError(
            "无法从 checkpoint_json 解析可用的 SB3 模型文件。"
        )

    resolved_vecnormalize_path = None
    for candidate_path in vecnorm_candidates:
        resolved_vecnormalize_path = _resolve_optional_checkpoint_sidecar_path(
            checkpoint_json=checkpoint_json,
            path_value=None if candidate_path is None else str(candidate_path),
            artifact_label="SB3 VecNormalize 统计文件",
        )
        if resolved_vecnormalize_path is not None:
            break
    return resolved_model_path, resolved_vecnormalize_path, resolved_source


def evaluate_sb3_policy(
    *,
    eval_df: pd.DataFrame,
    env_config: EnvConfig,
    checkpoint_json: str | Path,
    run_dir: str | Path,
    seed: int = 42,
    deterministic: bool = True,
    model_source: str = "best",
    device: str = "auto",
) -> dict[str, Any]:
    eval_start_time = time.perf_counter()
    _, _, PPO, SAC, TD3, DDPG, DQN, DummyVecEnv, VecNormalize = _require_sb3_modules()

    year = _extract_year(eval_df)
    if year != EVAL_YEAR:
        raise ValueError(f"评估必须使用 {EVAL_YEAR}，当前年份 {year}")

    checkpoint_json_path = Path(checkpoint_json)
    ckpt = json.loads(checkpoint_json_path.read_text(encoding="utf-8"))
    if ckpt.get("artifact_type") != "sb3_policy":
        raise ValueError("checkpoint_json 不是 sb3_policy。")

    algo = str(ckpt.get("algo", "")).strip().lower()
    if algo not in {"ppo", "sac", "td3", "ddpg", "dqn"}:
        raise ValueError(f"未知 algo: {algo}")

    algo_cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG, "dqn": DQN}[algo]
    resolved_model_path, resolved_vecnormalize_path, resolved_model_source = _resolve_sb3_eval_artifacts(
        checkpoint_json=checkpoint_json_path,
        checkpoint_payload=ckpt,
        model_source=model_source,
    )

    history_steps = int(ckpt.get("history_steps", 1))
    if history_steps <= 0:
        raise ValueError("checkpoint_json.history_steps 必须 > 0。")
    observation_keys = tuple(str(item) for item in (ckpt.get("observation_keys") or OBS_KEYS))
    residual_payload = ckpt.get("residual") or {}
    residual_enabled = bool(residual_payload.get("enabled", False))
    residual_policy_name = str(residual_payload.get("policy", "rule"))
    residual_scale = float(residual_payload.get("scale", 0.0))
    normalizer = None
    obs_norm = ckpt.get("obs_norm") or {}
    obs_norm_mode = ""
    if isinstance(obs_norm, dict):
        obs_norm_mode = str(obs_norm.get("mode", "")).strip().lower()
    train_statistics_path = ckpt.get("train_statistics_path")
    needs_train_statistics = (
        algo == "dqn"
        or (resolved_vecnormalize_path is None and obs_norm_mode in {"zscore_affine_v1", "affine_v1"})
        or (residual_enabled and residual_policy_name == "rule")
    )
    if needs_train_statistics:
        if not train_statistics_path:
            raise ValueError("checkpoint_json 缺少 train_statistics_path，无法重建评估所需 sidecar。")
        resolved_train_statistics_path = _resolve_checkpoint_sidecar_path(
            checkpoint_json=checkpoint_json_path,
            path_value=str(train_statistics_path),
            artifact_label="SB3 训练统计文件（train_statistics.json）",
        )
        train_statistics = json.loads(resolved_train_statistics_path.read_text(encoding="utf-8"))
        if resolved_vecnormalize_path is None and obs_norm_mode in {"zscore_affine_v1", "affine_v1"}:
            normalizer = _build_observation_normalizer(
                train_statistics=train_statistics,
                env_config=env_config,
                keys=observation_keys,
            )
    else:
        train_statistics = None

    discrete_action_mapper = None
    if algo == "dqn":
        dqn_hparams = ckpt.get("dqn_hyperparameters") or {}
        discrete_action_mapper = RuleBasedDiscreteActionMapper(
            env_config=env_config,
            train_statistics=train_statistics if train_statistics is not None else {},
            action_mode=str(dqn_hparams.get("action_mode", "rb_v1")),
        )

    output_run_dir = Path(run_dir)
    (output_run_dir / "eval").mkdir(parents=True, exist_ok=True)

    base_eval_env = DummyVecEnv(
        [
            make_eval_env_factory(
                eval_df=eval_df.reset_index(drop=True),
                env_config=env_config,
                seed=seed,
                history_steps=history_steps,
                observation_keys=observation_keys,
                normalizer=normalizer,
                algo=algo,
                discrete_action_mapper=discrete_action_mapper,
                residual_policy_name=residual_policy_name if residual_enabled else None,
                residual_scale=residual_scale,
                train_statistics=train_statistics,
            )
        ]
    )
    vec_env = base_eval_env
    if resolved_vecnormalize_path is not None:
        vec_env = VecNormalize.load(str(resolved_vecnormalize_path), base_eval_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = algo_cls.load(str(resolved_model_path), env=vec_env, device=device)
    observation = vec_env.reset()
    terminated = False
    total_reward = 0.0
    step_rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}

    while not terminated:
        action_vec, _ = model.predict(observation, deterministic=bool(deterministic))
        observation, rewards, dones, infos = vec_env.step(action_vec)
        reward = float(np.asarray(rewards).reshape(-1)[0])
        terminated = bool(np.asarray(dones).reshape(-1)[0])
        info = dict(infos[0] if infos else {})
        total_reward += float(reward)
        final_info = info
        log_row = {
            key: value
            for key, value in info.items()
            if key not in {"violation_flags", "diagnostic_flags", "state_diagnostic_flags", "terminal_observation"}
        }
        log_row["violation_flags_json"] = json.dumps(info.get("violation_flags", {}), ensure_ascii=False)
        log_row["diagnostic_flags_json"] = json.dumps(info.get("diagnostic_flags", {}), ensure_ascii=False)
        log_row["state_diagnostic_flags_json"] = json.dumps(
            info.get("state_diagnostic_flags", {}), ensure_ascii=False
        )
        step_rows.append(log_row)

    fallback_env = getattr(base_eval_env.envs[0], "env", base_eval_env.envs[0])
    summary = final_info.get("episode_summary", fallback_env.kpi.summary())
    summary = dict(summary)
    summary["total_reward_from_loop"] = float(total_reward)
    summary["mode"] = "eval"
    summary["year"] = int(EVAL_YEAR)
    summary["policy"] = "sb3"
    summary["algo"] = algo
    summary["paper_model_label"] = str(
        ckpt.get("paper_model_label")
        or _paper_model_label(
            algo=algo,
            dqn_action_mode=str((ckpt.get("dqn_hyperparameters") or {}).get("action_mode", "")),
            residual_enabled=residual_enabled,
            residual_policy=residual_policy_name,
        )
    )
    summary["backbone"] = str(ckpt.get("backbone", ""))
    summary["history_steps"] = int(history_steps)
    summary["seed"] = int(seed)
    summary["eval_wall_time_s"] = float(time.perf_counter() - eval_start_time)
    summary["eval_steps_per_second"] = float(
        len(step_rows) / max(1e-9, float(summary["eval_wall_time_s"]))
    )
    summary["checkpoint_json"] = str(checkpoint_json_path).replace("\\", "/")
    summary["model_source"] = str(resolved_model_source)
    summary["resolved_model_path"] = str(resolved_model_path).replace("\\", "/")
    summary["residual"] = {
        "enabled": bool(residual_enabled),
        "policy": str(residual_policy_name),
        "scale": float(residual_scale),
    }
    if resolved_vecnormalize_path is not None:
        summary["resolved_vecnormalize_path"] = str(resolved_vecnormalize_path).replace("\\", "/")

    step_df = pd.DataFrame(step_rows)
    step_df.to_csv(output_run_dir / "eval" / "step_log.csv", index=False)
    diagnostic_artifacts = _write_eval_diagnostics(
        eval_dir=output_run_dir / "eval",
        step_df=step_df,
        env_config=env_config,
    )
    if diagnostic_artifacts:
        summary["diagnostic_artifacts"] = diagnostic_artifacts
    (output_run_dir / "eval" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_paper_eval_artifacts(
        output_run_dir / "eval",
        summary=summary,
        step_log=step_df,
        dt_h=float(env_config.dt_hours),
    )
    vec_env.close()
    return summary


def load_dataframe(path: str | Path) -> pd.DataFrame:
    return load_exogenous_data(path)
