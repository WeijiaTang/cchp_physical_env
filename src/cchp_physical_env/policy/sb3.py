# Ref: docs/spec/task.md (Task-ID: 011)
# Ref: docs/spec/architecture.md (Pattern: Policy / Optional Dependency Integration)
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from ..core.data import (
    EVAL_YEAR,
    TRAIN_YEAR,
    compute_training_statistics,
    load_exogenous_data,
    make_episode_sampler,
)
from ..core.reporting import flatten_mapping, write_one_row_csv, write_paper_eval_artifacts
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
    "p_gt_prev_mw",
    "gt_ramp_headroom_up_mw",
    "gt_ramp_headroom_down_mw",
    "e_tes_mwh",
    "t_tes_hot_k",
    "abs_drive_margin_k",
    "sin_t",
    "cos_t",
    "sin_week",
    "cos_week",
)


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 torch。SAC/TD3/DDPG 训练需要 PyTorch。\n"
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


def _require_sb3_modules():
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import DDPG, PPO, SAC, TD3
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ModuleNotFoundError as error:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 sb3 相关依赖。请先安装（并确保 torch 可用）：\n"
            "  uv pip install -e '.[sb3]'\n"
            "或：\n"
            "  uv pip install stable-baselines3 gymnasium\n"
            "然后再运行 sb3-train/sb3-eval。"
        ) from error
    return gym, spaces, PPO, SAC, TD3, DDPG, DummyVecEnv, VecNormalize


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
    ppo_warm_start_enabled: bool = False
    residual_enabled: bool = False
    ppo_warm_start_samples: int = 16_384
    ppo_warm_start_epochs: int = 4
    ppo_warm_start_batch_size: int = 256
    ppo_warm_start_lr: float = 1e-4
    offpolicy_prefill_enabled: bool = False
    offpolicy_prefill_steps: int = 0
    ppo_n_steps: int = 2048
    ppo_gae_lambda: float = 0.95
    ppo_ent_coef: float = 0.0
    ppo_clip_range: float = 0.2
    learning_starts: int = 5_000
    train_freq: int = 1
    gradient_steps: int = 1
    tau: float = 0.005
    action_noise_std: float = 0.1
    buffer_size: int = 50_000
    optimize_memory_usage: bool = True
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        self.algo = str(self.algo).strip().lower()
        if self.algo not in {"ppo", "sac", "td3", "ddpg"}:
            raise ValueError("sb3 algo 仅支持 ppo/sac/td3/ddpg。")
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
        if self.residual_enabled:
            raise ValueError("residual_enabled 当前尚未实现，请保持 false。")
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
        self.seed = int(self.seed)
        self.device = str(self.device).strip().lower()


def _build_spaces(*, history_steps: int, obs_dim: int):
    _, spaces, *_ = _require_sb3_modules()
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
    fixed_episode_df: pd.DataFrame | None = None,
    fixed_episode_dfs: tuple[pd.DataFrame, ...] | None = None,
) -> Callable[[], Any]:
    gym, *_ = _require_sb3_modules()
    observation_space, action_space = _build_spaces(
        history_steps=history_steps, obs_dim=len(observation_keys)
    )

    class _CCHPSB3TrainEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = observation_space
            self.action_space = action_space
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
            action_dict = _action_vector_to_env_action(action)
            next_obs, reward, terminated, truncated, info = self.env.step(action_dict)
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
        ) -> None:
            shape = getattr(observation_space, "shape", None)
            if not shape or len(shape) != 2:
                raise ValueError("TransformerWindowExtractor 仅支持 Box(K,D) observation_space。")
            _, obs_dim = int(shape[0]), int(shape[1])
            super().__init__(observation_space, features_dim=int(d_model))
            self.input_norm = nn.LayerNorm(obs_dim)
            self.input_proj = nn.Linear(obs_dim, int(d_model))
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
            "features_extractor_kwargs": {"d_model": 128, "n_head": 4, "n_layer": 3, "dropout": 0.1},
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


def _warm_start_ppo_from_easy_rule(
    *,
    model: Any,
    observations: np.ndarray,
    targets: np.ndarray,
    config: SB3TrainConfig,
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

    model.policy.train()
    for _ in range(int(config.ppo_warm_start_epochs)):
        permutation = np.random.permutation(dataset_size)
        for start in range(0, dataset_size, batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch_obs = torch.as_tensor(observations[batch_indices], dtype=torch.float32, device=device)
            batch_targets = torch.as_tensor(targets[batch_indices], dtype=torch.float32, device=device)
            predicted_actions = _extract_actor_mean_actions(model, batch_obs)
            loss = F.mse_loss(predicted_actions[:, [2, 4]], batch_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=1.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            loss_history.append(last_loss)

    return {
        "enabled": True,
        "mode": "easy_rule_bc_v1",
        "target_action_keys": ["u_boiler", "u_ech"],
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
    observation_keys: tuple[str, ...],
    DummyVecEnv: Any,
) -> dict[str, Any]:
    from ..pipeline.runner import EasyRulePolicy

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
        )
        for idx in range(int(config.n_envs))
    ]
    prefill_env = DummyVecEnv(env_fns)
    observation = np.asarray(prefill_env.reset(), dtype=np.float32)
    carrier_envs = _resolve_observation_carrier_envs(prefill_env)
    if not carrier_envs:
        prefill_env.close()
        raise RuntimeError("off-policy prefill 失败：未找到底层 observation carrier。")

    easy_rule = EasyRulePolicy(
        p_gt_cap_mw=float(env_config.p_gt_cap_mw),
        q_boiler_cap_mw=float(env_config.q_boiler_cap_mw),
        q_ech_cap_mw=float(env_config.q_ech_cap_mw),
    )

    collected_steps = 0
    episode_resets = 0
    try:
        while collected_steps < target_steps:
            action_batch = np.asarray(
                [
                    _action_dict_to_vector(
                        easy_rule.act(getattr(carrier_env, "observation"))
                    )
                    for carrier_env in carrier_envs
                ],
                dtype=np.float32,
            )
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
                action_batch,
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
        "mode": "easy_rule_replay_prefill_v1",
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
    gym, _, PPO, SAC, TD3, DDPG, DummyVecEnv, VecNormalize = _require_sb3_modules()
    del gym

    try:
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
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
    vecnormalize_path = run_dir / "checkpoints" / "vecnormalize.pkl"
    best_vecnormalize_path = run_dir / "checkpoints" / "best_vecnormalize.pkl"
    eval_windows_path = run_dir / "checkpoints" / "eval_windows.json"

    train_statistics = compute_training_statistics(train_df)
    train_statistics_path.write_text(
        json.dumps(train_statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    observation_keys = tuple(OBS_KEYS)
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

    algo_cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG}[config.algo]
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
            warm_obs, warm_targets, warm_vec_env = _collect_easy_rule_warm_start_dataset(
                train_df=train_df,
                env_config=env_config,
                config=config,
                observation_keys=observation_keys,
                DummyVecEnv=DummyVecEnv,
                VecNormalize=VecNormalize,
                use_vecnormalize=use_vecnormalize,
            )
            try:
                if use_vecnormalize and warm_vec_env is not None:
                    _copy_vecnormalize_stats(warm_vec_env, train_env)
                warm_start_summary = _warm_start_ppo_from_easy_rule(
                    model=model,
                    observations=warm_obs,
                    targets=warm_targets,
                    config=config,
                )
                warm_start_summary["applied"] = True
            finally:
                if warm_vec_env is not None:
                    warm_vec_env.close()
    if bool(config.offpolicy_prefill_enabled):
        if config.algo not in {"sac", "td3", "ddpg"}:
            offpolicy_prefill_summary["status"] = "skipped_non_offpolicy"
        else:
            offpolicy_prefill_summary = _prefill_offpolicy_replay_buffer(
                model=model,
                train_df=train_df,
                env_config=env_config,
                config=config,
                observation_keys=observation_keys,
                DummyVecEnv=DummyVecEnv,
            )

    eval_factory = make_eval_env_factory(
        eval_df=eval_episode_df,
        env_config=env_config,
        seed=config.seed,
        history_steps=config.history_steps,
        observation_keys=observation_keys,
        normalizer=None,
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
            "learning_starts": int(config.learning_starts),
            "train_freq": int(config.train_freq),
            "gradient_steps": int(config.gradient_steps),
            "tau": float(config.tau),
            "action_noise_std": float(config.action_noise_std),
            "buffer_size": int(config.buffer_size),
            "optimize_memory_usage": bool(config.optimize_memory_usage),
        },
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
        "vecnormalize_path": (
            str(vecnormalize_path).replace("\\", "/") if use_vecnormalize else None
        ),
        "best_vecnormalize_path": (
            str(best_vecnormalize_path).replace("\\", "/") if use_vecnormalize else None
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

    class _BestModelEvalCallback(EvalCallback):
        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                _reset_eval_window_cursor(self.eval_env)
            previous_best = float(self.best_mean_reward)
            continue_training = super()._on_step()
            if self.best_mean_reward > previous_best:
                _save_vecnormalize_bundle(best_vecnormalize_path if use_vecnormalize else None)
            return continue_training

    eval_callback = _BestModelEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=int(eval_n_episodes),
        eval_freq=max(1, int(config.eval_freq) // max(1, int(config.n_envs))),
        log_path=str(run_dir / "train" / "eval_callback"),
        best_model_save_path=str(run_dir / "checkpoints"),
        deterministic=True,
        render=False,
        verbose=1,
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
    payload["training_complete"] = True
    payload["training_timesteps_completed"] = int(getattr(model, "num_timesteps", config.total_timesteps))
    best_reward = float(getattr(eval_callback, "best_mean_reward", float("-inf")))
    payload["best_mean_reward"] = None if not np.isfinite(best_reward) else best_reward
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
    _, _, PPO, SAC, TD3, DDPG, DummyVecEnv, VecNormalize = _require_sb3_modules()

    year = _extract_year(eval_df)
    if year != EVAL_YEAR:
        raise ValueError(f"评估必须使用 {EVAL_YEAR}，当前年份 {year}")

    checkpoint_json_path = Path(checkpoint_json)
    ckpt = json.loads(checkpoint_json_path.read_text(encoding="utf-8"))
    if ckpt.get("artifact_type") != "sb3_policy":
        raise ValueError("checkpoint_json 不是 sb3_policy。")

    algo = str(ckpt.get("algo", "")).strip().lower()
    if algo not in {"ppo", "sac", "td3", "ddpg"}:
        raise ValueError(f"未知 algo: {algo}")

    algo_cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG}[algo]
    resolved_model_path, resolved_vecnormalize_path, resolved_model_source = _resolve_sb3_eval_artifacts(
        checkpoint_json=checkpoint_json_path,
        checkpoint_payload=ckpt,
        model_source=model_source,
    )

    history_steps = int(ckpt.get("history_steps", 1))
    if history_steps <= 0:
        raise ValueError("checkpoint_json.history_steps 必须 > 0。")
    observation_keys = tuple(str(item) for item in (ckpt.get("observation_keys") or OBS_KEYS))
    normalizer = None
    obs_norm = ckpt.get("obs_norm") or {}
    obs_norm_mode = ""
    if isinstance(obs_norm, dict):
        obs_norm_mode = str(obs_norm.get("mode", "")).strip().lower()
    if resolved_vecnormalize_path is None and obs_norm_mode in {"zscore_affine_v1", "affine_v1"}:
        train_statistics_path = ckpt.get("train_statistics_path")
        if not train_statistics_path:
            raise ValueError("checkpoint_json 启用了 obs_norm 但缺少 train_statistics_path。")
        resolved_train_statistics_path = _resolve_checkpoint_sidecar_path(
            checkpoint_json=checkpoint_json_path,
            path_value=str(train_statistics_path),
            artifact_label="SB3 训练统计文件（train_statistics.json）",
        )
        train_statistics = json.loads(resolved_train_statistics_path.read_text(encoding="utf-8"))
        normalizer = _build_observation_normalizer(
            train_statistics=train_statistics,
            env_config=env_config,
            keys=observation_keys,
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
            if key not in {"violation_flags", "diagnostic_flags", "terminal_observation"}
        }
        log_row["violation_flags_json"] = json.dumps(info.get("violation_flags", {}), ensure_ascii=False)
        log_row["diagnostic_flags_json"] = json.dumps(info.get("diagnostic_flags", {}), ensure_ascii=False)
        step_rows.append(log_row)

    fallback_env = getattr(base_eval_env.envs[0], "env", base_eval_env.envs[0])
    summary = final_info.get("episode_summary", fallback_env.kpi.summary())
    summary = dict(summary)
    summary["total_reward_from_loop"] = float(total_reward)
    summary["mode"] = "eval"
    summary["year"] = int(EVAL_YEAR)
    summary["policy"] = "sb3"
    summary["algo"] = algo
    summary["backbone"] = str(ckpt.get("backbone", ""))
    summary["history_steps"] = int(history_steps)
    summary["seed"] = int(seed)
    summary["checkpoint_json"] = str(checkpoint_json_path).replace("\\", "/")
    summary["model_source"] = str(resolved_model_source)
    summary["resolved_model_path"] = str(resolved_model_path).replace("\\", "/")
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
