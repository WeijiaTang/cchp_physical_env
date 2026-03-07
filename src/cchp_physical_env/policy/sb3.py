# Ref: docs/spec/task.md (Task-ID: 011)
# Ref: docs/spec/architecture.md (Pattern: Policy / Optional Dependency Integration)
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..core.data import EVAL_YEAR, TRAIN_YEAR, load_exogenous_data, make_episode_sampler
from ..core.reporting import flatten_mapping, write_one_row_csv, write_paper_eval_artifacts
from ..env.cchp_env import CCHPPhysicalEnv, EnvConfig

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
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
    "e_tes_mwh",
    "t_tes_hot_k",
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
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ModuleNotFoundError as error:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 sb3 相关依赖。请先安装（并确保 torch 可用）：\n"
            "  uv pip install -e '.[sb3]'\n"
            "或：\n"
            "  uv pip install stable-baselines3 gymnasium\n"
            "然后再运行 sb3-train/sb3-eval。"
        ) from error
    return gym, spaces, PPO, SAC, TD3, DDPG, DummyVecEnv


def _timestamped_run_dir(
    run_root: str | Path, *, mode: str, algo: str, backbone: str, history_steps: int
) -> Path:
    root = Path(run_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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


def _observation_dict_to_vector(observation: dict[str, float]) -> np.ndarray:
    return np.asarray([float(observation[key]) for key in OBS_KEYS], dtype=np.float32)


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
        self.seed = int(self.seed)
        self.device = str(self.device).strip().lower()


def _build_spaces(*, history_steps: int):
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
    obs_dim = len(OBS_KEYS)
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
) -> Callable[[], Any]:
    gym, *_ = _require_sb3_modules()
    observation_space, action_space = _build_spaces(history_steps=history_steps)

    class _CCHPSB3TrainEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = observation_space
            self.action_space = action_space
            self.rng = np.random.default_rng(seed)
            self.sampler = make_episode_sampler(
                train_df, episode_days=episode_days, seed=int(seed)
            )
            self.env = CCHPPhysicalEnv(exogenous_df=train_df, config=env_config, seed=int(seed))
            self.observation: dict[str, float] | None = None
            self.buffer = WindowBuffer(history_steps=int(history_steps), obs_dim=len(OBS_KEYS))

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            del options
            if seed is not None:
                self.rng = np.random.default_rng(int(seed))
            _, episode_df = next(self.sampler)
            observation, _ = self.env.reset(seed=int(seed or 0), episode_df=episode_df)
            self.observation = observation
            vector = _observation_dict_to_vector(observation)
            window = self.buffer.reset(vector)
            return window.copy(), {}

        def step(self, action):
            if self.observation is None:
                raise RuntimeError("环境未 reset。")
            action_dict = _action_vector_to_env_action(action)
            next_obs, reward, terminated, truncated, info = self.env.step(action_dict)
            self.observation = next_obs
            vector = _observation_dict_to_vector(next_obs)
            window = self.buffer.push(vector)
            return window.copy(), float(reward), bool(terminated), bool(truncated), dict(info)

    return _CCHPSB3TrainEnv


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


def train_sb3_policy(
    *,
    train_df: pd.DataFrame,
    env_config: EnvConfig,
    config: SB3TrainConfig,
    run_root: str | Path = "runs",
) -> dict[str, Any]:
    gym, _, PPO, SAC, TD3, DDPG, DummyVecEnv = _require_sb3_modules()
    del gym

    try:
        from stable_baselines3.common.logger import configure as sb3_configure_logger
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as error:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 stable-baselines3 训练日志模块。请确认已安装：uv pip install -e '.[sb3]'"
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
    checkpoint_path = run_dir / "checkpoints" / "baseline_policy.json"
    model_path = run_dir / "checkpoints" / "sb3_model.zip"

    env_fns = [
        make_train_env_factory(
            train_df=train_df,
            env_config=env_config,
            seed=config.seed + idx,
            episode_days=config.episode_days,
            history_steps=config.history_steps,
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

    vec_env = DummyVecEnv(wrapped_env_fns)

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
    model = algo_cls(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        gamma=config.gamma,
        seed=config.seed,
        device=config.device,
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        verbose=1,
    )
    model.set_logger(sb3_configure_logger(folder=str(run_dir / "train"), format_strings=["stdout", "csv"]))
    model.learn(total_timesteps=int(config.total_timesteps))
    model.save(str(model_path))

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
        "learning_rate": float(config.learning_rate),
        "batch_size": int(config.batch_size),
        "gamma": float(config.gamma),
        "device": str(config.device),
        "observation_keys": list(OBS_KEYS),
        "policy_kwargs": policy_kwargs_serializable,
        "model_path": str(model_path).replace("\\", "/"),
        "run_dir": str(run_dir).replace("\\", "/"),
    }
    checkpoint_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    train_summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_one_row_csv(run_dir / "train" / "summary_flat.csv", flatten_mapping(payload))
    return payload


def evaluate_sb3_policy(
    *,
    eval_df: pd.DataFrame,
    env_config: EnvConfig,
    checkpoint_json: str | Path,
    run_dir: str | Path,
    seed: int = 42,
    deterministic: bool = True,
    device: str = "auto",
) -> dict[str, Any]:
    _, _, PPO, SAC, TD3, DDPG, _ = _require_sb3_modules()

    year = _extract_year(eval_df)
    if year != EVAL_YEAR:
        raise ValueError(f"评估必须使用 {EVAL_YEAR}，当前年份 {year}")

    ckpt = json.loads(Path(checkpoint_json).read_text(encoding="utf-8"))
    if ckpt.get("artifact_type") != "sb3_policy":
        raise ValueError("checkpoint_json 不是 sb3_policy。")

    algo = str(ckpt.get("algo", "")).strip().lower()
    if algo not in {"ppo", "sac", "td3", "ddpg"}:
        raise ValueError(f"未知 algo: {algo}")
    model_path = ckpt.get("model_path")
    if not model_path:
        raise ValueError("checkpoint_json 缺少 model_path。")

    algo_cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG}[algo]
    model = algo_cls.load(str(model_path), device=device)

    history_steps = int(ckpt.get("history_steps", 1))
    if history_steps <= 0:
        raise ValueError("checkpoint_json.history_steps 必须 > 0。")
    obs_dim = len(OBS_KEYS)
    buffer = WindowBuffer(history_steps=history_steps, obs_dim=obs_dim)

    output_run_dir = Path(run_dir)
    (output_run_dir / "eval").mkdir(parents=True, exist_ok=True)

    env = CCHPPhysicalEnv(exogenous_df=eval_df, config=env_config, seed=seed)
    observation, _ = env.reset(seed=seed, episode_df=eval_df)
    window = buffer.reset(_observation_dict_to_vector(observation)).copy()
    terminated = False
    total_reward = 0.0
    step_rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}

    while not terminated:
        action_vec, _ = model.predict(window, deterministic=bool(deterministic))
        action_dict = _action_vector_to_env_action(action_vec)
        observation, reward, terminated, _, info = env.step(action_dict)
        window = buffer.push(_observation_dict_to_vector(observation)).copy()
        total_reward += float(reward)
        final_info = dict(info)
        log_row = {key: value for key, value in info.items() if key not in {"violation_flags", "diagnostic_flags"}}
        log_row["violation_flags_json"] = json.dumps(info.get("violation_flags", {}), ensure_ascii=False)
        log_row["diagnostic_flags_json"] = json.dumps(info.get("diagnostic_flags", {}), ensure_ascii=False)
        step_rows.append(log_row)

    summary = final_info.get("episode_summary", env.kpi.summary())
    summary = dict(summary)
    summary["total_reward_from_loop"] = float(total_reward)
    summary["mode"] = "eval"
    summary["year"] = int(EVAL_YEAR)
    summary["policy"] = "sb3"
    summary["algo"] = algo
    summary["backbone"] = str(ckpt.get("backbone", ""))
    summary["history_steps"] = int(history_steps)
    summary["seed"] = int(seed)
    summary["checkpoint_json"] = str(Path(checkpoint_json)).replace("\\", "/")

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
        dt_h=float(env_config.dt_hours),
    )
    return summary


def load_dataframe(path: str | Path) -> pd.DataFrame:
    return load_exogenous_data(path)
