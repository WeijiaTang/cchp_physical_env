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
from ..env.cchp_env import CCHPPhysicalEnv, EnvConfig


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


def _timestamped_run_dir(run_root: str | Path, *, mode: str, algo: str) -> Path:
    root = Path(run_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{stamp}_{mode}_sb3_{algo}"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _extract_year(df: pd.DataFrame) -> int:
    years = sorted({int(value.year) for value in pd.to_datetime(df["timestamp"])})
    if len(years) != 1:
        raise ValueError(f"仅支持单年数据，当前年份集合: {years}")
    return years[0]


def _observation_to_gym(observation: dict[str, float]) -> dict[str, np.ndarray]:
    return {key: np.asarray([float(value)], dtype=np.float32) for key, value in observation.items()}


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


def _build_spaces():
    _, spaces, *_ = _require_sb3_modules()
    # action: [u_gt,u_bes,u_boiler,u_abs,u_ech,u_tes]
    action_space = spaces.Box(
        low=np.asarray([-1.0, -1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
        high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        shape=(6,),
        dtype=np.float32,
    )
    # observation: all scalar floats (wrapped as shape=(1,) arrays)
    # 这里采用保守的大范围有限边界，避免依赖外部统计文件；论文口径建议再加归一化消融。
    obs_low = np.asarray([-1e9], dtype=np.float32)
    obs_high = np.asarray([1e9], dtype=np.float32)
    observation_space = spaces.Dict(
        {
            "p_dem_mw": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "qh_dem_mw": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "qc_dem_mw": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "pv_mw": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "wt_mw": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "price_e": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "price_gas": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "carbon_tax": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "t_amb_k": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "sp_pa": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "rh_pct": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "wind_speed": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "wind_direction": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "ghi_wm2": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "dni_wm2": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "dhi_wm2": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "soc_bes": spaces.Box(np.asarray([0.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32), shape=(1,), dtype=np.float32),
            "gt_on": spaces.Box(np.asarray([0.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32), shape=(1,), dtype=np.float32),
            "e_tes_mwh": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "t_tes_hot_k": spaces.Box(obs_low, obs_high, shape=(1,), dtype=np.float32),
            "sin_t": spaces.Box(np.asarray([-1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32), shape=(1,), dtype=np.float32),
            "cos_t": spaces.Box(np.asarray([-1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32), shape=(1,), dtype=np.float32),
            "sin_week": spaces.Box(np.asarray([-1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32), shape=(1,), dtype=np.float32),
            "cos_week": spaces.Box(np.asarray([-1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32), shape=(1,), dtype=np.float32),
        }
    )
    return observation_space, action_space


def make_train_env_factory(
    *, train_df: pd.DataFrame, env_config: EnvConfig, seed: int, episode_days: int
) -> Callable[[], Any]:
    gym, *_ = _require_sb3_modules()
    observation_space, action_space = _build_spaces()

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

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            del options
            if seed is not None:
                self.rng = np.random.default_rng(int(seed))
            _, episode_df = next(self.sampler)
            observation, _ = self.env.reset(seed=int(seed or 0), episode_df=episode_df)
            self.observation = observation
            return _observation_to_gym(observation), {}

        def step(self, action):
            if self.observation is None:
                raise RuntimeError("环境未 reset。")
            action_dict = _action_vector_to_env_action(action)
            next_obs, reward, terminated, truncated, info = self.env.step(action_dict)
            self.observation = next_obs
            return _observation_to_gym(next_obs), float(reward), bool(terminated), bool(truncated), dict(info)

    return _CCHPSB3TrainEnv


def train_sb3_policy(
    *,
    train_df: pd.DataFrame,
    env_config: EnvConfig,
    config: SB3TrainConfig,
    run_root: str | Path = "runs",
) -> dict[str, Any]:
    gym, _, PPO, SAC, TD3, DDPG, DummyVecEnv = _require_sb3_modules()
    del gym

    year = _extract_year(train_df)
    if year != TRAIN_YEAR:
        raise ValueError(f"训练必须使用 {TRAIN_YEAR}，当前年份 {year}")

    run_dir = _timestamped_run_dir(run_root=run_root, mode="train", algo=config.algo)
    train_summary_path = run_dir / "train" / "summary.json"
    checkpoint_path = run_dir / "checkpoints" / "baseline_policy.json"
    model_path = run_dir / "checkpoints" / "sb3_model.zip"

    env_fns = [
        make_train_env_factory(
            train_df=train_df,
            env_config=env_config,
            seed=config.seed + idx,
            episode_days=config.episode_days,
        )
        for idx in range(config.n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    algo_cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG}[config.algo]
    model = algo_cls(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        gamma=config.gamma,
        seed=config.seed,
        device=config.device,
        verbose=1,
    )
    model.learn(total_timesteps=int(config.total_timesteps))
    model.save(str(model_path))

    payload = {
        "artifact_type": "sb3_policy",
        "algo": config.algo,
        "policy": "MultiInputPolicy",
        "seed": int(config.seed),
        "train_year": int(TRAIN_YEAR),
        "episode_days": int(config.episode_days),
        "n_envs": int(config.n_envs),
        "total_timesteps": int(config.total_timesteps),
        "learning_rate": float(config.learning_rate),
        "batch_size": int(config.batch_size),
        "gamma": float(config.gamma),
        "device": str(config.device),
        "model_path": str(model_path).replace("\\", "/"),
        "run_dir": str(run_dir).replace("\\", "/"),
    }
    checkpoint_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    train_summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
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

    output_run_dir = Path(run_dir)
    (output_run_dir / "eval").mkdir(parents=True, exist_ok=True)

    env = CCHPPhysicalEnv(exogenous_df=eval_df, config=env_config, seed=seed)
    observation, _ = env.reset(seed=seed, episode_df=eval_df)
    terminated = False
    total_reward = 0.0
    step_rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}

    while not terminated:
        obs_gym = _observation_to_gym(observation)
        action_vec, _ = model.predict(obs_gym, deterministic=bool(deterministic))
        action_dict = _action_vector_to_env_action(action_vec)
        observation, reward, terminated, _, info = env.step(action_dict)
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
    summary["seed"] = int(seed)
    summary["checkpoint_json"] = str(Path(checkpoint_json)).replace("\\", "/")

    (output_run_dir / "eval" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(step_rows).to_csv(output_run_dir / "eval" / "step_log.csv", index=False)
    return summary


def load_dataframe(path: str | Path) -> pd.DataFrame:
    return load_exogenous_data(path)

