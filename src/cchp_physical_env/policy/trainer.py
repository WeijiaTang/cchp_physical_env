# Ref: docs/spec/task.md
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.data import TRAIN_YEAR, dump_statistics_json, make_episode_sampler
from ..env.cchp_env import CCHPPhysicalEnv, EnvConfig
from ..pipeline.sequence import (
    DEFAULT_SEQUENCE_ACTION_KEYS,
    DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS,
    SequenceWindowBuffer,
    normalized_action_vector_to_env_action_dict,
)
from .checkpoint import resolve_torch_device, save_policy
from .models import SUPPORTED_POLICY_BACKBONES, build_policy_network

_SEQUENCE_CHECKPOINT_EVERY_STEPS = 100_000
_NORM_EPS = 1e-6


def _require_torch_modules():
    try:
        import torch
        import torch.nn.utils
        from torch.optim import AdamW
    except ModuleNotFoundError as error:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 torch，无法进行 Task-007 深度策略训练。"
        ) from error
    return torch, AdamW


@dataclass(slots=True)
class SequenceTrainerConfig:
    policy_backbone: str
    history_steps: int = 16
    episode_days: int = 14
    train_steps: int = 4096
    batch_size: int = 128
    update_epochs: int = 4
    gamma: float = 0.99
    clip_ratio: float = 0.2
    entropy_coef: float = 0.001
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = self.policy_backbone.strip().lower()
        if normalized not in SUPPORTED_POLICY_BACKBONES:
            raise ValueError(
                f"不支持的 policy_backbone: {self.policy_backbone}，支持 {SUPPORTED_POLICY_BACKBONES}"
            )
        self.policy_backbone = normalized
        if self.history_steps <= 0:
            raise ValueError("history_steps 必须 > 0。")
        if self.episode_days < 7 or self.episode_days > 30:
            raise ValueError("episode_days 必须在 [7,30]。")
        if self.train_steps <= 0:
            raise ValueError("train_steps 必须 > 0。")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须 > 0。")
        if self.update_epochs <= 0:
            raise ValueError("update_epochs 必须 > 0。")
        if self.lr <= 0.0:
            raise ValueError("lr 必须 > 0。")


@dataclass(slots=True)
class RolloutBatch:
    windows: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    rewards: np.ndarray

    @property
    def n_steps(self) -> int:
        return int(self.rewards.shape[0])


class SequencePolicyTrainer:
    """Task-007：PPO-style 序列策略训练器。"""

    def __init__(
        self,
        *,
        train_df: pd.DataFrame,
        train_statistics: dict,
        env_config: EnvConfig,
        config: SequenceTrainerConfig,
        run_root: str | Path = "runs",
    ) -> None:
        year = sorted({int(value.year) for value in pd.to_datetime(train_df["timestamp"])})
        if year != [TRAIN_YEAR]:
            raise ValueError(f"深度策略训练仅支持 {TRAIN_YEAR}，当前年份集合: {year}")

        self.torch, adam_w_cls = _require_torch_modules()
        self.train_df = train_df
        self.train_statistics = train_statistics
        self.env_config = env_config
        self.config = config
        self.device = resolve_torch_device(config.device)
        try:
            self.torch.manual_seed(int(self.config.seed))
            if str(self.device).startswith("cuda") and getattr(self.torch, "cuda", None) is not None:
                self.torch.cuda.manual_seed_all(int(self.config.seed))
        except Exception:
            pass
        self.rng = np.random.default_rng(config.seed)

        self.feature_keys = DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS
        self.action_keys = DEFAULT_SEQUENCE_ACTION_KEYS
        self.window_buffer = SequenceWindowBuffer(
            history_steps=self.config.history_steps,
            feature_keys=self.feature_keys,
            action_feature_keys=self.action_keys,
        )

        self.obs_norm_payload, self._obs_offset, self._obs_scale = _build_sequence_obs_norm(
            feature_keys=self.feature_keys,
            train_statistics=self.train_statistics,
            env_config=self.env_config,
        )

        model_kwargs = dict(self.config.model_kwargs)
        model_kwargs.setdefault("history_steps", int(self.config.history_steps))
        self.model = build_policy_network(
            policy_backbone=self.config.policy_backbone,
            n_features=self.window_buffer.n_features,
            n_actions=len(self.action_keys),
            model_kwargs=model_kwargs,
        ).to(self.device)
        self.optimizer = adam_w_cls(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = self._build_scheduler()

        self.run_dir = self._create_run_directory(run_root=run_root)
        dump_statistics_json(
            self.train_statistics, self.run_dir / "train" / "train_statistics.json"
        )
        self.checkpoint_json = self.run_dir / "checkpoints" / "baseline_policy.json"

    def _create_run_directory(self, run_root: str | Path) -> Path:
        root = Path(run_root)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root / f"{stamp}_train_sequence_{self.config.policy_backbone}"
        (run_dir / "train").mkdir(parents=True, exist_ok=True)
        (run_dir / "eval").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        return run_dir

    def _normalize_window(self, window: np.ndarray) -> np.ndarray:
        """对序列窗口的观测部分做 affine 归一化（动作部分不处理）。"""

        normalized = np.asarray(window, dtype=np.float32).copy()
        n_obs = len(self.feature_keys)
        normalized[:, :n_obs] = (normalized[:, :n_obs] - self._obs_offset) / self._obs_scale
        return normalized

    def _build_checkpoint_metadata(self, *, total_env_steps: int, update_idx: int, training_complete: bool) -> dict[str, Any]:
        model_kwargs = {
            "history_steps": int(self.config.history_steps),
            **dict(self.config.model_kwargs),
        }
        return {
            "artifact_type": "sequence_torch_policy",
            "policy_name": "sequence_rule",
            "sequence_adapter": self.config.policy_backbone,
            "policy_backbone": self.config.policy_backbone,
            "history_steps": int(self.config.history_steps),
            "n_features": int(self.window_buffer.n_features),
            "n_actions": int(len(self.action_keys)),
            "feature_keys": list(self.feature_keys),
            "action_keys": list(self.action_keys),
            "model_kwargs": model_kwargs,
            "obs_norm": dict(self.obs_norm_payload),
            "total_env_steps": int(total_env_steps),
            "update_idx": int(update_idx),
            "training_complete": bool(training_complete),
        }

    def _write_checkpoint_json(
        self,
        *,
        model_checkpoint_path: Path,
        total_env_steps: int,
        update_idx: int,
        training_complete: bool,
    ) -> None:
        checkpoint_payload = {
            "artifact_type": "sequence_torch_policy",
            "policy_name": "sequence_rule",
            "sequence_adapter": self.config.policy_backbone,
            "history_steps": int(self.config.history_steps),
            "seed": int(self.config.seed),
            "train_year": TRAIN_YEAR,
            "train_steps": int(self.config.train_steps),
            "batch_size": int(self.config.batch_size),
            "lr": float(self.config.lr),
            "train_statistics_path": str(self.run_dir / "train" / "train_statistics.json"),
            "model_checkpoint_path": str(model_checkpoint_path),
            "device": self.device,
            "total_env_steps": int(total_env_steps),
            "update_idx": int(update_idx),
            "training_complete": bool(training_complete),
        }
        self.checkpoint_json.write_text(
            json.dumps(checkpoint_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _build_scheduler(self):
        try:
            from transformers import get_cosine_schedule_with_warmup
        except ModuleNotFoundError:
            return None

        total_batches = math.ceil(self.config.train_steps / self.config.batch_size)
        total_updates = max(1, total_batches * self.config.update_epochs)
        warmup_steps = max(1, int(total_updates * 0.1))
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )

    def _compute_discounted_returns(
        self, rewards: list[float], dones: list[bool]
    ) -> np.ndarray:
        returns = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for index in reversed(range(len(rewards))):
            if dones[index]:
                running = 0.0
            running = rewards[index] + self.config.gamma * running
            returns[index] = running
        return returns

    def _collect_rollout(
        self, *, episode_df: pd.DataFrame, seed: int, max_steps: int
    ) -> RolloutBatch:
        env = CCHPPhysicalEnv(exogenous_df=self.train_df, config=self.env_config, seed=seed)
        observation, _ = env.reset(seed=seed, episode_df=episode_df)
        previous_action = {key: 0.0 for key in self.action_keys}
        self.window_buffer.reset(observation, previous_action=previous_action)

        windows: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        old_log_probs: list[float] = []
        rewards: list[float] = []
        dones: list[bool] = []

        terminated = False
        step_count = 0
        self.model.eval()
        with self.torch.no_grad():
            while not terminated and step_count < max_steps:
                window_raw = self.window_buffer.current_window()
                window_np = self._normalize_window(window_raw)
                window_tensor = (
                    self.torch.as_tensor(window_np, dtype=self.torch.float32, device=self.device)
                    .unsqueeze(0)
                )
                dist = self.model.action_distribution(window_tensor)
                sampled = dist.sample()
                action_norm = self.torch.clamp(sampled, -1.0, 1.0)
                log_prob = dist.log_prob(action_norm).sum(dim=-1)

                action_norm_np = action_norm.squeeze(0).detach().cpu().numpy().astype(np.float32)
                action_dict = normalized_action_vector_to_env_action_dict(
                    action_norm_np, action_keys=self.action_keys
                )
                next_observation, reward, terminated, _, _ = env.step(action_dict)

                windows.append(window_np.astype(np.float32))
                actions.append(action_norm_np)
                old_log_probs.append(float(log_prob.item()))
                rewards.append(float(reward))
                dones.append(bool(terminated))

                step_count += 1
                previous_action = action_dict
                observation = next_observation
                self.window_buffer.push(
                    observation, previous_action=previous_action
                )

        returns = self._compute_discounted_returns(rewards=rewards, dones=dones)
        advantages = returns - float(np.mean(returns))
        advantages = advantages / (float(np.std(advantages)) + 1e-8)

        return RolloutBatch(
            windows=np.asarray(windows, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            old_log_probs=np.asarray(old_log_probs, dtype=np.float32),
            advantages=np.asarray(advantages, dtype=np.float32),
            returns=np.asarray(returns, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
        )

    def _ppo_update(self, rollout: RolloutBatch) -> dict[str, float]:
        self.model.train()
        windows = self.torch.as_tensor(rollout.windows, dtype=self.torch.float32, device=self.device)
        actions = self.torch.as_tensor(rollout.actions, dtype=self.torch.float32, device=self.device)
        old_log_probs = self.torch.as_tensor(
            rollout.old_log_probs, dtype=self.torch.float32, device=self.device
        )
        advantages = self.torch.as_tensor(
            rollout.advantages, dtype=self.torch.float32, device=self.device
        )

        n_steps = rollout.n_steps
        losses: list[float] = []
        entropy_terms: list[float] = []
        for _ in range(self.config.update_epochs):
            indices = self.torch.randperm(n_steps, device=self.device)
            for start in range(0, n_steps, self.config.batch_size):
                batch_idx = indices[start : start + self.config.batch_size]
                if batch_idx.numel() == 0:
                    continue
                dist = self.model.action_distribution(windows[batch_idx])
                new_log_prob = dist.log_prob(actions[batch_idx]).sum(dim=-1)
                ratio = self.torch.exp(new_log_prob - old_log_probs[batch_idx])
                unclipped = ratio * advantages[batch_idx]
                clipped = self.torch.clamp(
                    ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                ) * advantages[batch_idx]
                policy_loss = -self.torch.min(unclipped, clipped).mean()
                entropy = dist.entropy().sum(dim=-1).mean()
                loss = policy_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                losses.append(float(policy_loss.detach().cpu().item()))
                entropy_terms.append(float(entropy.detach().cpu().item()))

        return {
            "policy_loss": float(np.mean(losses)) if losses else 0.0,
            "entropy": float(np.mean(entropy_terms)) if entropy_terms else 0.0,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def train(self) -> dict[str, Any]:
        sampler = make_episode_sampler(
            self.train_df, episode_days=self.config.episode_days, seed=self.config.seed
        )
        total_env_steps = 0
        update_idx = 0
        rows: list[dict[str, Any]] = []
        checkpoint_every = int(
            min(
                _SEQUENCE_CHECKPOINT_EVERY_STEPS,
                max(2_000, int(self.config.train_steps // 5)),
            )
        )
        next_checkpoint_step = checkpoint_every
        latest_checkpoint_path = self.run_dir / "checkpoints" / "sequence_policy_latest.pt"

        tqdm = None
        try:
            from tqdm.auto import tqdm as _tqdm  # type: ignore

            tqdm = _tqdm
        except Exception:
            tqdm = None

        pbar = None
        if tqdm is not None:
            pbar = tqdm(
                total=int(self.config.train_steps),
                desc=f"sequence_train({self.config.policy_backbone})",
                unit="step",
                dynamic_ncols=True,
            )
            pbar.update(0)
        else:
            last_print = 0
            print_every = max(1, int(self.config.train_steps // 20))

        while total_env_steps < self.config.train_steps:
            _, episode_df = next(sampler)
            remaining = self.config.train_steps - total_env_steps
            rollout = self._collect_rollout(
                episode_df=episode_df,
                seed=self.config.seed + update_idx,
                max_steps=min(len(episode_df), remaining),
            )
            update_metrics = self._ppo_update(rollout)
            total_env_steps += rollout.n_steps

            if pbar is not None:
                pbar.update(int(rollout.n_steps))
                try:
                    pbar.set_postfix(
                        {
                            "r_mean": float(np.mean(rollout.rewards)),
                            "loss": float(update_metrics.get("policy_loss", 0.0)),
                            "lr": float(update_metrics.get("lr", 0.0)),
                        },
                        refresh=False,
                    )
                except Exception:
                    pass
            else:
                if total_env_steps - last_print >= print_every:
                    last_print = total_env_steps
                    print(
                        f"[sequence_train] steps={total_env_steps}/{self.config.train_steps} "
                        f"update={update_idx} r_mean={float(np.mean(rollout.rewards)):.3f} "
                        f"loss={float(update_metrics.get('policy_loss', 0.0)):.5f} "
                        f"lr={float(update_metrics.get('lr', 0.0)):.6f}"
                    )

            rows.append(
                {
                    "update_idx": int(update_idx),
                    "rollout_steps": int(rollout.n_steps),
                    "reward_mean": float(np.mean(rollout.rewards)),
                    "reward_sum": float(np.sum(rollout.rewards)),
                    "return_mean": float(np.mean(rollout.returns)),
                    "policy_loss": update_metrics["policy_loss"],
                    "entropy": update_metrics["entropy"],
                    "lr": update_metrics["lr"],
                    "total_env_steps": int(total_env_steps),
                }
            )

            if total_env_steps >= next_checkpoint_step:
                metadata = self._build_checkpoint_metadata(
                    total_env_steps=total_env_steps,
                    update_idx=update_idx,
                    training_complete=False,
                )
                save_policy(
                    model=self.model,
                    checkpoint_path=latest_checkpoint_path,
                    metadata=metadata,
                )
                self._write_checkpoint_json(
                    model_checkpoint_path=latest_checkpoint_path,
                    total_env_steps=total_env_steps,
                    update_idx=update_idx,
                    training_complete=False,
                )
                next_checkpoint_step += checkpoint_every

            update_idx += 1

        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

        pd.DataFrame(rows).to_csv(self.run_dir / "train" / "sequence_updates.csv", index=False)

        final_checkpoint_path = self.run_dir / "checkpoints" / "sequence_policy.pt"
        final_metadata = self._build_checkpoint_metadata(
            total_env_steps=total_env_steps,
            update_idx=update_idx,
            training_complete=True,
        )
        model_checkpoint_path = save_policy(
            model=self.model,
            checkpoint_path=final_checkpoint_path,
            metadata=final_metadata,
        )
        self._write_checkpoint_json(
            model_checkpoint_path=model_checkpoint_path,
            total_env_steps=total_env_steps,
            update_idx=update_idx,
            training_complete=True,
        )

        summary = {
            "mode": "train",
            "policy": "sequence_rule",
            "sequence_adapter": self.config.policy_backbone,
            "train_year": TRAIN_YEAR,
            "history_steps": int(self.config.history_steps),
            "train_steps": int(total_env_steps),
            "updates": int(update_idx),
            "batch_size": int(self.config.batch_size),
            "lr": float(self.config.lr),
            "model_checkpoint_path": str(model_checkpoint_path),
            "checkpoint_json_path": str(self.checkpoint_json),
            "run_dir": str(self.run_dir),
            "observation_feature_keys": list(self.feature_keys),
            "obs_norm": dict(self.obs_norm_payload),
        }
        (self.run_dir / "train" / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary


def _build_sequence_obs_norm(
    *,
    feature_keys: tuple[str, ...],
    train_statistics: dict,
    env_config: EnvConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """
    为序列深度策略构建观测归一化参数（默认启用）。

    规则：
    - 外生变量：使用 training_statistics 的 mean/std（仅训练年拟合）
    - 内部状态：使用 EnvConfig 的物理边界做中心化/缩放
    """
    stats = dict(train_statistics.get("stats", {}) or {})

    offsets: list[float] = []
    scales: list[float] = []
    for key in feature_keys:
        offset = 0.0
        scale = 1.0
        if key in stats:
            entry = stats.get(key, {}) or {}
            offset = float(entry.get("mean", 0.0))
            scale = float(entry.get("std", 1.0))
        elif key == "soc_bes":
            low = float(env_config.bes_soc_min)
            high = float(env_config.bes_soc_max)
            offset = 0.5 * (low + high)
            scale = 0.5 * max(_NORM_EPS, high - low)
        elif key == "gt_on":
            offset = 0.5
            scale = 0.5
        elif key == "gt_state":
            offset = 1.0
            scale = 1.0
        elif key == "p_gt_prev_mw":
            cap = float(env_config.p_gt_cap_mw)
            offset = 0.5 * cap
            scale = 0.5 * max(_NORM_EPS, cap)
        elif key in {"gt_ramp_headroom_up_mw", "gt_ramp_headroom_down_mw"}:
            ramp = float(env_config.gt_ramp_mw_per_step)
            offset = 0.5 * ramp
            scale = 0.5 * max(_NORM_EPS, ramp)
        elif key == "e_tes_mwh":
            cap = float(env_config.e_tes_cap_mwh)
            offset = 0.5 * cap
            scale = 0.5 * max(_NORM_EPS, cap)
        elif key == "t_tes_hot_k":
            # TES 热端温度大致在 [hrsg_water_inlet_k, hrsg_water_inlet_k + 60] 附近。
            center = float(env_config.hrsg_water_inlet_k) + 30.0
            offset = center
            scale = 30.0
        elif key == "abs_drive_margin_k":
            offset = 0.0
            scale = max(2.0, float(env_config.abs_gate_scale_k))
        elif key in {"sin_t", "cos_t", "sin_week", "cos_week"}:
            offset = 0.0
            scale = 1.0

        if not np.isfinite(scale) or abs(scale) < _NORM_EPS:
            scale = 1.0
        offsets.append(float(offset))
        scales.append(float(scale))

    offset_arr = np.asarray(offsets, dtype=np.float32)
    scale_arr = np.asarray(scales, dtype=np.float32)
    payload = {
        "kind": "affine_v1",
        "observation_feature_keys": list(feature_keys),
        "offset": offsets,
        "scale": scales,
    }
    return payload, offset_arr, scale_arr


def train_sequence_policy(
    *,
    train_df: pd.DataFrame,
    train_statistics: dict,
    env_config: EnvConfig,
    trainer_config: SequenceTrainerConfig,
    run_root: str | Path = "runs",
) -> dict[str, Any]:
    trainer = SequencePolicyTrainer(
        train_df=train_df,
        train_statistics=train_statistics,
        env_config=env_config,
        config=trainer_config,
        run_root=run_root,
    )
    return trainer.train()
