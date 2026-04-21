from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.data import EVAL_YEAR, TRAIN_YEAR
from ..env.cchp_env import CCHPPhysicalEnv, EnvConfig
from .pafc_td3 import _ACTION_BOUNDS, _action_vector_to_dict, load_pafc_td3_predictor
from .sb3 import (
    RuleBasedDiscreteActionMapper,
    WindowBuffer,
    _build_observation_normalizer,
    _extract_year,
    _observation_dict_to_vector,
    _resolve_checkpoint_sidecar_path,
    _require_sb3_modules,
    _resolve_sb3_eval_artifacts,
    make_eval_env_factory,
)

_HYBRID_ACTION_KEYS = ("u_gt", "u_bes", "u_boiler", "u_abs", "u_ech", "u_tes")
_HYBRID_MODEL_LABEL = "DPAR"
_HYBRID_MODEL_NAME = "DQN-PAFC Anchored Refinement"


def _normalize_action_key_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
    elif isinstance(value, (list, tuple, set)):
        tokens = [str(token).strip() for token in value if str(token).strip()]
    else:
        tokens = [str(value).strip()]
    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = str(token).strip().lower().replace("-", "_")
        if not key or key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    return tuple(normalized)


def _clip_action_value(action_key: str, value: float) -> float:
    low, high = _ACTION_BOUNDS[str(action_key)]
    return float(np.clip(float(value), float(low), float(high)))


def _normalize_action_scale_map(value: object) -> dict[str, float]:
    if value is None:
        return {}
    items: list[tuple[str, object]] = []
    if isinstance(value, Mapping):
        items = [(str(key), raw_value) for key, raw_value in value.items()]
    elif isinstance(value, str):
        tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
        for token in tokens:
            if "=" not in token:
                raise ValueError(
                    f"refine_action_scales 项 {token!r} 缺少 '='，示例: u_boiler=1.0,u_abs=0.05"
                )
            key, raw_value = token.split("=", 1)
            items.append((key, raw_value))
    else:
        raise ValueError("refine_action_scales 仅支持 dict 或 'u_key=value,...' 字符串。")

    normalized: dict[str, float] = {}
    for raw_key, raw_value in items:
        key = str(raw_key).strip().lower().replace("-", "_")
        if key not in _HYBRID_ACTION_KEYS:
            raise ValueError(f"refine_action_scales 包含未知动作键: {raw_key!r}")
        scale = float(raw_value)
        if scale < 0.0 or scale > 1.0:
            raise ValueError(f"{key} 的 scale 必须在 [0,1]，收到 {scale}")
        normalized[key] = float(scale)
    return normalized


def _timestamped_run_dir(run_root: str | Path) -> Path:
    root = Path(run_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"{stamp}_train_hybrid_pafc"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


@dataclass(slots=True)
class HybridPAFCConfig:
    dqn_checkpoint_path: str | Path
    pafc_checkpoint_path: str | Path
    discrete_hold_steps: int = 1
    residual_scale: float = 1.0
    refine_action_keys: tuple[str, ...] = field(default_factory=lambda: ("u_boiler",))
    refine_action_scales: dict[str, float] = field(default_factory=dict)
    dqn_model_source: str = "best"
    dqn_deterministic: bool = True
    use_safe_residual: bool = False
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        self.dqn_checkpoint_path = str(Path(self.dqn_checkpoint_path))
        self.pafc_checkpoint_path = str(Path(self.pafc_checkpoint_path))
        self.discrete_hold_steps = int(self.discrete_hold_steps)
        self.residual_scale = float(self.residual_scale)
        self.refine_action_keys = _normalize_action_key_tuple(self.refine_action_keys)
        self.refine_action_scales = _normalize_action_scale_map(self.refine_action_scales)
        self.dqn_model_source = str(self.dqn_model_source).strip().lower()
        self.dqn_deterministic = bool(self.dqn_deterministic)
        self.use_safe_residual = bool(self.use_safe_residual)
        self.seed = int(self.seed)
        self.device = str(self.device).strip().lower()
        if self.discrete_hold_steps <= 0:
            raise ValueError("discrete_hold_steps 必须 > 0。")
        if self.residual_scale < 0.0 or self.residual_scale > 1.0:
            raise ValueError("residual_scale 必须在 [0,1]。")
        if self.dqn_model_source not in {"best", "last"}:
            raise ValueError("dqn_model_source 仅支持 best/last。")
        if not self.refine_action_keys and self.refine_action_scales:
            self.refine_action_keys = tuple(self.refine_action_scales.keys())
        if not self.refine_action_keys:
            self.refine_action_keys = ("u_boiler",)
        invalid_keys = [key for key in self.refine_action_keys if key not in _HYBRID_ACTION_KEYS]
        if invalid_keys:
            raise ValueError(f"refine_action_keys 包含未知动作键: {invalid_keys}")
        for key in self.refine_action_keys:
            self.refine_action_scales.setdefault(key, float(self.residual_scale))


class _DQNAnchorPredictor:
    def __init__(
        self,
        *,
        checkpoint_json: str | Path,
        eval_df: pd.DataFrame,
        env_config: EnvConfig,
        model_source: str,
        deterministic: bool,
        device: str,
        seed: int,
    ) -> None:
        _, _, _, _, _, _, DQN, DummyVecEnv, VecNormalize = _require_sb3_modules()
        checkpoint_json_path = Path(checkpoint_json)
        ckpt = json.loads(checkpoint_json_path.read_text(encoding="utf-8"))
        if str(ckpt.get("artifact_type", "")).strip().lower() != "sb3_policy":
            raise ValueError("DQN anchor checkpoint 必须是 sb3_policy json。")
        algo = str(ckpt.get("algo", "")).strip().lower()
        if algo != "dqn":
            raise ValueError(f"当前 hybrid 高层仅支持 DQN，收到 algo={algo!r}")
        self.history_steps = int(ckpt.get("history_steps", 1))
        if self.history_steps <= 0:
            raise ValueError("DQN checkpoint history_steps 必须 > 0。")
        self.observation_keys = tuple(str(item) for item in (ckpt.get("observation_keys") or ()))
        if not self.observation_keys:
            raise ValueError("DQN checkpoint 缺少 observation_keys。")
        train_statistics_path = ckpt.get("train_statistics_path")
        if not train_statistics_path:
            raise ValueError("DQN checkpoint 缺少 train_statistics_path。")
        train_statistics_resolved = _resolve_checkpoint_sidecar_path(
            checkpoint_json=checkpoint_json_path,
            path_value=str(train_statistics_path),
            artifact_label="DQN train_statistics",
        )
        train_statistics = json.loads(train_statistics_resolved.read_text(encoding="utf-8"))
        normalizer = None
        obs_norm = ckpt.get("obs_norm") or {}
        if str((obs_norm or {}).get("mode", "")).strip().lower() in {"zscore_affine_v1", "affine_v1"}:
            normalizer = _build_observation_normalizer(
                train_statistics=train_statistics,
                env_config=env_config,
                keys=self.observation_keys,
            )
        self.normalizer = normalizer
        dqn_hparams = ckpt.get("dqn_hyperparameters") or {}
        self.discrete_action_mapper = RuleBasedDiscreteActionMapper(
            env_config=env_config,
            train_statistics=train_statistics,
            action_mode=str(dqn_hparams.get("action_mode", "rb_v1")),
        )
        self.action_labels = tuple(self.discrete_action_mapper.action_labels)
        base_eval_env = DummyVecEnv(
            [
                make_eval_env_factory(
                    eval_df=eval_df.reset_index(drop=True),
                    env_config=env_config,
                    seed=int(seed),
                    history_steps=self.history_steps,
                    observation_keys=self.observation_keys,
                    normalizer=normalizer,
                    algo="dqn",
                    discrete_action_mapper=self.discrete_action_mapper,
                    train_statistics=train_statistics,
                )
            ]
        )
        resolved_model_path, resolved_vecnormalize_path, _ = _resolve_sb3_eval_artifacts(
            checkpoint_json=checkpoint_json_path,
            checkpoint_payload=ckpt,
            model_source=model_source,
        )
        self.vec_env = None
        if resolved_vecnormalize_path is not None:
            self.vec_env = VecNormalize.load(str(resolved_vecnormalize_path), base_eval_env)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
        self.model = DQN.load(
            str(resolved_model_path),
            env=self.vec_env if self.vec_env is not None else base_eval_env,
            device=device,
        )
        self.buffer = WindowBuffer(
            history_steps=self.history_steps,
            obs_dim=len(self.observation_keys),
        )
        self.deterministic = bool(deterministic)
        self._fresh_reset = True

    def reset(self, observation: Mapping[str, float]) -> None:
        obs_vector = _observation_dict_to_vector(observation, keys=self.observation_keys)
        if self.normalizer is not None:
            obs_vector = self.normalizer.apply(obs_vector)
        self.buffer.reset(obs_vector)
        self._fresh_reset = True

    def _current_model_observation(self) -> np.ndarray:
        window = self.buffer.window.astype(np.float32, copy=True)
        model_obs = window.reshape(1, self.history_steps, len(self.observation_keys))
        if self.vec_env is not None:
            model_obs = self.vec_env.normalize_obs(model_obs)
        return np.asarray(model_obs, dtype=np.float32)

    def consume_observation(self, observation: Mapping[str, float]) -> None:
        if self._fresh_reset:
            self._fresh_reset = False
            return
        obs_vector = _observation_dict_to_vector(observation, keys=self.observation_keys)
        if self.normalizer is not None:
            obs_vector = self.normalizer.apply(obs_vector)
        self.buffer.push(obs_vector)

    def predict_anchor(self, observation: Mapping[str, float]) -> tuple[dict[str, float], dict[str, Any]]:
        model_obs = self._current_model_observation()
        action_vec, _ = self.model.predict(model_obs, deterministic=bool(self.deterministic))
        action_index = int(np.asarray(action_vec).reshape(-1)[0])
        anchor_action = dict(self.discrete_action_mapper.decode(action_index, observation))
        action_label = str(self.action_labels[action_index])
        return anchor_action, {
            "anchor_action_index": int(action_index),
            "anchor_action_label": action_label,
        }


class _HybridPolicyController:
    def __init__(
        self,
        *,
        eval_df: pd.DataFrame,
        env_config: EnvConfig,
        config: HybridPAFCConfig,
    ) -> None:
        self.config = config
        self.anchor_predictor = _DQNAnchorPredictor(
            checkpoint_json=config.dqn_checkpoint_path,
            eval_df=eval_df,
            env_config=env_config,
            model_source=config.dqn_model_source,
            deterministic=config.dqn_deterministic,
            device=config.device,
            seed=config.seed,
        )
        self.pafc_predictor, self.pafc_metadata = load_pafc_td3_predictor(
            checkpoint_path=config.pafc_checkpoint_path,
            device=config.device,
            env_config=env_config,
        )
        self.safe_predictor = None
        self.safe_checkpoint_path = ""
        if bool(config.use_safe_residual):
            safe_checkpoint = str(
                self.pafc_metadata.get("frozen_action_safe_checkpoint_path", "")
            ).strip()
            if safe_checkpoint:
                self.safe_predictor, _ = load_pafc_td3_predictor(
                    checkpoint_path=safe_checkpoint,
                    device=config.device,
                    env_config=env_config,
                )
                self.safe_checkpoint_path = str(Path(safe_checkpoint).resolve()).replace("\\", "/")
        self.refine_action_keys = tuple(
            key for key in config.refine_action_keys if key in _HYBRID_ACTION_KEYS
        )
        self.refine_action_scales = {
            key: float(config.refine_action_scales.get(key, config.residual_scale))
            for key in self.refine_action_keys
        }
        self.current_anchor_action: dict[str, float] | None = None
        self.current_anchor_info: dict[str, Any] = {}
        self.anchor_hold_remaining = 0

    def reset(self, observation: Mapping[str, float]) -> None:
        self.anchor_predictor.reset(observation)
        self.current_anchor_action = None
        self.current_anchor_info = {}
        self.anchor_hold_remaining = 0

    def act(self, observation: Mapping[str, float]) -> tuple[dict[str, float], dict[str, Any]]:
        self.anchor_predictor.consume_observation(observation)
        refreshed = False
        if self.current_anchor_action is None or self.anchor_hold_remaining <= 0:
            self.current_anchor_action, self.current_anchor_info = self.anchor_predictor.predict_anchor(
                observation
            )
            self.anchor_hold_remaining = int(self.config.discrete_hold_steps)
            refreshed = True
        self.anchor_hold_remaining -= 1
        anchor_action = dict(self.current_anchor_action or {})
        pafc_action = dict(self.pafc_predictor(observation))
        if self.safe_predictor is not None:
            safe_action = dict(self.safe_predictor(observation))
            residual_reference = dict(safe_action)
        else:
            safe_action = {}
            residual_reference = dict(anchor_action)
        final_action = dict(anchor_action)
        residual_action: dict[str, float] = {}
        for key in _HYBRID_ACTION_KEYS:
            residual_value = float(pafc_action.get(key, 0.0)) - float(
                residual_reference.get(key, anchor_action.get(key, 0.0))
            )
            residual_action[key] = float(residual_value)
            if key in self.refine_action_keys:
                key_scale = float(self.refine_action_scales.get(key, self.config.residual_scale))
                final_action[key] = _clip_action_value(
                    key,
                    float(anchor_action.get(key, 0.0))
                    + key_scale * residual_value,
                )
            else:
                final_action[key] = _clip_action_value(key, float(anchor_action.get(key, 0.0)))
        residual_l1 = float(sum(abs(float(residual_action[key])) for key in self.refine_action_keys))
        final_delta_l1 = float(
            sum(
                abs(float(final_action[key]) - float(anchor_action.get(key, 0.0)))
                for key in self.refine_action_keys
            )
        )
        info = {
            "anchor_refreshed": bool(refreshed),
            "anchor_hold_remaining": int(self.anchor_hold_remaining),
            "anchor_action_index": int(self.current_anchor_info.get("anchor_action_index", -1)),
            "anchor_action_label": str(self.current_anchor_info.get("anchor_action_label", "")),
            "residual_l1": residual_l1,
            "final_delta_l1": final_delta_l1,
            "refine_action_keys": list(self.refine_action_keys),
            "refine_action_scales": dict(self.refine_action_scales),
            "residual_scale": float(self.config.residual_scale),
            "pafc_action": dict(pafc_action),
            "anchor_action": dict(anchor_action),
            "safe_action": dict(safe_action),
            "residual_action": dict(residual_action),
            "final_action": dict(final_action),
        }
        return final_action, info


def _load_hybrid_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    if str(payload.get("artifact_type", "")).strip().lower() != "hybrid_pafc_policy":
        raise ValueError("hybrid checkpoint 必须是 hybrid_pafc_policy json。")
    return payload


def train_hybrid_pafc(
    *,
    train_df: pd.DataFrame,
    env_config: EnvConfig,
    trainer_config: HybridPAFCConfig,
    run_root: str | Path = "runs",
) -> dict[str, Any]:
    year = _extract_year(train_df)
    if year != TRAIN_YEAR:
        raise ValueError(f"Hybrid train 仅支持 {TRAIN_YEAR}，当前年份 {year}")
    run_dir = _timestamped_run_dir(run_root)
    checkpoint_json_path = run_dir / "checkpoints" / "hybrid_pafc_policy.json"
    checkpoint_payload = {
        "artifact_type": "hybrid_pafc_policy",
        "paper_model_label": _HYBRID_MODEL_LABEL,
        "paper_algorithm_name": _HYBRID_MODEL_NAME,
        "dqn_checkpoint_path": str(Path(trainer_config.dqn_checkpoint_path).resolve()).replace("\\", "/"),
        "pafc_checkpoint_path": str(Path(trainer_config.pafc_checkpoint_path).resolve()).replace("\\", "/"),
        "dqn_model_source": str(trainer_config.dqn_model_source),
        "dqn_deterministic": bool(trainer_config.dqn_deterministic),
        "discrete_hold_steps": int(trainer_config.discrete_hold_steps),
        "residual_scale": float(trainer_config.residual_scale),
        "refine_action_keys": list(trainer_config.refine_action_keys),
        "refine_action_scales": dict(trainer_config.refine_action_scales),
        "use_safe_residual": bool(trainer_config.use_safe_residual),
        "seed": int(trainer_config.seed),
        "device": str(trainer_config.device),
        "train_year": int(TRAIN_YEAR),
        "created_at": datetime.now().isoformat(),
    }
    checkpoint_json_path.write_text(
        json.dumps(checkpoint_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "mode": "train",
        "policy": "hybrid_pafc",
        "status": "assembled",
        "train_year": int(TRAIN_YEAR),
        "run_dir": str(run_dir.resolve()).replace("\\", "/"),
        "checkpoint_json_path": str(checkpoint_json_path.resolve()).replace("\\", "/"),
        "paper_model_label": _HYBRID_MODEL_LABEL,
        "step_count": int(len(train_df)),
        "hybrid_config": dict(checkpoint_payload),
        "notes": [
            "当前版本为最小可运行 Hybrid 骨架：组装 DQN 高层 anchor 与 PAFC 低层 residual。",
            "该 train 阶段不做联合反向传播训练，主要产出可复现实验的 hybrid checkpoint json。",
        ],
    }
    (run_dir / "train" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def evaluate_hybrid_pafc(
    eval_df: pd.DataFrame,
    *,
    run_dir: str | Path,
    checkpoint_path: str | Path,
    seed: int = 42,
    device: str = "auto",
    config: EnvConfig,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    year = _extract_year(eval_df)
    if year != EVAL_YEAR:
        raise ValueError(f"评估必须使用 {EVAL_YEAR}，当前年份 {year}")
    payload = _load_hybrid_payload(checkpoint_path)
    hybrid_config = HybridPAFCConfig(
        dqn_checkpoint_path=str(payload["dqn_checkpoint_path"]),
        pafc_checkpoint_path=str(payload["pafc_checkpoint_path"]),
        dqn_model_source=str(payload.get("dqn_model_source", "best")),
        dqn_deterministic=bool(payload.get("dqn_deterministic", True)),
        discrete_hold_steps=int(payload.get("discrete_hold_steps", 1)),
        residual_scale=float(payload.get("residual_scale", 1.0)),
        refine_action_keys=_normalize_action_key_tuple(payload.get("refine_action_keys", ("u_boiler",))),
        refine_action_scales=_normalize_action_scale_map(payload.get("refine_action_scales", {})),
        use_safe_residual=bool(payload.get("use_safe_residual", False)),
        seed=int(seed),
        device=str(device),
    )
    controller = _HybridPolicyController(
        eval_df=eval_df,
        env_config=config,
        config=hybrid_config,
    )
    output_run_dir = Path(run_dir)
    (output_run_dir / "eval").mkdir(parents=True, exist_ok=True)

    env = CCHPPhysicalEnv(exogenous_df=eval_df, config=config, seed=int(seed))
    observation, _ = env.reset(seed=int(seed), episode_df=eval_df)
    controller.reset(observation)
    terminated = False
    total_reward = 0.0
    step_rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    anchor_counter: Counter[str] = Counter()
    residual_l1_rows: list[float] = []
    final_delta_l1_rows: list[float] = []

    while not terminated:
        action, hybrid_info = controller.act(observation)
        next_observation, reward, terminated, _, info = env.step(action)
        total_reward += float(reward)
        final_info = dict(info)
        anchor_label = str(hybrid_info.get("anchor_action_label", ""))
        if anchor_label:
            anchor_counter[anchor_label] += 1
        residual_l1_rows.append(float(hybrid_info.get("residual_l1", 0.0)))
        final_delta_l1_rows.append(float(hybrid_info.get("final_delta_l1", 0.0)))
        log_row = {
            key: value
            for key, value in info.items()
            if key not in {"violation_flags", "diagnostic_flags", "state_diagnostic_flags", "terminal_observation"}
        }
        log_row["hybrid_anchor_label"] = anchor_label
        log_row["hybrid_anchor_index"] = int(hybrid_info.get("anchor_action_index", -1))
        log_row["hybrid_anchor_refreshed"] = bool(hybrid_info.get("anchor_refreshed", False))
        log_row["hybrid_anchor_hold_remaining"] = int(hybrid_info.get("anchor_hold_remaining", 0))
        log_row["hybrid_residual_l1"] = float(hybrid_info.get("residual_l1", 0.0))
        log_row["hybrid_final_delta_l1"] = float(hybrid_info.get("final_delta_l1", 0.0))
        step_rows.append(log_row)
        observation = next_observation

    step_df = pd.DataFrame(step_rows)
    step_df.to_csv(output_run_dir / "eval" / "step_log.csv", index=False)
    summary = dict(final_info.get("episode_summary", env.kpi.summary()))
    summary["mode"] = "eval"
    summary["year"] = int(EVAL_YEAR)
    summary["policy"] = "hybrid_pafc"
    summary["paper_model_label"] = str(payload.get("paper_model_label", _HYBRID_MODEL_LABEL))
    summary["paper_algorithm_name"] = str(payload.get("paper_algorithm_name", _HYBRID_MODEL_NAME))
    summary["seed"] = int(seed)
    summary["device"] = str(device)
    summary["total_reward"] = float(total_reward)
    summary["eval_wall_time_s"] = float(time.perf_counter() - start_time)
    summary["eval_steps_per_second"] = float(len(step_rows) / max(1e-9, float(summary["eval_wall_time_s"])))
    summary["checkpoint_path"] = str(Path(checkpoint_path).resolve()).replace("\\", "/")
    summary["hybrid_policy_details"] = {
        "dqn_checkpoint_path": str(payload["dqn_checkpoint_path"]),
        "pafc_checkpoint_path": str(payload["pafc_checkpoint_path"]),
        "safe_checkpoint_path": str(controller.safe_checkpoint_path),
        "dqn_model_source": str(hybrid_config.dqn_model_source),
        "dqn_deterministic": bool(hybrid_config.dqn_deterministic),
        "discrete_hold_steps": int(hybrid_config.discrete_hold_steps),
        "residual_scale": float(hybrid_config.residual_scale),
        "refine_action_keys": list(hybrid_config.refine_action_keys),
        "refine_action_scales": dict(controller.refine_action_scales),
        "use_safe_residual": bool(hybrid_config.use_safe_residual),
        "anchor_action_counts": dict(anchor_counter),
        "anchor_refresh_count": int(sum(1 for row in step_rows if bool(row.get("hybrid_anchor_refreshed", False)))),
        "residual_l1_mean": float(np.mean(residual_l1_rows)) if residual_l1_rows else 0.0,
        "final_delta_l1_mean": float(np.mean(final_delta_l1_rows)) if final_delta_l1_rows else 0.0,
    }
    (output_run_dir / "eval" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
