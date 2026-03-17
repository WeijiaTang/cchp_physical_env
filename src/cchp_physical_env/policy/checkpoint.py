# Ref: docs/spec/task.md
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .models import build_policy_network


def _require_torch_module():
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover
        raise ModuleNotFoundError("未检测到 torch，无法读写深度策略 checkpoint。") from error
    return torch


def resolve_torch_device(device: str = "auto") -> str:
    torch = _require_torch_module()
    normalized = device.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 torch 未检测到可用 GPU。")
    return normalized


def save_policy(
    *,
    model,
    checkpoint_path: str | Path,
    metadata: Mapping[str, Any],
) -> Path:
    """保存模型参数与元信息。"""

    torch = _require_torch_module()
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": dict(metadata),
    }
    torch.save(payload, target)
    return target


def load_policy(
    checkpoint_path: str | Path, *, map_location: str = "cpu"
) -> dict[str, Any]:
    """加载模型 payload（state_dict + metadata）。"""

    torch = _require_torch_module()
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint 格式错误：应为 dict。")
    if "state_dict" not in payload or "metadata" not in payload:
        raise ValueError("checkpoint 缺少 state_dict 或 metadata。")
    return payload


def load_policy_predictor(
    *,
    checkpoint_path: str | Path,
    device: str = "auto",
):
    """从 checkpoint 恢复模型并构建 sequence predictor。"""

    from ..pipeline.sequence import (
        DEFAULT_SEQUENCE_ACTION_KEYS,
        build_torch_module_predictor,
    )

    target_device = resolve_torch_device(device=device)
    payload = load_policy(checkpoint_path, map_location=target_device)
    metadata = dict(payload["metadata"])
    policy_backbone = str(metadata["policy_backbone"])
    n_features = int(metadata["n_features"])
    n_actions = int(metadata["n_actions"])
    model_kwargs = dict(metadata.get("model_kwargs", {}))
    model_kwargs.setdefault("history_steps", int(metadata.get("history_steps", 0)))
    action_keys = tuple(metadata.get("action_keys", DEFAULT_SEQUENCE_ACTION_KEYS))

    model = build_policy_network(
        policy_backbone=policy_backbone,
        n_features=n_features,
        n_actions=n_actions,
        model_kwargs=model_kwargs,
    )
    model.load_state_dict(payload["state_dict"])
    predictor = build_torch_module_predictor(
        model=model, device=target_device, action_keys=action_keys
    )

    obs_norm = metadata.get("obs_norm")
    if isinstance(obs_norm, Mapping) and str(obs_norm.get("kind", "")).strip().lower() == "affine_v1":
        try:
            import numpy as np

            offsets = np.asarray(list(obs_norm.get("offset", [])), dtype=np.float32)
            scales = np.asarray(list(obs_norm.get("scale", [])), dtype=np.float32)
            feature_keys = list(obs_norm.get("observation_feature_keys") or metadata.get("feature_keys") or [])
            n_obs = int(len(feature_keys))
            if n_obs > 0 and offsets.shape == (n_obs,) and scales.shape == (n_obs,):
                scales = np.where(np.abs(scales) < 1e-6, 1.0, scales)
                base_predictor = predictor

                def predictor(window, observation):  # type: ignore[no-redef]
                    window_np = np.asarray(window, dtype=np.float32).copy()
                    window_np[:, :n_obs] = (window_np[:, :n_obs] - offsets) / scales
                    return base_predictor(window_np, observation)
        except Exception:
            # 归一化为“增强项”，异常时退化为原 predictor，保证旧 checkpoint/轻量环境可用。
            pass
    return predictor, metadata
