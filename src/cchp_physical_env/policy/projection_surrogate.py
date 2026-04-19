# Ref: docs/spec/task.md (Task-ID: 012)
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .checkpoint import load_policy, resolve_torch_device, save_policy

_NORM_EPS = 1e-6
_ACTION_KEYS = ("u_gt", "u_bes", "u_boiler", "u_abs", "u_ech", "u_tes")
DEFAULT_PROJECTION_ACTION_INPUT_KEYS = tuple(f"action_raw_{key}" for key in _ACTION_KEYS)
DEFAULT_PROJECTION_TARGET_KEYS = tuple(f"action_exec_{key}" for key in _ACTION_KEYS)
DEFAULT_PROJECTION_STATE_FEATURE_CANDIDATES = (
    "energy_demand_e_mwh",
    "energy_demand_h_mwh",
    "energy_demand_c_mwh",
    "price_e_buy",
    "t_tes_hot_k",
    "e_tes_mwh",
    "p_re_mw",
    "p_gt_mw",
    "q_hrsg_rec_mw",
    "q_tes_discharge_feasible_mw",
    "heat_backup_min_needed_mw",
    "heat_deficit_if_boiler_off_mw",
    "u_boiler_lower_bound",
    "u_abs_gate",
    "abs_drive_margin_k",
    "p_gt_ramp_delta_mw",
)
SUPPORTED_SURROGATE_LOSSES = ("mse", "smooth_l1")


def _require_torch_modules():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as error:  # pragma: no cover
        raise ModuleNotFoundError("未检测到 torch，无法训练 projection surrogate。") from error
    return torch, nn, F, AdamW, DataLoader, TensorDataset


def _resolve_step_log_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    probes = (
        candidate / "eval" / "step_log.csv",
        candidate / "step_log.csv",
    )
    for probe in probes:
        if probe.exists():
            return probe
    raise FileNotFoundError(f"未找到 step_log.csv: {candidate}")


def load_projection_step_logs(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    resolved_paths: list[str] = []
    for item in paths:
        target = _resolve_step_log_path(item)
        frame = pd.read_csv(target)
        frame["__source_path__"] = str(target).replace("\\", "/")
        frames.append(frame)
        resolved_paths.append(str(target).replace("\\", "/"))
    if not frames:
        raise ValueError("paths 不能为空：至少需要一个 step_log.csv 或 run/eval 目录。")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged.attrs["source_paths"] = resolved_paths
    return merged


@dataclass(slots=True)
class ProjectionDatasetBundle:
    features: np.ndarray
    targets: np.ndarray
    feature_keys: tuple[str, ...]
    target_keys: tuple[str, ...]
    state_feature_keys: tuple[str, ...]
    action_feature_keys: tuple[str, ...]
    feature_norm: dict[str, Any]
    source_rows: int
    kept_rows: int
    source_paths: tuple[str, ...] = ()

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.targets.shape[1])


def build_projection_dataset(
    step_log: pd.DataFrame,
    *,
    state_feature_keys: Sequence[str] | None = None,
    action_feature_keys: Sequence[str] = DEFAULT_PROJECTION_ACTION_INPUT_KEYS,
    target_keys: Sequence[str] = DEFAULT_PROJECTION_TARGET_KEYS,
) -> ProjectionDatasetBundle:
    if step_log.empty:
        raise ValueError("step_log 为空，无法构建 projection surrogate 数据集。")

    if state_feature_keys is None:
        selected_state_keys = tuple(
            key for key in DEFAULT_PROJECTION_STATE_FEATURE_CANDIDATES if key in step_log.columns
        )
    else:
        selected_state_keys = tuple(str(key) for key in state_feature_keys)
        missing = [key for key in selected_state_keys if key not in step_log.columns]
        if missing:
            raise ValueError(f"state_feature_keys 缺少列: {missing}")

    selected_action_keys = tuple(str(key) for key in action_feature_keys)
    selected_target_keys = tuple(str(key) for key in target_keys)
    missing_action = [key for key in selected_action_keys if key not in step_log.columns]
    missing_target = [key for key in selected_target_keys if key not in step_log.columns]
    if missing_action:
        raise ValueError(f"action_feature_keys 缺少列: {missing_action}")
    if missing_target:
        raise ValueError(f"target_keys 缺少列: {missing_target}")

    feature_keys = tuple([*selected_state_keys, *selected_action_keys])
    numeric_frame = step_log.loc[:, [*feature_keys, *selected_target_keys]].copy()
    for column in numeric_frame.columns:
        numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")
    valid_mask = numeric_frame.notna().all(axis=1)
    filtered = numeric_frame.loc[valid_mask].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("构建 projection surrogate 数据集失败：有效样本数为 0。")

    features = filtered.loc[:, feature_keys].to_numpy(dtype=np.float32)
    targets = filtered.loc[:, selected_target_keys].to_numpy(dtype=np.float32)

    offset = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale = np.where(np.abs(scale) < _NORM_EPS, 1.0, scale)
    feature_norm = {
        "kind": "affine_v1",
        "feature_keys": list(feature_keys),
        "offset": offset.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
    }

    source_paths = tuple(str(item) for item in (step_log.attrs.get("source_paths") or []))
    return ProjectionDatasetBundle(
        features=features,
        targets=targets,
        feature_keys=feature_keys,
        target_keys=selected_target_keys,
        state_feature_keys=selected_state_keys,
        action_feature_keys=selected_action_keys,
        feature_norm=feature_norm,
        source_rows=int(len(step_log)),
        kept_rows=int(len(filtered)),
        source_paths=source_paths,
    )


def build_projection_surrogate_network(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int] = (256, 256),
):
    torch, nn, *_ = _require_torch_modules()
    del torch

    if input_dim <= 0:
        raise ValueError("input_dim 必须 > 0。")
    if output_dim <= 0:
        raise ValueError("output_dim 必须 > 0。")
    normalized_hidden_dims = tuple(int(dim) for dim in hidden_dims)
    if any(dim <= 0 for dim in normalized_hidden_dims):
        raise ValueError("hidden_dims 必须全部 > 0。")

    class _ProjectionSurrogateNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev_dim = int(input_dim)
            for hidden_dim in normalized_hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, int(output_dim)))
            self.net = nn.Sequential(*layers)

        def forward(self, inputs):
            return self.net(inputs)

    return _ProjectionSurrogateNet()


@dataclass(slots=True)
class ProjectionSurrogateTrainConfig:
    batch_size: int = 256
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.1
    hidden_dims: tuple[int, ...] = (256, 256)
    loss_name: str = "smooth_l1"
    seed: int = 42
    device: str = "auto"
    state_feature_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        self.val_ratio = float(self.val_ratio)
        self.hidden_dims = tuple(int(dim) for dim in self.hidden_dims)
        self.loss_name = str(self.loss_name).strip().lower()
        self.seed = int(self.seed)
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须 > 0。")
        if self.epochs <= 0:
            raise ValueError("epochs 必须 > 0。")
        if self.lr <= 0.0:
            raise ValueError("lr 必须 > 0。")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay 必须 >= 0。")
        if not (0.0 <= self.val_ratio < 1.0):
            raise ValueError("val_ratio 必须在 [0,1) 内。")
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims 必须全部 > 0。")
        if self.loss_name not in SUPPORTED_SURROGATE_LOSSES:
            raise ValueError(
                f"不支持的 loss_name: {self.loss_name}，支持 {SUPPORTED_SURROGATE_LOSSES}"
            )
        self.state_feature_keys = tuple(str(key) for key in self.state_feature_keys)


def _create_run_directory(run_root: str | Path) -> Path:
    root = Path(run_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"{stamp}_train_projection_surrogate"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _split_indices(n_samples: int, *, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 2:
        raise ValueError("projection surrogate 至少需要 2 条样本。")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(indices)
    val_size = int(round(n_samples * float(val_ratio)))
    val_size = max(1 if n_samples >= 5 and val_ratio > 0.0 else 0, val_size)
    val_size = min(max(0, n_samples - 1), val_size)
    if val_size <= 0:
        return indices, np.empty((0,), dtype=np.int64)
    return indices[val_size:], indices[:val_size]


def _normalize_features(features: np.ndarray, feature_norm: Mapping[str, Any]) -> np.ndarray:
    offset = np.asarray(feature_norm.get("offset", []), dtype=np.float32)
    scale = np.asarray(feature_norm.get("scale", []), dtype=np.float32)
    if offset.shape != (features.shape[1],) or scale.shape != (features.shape[1],):
        raise ValueError("feature_norm 维度与输入不匹配。")
    scale = np.where(np.abs(scale) < _NORM_EPS, 1.0, scale)
    return ((features - offset) / scale).astype(np.float32)


def train_projection_surrogate(
    *,
    step_log: pd.DataFrame | None = None,
    step_log_paths: Sequence[str | Path] | None = None,
    train_config: ProjectionSurrogateTrainConfig | None = None,
    run_root: str | Path = "runs",
) -> dict[str, Any]:
    if step_log is None and not step_log_paths:
        raise ValueError("step_log 或 step_log_paths 至少提供一个。")
    config = ProjectionSurrogateTrainConfig() if train_config is None else train_config
    frame = step_log if step_log is not None else load_projection_step_logs(step_log_paths or ())
    dataset = build_projection_dataset(
        frame,
        state_feature_keys=config.state_feature_keys or None,
    )

    torch, _, F, AdamW, DataLoader, TensorDataset = _require_torch_modules()
    device = resolve_torch_device(config.device)
    try:
        torch.manual_seed(int(config.seed))
        if str(device).startswith("cuda") and getattr(torch, "cuda", None) is not None:
            torch.cuda.manual_seed_all(int(config.seed))
    except Exception:
        pass

    normalized_features = _normalize_features(dataset.features, dataset.feature_norm)
    train_idx, val_idx = _split_indices(
        normalized_features.shape[0],
        val_ratio=float(config.val_ratio),
        seed=int(config.seed),
    )

    train_x = torch.as_tensor(normalized_features[train_idx], dtype=torch.float32)
    train_y = torch.as_tensor(dataset.targets[train_idx], dtype=torch.float32)
    val_x = torch.as_tensor(normalized_features[val_idx], dtype=torch.float32)
    val_y = torch.as_tensor(dataset.targets[val_idx], dtype=torch.float32)

    model = build_projection_surrogate_network(
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        hidden_dims=config.hidden_dims,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(int(config.batch_size), int(len(train_idx))),
        shuffle=True,
        drop_last=False,
    )

    def _loss_fn(prediction, target):
        if config.loss_name == "mse":
            return F.mse_loss(prediction, target)
        return F.smooth_l1_loss(prediction, target)

    run_dir = _create_run_directory(run_root=run_root)
    history_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        train_maes: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model(batch_x)
            loss = _loss_fn(prediction, batch_y)
            mae = torch.mean(torch.abs(prediction - batch_y))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            train_maes.append(float(mae.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            if int(len(val_idx)) > 0:
                val_prediction = model(val_x.to(device))
                val_loss = float(_loss_fn(val_prediction, val_y.to(device)).detach().cpu().item())
                val_mae = float(torch.mean(torch.abs(val_prediction - val_y.to(device))).detach().cpu().item())
            else:
                train_prediction = model(train_x.to(device))
                val_loss = float(_loss_fn(train_prediction, train_y.to(device)).detach().cpu().item())
                val_mae = float(torch.mean(torch.abs(train_prediction - train_y.to(device))).detach().cpu().item())

        mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        mean_train_mae = float(np.mean(train_maes)) if train_maes else 0.0
        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": mean_train_loss,
                "train_mae": mean_train_mae,
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if float(val_loss) <= best_val_loss:
            best_val_loss = float(val_loss)
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    checkpoint_path = run_dir / "checkpoints" / "projection_surrogate.pt"
    metadata = {
        "artifact_type": "projection_surrogate",
        "input_dim": int(dataset.input_dim),
        "output_dim": int(dataset.output_dim),
        "feature_keys": list(dataset.feature_keys),
        "target_keys": list(dataset.target_keys),
        "state_feature_keys": list(dataset.state_feature_keys),
        "action_feature_keys": list(dataset.action_feature_keys),
        "feature_norm": dict(dataset.feature_norm),
        "hidden_dims": list(config.hidden_dims),
        "loss_name": str(config.loss_name),
        "seed": int(config.seed),
        "device": str(device),
        "source_rows": int(dataset.source_rows),
        "kept_rows": int(dataset.kept_rows),
        "source_paths": list(dataset.source_paths),
    }
    save_policy(
        model=model,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )

    checkpoint_json = run_dir / "checkpoints" / "projection_surrogate.json"
    checkpoint_json.write_text(
        json.dumps(
            {
                "artifact_type": "projection_surrogate",
                "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
                "feature_keys": list(dataset.feature_keys),
                "target_keys": list(dataset.target_keys),
                "train_history_path": str(run_dir / "train" / "history.csv").replace("\\", "/"),
                "summary_path": str(run_dir / "train" / "summary.json").replace("\\", "/"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(history_rows).to_csv(run_dir / "train" / "history.csv", index=False)
    summary = {
        "mode": "train",
        "artifact_type": "projection_surrogate",
        "run_dir": str(run_dir).replace("\\", "/"),
        "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
        "checkpoint_json_path": str(checkpoint_json).replace("\\", "/"),
        "device": str(device),
        "epochs": int(config.epochs),
        "batch_size": int(config.batch_size),
        "loss_name": str(config.loss_name),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "hidden_dims": list(config.hidden_dims),
        "input_dim": int(dataset.input_dim),
        "output_dim": int(dataset.output_dim),
        "feature_keys": list(dataset.feature_keys),
        "target_keys": list(dataset.target_keys),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "source_rows": int(dataset.source_rows),
        "kept_rows": int(dataset.kept_rows),
        "best_val_loss": float(best_val_loss),
        "last_val_loss": float(history_rows[-1]["val_loss"]) if history_rows else 0.0,
        "last_val_mae": float(history_rows[-1]["val_mae"]) if history_rows else 0.0,
        "source_paths": list(dataset.source_paths),
    }
    (run_dir / "train" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def load_projection_surrogate_predictor(
    *,
    checkpoint_path: str | Path,
    device: str = "auto",
):
    resolved_device = resolve_torch_device(device)
    payload = load_policy(checkpoint_path, map_location=resolved_device)
    metadata = dict(payload["metadata"])
    model = build_projection_surrogate_network(
        input_dim=int(metadata["input_dim"]),
        output_dim=int(metadata["output_dim"]),
        hidden_dims=tuple(int(dim) for dim in metadata.get("hidden_dims", (256, 256))),
    ).to(resolved_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    feature_keys = tuple(str(key) for key in metadata.get("feature_keys", ()))
    target_keys = tuple(str(key) for key in metadata.get("target_keys", ()))
    feature_norm = dict(metadata.get("feature_norm", {}))
    offset = np.asarray(feature_norm.get("offset", []), dtype=np.float32)
    scale = np.asarray(feature_norm.get("scale", []), dtype=np.float32)
    scale = np.where(np.abs(scale) < _NORM_EPS, 1.0, scale)
    torch, *_ = _require_torch_modules()

    def predictor(features: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]):
        if isinstance(features, pd.DataFrame):
            if not feature_keys:
                raise ValueError("checkpoint 缺少 feature_keys，无法从 DataFrame 预测。")
            missing = [key for key in feature_keys if key not in features.columns]
            if missing:
                raise ValueError(f"预测输入缺少列: {missing}")
            feature_array = features.loc[:, feature_keys].to_numpy(dtype=np.float32)
        else:
            feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(1, -1)
        if feature_array.shape[1] != len(feature_keys):
            raise ValueError(
                f"输入维度不匹配：期望 {len(feature_keys)}，实际 {feature_array.shape[1]}"
            )
        normalized = ((feature_array - offset) / scale).astype(np.float32)
        with torch.no_grad():
            tensor = torch.as_tensor(normalized, dtype=torch.float32, device=resolved_device)
            prediction = model(tensor).detach().cpu().numpy()
        return prediction

    return predictor, {
        **metadata,
        "feature_keys": list(feature_keys),
        "target_keys": list(target_keys),
    }
