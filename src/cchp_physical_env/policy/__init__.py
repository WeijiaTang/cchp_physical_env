# Ref: docs/spec/task.md (Task-ID: 012)
"""Task-007 / Task-012 策略与学习模块。"""

from .checkpoint import load_policy, load_policy_predictor, resolve_torch_device, save_policy
from .models import (
    MambaPolicyNet,
    SUPPORTED_POLICY_BACKBONES,
    TransformerPolicyNet,
    build_policy_network,
)
from .pafc_td3 import (
    PAFCTD3TrainConfig,
    evaluate_pafc_td3,
    load_pafc_td3_predictor,
    train_pafc_td3,
)
from .projection_surrogate import (
    DEFAULT_PROJECTION_ACTION_INPUT_KEYS,
    DEFAULT_PROJECTION_STATE_FEATURE_CANDIDATES,
    DEFAULT_PROJECTION_TARGET_KEYS,
    ProjectionDatasetBundle,
    ProjectionSurrogateTrainConfig,
    build_projection_dataset,
    build_projection_surrogate_network,
    load_projection_step_logs,
    load_projection_surrogate_predictor,
    train_projection_surrogate,
)
from .trainer import SequencePolicyTrainer, SequenceTrainerConfig, train_sequence_policy

__all__ = [
    "SUPPORTED_POLICY_BACKBONES",
    "TransformerPolicyNet",
    "MambaPolicyNet",
    "build_policy_network",
    "save_policy",
    "load_policy",
    "load_policy_predictor",
    "resolve_torch_device",
    "SequenceTrainerConfig",
    "SequencePolicyTrainer",
    "train_sequence_policy",
    "PAFCTD3TrainConfig",
    "train_pafc_td3",
    "evaluate_pafc_td3",
    "load_pafc_td3_predictor",
    "DEFAULT_PROJECTION_STATE_FEATURE_CANDIDATES",
    "DEFAULT_PROJECTION_ACTION_INPUT_KEYS",
    "DEFAULT_PROJECTION_TARGET_KEYS",
    "ProjectionDatasetBundle",
    "ProjectionSurrogateTrainConfig",
    "load_projection_step_logs",
    "build_projection_dataset",
    "build_projection_surrogate_network",
    "train_projection_surrogate",
    "load_projection_surrogate_predictor",
]
