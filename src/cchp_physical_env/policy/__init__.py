# Ref: docs/spec/task.md
"""Task-007 深度序列策略模块。"""

from .checkpoint import load_policy, load_policy_predictor, resolve_torch_device, save_policy
from .models import (
    MambaPolicyNet,
    SUPPORTED_POLICY_BACKBONES,
    TransformerPolicyNet,
    build_policy_network,
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
]
