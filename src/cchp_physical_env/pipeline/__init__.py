# Ref: docs/spec/task.md
"""训练与评估流程层。"""

from .ablation import run_constraint_ablation
from .calibration import (
    load_calibration_config,
    run_calibration_search,
    run_calibration_trial,
    sample_physical_params,
    validate_calibration_config,
)
from .runner import evaluate_baseline, train_baseline
from .sequence import (
    DEFAULT_SEQUENCE_ACTION_KEYS,
    DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS,
    SUPPORTED_SEQUENCE_ADAPTERS,
    MambaSequenceAdapter,
    RuleSequenceAdapter,
    SequenceAdapter,
    SequenceRulePolicy,
    SequenceWindowBuffer,
    TransformerSequenceAdapter,
    build_torch_module_predictor,
    build_action_vector,
    build_feature_vector,
    build_sequence_adapter,
    normalized_action_vector_to_env_action_dict,
)

__all__ = [
    "evaluate_baseline",
    "train_baseline",
    "run_constraint_ablation",
    "load_calibration_config",
    "validate_calibration_config",
    "sample_physical_params",
    "run_calibration_trial",
    "run_calibration_search",
    "DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS",
    "DEFAULT_SEQUENCE_ACTION_KEYS",
    "SUPPORTED_SEQUENCE_ADAPTERS",
    "build_action_vector",
    "build_feature_vector",
    "normalized_action_vector_to_env_action_dict",
    "build_torch_module_predictor",
    "build_sequence_adapter",
    "SequenceAdapter",
    "RuleSequenceAdapter",
    "TransformerSequenceAdapter",
    "MambaSequenceAdapter",
    "SequenceWindowBuffer",
    "SequenceRulePolicy",
]
