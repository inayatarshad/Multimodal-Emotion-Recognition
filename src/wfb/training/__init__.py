"""Training loop, run bookkeeping and the modality-dropout mitigation protocol."""

from wfb.training.modality_dropout import (
    DROPOUT_PROBABILITIES,
    MitigationVariant,
    apply_variant,
    pair_with_control,
    variants_for,
)
from wfb.training.trainer import (
    TrainConfig,
    TrainResult,
    evaluate_clean,
    load_checkpoint,
    run_name,
    train,
)

__all__ = [
    "DROPOUT_PROBABILITIES",
    "MitigationVariant",
    "TrainConfig",
    "TrainResult",
    "apply_variant",
    "evaluate_clean",
    "load_checkpoint",
    "pair_with_control",
    "run_name",
    "train",
    "variants_for",
]
