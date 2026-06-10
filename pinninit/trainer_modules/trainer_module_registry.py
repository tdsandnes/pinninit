from __future__ import annotations

from enum import Enum

from .adaptive_weights_grad_balance import AdaptiveWeightsGradBalance
from .adaptive_weights_W21 import AdaptiveWeightsW21
from .early_stopping import EarlyStopping
from .learning_rate_schedule import LearningRateSchedule
from .loss_term_staggering import LossTermStaggering
from .trainer_module import TrainerModule


class TrainerModuleLabels(str, Enum):
    """Enumeration of builtin trainer module labels."""

    EARLY_STOPPING = "early_stopping"
    LEARNING_RATE_SCHEDULE = "learning_rate_schedule"
    LOSS_TERM_STAGGERING = "loss_term_staggering"
    ADAPTIVE_WEIGHTS_W21 = "adaptive_weights_w21"
    ADAPTIVE_WEIGHTS_GRAD_BALANCE = "adaptive_weights_grad_balance"


# Mapping from trainer module labels to classes
### UpdateOptimizer and R3D23 are not yet config-buildable since they require
### constructed objects (an Optimizer / a sampler) rather than plain values
TRAINER_MODULE_REGISTRY: dict[TrainerModuleLabels, type[TrainerModule]] = {
    TrainerModuleLabels.EARLY_STOPPING: EarlyStopping,
    TrainerModuleLabels.LEARNING_RATE_SCHEDULE: LearningRateSchedule,
    TrainerModuleLabels.LOSS_TERM_STAGGERING: LossTermStaggering,
    TrainerModuleLabels.ADAPTIVE_WEIGHTS_W21: AdaptiveWeightsW21,
    TrainerModuleLabels.ADAPTIVE_WEIGHTS_GRAD_BALANCE: AdaptiveWeightsGradBalance,
}
