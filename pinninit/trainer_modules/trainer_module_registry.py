from __future__ import annotations

from enum import Enum

from .adaptive_weights_W21 import AdaptiveWeightsW21
from .adaptive_weights_W23 import AdaptiveWeightsW23
from .balanced_residual_decay_rate_C25 import BalancedResidualDecayRateC25
from .early_stopping import EarlyStopping
from .learning_rate_schedule import LearningRateSchedule
from .loss_term_staggering import LossTermStaggering
from .residual_based_attention_A24 import ResidualBasedAttentionA24
from .trainer_module import TrainerModule


class TrainerModuleLabels(str, Enum):
    """Enumeration of builtin trainer module labels."""

    EARLY_STOPPING = "early_stopping"
    LEARNING_RATE_SCHEDULE = "learning_rate_schedule"
    LOSS_TERM_STAGGERING = "loss_term_staggering"
    ADAPTIVE_WEIGHTS_W21 = "adaptive_weights_w21"
    ADAPTIVE_WEIGHTS_W23 = "adaptive_weights_w23"
    RESIDUAL_BASED_ATTENTION_A24 = "residual_based_attention_a24"
    BALANCED_RESIDUAL_DECAY_RATE_C25 = "balanced_residual_decay_rate_c25"


# Mapping from trainer module labels to classes
# UpdateOptimizer isn't here, since it needs a constructed Optimizer rather
# than values from the parameter file
TRAINER_MODULE_REGISTRY: dict[TrainerModuleLabels, type[TrainerModule]] = {
    TrainerModuleLabels.EARLY_STOPPING: EarlyStopping,
    TrainerModuleLabels.LEARNING_RATE_SCHEDULE: LearningRateSchedule,
    TrainerModuleLabels.LOSS_TERM_STAGGERING: LossTermStaggering,
    TrainerModuleLabels.ADAPTIVE_WEIGHTS_W21: AdaptiveWeightsW21,
    TrainerModuleLabels.ADAPTIVE_WEIGHTS_W23: AdaptiveWeightsW23,
    TrainerModuleLabels.RESIDUAL_BASED_ATTENTION_A24: ResidualBasedAttentionA24,
    TrainerModuleLabels.BALANCED_RESIDUAL_DECAY_RATE_C25: BalancedResidualDecayRateC25,
}
