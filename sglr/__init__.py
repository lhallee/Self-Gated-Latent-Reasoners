"""Self-Gated Latent Reasoners."""

from sglr.config import (
    ExperimentConfig,
    ExpertSpec,
    ModelConfig,
    SweepConfig,
    TrainingConfig,
    experiment_from_dict,
)
from sglr.model import (
    MNISTSGLR,
    MNISTOutput,
    SGLRCore,
    SGLRCoreOutput,
    build_mnist_model,
)
from sglr.presets.mnist import MNIST_PRESET_NAMES, get_mnist_preset, make_expert_pool
from sglr.router import RoutingTrace


__all__ = [
    "ExperimentConfig",
    "ExpertSpec",
    "MNISTSGLR",
    "MNISTOutput",
    "MNIST_PRESET_NAMES",
    "ModelConfig",
    "RoutingTrace",
    "SGLRCore",
    "SGLRCoreOutput",
    "SweepConfig",
    "TrainingConfig",
    "build_mnist_model",
    "experiment_from_dict",
    "get_mnist_preset",
    "make_expert_pool",
]
