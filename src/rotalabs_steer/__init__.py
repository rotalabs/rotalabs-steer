"""
rotalabs-steer: Control agent behaviors through activation steering.

This package provides tools for extracting and applying steering vectors
to control LLM agent behaviors at inference time, without retraining.
"""

from rotalabs_steer._version import __version__
from rotalabs_steer.core import (
    MODEL_CONFIGS,
    ActivationCache,
    ActivationHook,
    ActivationInjector,
    ModelConfig,
    MultiVectorInjector,
    SteeringVector,
    SteeringVectorSet,
    extract_activations,
    get_model_config,
    infer_model_config,
)

__all__ = [
    "__version__",
    "ActivationCache",
    "ActivationHook",
    "ActivationInjector",
    "MODEL_CONFIGS",
    "ModelConfig",
    "MultiVectorInjector",
    "SteeringVector",
    "SteeringVectorSet",
    "extract_activations",
    "get_model_config",
    "infer_model_config",
]
