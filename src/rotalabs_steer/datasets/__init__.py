"""Contrast pair datasets for steering vector extraction."""

from .base import ContrastPair, ContrastPairDataset, EvaluationDataset, EvaluationExample
from .hierarchy_pairs import HIERARCHY_PAIRS, load_hierarchy_pairs
from .refusal_pairs import get_refusal_pairs, load_refusal_pairs
from .tool_pairs import TOOL_RESTRAINT_PAIRS, load_tool_restraint_pairs
from .uncertainty_pairs import UNCERTAINTY_PAIRS, load_uncertainty_pairs

# Style behaviors
from .formality_pairs import FORMALITY_PAIRS, load_formality_pairs
from .conciseness_pairs import CONCISENESS_PAIRS, load_conciseness_pairs
from .creativity_pairs import CREATIVITY_PAIRS, load_creativity_pairs
from .assertiveness_pairs import ASSERTIVENESS_PAIRS, load_assertiveness_pairs

# Personality behaviors
from .humor_pairs import HUMOR_PAIRS, load_humor_pairs
from .empathy_pairs import EMPATHY_PAIRS, load_empathy_pairs
from .technical_depth_pairs import TECHNICAL_DEPTH_PAIRS, load_technical_depth_pairs

__all__ = [
    "ContrastPair",
    "ContrastPairDataset",
    "EvaluationDataset",
    "EvaluationExample",
    # Refusal
    "get_refusal_pairs",
    "load_refusal_pairs",
    # Tool restraint
    "load_tool_restraint_pairs",
    "TOOL_RESTRAINT_PAIRS",
    # Instruction hierarchy
    "load_hierarchy_pairs",
    "HIERARCHY_PAIRS",
    # Uncertainty
    "load_uncertainty_pairs",
    "UNCERTAINTY_PAIRS",
    # Formality
    "load_formality_pairs",
    "FORMALITY_PAIRS",
    # Conciseness
    "load_conciseness_pairs",
    "CONCISENESS_PAIRS",
    # Creativity
    "load_creativity_pairs",
    "CREATIVITY_PAIRS",
    # Assertiveness
    "load_assertiveness_pairs",
    "ASSERTIVENESS_PAIRS",
    # Humor
    "load_humor_pairs",
    "HUMOR_PAIRS",
    # Empathy
    "load_empathy_pairs",
    "EMPATHY_PAIRS",
    # Technical depth
    "load_technical_depth_pairs",
    "TECHNICAL_DEPTH_PAIRS",
]
