"""NeuroTwin: a predictions-only identifiability audit of TRIBE v2 brain twins.

The central object is :class:`Responses`: predicted brain responses indexed by
(subject, stimulus, feature), where ``feature`` is a temporally-reduced vector of
cortical-surface vertices. Every analysis below operates on the *model's own
predictions* — no real fMRI data is involved.
"""

from .anchor import anatomy_predictability, anchor_correlation, differential_identifiability
from .fingerprint import (
    fingerprint_pair,
    identification_accuracy,
    identification_matrix,
    score_identification,
    stimulus_removed_residuals,
    subject_fingerprints,
)
from .geometry import effective_dimensionality, subject_manifold
from .guidance import guidance_rule
from .io import REDUCERS, roi_functional_connectivity
from .responses import Responses
from .stats import bootstrap_ci, permutation_null
from .variance import variance_partition

__all__ = [
    "Responses",
    "REDUCERS",
    "roi_functional_connectivity",
    "stimulus_removed_residuals",
    "subject_fingerprints",
    "fingerprint_pair",
    "score_identification",
    "identification_accuracy",
    "identification_matrix",
    "variance_partition",
    "guidance_rule",
    "differential_identifiability",
    "anchor_correlation",
    "anatomy_predictability",
    "effective_dimensionality",
    "subject_manifold",
    "permutation_null",
    "bootstrap_ci",
]
