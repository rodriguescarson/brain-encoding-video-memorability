"""Geometry of the subject manifold.

If individuality is "twin-like", the across-subject variation should occupy a rich,
high-dimensional subspace. If it is "nuisance-like" (a low-rank gain / anatomical-
alignment correction), it should collapse into a handful of dimensions.

We quantify this with the **participation ratio** (effective dimensionality) of the
subject-mean residual matrix.
"""

from __future__ import annotations

import numpy as np

from .fingerprint import stimulus_removed_residuals


def subject_manifold(data: np.ndarray) -> np.ndarray:
    """Subject-mean residual matrix ``(n_subjects, n_features)``.

    Each row is a subject's stimulus-removed residual, averaged over all stimuli — i.e.
    that subject's overall fingerprint. The geometry of these rows is the subject
    manifold.
    """
    residuals = stimulus_removed_residuals(np.asarray(data, dtype=np.float64))
    return residuals.mean(axis=1)


def _eigvals(mat: np.ndarray) -> np.ndarray:
    """Non-negative eigenvalues of the row covariance, descending."""
    centered = mat - mat.mean(axis=0, keepdims=True)
    # svd returns singular values in descending order; eigenvalues of the
    # covariance are ∝ s**2 (also descending).
    s = np.linalg.svd(centered, compute_uv=False)
    return s ** 2


def effective_dimensionality(data: np.ndarray) -> dict:
    """Participation ratio of the subject manifold.

    PR = ``(sum λ)^2 / sum(λ^2)`` over the eigenvalues λ of the subject-residual
    covariance. PR ranges from 1 (all variance in one mode → nuisance-like) up to
    ``min(n_subjects - 1, n_features)`` (variance spread evenly → rich/twin-like).

    Returns:
        dict with ``participation_ratio``, ``max_possible`` (= ``n_subjects - 1``),
        ``normalized`` (PR / max_possible, in [0, 1]), and ``eigenvalues`` (descending,
        normalized to sum to 1).
    """
    mat = subject_manifold(data)
    lam = _eigvals(mat)
    lam = lam[lam > 1e-12]
    if lam.size == 0:
        return {
            "participation_ratio": 0.0,
            "max_possible": max(mat.shape[0] - 1, 1),
            "normalized": 0.0,
            "eigenvalues": np.array([]),
        }
    pr = float((lam.sum() ** 2) / (lam ** 2).sum())
    max_possible = max(mat.shape[0] - 1, 1)
    return {
        "participation_ratio": pr,
        "max_possible": max_possible,
        "normalized": pr / max_possible,
        "eigenvalues": lam / lam.sum(),
    }
