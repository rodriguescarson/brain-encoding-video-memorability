"""Core data container for predicted brain responses.

A :class:`Responses` holds a dense tensor ``data`` of shape
``(n_subjects, n_stimuli, n_features)`` plus aligned subject / stimulus metadata.

``n_features`` is whatever per-stimulus summary you choose for fingerprinting: the
default pipeline reduces TRIBE v2's ``(n_timesteps, n_vertices)`` prediction to a
per-vertex mean (or another temporal statistic), so ``n_features == n_vertices``.
Keeping the container agnostic to that choice lets the same analyses run on
vertex-means, time-binned features, or ROI-averaged features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Responses:
    """Predicted responses for every (subject, stimulus) pair.

    Args:
        data: float array ``(n_subjects, n_stimuli, n_features)``.
        subjects: subject ids, length ``n_subjects``.
        stimuli: stimulus ids, length ``n_stimuli``.
        domains: optional per-stimulus domain tag (e.g. ``"movie"``, ``"podcast"``,
            ``"silent"``), length ``n_stimuli``. Used by the cross-domain
            consolidation analysis. Defaults to all ``"default"``.
        feature_ids: optional per-feature label (e.g. vertex index), length
            ``n_features``.
    """

    data: np.ndarray
    subjects: list[str]
    stimuli: list[str]
    domains: list[str] = field(default_factory=list)
    feature_ids: list[str] | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim != 3:
            raise ValueError(
                f"data must be 3-D (subjects, stimuli, features); got {self.data.shape}"
            )
        n_s, n_c, n_f = self.data.shape
        if len(self.subjects) != n_s:
            raise ValueError(f"{len(self.subjects)} subject ids for {n_s} subjects")
        if len(self.stimuli) != n_c:
            raise ValueError(f"{len(self.stimuli)} stimulus ids for {n_c} stimuli")
        if not self.domains:
            self.domains = ["default"] * n_c
        if len(self.domains) != n_c:
            raise ValueError(f"{len(self.domains)} domain tags for {n_c} stimuli")
        if self.feature_ids is not None and len(self.feature_ids) != n_f:
            raise ValueError(f"{len(self.feature_ids)} feature ids for {n_f} features")

    @property
    def n_subjects(self) -> int:
        return self.data.shape[0]

    @property
    def n_stimuli(self) -> int:
        return self.data.shape[1]

    @property
    def n_features(self) -> int:
        return self.data.shape[2]

    def stimulus_indices(self, domain: str | None = None) -> np.ndarray:
        """Indices of stimuli, optionally restricted to one domain."""
        if domain is None:
            return np.arange(self.n_stimuli)
        return np.array([i for i, d in enumerate(self.domains) if d == domain])

    def select_stimuli(self, idx: np.ndarray) -> Responses:
        """Return a new Responses over a subset of stimulus indices."""
        idx = np.asarray(idx, dtype=int)
        return Responses(
            data=self.data[:, idx, :],
            subjects=list(self.subjects),
            stimuli=[self.stimuli[i] for i in idx],
            domains=[self.domains[i] for i in idx],
            feature_ids=self.feature_ids,
        )
