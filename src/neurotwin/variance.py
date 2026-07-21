"""Per-feature variance partitioning.

For each feature (vertex), decompose the variance of predicted responses over the
(subject x stimulus) grid into three additive components:

- ``stimulus``: variance explained by the stimulus main effect (across-stimulus
  variance of the subject-averaged response).
- ``subject``: variance explained by the subject main effect (across-subject variance
  of the stimulus-averaged response). **This is "how much individuality lives here."**
- ``interaction``: the remaining (subject x stimulus) variation.

Uses the standard two-way ANOVA sum-of-squares identity on a balanced grid:
``SS_total = SS_subject + SS_stimulus + SS_interaction`` (no replicates, so the
interaction term absorbs residual).
"""

from __future__ import annotations

import numpy as np


def variance_partition(data: np.ndarray) -> dict[str, np.ndarray]:
    """Partition per-feature variance into subject / stimulus / interaction fractions.

    Args:
        data: ``(n_subjects, n_stimuli, n_features)``.

    Returns:
        dict of per-feature arrays (each length ``n_features``):
        ``subject``, ``stimulus``, ``interaction`` — fractions that sum to ~1 where
        total variance > 0 (0 where a feature is constant).
    """
    data = np.asarray(data, dtype=np.float64)
    n_s, n_c, _ = data.shape

    grand = data.mean(axis=(0, 1), keepdims=True)            # (1,1,F)
    subj_mean = data.mean(axis=1, keepdims=True)             # (S,1,F)
    stim_mean = data.mean(axis=0, keepdims=True)             # (1,C,F)

    ss_subject = n_c * ((subj_mean - grand)[:, 0, :] ** 2).sum(axis=0)   # (F,)
    ss_stimulus = n_s * ((stim_mean - grand)[0, :, :] ** 2).sum(axis=0)  # (F,)
    resid = data - subj_mean - stim_mean + grand
    ss_interaction = (resid ** 2).sum(axis=(0, 1))                        # (F,)

    ss_total = ss_subject + ss_stimulus + ss_interaction
    safe = ss_total > 0

    def frac(ss: np.ndarray) -> np.ndarray:
        out = np.zeros_like(ss)
        out[safe] = ss[safe] / ss_total[safe]
        return out

    return {
        "subject": frac(ss_subject),
        "stimulus": frac(ss_stimulus),
        "interaction": frac(ss_interaction),
    }
