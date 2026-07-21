"""Ground-truth anchor — does predicted identity track *real* identity?

The audit is predictions-only, which the council flagged as potentially circular (it could
just describe the model's own conditioning). The anchor breaks the circularity: using the
repeat sessions present in HCP/CNeuroMod, compute each subject's identifiability from REAL
fMRI and from the MODEL's predictions, then test whether they correlate across subjects.

- High correlation → the predicted "twin" recapitulates a real individuating signal; the
  predictions-only metrics earn the right to speak about subjects.
- Near-zero correlation → the per-subject component is model-internal bookkeeping; reframe
  strictly as Subject-Block cartography.

``anatomy_predictability`` echoes Borovykh 2026: how much of the subject-variance map is
predictable from cortical anatomy (the "nuisance" reading, in real brains).

These run on synthetic data today; plug real repeat-session arrays in for the real anchor.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from .fingerprint import fingerprint_pair, identification_matrix
from .responses import Responses


def differential_identifiability(resp: Responses, split_seed: int = 0) -> np.ndarray:
    """Per-subject differential identifiability (Amico & Goñi style), on disjoint halves.

    For each subject ``s``: self-similarity ``corr(fp_A[s], fp_B[s])`` minus the mean
    similarity to other subjects. Positive ⇒ that subject is individually identifiable.

    Returns: ``(n_subjects,)`` array.
    """
    fp_a, fp_b = fingerprint_pair(resp, split_seed=split_seed)
    m = identification_matrix(fp_a, fp_b)  # [db i, target j]
    n = m.shape[0]
    out = np.empty(n)
    for s in range(n):
        others = np.delete(m[:, s], s)
        out[s] = m[s, s] - others.mean()
    return out


def anchor_correlation(pred_resp: Responses, real_resp: Responses, split_seed: int = 0) -> dict:
    """Correlate predicted-identifiability with real-fMRI identifiability across subjects.

    ``pred_resp`` and ``real_resp`` must share the same subject order (the model's
    predicted responses and the subjects' real repeat-session responses). The stimulus axis
    can differ (clips vs. sessions) — identifiability is computed within each separately.

    Returns: dict with Pearson ``r`` and ``p``, the two per-subject vectors, and ``n``.
    """
    if pred_resp.n_subjects != real_resp.n_subjects:
        raise ValueError("pred and real must have the same subjects in the same order")
    di_pred = differential_identifiability(pred_resp, split_seed)
    di_real = differential_identifiability(real_resp, split_seed)
    r, p = pearsonr(di_pred, di_real)
    return {"r": float(r), "p": float(p), "di_pred": di_pred, "di_real": di_real, "n": di_pred.size}


def anatomy_predictability(target: np.ndarray, anatomy: np.ndarray) -> float:
    """R² of predicting a per-ROI quantity (e.g. subject-variance) from anatomy features.

    Echoes Borovykh 2026's "identity dimensions are predictable from cortical anatomy".
    ``anatomy`` is ``(n_roi, k)`` (e.g. parcel surface area / volume); ``target`` is
    ``(n_roi,)``. Returns the in-sample OLS R² (a high value supports the nuisance reading).
    """
    target = np.asarray(target, dtype=np.float64)
    anatomy = np.atleast_2d(np.asarray(anatomy, dtype=np.float64))
    if anatomy.shape[0] != target.shape[0]:
        anatomy = anatomy.T
    design = np.column_stack([np.ones(target.shape[0]), anatomy])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    pred = design @ beta
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
