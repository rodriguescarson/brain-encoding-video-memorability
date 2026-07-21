"""Subject fingerprinting / identification (Finn et al., 2015, applied to predictions).

The question: given TRIBE v2's *predicted* responses, can we re-identify a subject
across a disjoint set of stimuli? If yes, the per-subject component carries a stable,
individuating signature. If identification is at chance, the "twin" is not individuating.

Pipeline
--------
1. ``stimulus_removed_residuals``: subtract the across-subject mean at each stimulus,
   removing the (dominant) stimulus-driven component and leaving the subject-specific
   residual.
2. ``subject_fingerprints``: average a subject's residuals across a set of stimuli to
   form that subject's fingerprint vector.
3. ``identification_accuracy``: split stimuli into disjoint halves A/B; for each subject's
   fingerprint in B, find the most-correlated fingerprint in A. Top-1 accuracy is the
   identifiability score.
"""

from __future__ import annotations

import numpy as np

from .responses import Responses


def stimulus_removed_residuals(data: np.ndarray, leave_one_out: bool = True) -> np.ndarray:
    """Remove the stimulus-main effect.

    Args:
        data: ``(n_subjects, n_stimuli, n_features)``.
        leave_one_out: if True (default), subtract the mean of the *other* subjects at
            each stimulus, so the subtracted reference never includes the target
            subject. The naive grand-mean subtraction (``False``) leaves residuals that
            sum to zero across subjects, which mechanically makes each subject look
            maximally unlike the others and inflates identification at the small subject
            counts typical of Algonauts/CNeuroMod (N≈4–10). LOO removes that bias.

    Returns:
        residuals of the same shape.
    """
    data = np.asarray(data, dtype=np.float64)
    if not leave_one_out:
        return data - data.mean(axis=0, keepdims=True)
    n = data.shape[0]
    if n < 2:
        raise ValueError("leave_one_out needs >=2 subjects")
    total = data.sum(axis=0, keepdims=True)          # (1, stimuli, features)
    loo_mean = (total - data) / (n - 1)              # mean of the other n-1 subjects
    return data - loo_mean


def subject_fingerprints(residuals: np.ndarray, stim_idx: np.ndarray) -> np.ndarray:
    """Average residuals over a set of stimuli to form per-subject fingerprints.

    Args:
        residuals: ``(n_subjects, n_stimuli, n_features)`` from
            :func:`stimulus_removed_residuals`.
        stim_idx: indices of stimuli to average over.

    Returns:
        ``(n_subjects, n_features)`` fingerprint matrix.
    """
    stim_idx = np.asarray(stim_idx, dtype=int)
    if stim_idx.size == 0:
        raise ValueError("stim_idx is empty")
    return residuals[:, stim_idx, :].mean(axis=1)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    """Center and L2-normalize each row; zero-variance rows become all-zero.

    Returning unit vectors makes the row-row dot product exactly the Pearson
    correlation, bounded in [-1, 1] with no division blow-ups.
    """
    centered = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    out = np.zeros_like(centered)
    nz = norm[:, 0] > 0
    out[nz] = centered[nz] / norm[nz]
    return out


def _corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between every row of ``a`` and every row of ``b``.

    Returns ``(n_a, n_b)`` with entries in [-1, 1].
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        corr = _unit_rows(a) @ _unit_rows(b).T
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def identification_matrix(fp_a: np.ndarray, fp_b: np.ndarray) -> np.ndarray:
    """Correlation matrix between database fingerprints A and target fingerprints B.

    Entry ``[i, j]`` = corr(database subject ``i``, target subject ``j``).
    """
    return _corr_matrix(fp_a, fp_b)


def fingerprint_pair(
    resp: Responses,
    split_seed: int = 0,
    domain_a: str | None = None,
    domain_b: str | None = None,
    leave_one_out: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a database/target fingerprint pair (fp_a, fp_b), each ``(n_subjects, n_features)``.

    With ``domain_a``/``domain_b`` set, A and B are the two domains (cross-domain transfer);
    otherwise stimuli are randomly split into disjoint halves under ``split_seed``.
    """
    residuals = stimulus_removed_residuals(resp.data, leave_one_out=leave_one_out)
    if domain_a is not None and domain_b is not None:
        idx_a = resp.stimulus_indices(domain_a)
        idx_b = resp.stimulus_indices(domain_b)
        if idx_a.size == 0 or idx_b.size == 0:
            raise ValueError(f"empty domain: {domain_a}={idx_a.size}, {domain_b}={idx_b.size}")
    else:
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(resp.n_stimuli)
        half = len(perm) // 2
        idx_a, idx_b = perm[:half], perm[half : 2 * half]
    return subject_fingerprints(residuals, idx_a), subject_fingerprints(residuals, idx_b)


def score_identification(fp_a: np.ndarray, fp_b: np.ndarray) -> tuple[float, float]:
    """Top-1 accuracy and mean true-match rank for a fingerprint pair (database, target)."""
    return _score(fp_a, fp_b)


def identification_accuracy(
    resp: Responses,
    split_seed: int = 0,
    n_splits: int = 1,
    domain_a: str | None = None,
    domain_b: str | None = None,
) -> dict:
    """Top-1 subject identification accuracy across disjoint stimulus halves.

    Args:
        resp: the responses to audit.
        split_seed: RNG seed for the A/B stimulus split (within-domain case).
        n_splits: number of random A/B splits to average over (within-domain case).
        domain_a / domain_b: if both given, identify *across domains* — fingerprints A
            from ``domain_a`` stimuli, fingerprints B from ``domain_b`` stimuli (no
            split needed). This is the cross-domain consolidation test.

    Returns:
        dict with ``accuracy`` (top-1), ``mean_rank`` (1 = perfect), ``chance``
        (``1/n_subjects``), and ``n_subjects``.
    """
    n_subj = resp.n_subjects
    accs: list[float] = []
    ranks: list[float] = []

    if domain_a is not None and domain_b is not None:
        fp_a, fp_b = fingerprint_pair(resp, domain_a=domain_a, domain_b=domain_b)
        acc, rank = _score(fp_a, fp_b)
        accs.append(acc)
        ranks.append(rank)
    else:
        for i in range(n_splits):
            fp_a, fp_b = fingerprint_pair(resp, split_seed=split_seed + i)
            acc, rank = _score(fp_a, fp_b)
            accs.append(acc)
            ranks.append(rank)

    return {
        "accuracy": float(np.mean(accs)),
        "mean_rank": float(np.mean(ranks)),
        "chance": 1.0 / n_subj,
        "n_subjects": n_subj,
        "n_splits": len(accs),
    }


def _score(fp_a: np.ndarray, fp_b: np.ndarray) -> tuple[float, float]:
    """Top-1 accuracy and mean true-match rank for one fingerprint pair."""
    corr = identification_matrix(fp_a, fp_b)  # (db, target)
    n = corr.shape[0]
    # for each target j, rank database subjects by correlation (desc)
    order = np.argsort(-corr, axis=0)  # (db, target)
    pred = order[0, :]  # top db match per target
    truth = np.arange(n)
    acc = float(np.mean(pred == truth))
    # rank of the true match (1-indexed)
    ranks = np.empty(n)
    for j in range(n):
        ranks[j] = int(np.where(order[:, j] == j)[0][0]) + 1
    return acc, float(np.mean(ranks))
