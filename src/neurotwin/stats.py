"""Significance and uncertainty for the identifiability metrics.

- ``permutation_null``: the null distribution of identification accuracy under "no
  individuality". We build the fingerprint pair once, then permute which *target*
  identity each column is matched against. Operating on the already-aggregated
  fingerprints (rather than shuffling raw, temporally-autocorrelated stimulus windows)
  keeps the null calibrated — window autocorrelation cannot inflate it.
- ``bootstrap_ci``: a confidence interval that resamples **stimulus blocks** (e.g.
  whole movies) with replacement, since windows within a stimulus are not independent.
  The number of independent blocks is the true sample size and is reported.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .fingerprint import fingerprint_pair, score_identification
from .responses import Responses


def permutation_null(
    resp: Responses,
    n_perm: int = 200,
    seed: int = 0,
    split_seed: int = 0,
    domain_a: str | None = None,
    domain_b: str | None = None,
    leave_one_out: bool = True,
) -> dict:
    """Label-permutation null for identification accuracy.

    Args:
        resp: responses to test.
        n_perm: number of label permutations.
        seed: RNG seed for the permutations.
        split_seed: seed for the single A/B stimulus split (ignored if domains given).
        domain_a / domain_b: cross-domain identification (else within, via split).
        leave_one_out: LOO stimulus-removed residuals (default).

    Returns:
        dict with ``observed`` (single-split top-1 accuracy), ``null_mean``,
        ``null_std``, ``p_value`` (one-sided), ``chance``, and the raw ``null`` array.
    """
    rng = np.random.default_rng(seed)
    fp_a, fp_b = fingerprint_pair(
        resp, split_seed=split_seed, domain_a=domain_a, domain_b=domain_b,
        leave_one_out=leave_one_out,
    )
    observed, _ = score_identification(fp_a, fp_b)
    n = fp_a.shape[0]

    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n)
        null[k], _ = score_identification(fp_a, fp_b[perm])

    p = (np.sum(null >= observed) + 1) / (n_perm + 1)  # +1 smoothing avoids p=0
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value": float(p),
        "chance": 1.0 / n,
        "null": null,
    }


def bootstrap_ci(
    resp: Responses,
    metric: Callable[[Responses], float],
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
    blocks: np.ndarray | None = None,
) -> dict:
    """Block-resampled bootstrap CI for a scalar metric.

    Resamples **stimuli** (default) or **stimulus blocks** (``blocks`` given, one block
    id per stimulus — e.g. the movie a clip came from) with replacement. We never
    resample subjects: duplicated subjects are perfectly confusable and would bias any
    identification metric downward.

    Returns:
        dict with ``estimate``, ``lo``, ``hi``, ``alpha``, and ``n_units`` (number of
        independent resampling units = the effective sample size).
    """
    rng = np.random.default_rng(seed)
    estimate = float(metric(resp))
    boots = np.empty(n_boot)

    if blocks is not None:
        blocks = np.asarray(blocks)
        uniq = np.unique(blocks)
        member_idx = {b: np.where(blocks == b)[0] for b in uniq}
        for k in range(n_boot):
            chosen = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([member_idx[b] for b in chosen])
            boots[k] = metric(resp.select_stimuli(idx))
        n_units = int(len(uniq))
    else:
        n_c = resp.n_stimuli
        for k in range(n_boot):
            idx = rng.integers(0, n_c, size=n_c)
            boots[k] = metric(resp.select_stimuli(idx))
        n_units = int(n_c)

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {"estimate": estimate, "lo": lo, "hi": hi, "alpha": alpha, "n_units": n_units}
