"""The operationalized deliverable: when to trust a per-subject TRIBE prediction.

The council's archival condition was that "use per-subject vs. group-average" must be a
*thresholded, validated decision procedure*, not a discussion paragraph. This is it.

Two registered gates:

1. **Global consolidation gate.** If subject identity does not transfer across stimulus
   domains (cross-domain identification ≤ chance + ``delta``), the per-subject component is
   a surface episode / nuisance — use the group average everywhere.
2. **Per-ROI individuation gate.** Where identity *does* consolidate, trust the per-subject
   prediction only in ROIs whose subject-variance share ≥ ``tau_var``; elsewhere the
   per-subject signal is too thin to be reliable — fall back to the group average.

Thresholds are pre-registered defaults (see ``paper/preregistration.md``); pass overrides
explicitly and report them.
"""

from __future__ import annotations

import itertools

import numpy as np

from .fingerprint import identification_accuracy
from .responses import Responses
from .rois import aggregate_by_atlas
from .variance import variance_partition

# Pre-registered defaults.
DEFAULT_TAU_VAR = 0.15  # subject-variance share for an ROI to count as individuating
DEFAULT_DELTA = 0.10  # cross-domain identification must beat chance by this margin


def _cross_domain_accuracy(resp: Responses) -> float:
    """Mean cross-domain identification, or within-domain if only one domain exists."""
    domains = sorted(set(resp.domains))
    if len(domains) >= 2:
        accs = [
            identification_accuracy(resp, domain_a=a, domain_b=b)["accuracy"]
            for a, b in itertools.permutations(domains, 2)
        ]
        return float(np.mean(accs))
    return identification_accuracy(resp, n_splits=10)["accuracy"]


def guidance_rule(
    resp: Responses,
    labels: np.ndarray,
    tau_var: float = DEFAULT_TAU_VAR,
    delta: float = DEFAULT_DELTA,
) -> dict:
    """Decide, per ROI, whether per-subject TRIBE predictions are trustworthy.

    Args:
        resp: predicted responses (subject × stimulus × vertex).
        labels: ``(n_vertices,)`` integer atlas label per vertex (0 = unassigned).
        tau_var: per-ROI subject-variance-share threshold.
        delta: cross-domain-identification margin over chance for the global gate.

    Returns:
        dict with the global decision, the cross-domain accuracy + chance, a per-ROI table
        (subject-variance share + trust flag), the fraction of ROIs flagged trustworthy,
        and the thresholds used.
    """
    parts = variance_partition(resp.data)
    subj_per_roi = aggregate_by_atlas(parts["subject"], labels)

    cross = _cross_domain_accuracy(resp)
    chance = 1.0 / resp.n_subjects
    consolidated = cross >= chance + delta

    per_roi = {
        int(roi): {
            "subject_var_share": float(share),
            "trust_per_subject": bool(consolidated and share >= tau_var),
        }
        for roi, share in subj_per_roi.items()
    }
    n_trust = sum(d["trust_per_subject"] for d in per_roi.values())
    n_rois = len(per_roi)

    decision = "per-subject-where-flagged" if consolidated else "group-average-everywhere"
    return {
        "global_decision": decision,
        "consolidated": consolidated,
        "cross_domain_accuracy": cross,
        "chance": chance,
        "per_roi": per_roi,
        "n_rois": n_rois,
        "n_trust_per_subject": int(n_trust),
        "frac_trusted": (n_trust / n_rois) if n_rois else 0.0,
        "thresholds": {"tau_var": tau_var, "delta": delta},
    }
