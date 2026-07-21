"""ROI mapping for the per-feature subject-variance maps.

The real pipeline maps per-vertex subject-variance onto the fsaverage5 surface and
aggregates by a standard atlas (Yeo 7/17-network or Glasser MMP1.0) via nilearn. nilearn
is an optional (``viz``) dependency, so it is imported lazily — the core analysis and
the test suite never require it.
"""

from __future__ import annotations

import numpy as np


def aggregate_by_atlas(per_vertex: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Mean of a per-vertex quantity within each atlas label.

    Args:
        per_vertex: ``(n_vertices,)`` quantity (e.g. subject-variance fraction).
        labels: ``(n_vertices,)`` integer atlas label per vertex (0 = medial wall / unassigned).

    Returns:
        ``{label: mean_value}`` for every non-zero label.
    """
    per_vertex = np.asarray(per_vertex)
    labels = np.asarray(labels)
    if per_vertex.shape != labels.shape:
        raise ValueError(f"shape mismatch: {per_vertex.shape} vs {labels.shape}")
    out: dict[int, float] = {}
    for lab in np.unique(labels):
        if lab == 0:
            continue
        out[int(lab)] = float(per_vertex[labels == lab].mean())
    return out


def fetch_yeo_fsaverage5_labels(networks: int = 7):  # pragma: no cover - needs nilearn+net
    """Fetch Yeo network labels resampled to fsaverage5 (requires nilearn).

    Returns a ``(n_vertices,)`` integer label array. Raises a clear error if nilearn is
    not installed so the message is actionable.
    """
    try:
        from nilearn import datasets  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "ROI atlas fetching needs the 'viz' extra: pip install -e 'research[viz]'"
        ) from e
    raise NotImplementedError(
        "Wire up nilearn Yeo->fsaverage5 resampling here once Phase 2 surface maps are run; "
        "the analysis layer only needs aggregate_by_atlas() with a label vector."
    )
