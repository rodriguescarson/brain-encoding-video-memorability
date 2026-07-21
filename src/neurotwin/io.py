"""Loading predicted responses + building the (subject, stimulus, feature) tensor.

The remote extractor (``remote/extract_backbone.py``) writes one array per
(subject, stimulus) prediction plus a small index. Here we assemble those into a dense
:class:`Responses`, applying a temporal reduction to turn each
``(n_timesteps, n_vertices)`` prediction into a per-stimulus feature vector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .responses import Responses


def _flatten_fc(x: np.ndarray, max_features: int = 2000) -> np.ndarray:
    """Flattened upper-triangle temporal correlation across features (FC-style).

    ``x`` is ``(T, V)``; returns the ``V*(V-1)/2`` upper-triangle of the V×V Pearson
    correlation over time. Finn-style fingerprinting uses connectivity, not means, so
    this captures the subject×stimulus *interaction* structure a temporal mean discards.

    Guarded for size: at whole-cortex resolution (V≈20k) the V×V matrix is infeasible —
    parcellate to ROIs first (see :func:`roi_functional_connectivity`) and pass the
    ROI-time matrix here.
    """
    v = x.shape[1]
    if v > max_features:
        raise ValueError(
            f"FC on {v} features is too large; parcellate to <= {max_features} ROIs first "
            "(see roi_functional_connectivity)"
        )
    if x.shape[0] < 2:
        return np.zeros(v * (v - 1) // 2)
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    iu = np.triu_indices(v, k=1)
    return corr[iu]


# temporal reductions: (n_timesteps, n_vertices) -> (n_features,)
REDUCERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": lambda x: x.mean(axis=0),
    "std": lambda x: x.std(axis=0),
    "var": lambda x: x.var(axis=0),
    "meanstd": lambda x: np.concatenate([x.mean(axis=0), x.std(axis=0)]),
    "fc": _flatten_fc,
}


def roi_functional_connectivity(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """ROI-averaged timecourses then flattened FC — the real-data FC reducer.

    Args:
        pred: ``(T, V)`` whole-cortex prediction.
        labels: ``(V,)`` integer atlas label per vertex (0 = unassigned, dropped).

    Returns:
        flattened upper-triangle of the ROI×ROI temporal correlation.
    """
    pred = np.asarray(pred, dtype=np.float64)
    labels = np.asarray(labels)
    rois = [lab for lab in np.unique(labels) if lab != 0]
    roi_ts = np.stack([pred[:, labels == lab].mean(axis=1) for lab in rois], axis=1)
    return _flatten_fc(roi_ts, max_features=len(rois) + 1)


def reduce_prediction(pred: np.ndarray, reducer: str = "mean") -> np.ndarray:
    """Reduce a ``(T, V)`` prediction to a fixed-length feature vector."""
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim != 2:
        raise ValueError(f"prediction must be 2-D (time, vertices); got {pred.shape}")
    if reducer not in REDUCERS:
        raise ValueError(f"unknown reducer {reducer!r}; choose from {list(REDUCERS)}")
    return REDUCERS[reducer](pred)


def load_responses_from_dir(
    pred_dir: str | Path,
    reducer: str = "mean",
) -> Responses:
    """Assemble a Responses from a directory of per-pair prediction files.

    Expects ``pred_dir/index.json`` of the form::

        {"reducer_default": "mean",
         "items": [{"subject": "sub-01", "stimulus": "friends_s01e01",
                    "domain": "movie", "file": "sub-01__friends_s01e01.npy"}, ...]}

    Each ``file`` is a ``(n_timesteps, n_vertices)`` array. All (subject x stimulus)
    pairs must be present (balanced grid).
    """
    pred_dir = Path(pred_dir)
    index = json.loads((pred_dir / "index.json").read_text())
    items = index["items"]

    subjects = sorted({it["subject"] for it in items})
    stimuli = sorted({it["stimulus"] for it in items})
    domain_of = {it["stimulus"]: it.get("domain", "default") for it in items}
    s_pos = {s: i for i, s in enumerate(subjects)}
    c_pos = {c: i for i, c in enumerate(stimuli)}

    feat_dim: int | None = None
    grid: dict[tuple[int, int], np.ndarray] = {}
    for it in items:
        vec = reduce_prediction(np.load(pred_dir / it["file"]), reducer)
        if feat_dim is None:
            feat_dim = vec.shape[0]
        elif vec.shape[0] != feat_dim:
            raise ValueError(f"feature dim mismatch for {it['file']}: {vec.shape[0]} != {feat_dim}")
        grid[(s_pos[it["subject"]], c_pos[it["stimulus"]])] = vec

    expected = len(subjects) * len(stimuli)
    if len(grid) != expected:
        raise ValueError(f"unbalanced grid: {len(grid)} pairs, expected {expected}")

    data = np.empty((len(subjects), len(stimuli), feat_dim))
    for (si, ci), vec in grid.items():
        data[si, ci, :] = vec

    domains = [domain_of[c] for c in stimuli]
    return Responses(data, subjects, stimuli, domains)


def save_results(results: dict, path: str | Path) -> None:
    """Write a metrics dict to JSON, coercing numpy types to native Python."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def coerce(o):
        if isinstance(o, dict):
            return {k: coerce(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [coerce(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    path.write_text(json.dumps(coerce(results), indent=2))
