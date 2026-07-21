#!/usr/bin/env python3
"""Reduce the full per-clip TRIBE temporal responses to fixed-length feature vectors.

The A100 re-extraction (remote/tribe_temporal_reextract.py) stored, for each clip, the
FULL predicted cortical response as a (T, 20484) float16 array on Drive, one .npy per clip
(T = 3-4 timepoints for a 3 s Memento clip). Pull that raw_temporal/ folder into
data/predictions/raw_temporal/ (via Drive desktop / rclone / a Colab zip-download), then run
this script to compute the temporal reductions that analysis/temporal_neuroforecast.py consumes:

  mean   time-average (== the run-1 feature; the thing we want to beat)
  std    temporal variability across the trajectory
  early  first timepoint (early perceptual response)
  late   last timepoint (the LATE, sustained memorability response; Bainbridge/PLOS-Biol 2024)
  slope  per-vertex linear trend over time (rise/decay of the response)

Each reduction is written to memento_brain_<reduction>.npz with key "X" of shape (N, 20484),
row-aligned to memento_labels.csv (same order as memento_vjepa.npz), so the downstream
partial-correlation tests line up. Rows whose raw file is missing are mean-imputed and logged.

Non-commercial data note: raw_temporal/ and these .npz stay gitignored (Memento license).
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "data" / "predictions"
RAW = PRED / "raw_temporal"
V_CORTEX = 20484


def _reduce_one(arr: np.ndarray) -> dict[str, np.ndarray]:
    """arr: (T, V) float32 -> dict of (V,) reductions."""
    T = arr.shape[0]
    mean = arr.mean(0)
    std = arr.std(0)
    early = arr[0]
    late = arr[-1]
    if T >= 2:
        # per-vertex least-squares slope over centered, unit-spaced time
        t = np.arange(T, dtype=np.float32)
        t -= t.mean()
        slope = (t[:, None] * arr).sum(0) / (t @ t)
    else:
        slope = np.zeros_like(mean)
    return {"mean": mean, "std": std, "early": early, "late": late, "slope": slope}


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"missing {RAW} - pull the raw_temporal/ folder from Drive first")

    labels_path = PRED / "memento_labels.csv"
    rows = list(csv.reader(open(labels_path)))[1:]
    ids = [r[0] for r in rows]
    print(f"{len(ids)} label rows; {len(list(RAW.glob('*.npy')))} raw .npy on disk")

    keys = ["mean", "std", "early", "late", "slope"]
    cols: dict[str, list] = {k: [] for k in keys}
    missing, Ts = [], []
    for sid in ids:
        f = RAW / f"{sid}.npy"
        if not f.exists():
            missing.append(sid)
            for k in keys:
                cols[k].append(None)  # placeholder, mean-imputed below
            continue
        arr = np.load(f).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = arr[:, :V_CORTEX]
        Ts.append(arr.shape[0])
        red = _reduce_one(arr)
        for k in keys:
            cols[k].append(red[k])

    if Ts:
        print(f"T range across clips: {min(Ts)}-{max(Ts)}")
    if missing:
        print(f"WARNING: {len(missing)} clips missing raw files (mean-imputed): {missing[:5]}")

    PRED.mkdir(parents=True, exist_ok=True)
    for k in keys:
        present = [c for c in cols[k] if c is not None]
        fill = np.mean(present, axis=0)
        X = np.stack([c if c is not None else fill for c in cols[k]]).astype(np.float32)
        out = PRED / f"memento_brain_{k}.npz"
        np.savez_compressed(out, X=X)
        print(f"wrote {out.name}  shape {X.shape}")

    print("\nnext: python analysis/temporal_neuroforecast.py")


if __name__ == "__main__":
    main()
