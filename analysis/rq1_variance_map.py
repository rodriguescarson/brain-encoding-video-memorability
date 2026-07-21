"""RQ1b — Where does individuality live? Per-feature variance partition + geometry.

Partitions per-vertex predicted-response variance into subject / stimulus / interaction,
and measures the effective dimensionality (participation ratio) of the subject manifold.
A low participation ratio => the "twin" is a low-rank nuisance correction; a high one =>
a rich, individuating subspace.

    python research/analysis/rq1_variance_map.py --demo
"""

from __future__ import annotations

import argparse

import numpy as np
from _common import add_common_args, load

from neurotwin.geometry import effective_dimensionality
from neurotwin.io import save_results
from neurotwin.variance import variance_partition


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()

    resp, meta = load(args)
    parts = variance_partition(resp.data)
    geom = effective_dimensionality(resp.data)

    subj = parts["subject"]
    result = {
        "meta": meta,
        "subject_variance": {
            "mean": float(np.mean(subj)),
            "median": float(np.median(subj)),
            "p90": float(np.percentile(subj, 90)),
            "max": float(np.max(subj)),
        },
        "stimulus_variance_mean": float(np.mean(parts["stimulus"])),
        "interaction_variance_mean": float(np.mean(parts["interaction"])),
        "participation_ratio": geom["participation_ratio"],
        "participation_ratio_normalized": geom["normalized"],
        "max_possible_dim": geom["max_possible"],
        # full per-feature subject-variance vector for the surface map (Phase 2, nilearn)
        "subject_variance_per_feature": subj,
    }
    out_path = f"{args.out}/rq1_variance_map.json"
    save_results(result, out_path)

    print(f"subject variance (mean) : {result['subject_variance']['mean']:.3f}")
    print(f"stimulus variance (mean): {result['stimulus_variance_mean']:.3f}")
    print(
        f"participation ratio     : {geom['participation_ratio']:.2f} "
        f"/ {geom['max_possible']} (norm {geom['normalized']:.3f})"
    )
    print(f"written                 : {out_path}")


if __name__ == "__main__":
    main()
