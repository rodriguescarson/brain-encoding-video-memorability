"""RQ1a (SANITY CHECK) — within-domain subject fingerprinting.

NOTE (post-council): within-domain identification of a model that has an explicit
per-subject Subject Block is near-guaranteed to succeed — it largely re-reads the
model's own conditioning variable, so it does NOT by itself distinguish an individuating
fingerprint from a low-rank nuisance. Treat this as a sanity check; the load-bearing
results are the cross-domain transfer (rq1_crossdomain.py) and the per-ROI variance map.

Reports top-1 identification accuracy with a label-permutation null and a bootstrap CI.
Run on synthetic demo data:

    python research/analysis/rq1_fingerprint.py --demo

or on real TRIBE v2 predictions once extracted (see configs/experiment_rq1.yaml).
"""

from __future__ import annotations

import argparse

from _common import add_common_args, load

from neurotwin.fingerprint import identification_accuracy
from neurotwin.io import save_results
from neurotwin.stats import bootstrap_ci, permutation_null


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()

    resp, meta = load(args)
    acc = identification_accuracy(resp, n_splits=args.n_splits)
    null = permutation_null(resp, n_perm=args.n_perm)
    ci = bootstrap_ci(
        resp,
        lambda r: identification_accuracy(r, n_splits=args.n_splits)["accuracy"],
        n_boot=args.n_boot,
    )

    result = {
        "meta": meta,
        "n_subjects": resp.n_subjects,
        "n_stimuli": resp.n_stimuli,
        "accuracy": acc["accuracy"],
        "mean_rank": acc["mean_rank"],
        "chance": acc["chance"],
        "ci95": [ci["lo"], ci["hi"]],
        "permutation": {
            "null_mean": null["null_mean"],
            "null_std": null["null_std"],
            "p_value": null["p_value"],
        },
    }
    out_path = f"{args.out}/rq1_fingerprint.json"
    save_results(result, out_path)

    significant = null["p_value"] < 0.05 and acc["accuracy"] > 2 * acc["chance"]
    verdict = "IDENTITY-BEARING" if significant else "near chance"
    print(f"identification accuracy : {acc['accuracy']:.3f}  (chance {acc['chance']:.3f})")
    print(f"95% CI                  : [{ci['lo']:.3f}, {ci['hi']:.3f}]")
    print(f"permutation p-value     : {null['p_value']:.4f}")
    print(f"verdict                 : {verdict}")
    print(f"written                 : {out_path}")


if __name__ == "__main__":
    main()
