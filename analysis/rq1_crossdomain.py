"""RQ1c — PRIMARY RESULT. Cross-domain subject identification.

Does a fingerprint learned on one stimulus domain (e.g. movies) re-identify the same
subject in another (e.g. podcasts)? This is the one outcome the model card cannot
predict: cross-domain transfer => a stimulus-general, modality-invariant subject signature;
domain-locked (high within, chance across) => the per-subject signal is modality-specific.
Reported with a label-permutation null and a block bootstrap (effective n = #stimuli, or
#movies on real data via a block column).

    python research/analysis/rq1_crossdomain.py --demo
"""

from __future__ import annotations

import argparse
import itertools

from _common import add_common_args, load

from neurotwin.fingerprint import identification_accuracy
from neurotwin.io import save_results
from neurotwin.stats import bootstrap_ci, permutation_null


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()

    resp, meta = load(args)
    domains = sorted(set(resp.domains))
    if len(domains) < 2:
        raise SystemExit(f"need >=2 stimulus domains for cross-domain test; got {domains}")

    within = {
        d: identification_accuracy(resp.select_stimuli(resp.stimulus_indices(d)), n_splits=20)[
            "accuracy"
        ]
        for d in domains
    }

    # cross-domain accuracy + a permutation null per direction
    cross, cross_p = {}, {}
    for a, b in itertools.permutations(domains, 2):
        key = f"{a}->{b}"
        cross[key] = identification_accuracy(resp, domain_a=a, domain_b=b)["accuracy"]
        cross_p[key] = permutation_null(resp, n_perm=args.n_perm, domain_a=a, domain_b=b)["p_value"]

    mean_within = sum(within.values()) / len(within)
    mean_cross = sum(cross.values()) / len(cross)
    consolidation = mean_cross / mean_within if mean_within > 0 else 0.0

    def mean_cross_metric(r):
        accs = [
            identification_accuracy(r, domain_a=a, domain_b=b)["accuracy"]
            for a, b in itertools.permutations(domains, 2)
        ]
        return sum(accs) / len(accs)

    ci = bootstrap_ci(resp, mean_cross_metric, n_boot=args.n_boot)

    result = {
        "meta": meta,
        "domains": domains,
        "within_domain_accuracy": within,
        "cross_domain_accuracy": cross,
        "cross_domain_p_values": cross_p,
        "mean_within": mean_within,
        "mean_cross": mean_cross,
        "mean_cross_ci95": [ci["lo"], ci["hi"]],
        "effective_n": ci["n_units"],
        "consolidation_index": consolidation,
        "chance": 1.0 / resp.n_subjects,
    }
    out_path = f"{args.out}/rq1_crossdomain.json"
    save_results(result, out_path)

    label = "MODALITY-INVARIANT" if consolidation > 0.6 else "modality-specific"
    pvals = ", ".join(f"{k}:{v:.3f}" for k, v in cross_p.items())
    print(f"mean within-domain acc  : {mean_within:.3f}")
    print(f"mean cross-domain acc   : {mean_cross:.3f}  (chance {result['chance']:.3f})")
    print(f"  95% CI (n={ci['n_units']:>2})        : [{ci['lo']:.3f}, {ci['hi']:.3f}]")
    print(f"  perm p-values         : {pvals}")
    print(f"consolidation index     : {consolidation:.3f}  -> {label}")
    print(f"written                 : {out_path}")


if __name__ == "__main__":
    main()
