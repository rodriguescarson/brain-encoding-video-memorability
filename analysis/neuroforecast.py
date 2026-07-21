#!/usr/bin/env python3
"""Pre-registered pilot analysis for the Brain Bottleneck (Paper 13 pivot).

Question: do TRIBE's predicted-brain features forecast video memorability BETTER than a matched
stimulus-feature baseline (V-JEPA2, TRIBE's own backbone) — within a dataset and, the real prize,
ACROSS datasets (Memento10k -> VideoMem)?

Decision rule (locked in PLAN.md, do not move the goalposts):
  GO    if  SRCC(brain) - SRCC(baseline) >= +0.02  AND the paired-bootstrap 95% CI excludes 0
            (ideally within-dataset AND cross-dataset)
  NO-GO if  |delta| < 0.02  (brain projection is decorative; pivot to a negative-result paper)

Feature-file contract (what the A100 pilot notebook emits): an .npz with
  'stim_ids' : (N,) array of stimulus ids (str)
  'X'        : (N, d) float feature matrix
Labels: a CSV with columns  stimulus_id,score  (memorability in [0,1]).

Usage:
    python3 analysis/neuroforecast.py --synthetic              # self-test the pipeline (no data)
    python3 analysis/neuroforecast.py \
        --brain data/features/memento_brain.npz \
        --baseline data/features/memento_vjepa.npz \
        --labels data/labels/memento.csv \
        [--brain-test ... --baseline-test ... --labels-test ...]   # cross-dataset
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-1, 4, 12)
PREREG_MARGIN = 0.02   # SRCC delta that counts as "brain beats baseline" (pre-registered)
N_SPLITS = 5
N_BOOT = 2000


def srcc(a, b) -> float:
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0


def oof_predict(X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Out-of-fold ridge predictions (standardize inside each fold; RidgeCV picks alpha)."""
    oof = np.zeros_like(y, dtype=float)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return oof


def fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    """Train on all of (Xtr,ytr), predict Xte — for the cross-dataset transfer test."""
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def paired_bootstrap(y, pred_brain, pred_base, seed: int = 0) -> dict:
    """95% CI on SRCC(brain) - SRCC(baseline) by resampling stimuli with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        deltas[i] = srcc(pred_brain[idx], y[idx]) - srcc(pred_base[idx], y[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_mean": float(deltas.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_brain_not_better": float((deltas <= 0).mean())}


def compare(name, Xb, Xv, y, seed=0) -> dict:
    """Within-dataset: OOF SRCC for brain vs baseline + paired bootstrap on the delta."""
    pb, pv = oof_predict(Xb, y, seed), oof_predict(Xv, y, seed)
    sb, sv = srcc(pb, y), srcc(pv, y)
    boot = paired_bootstrap(y, pb, pv, seed)
    verdict = "GO" if (sb - sv >= PREREG_MARGIN and boot["ci_lo"] > 0) else "NO-GO"
    return {"dataset": name, "n": int(len(y)), "srcc_brain": round(sb, 4),
            "srcc_baseline": round(sv, 4), "delta": round(sb - sv, 4),
            "bootstrap": {k: round(v, 4) for k, v in boot.items()}, "within_verdict": verdict}


def cross_dataset(Xb_tr, Xv_tr, y_tr, Xb_te, Xv_te, y_te) -> dict:
    """Fit on the training dataset, score the held-out dataset — the generalization prize (H1b)."""
    sb = srcc(fit_predict(Xb_tr, y_tr, Xb_te), y_te)
    sv = srcc(fit_predict(Xv_tr, y_tr, Xv_te), y_te)
    return {"srcc_brain": round(sb, 4), "srcc_baseline": round(sv, 4), "delta": round(sb - sv, 4),
            "cross_verdict": "GO" if sb - sv >= PREREG_MARGIN else "NO-GO"}


# ----------------------------- loading -----------------------------
def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["stim_ids"]).astype(str), np.asarray(d["X"], dtype=float)


def load_labels(path: str) -> dict:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[str(row["stimulus_id"])] = float(row["score"])
    return out


def align(brain_npz, base_npz, labels_csv):
    """Intersect stimulus ids across brain features, baseline features, and labels; return aligned X,y."""
    bid, Xb = load_npz(brain_npz)
    vid, Xv = load_npz(base_npz)
    lab = load_labels(labels_csv)
    bi, vi = {s: i for i, s in enumerate(bid)}, {s: i for i, s in enumerate(vid)}
    keep = [s for s in bid if s in vi and s in lab]
    if not keep:
        sys.exit("no overlapping stimulus ids across brain / baseline / labels")
    Xb = np.vstack([Xb[bi[s]] for s in keep])
    Xv = np.vstack([Xv[vi[s]] for s in keep])
    y = np.array([lab[s] for s in keep])
    return Xb, Xv, y


# ----------------------------- synthetic self-test -----------------------------
def synthetic(seed=0):
    """Two regimes to prove the pipeline: a GO world (brain carries extra memorability signal beyond
    the baseline) and a NULL world (brain is just a rotation of the baseline = no added value)."""
    rng = np.random.default_rng(seed)
    n, d = 400, 60
    z = rng.standard_normal(n)                 # latent memorability
    base = z[:, None] * rng.standard_normal((1, d)) * 0.5 + rng.standard_normal((n, d))
    extra = rng.standard_normal(n)             # signal ONLY the brain space sees
    brain_go = base + extra[:, None] * rng.standard_normal((1, d)) * 0.8
    y = 0.6 * z + 0.5 * extra + 0.3 * rng.standard_normal(n)
    y = (y - y.min()) / (y.max() - y.min())
    brain_null = base + rng.standard_normal((n, d))   # baseline + fresh noise: no new signal -> NO-GO
    print("[synthetic] GO world  :", json.dumps(compare("synthetic-GO", brain_go, base, y, seed)))
    print("[synthetic] NULL world:", json.dumps(compare("synthetic-NULL", brain_null, base, y, seed)))
    print("\nExpect: GO world -> within_verdict GO (brain delta clearly > 0); "
          "NULL world -> NO-GO. If so, the analysis is sound.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="run the self-test, no data needed")
    ap.add_argument("--brain"); ap.add_argument("--baseline"); ap.add_argument("--labels")
    ap.add_argument("--brain-test"); ap.add_argument("--baseline-test"); ap.add_argument("--labels-test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/results/neuroforecast_pilot.json")
    args = ap.parse_args()

    if args.synthetic:
        synthetic(args.seed); return
    if not (args.brain and args.baseline and args.labels):
        sys.exit("need --brain --baseline --labels (or --synthetic)")

    Xb, Xv, y = align(args.brain, args.baseline, args.labels)
    res = {"prereg_margin": PREREG_MARGIN, "within": compare("primary", Xb, Xv, y, args.seed)}

    if args.brain_test and args.baseline_test and args.labels_test:
        Xb2, Xv2, y2 = align(args.brain_test, args.baseline_test, args.labels_test)
        res["cross_dataset"] = cross_dataset(Xb, Xv, y, Xb2, Xv2, y2)

    w = res["within"]["within_verdict"]
    c = res.get("cross_dataset", {}).get("cross_verdict", "n/a")
    res["GO_NO_GO"] = "GO" if w == "GO" and c in ("GO", "n/a") else "NO-GO"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nDECISION: {res['GO_NO_GO']}  (within={w}, cross={c}) -> {args.out}")


if __name__ == "__main__":
    main()
