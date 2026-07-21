#!/usr/bin/env python3
"""Localize the *unique* memorability signal in TRIBE's predicted brain responses.

We already know (neuroforecast.py) that the time-averaged brain features do NOT beat the
V-JEPA2 backbone, yet carry a component of memorability that is orthogonal to vision
(partial SRCC ~0.19, permutation p<1e-3). This script asks WHERE on the cortex that unique
signal lives, using the Destrieux fsaverage5 surface atlas, and tests the pre-registered
neuroscience prediction: it should concentrate in the ventral visual stream (fusiform / IT /
lateral-occipital) and medial-temporal cortex (Bainbridge 2017; Rust 2019; PLOS Biol 2024).

Method per ROI r:
  vpred = out-of-fold RidgeCV prediction from V-JEPA2 (the vision nuisance to control for)
  rpred = out-of-fold RidgeCV prediction from ROI r's vertices only
  unique_r = partial Spearman(rpred, y | vpred)   (residualize both on vpred, then rank-corr)
  permutation p (shuffle the vision-residualized label), BH-FDR across ROIs.

Assumption: TRIBE outputs the standard fsaverage5 vertex order (LH 0-10241, RH 10242-20483),
matching nilearn's Destrieux map_left/map_right. Flagged in the header of the JSON output.

Reads gitignored data/predictions/*, writes data/results/roi_localization.json (aggregate
numbers only; no per-subject Memento data leaves the repo boundary).
"""
from __future__ import annotations
import csv, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "data" / "predictions"
ALPHAS = np.logspace(-1, 5, 10)
NPERM = 4000
SEED = 0

# Destrieux region-name substrings that the memorability literature implicates.
VENTRAL_MTL = ("fusifor", "temporal_inf", "oc-temp", "Lingual", "collat", "Pole_occipital",
               "occipital_inf", "occipital_middle", "Parahip", "temp_sup")  # VVS + MTL-cortex + STS


def load():
    Xb = np.load(PRED / "memento_brain.npz")["X"].astype(np.float64)
    Xv = np.load(PRED / "memento_vjepa.npz")["X"].astype(np.float64)
    y = np.array([float(r[1]) for r in list(csv.reader(open(PRED / "memento_labels.csv")))[1:]])
    return Xb, Xv, y


def atlas_vertex_labels():
    from nilearn import datasets
    d = datasets.fetch_atlas_surf_destrieux()
    names = [l.decode() if isinstance(l, bytes) else l for l in d["labels"]]
    lh, rh = np.asarray(d["map_left"]), np.asarray(d["map_right"])
    # (hemi, label_id) -> vertex indices in the 20484 concat space
    rois = {}
    for hemi, arr, off in (("L", lh, 0), ("R", rh, 10242)):
        for lab in np.unique(arr):
            nm = names[lab]
            if nm in ("Unknown", "Medial_wall"):
                continue
            idx = np.where(arr == lab)[0] + off
            rois[f"{hemi}.{nm}"] = idx
    return rois, names


def oof(X, y, seed=SEED):
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return pred


def _z_ranks(a):
    r = rankdata(a).astype(float)
    return (r - r.mean()) / r.std()


def _resid(a, b):  # residual of a after linear regression on b (single column)
    B = np.c_[np.ones_like(b), b]
    beta = np.linalg.lstsq(B, a, rcond=None)[0]
    return a - B @ beta


def partial_spearman_perm(rpred, y, vpred, rng):
    """partial Spearman(rpred, y | vpred) with a permutation p-value."""
    ry, rr = _resid(y, vpred), _resid(rpred, vpred)
    zy, zr = _z_ranks(ry), _z_ranks(rr)
    n = len(zy)
    obs = float(zr @ zy / n)
    perms = np.array([rng.permutation(n) for _ in range(NPERM)])
    null = (zy[perms] @ zr) / n  # (NPERM,)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (NPERM + 1)
    return obs, float(p)


def bh_fdr(pvals):
    p = np.asarray(pvals); m = len(p); order = np.argsort(p)
    q = np.empty(m); prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, p[i] * m / (m - rank + 1)); q[i] = prev
    return q


def main():
    rng = np.random.default_rng(SEED)
    Xb, Xv, y = load()
    rois, _ = atlas_vertex_labels()
    print(f"loaded brain {Xb.shape}, vjepa {Xv.shape}; {len(rois)} Destrieux ROIs")

    vpred = oof(Xv, y)                      # vision nuisance
    bpred = oof(Xb, y)                      # whole-brain (for reference)
    srcc = lambda a: float(spearmanr(a, y).correlation)

    # aggregate: does the orthogonal signal survive anatomical ROI-mean pooling?
    pooled = np.column_stack([Xb[:, idx].mean(1) for idx in rois.values()])  # (499, nROI)
    ppred = oof(pooled, y)
    agg_partial, agg_p = partial_spearman_perm(ppred, y, vpred, np.random.default_rng(1))
    print(f"whole-brain SRCC {srcc(bpred):.4f} | vision {srcc(vpred):.4f} | "
          f"ROI-pooled {srcc(ppred):.4f} | ROI-pooled unique|vision {agg_partial:.4f} (p={agg_p:.4g})")

    # per-ROI localization
    rows = []
    for name, idx in rois.items():
        rp = oof(Xb[:, idx], y, seed=SEED)
        uniq, p = partial_spearman_perm(rp, y, vpred, rng)
        rows.append({"roi": name, "n_vert": int(len(idx)), "srcc_raw": srcc(rp),
                     "unique_srcc": uniq, "p": p,
                     "ventral_mtl": any(k in name for k in VENTRAL_MTL)})
    q = bh_fdr([r["p"] for r in rows])
    for r, qi in zip(rows, q):
        r["q_fdr"] = float(qi)
    rows.sort(key=lambda r: r["unique_srcc"], reverse=True)

    top = rows[:15]
    sig = [r for r in rows if r["q_fdr"] < 0.05 and r["unique_srcc"] > 0]
    vm_in_top = sum(r["ventral_mtl"] for r in top)
    vm_in_sig = sum(r["ventral_mtl"] for r in sig)
    print(f"\nTOP 15 ROIs by unique (vision-controlled) memorability signal:")
    print(f"{'ROI':34s} {'nvert':>5s} {'uniqSRCC':>9s} {'rawSRCC':>8s} {'p':>8s} {'q':>7s} VVS/MTL")
    for r in top:
        print(f"{r['roi']:34s} {r['n_vert']:5d} {r['unique_srcc']:9.4f} {r['srcc_raw']:8.4f} "
              f"{r['p']:8.4g} {r['q_fdr']:7.3f} {'*' if r['ventral_mtl'] else ''}")
    print(f"\nFDR<0.05 ROIs: {len(sig)} | ventral/MTL among them: {vm_in_sig}/{len(sig)} "
          f"| ventral/MTL in top-15: {vm_in_top}/15")

    out = {
        "assumption": "TRIBE fsaverage5 order == nilearn Destrieux (LH 0-10241, RH 10242-20483)",
        "n": len(y), "atlas": "Destrieux fsaverage5 (aparc.a2009s)", "n_rois": len(rois),
        "nperm": NPERM,
        "whole_brain_srcc": srcc(bpred), "vision_srcc": srcc(vpred),
        "roi_pooled_srcc": srcc(ppred),
        "roi_pooled_unique_given_vision": {"partial_srcc": agg_partial, "p": agg_p},
        "n_sig_fdr05": len(sig), "ventral_mtl_in_sig": vm_in_sig,
        "ventral_mtl_in_top15": vm_in_top,
        "ranked_rois": rows,
    }
    res = ROOT / "data" / "results"; res.mkdir(parents=True, exist_ok=True)
    (res / "roi_localization.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {res / 'roi_localization.json'}")


if __name__ == "__main__":
    main()
