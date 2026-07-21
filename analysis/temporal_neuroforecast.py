#!/usr/bin/env python3
"""Does the TEMPORAL structure of TRIBE's predicted response carry memorability
that the time-MEAN (and the vision backbone) discard?

Consumes the reductions written by remote/tribe_temporal_reextract.py, unzipped into
data/predictions/: memento_brain_mean.npz, _std.npz, _late.npz, _early.npz, _slope.npz
(any subset present is used), plus memento_vjepa.npz + memento_labels.csv.

Pre-registered temporal test:
  H_temporal: [mean+std+late+slope] predicts memorability better than [mean alone],
  AND the temporal-only signal is orthogonal to BOTH vision and the mean.
This is the claim only a brain-encoding model of naturalistic video can make; a
frame-averaged visual feature has no per-timepoint predicted-response trajectory.
"""
from __future__ import annotations
import csv, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "data" / "predictions"
ALPHAS = np.logspace(-1, 5, 10)
REDUCTIONS = ["mean", "std", "early", "late", "slope"]
PCA_DIM = 100          # denoise each high-dim reduction before fusing (matches roi/robustness work)
NPERM = 4000


def load():
    y = np.array([float(r[1]) for r in list(csv.reader(open(PRED / "memento_labels.csv")))[1:]])
    Xv = np.load(PRED / "memento_vjepa.npz")["X"].astype(np.float64)
    red = {}
    for k in REDUCTIONS:
        f = PRED / f"memento_brain_{k}.npz"
        if f.exists():
            red[k] = np.load(f)["X"].astype(np.float64)
    if "mean" not in red and (PRED / "memento_brain.npz").exists():
        red["mean"] = np.load(PRED / "memento_brain.npz")["X"].astype(np.float64)  # from run 1
    return Xv, red, y


def oof(X, y, k=None, seed=0):
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr]); Xt, Xe = sc.transform(X[tr]), sc.transform(X[te])
        if k and X.shape[1] > k:
            p = PCA(k, random_state=0).fit(Xt); Xt, Xe = p.transform(Xt), p.transform(Xe)
        pred[te] = RidgeCV(alphas=ALPHAS).fit(Xt, y[tr]).predict(Xe)
    return pred


def _z(a):
    r = rankdata(a).astype(float); return (r - r.mean()) / r.std()


def _resid(a, *ctrl):
    B = np.c_[np.ones_like(a), np.column_stack(ctrl)]
    return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]


def partial_perm(pred, y, ctrl, rng):
    ry, rr = _resid(y, *ctrl), _resid(pred, *ctrl)
    zy, zr = _z(ry), _z(rr); n = len(zy)
    obs = float(zr @ zy / n)
    perms = np.array([rng.permutation(n) for _ in range(NPERM)])
    null = (zy[perms] @ zr) / n
    return obs, float((np.sum(np.abs(null) >= abs(obs)) + 1) / (NPERM + 1))


def main():
    rng = np.random.default_rng(0)
    Xv, red, y = load()
    srcc = lambda a: float(spearmanr(a, y).correlation)
    print("reductions present:", list(red))
    if "mean" not in red:
        raise SystemExit("need at least memento_brain_mean.npz")

    pv = oof(Xv, y)                                   # vision
    pmean = oof(red["mean"], y)                       # brain time-mean (run-1 result)
    print(f"vision SRCC {srcc(pv):.4f} | brain-mean SRCC {srcc(pmean):.4f}")

    # each temporal reduction: unique over vision, and unique over (vision + mean)
    out = {"vision_srcc": srcc(pv), "brain_mean_srcc": srcc(pmean), "reductions": {}}
    for k in [r for r in red if r != "mean"]:
        pk = oof(red[k], y)
        u_v, p_v = partial_perm(pk, y, [pv], rng)
        u_vm, p_vm = partial_perm(pk, y, [pv, pmean], rng)
        out["reductions"][k] = {"srcc": srcc(pk), "unique_vs_vision": u_v, "p_vs_vision": p_v,
                                "unique_vs_vision_and_mean": u_vm, "p_vs_vision_and_mean": p_vm}
        print(f"  {k:6s} SRCC {srcc(pk):.4f} | unique|vision {u_v:+.4f} (p={p_v:.3g}) | "
              f"unique|vision+mean {u_vm:+.4f} (p={p_vm:.3g})")

    # fused: vision + PCA-denoised(all brain reductions).  compare vs vision, and vs vision+mean.
    def fuse(keys, seed):
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xv):
            cols_t, cols_e = [], []
            sv = StandardScaler().fit(Xv[tr]); cols_t.append(sv.transform(Xv[tr])); cols_e.append(sv.transform(Xv[te]))
            for k in keys:
                sc = StandardScaler().fit(red[k][tr]); Bt, Be = sc.transform(red[k][tr]), sc.transform(red[k][te])
                pc = PCA(min(PCA_DIM, Bt.shape[1], Bt.shape[0] - 1), random_state=0).fit(Bt)
                cols_t.append(pc.transform(Bt)); cols_e.append(pc.transform(Be))
            pred[te] = RidgeCV(alphas=ALPHAS).fit(np.hstack(cols_t), y[tr]).predict(np.hstack(cols_e))
        return srcc(pred)

    seeds = range(5)
    f_mean = np.mean([fuse(["mean"], s) for s in seeds])
    f_all = np.mean([fuse(list(red), s) for s in seeds])
    out["fusion"] = {"vision_plus_mean": float(f_mean), "vision_plus_all_temporal": float(f_all),
                     "temporal_gain_over_mean": float(f_all - f_mean)}
    print(f"\nfusion  vision+mean {f_mean:.4f} | vision+ALL-temporal {f_all:.4f} | "
          f"temporal gain over mean {f_all - f_mean:+.4f}")

    res = ROOT / "data" / "results"; res.mkdir(parents=True, exist_ok=True)
    (res / "temporal_neuroforecast.json").write_text(json.dumps(out, indent=2))
    print("wrote", res / "temporal_neuroforecast.json")


if __name__ == "__main__":
    main()
