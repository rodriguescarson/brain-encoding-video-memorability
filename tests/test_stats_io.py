import json

import numpy as np
import pytest

from neurotwin.fingerprint import identification_accuracy
from neurotwin.io import (
    load_responses_from_dir,
    reduce_prediction,
    roi_functional_connectivity,
    save_results,
)
from neurotwin.stats import bootstrap_ci, permutation_null
from neurotwin.synthetic import make_synthetic


def test_permutation_null_flags_real_identity():
    r = make_synthetic(identity_strength=3.0, noise=0.3, seed=8)
    out = permutation_null(r, n_perm=100, seed=0)
    assert out["observed"] > out["null_mean"]
    assert out["p_value"] < 0.05


def test_permutation_null_not_significant_for_noise():
    r = make_synthetic(identity_strength=0.0, noise=1.0, n_features=400, seed=9)
    out = permutation_null(r, n_perm=200, seed=0)
    assert out["p_value"] > 0.05


def test_bootstrap_ci_brackets_estimate():
    r = make_synthetic(identity_strength=3.0, noise=0.3, seed=10)
    ci = bootstrap_ci(r, lambda x: identification_accuracy(x, n_splits=3)["accuracy"], n_boot=100)
    assert ci["lo"] <= ci["estimate"] <= ci["hi"]
    assert ci["n_units"] == r.n_stimuli


def test_block_bootstrap_reports_block_count():
    r = make_synthetic(identity_strength=2.0, noise=0.3, seed=11)
    blocks = [i // 6 for i in range(r.n_stimuli)]  # group stimuli into movies of 6
    ci = bootstrap_ci(
        r,
        lambda x: identification_accuracy(x, n_splits=3)["accuracy"],
        n_boot=60,
        blocks=blocks,
    )
    assert ci["n_units"] == len(set(blocks))
    assert ci["lo"] <= ci["estimate"] <= ci["hi"]


def test_reduce_prediction_shapes():
    pred = np.random.default_rng(0).standard_normal((50, 200))
    assert reduce_prediction(pred, "mean").shape == (200,)
    assert reduce_prediction(pred, "var").shape == (200,)
    assert reduce_prediction(pred, "meanstd").shape == (400,)


def test_fc_reducer_and_guard():
    pred = np.random.default_rng(0).standard_normal((40, 12))
    assert reduce_prediction(pred, "fc").shape == (12 * 11 // 2,)
    with pytest.raises(ValueError):
        reduce_prediction(np.zeros((10, 3000)), "fc")  # too large -> must parcellate


def test_roi_functional_connectivity():
    pred = np.random.default_rng(0).standard_normal((30, 20))
    labels = np.array([0] * 4 + [1] * 8 + [2] * 8)  # label 0 dropped -> 2 ROIs -> 1 pair
    assert roi_functional_connectivity(pred, labels).shape == (1,)


def test_io_roundtrip(tmp_path):
    # write a tiny balanced grid of (T, V) predictions + index, then load it back
    subjects, stimuli = ["sub-00", "sub-01"], ["movieA", "podB"]
    domains = {"movieA": "movie", "podB": "podcast"}
    rng = np.random.default_rng(0)
    items = []
    for s in subjects:
        for c in stimuli:
            arr = rng.standard_normal((10, 32))
            fname = f"{s}__{c}.npy"
            np.save(tmp_path / fname, arr)
            items.append({"subject": s, "stimulus": c, "domain": domains[c], "file": fname})
    (tmp_path / "index.json").write_text(json.dumps({"items": items}))

    r = load_responses_from_dir(tmp_path, reducer="mean")
    assert r.n_subjects == 2 and r.n_stimuli == 2 and r.n_features == 32
    assert set(r.domains) == {"movie", "podcast"}


def test_save_results_coerces_numpy(tmp_path):
    out = {"acc": np.float64(0.5), "arr": np.array([1, 2, 3])}
    p = tmp_path / "r.json"
    save_results(out, p)
    loaded = json.loads(p.read_text())
    assert loaded["acc"] == 0.5 and loaded["arr"] == [1, 2, 3]
