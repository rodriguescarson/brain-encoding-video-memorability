import numpy as np

from neurotwin.geometry import effective_dimensionality
from neurotwin.synthetic import make_synthetic
from neurotwin.variance import variance_partition


def test_variance_fractions_sum_to_one():
    r = make_synthetic(seed=5)
    parts = variance_partition(r.data)
    total = parts["subject"] + parts["stimulus"] + parts["interaction"]
    assert np.allclose(total[total > 0], 1.0, atol=1e-9)


def test_subject_variance_tracks_identity_strength():
    weak = variance_partition(make_synthetic(identity_strength=0.2, seed=6).data)
    strong = variance_partition(make_synthetic(identity_strength=4.0, seed=6).data)
    assert strong["subject"].mean() > weak["subject"].mean()


def test_low_rank_identity_collapses_dimensionality():
    # nuisance-like: every subject differs along ONE shared direction (rank-1 identity)
    rng = np.random.default_rng(0)
    n_s, n_c, n_f = 10, 30, 100
    direction = rng.standard_normal(n_f)
    gains = rng.standard_normal(n_s)
    stim = rng.standard_normal((n_c, n_f))
    data = np.empty((n_s, n_c, n_f))
    for s in range(n_s):
        for c in range(n_c):
            data[s, c] = stim[c] + gains[s] * direction + 0.01 * rng.standard_normal(n_f)
    pr_low = effective_dimensionality(data)["participation_ratio"]
    assert pr_low < 2.0  # collapses toward a single mode

    # twin-like: full-rank independent identity per subject
    rich = make_synthetic(n_subjects=10, identity_strength=3.0, noise=0.1, seed=7)
    pr_high = effective_dimensionality(rich.data)["participation_ratio"]
    assert pr_high > pr_low + 1.0
