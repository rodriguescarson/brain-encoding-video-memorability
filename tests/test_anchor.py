import numpy as np

from neurotwin.anchor import (
    anatomy_predictability,
    anchor_correlation,
    differential_identifiability,
)
from neurotwin.synthetic import make_synthetic


def test_diff_identifiability_positive_when_identity_strong():
    r = make_synthetic(identity_strength=3.0, noise=0.3, seed=1)
    di = differential_identifiability(r)
    assert di.shape == (r.n_subjects,)
    assert di.mean() > 0  # subjects are self-similar across halves


def test_diff_identifiability_near_zero_for_noise():
    r = make_synthetic(identity_strength=0.0, noise=1.0, n_features=400, seed=2)
    di = differential_identifiability(r)
    assert abs(di.mean()) < 0.3


def test_anchor_correlation_perfect_for_identical_inputs():
    r = make_synthetic(identity_strength=2.0, noise=0.4, seed=3)
    out = anchor_correlation(r, r)
    assert out["n"] == r.n_subjects
    assert out["r"] > 0.99  # identical pred/real -> identifiability tracks perfectly


def test_anchor_correlation_requires_aligned_subjects():
    a = make_synthetic(n_subjects=8, seed=4)
    b = make_synthetic(n_subjects=6, seed=4)
    try:
        anchor_correlation(a, b)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_anatomy_predictability_recovers_linear_signal():
    rng = np.random.default_rng(0)
    anatomy = rng.standard_normal((40, 2))
    target = 1.3 * anatomy[:, 0] - 0.7 * anatomy[:, 1] + 0.5  # exact linear function
    assert anatomy_predictability(target, anatomy) > 0.99
    # random target is poorly predicted
    assert anatomy_predictability(rng.standard_normal(40), anatomy) < 0.5
