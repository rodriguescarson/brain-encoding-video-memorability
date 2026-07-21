import numpy as np

from neurotwin.fingerprint import identification_accuracy, stimulus_removed_residuals
from neurotwin.synthetic import make_synthetic


def test_grand_mean_removal_sums_to_zero():
    r = make_synthetic(seed=1)
    res = stimulus_removed_residuals(r.data, leave_one_out=False)
    # across-subject mean at each stimulus is ~0 after grand-mean removal
    assert np.allclose(res.mean(axis=0), 0.0, atol=1e-10)


def test_leave_one_out_residual_is_other_subjects_mean():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 3, 5))
    res = stimulus_removed_residuals(data, leave_one_out=True)
    # subject 0's reference is the mean of subjects 1..3 (never itself)
    assert np.allclose(res[0], data[0] - data[1:].mean(axis=0))


def test_strong_identity_is_identifiable():
    r = make_synthetic(identity_strength=3.0, noise=0.3, seed=2)
    out = identification_accuracy(r, n_splits=5)
    assert out["accuracy"] > 0.9
    assert out["mean_rank"] < 1.5


def test_no_identity_is_near_chance():
    r = make_synthetic(identity_strength=0.0, noise=1.0, seed=3)
    out = identification_accuracy(r, n_splits=10)
    # 8 subjects -> chance 0.125; pure-noise identity should stay low
    assert out["accuracy"] < 0.4
    assert abs(out["chance"] - 0.125) < 1e-9


def test_cross_domain_consolidated_vs_locked():
    # consolidated identity transfers across domains
    consolidated = make_synthetic(
        identity_strength=3.0, domain_specificity=0.0, noise=0.3, seed=4
    )
    cross_c = identification_accuracy(consolidated, domain_a="movie", domain_b="podcast")

    # domain-locked identity ("surface episode") does NOT transfer
    locked = make_synthetic(
        identity_strength=0.1, domain_specificity=3.0, noise=0.3, seed=4
    )
    cross_l = identification_accuracy(locked, domain_a="movie", domain_b="podcast")
    within_l = identification_accuracy(locked, n_splits=5)

    assert cross_c["accuracy"] > 0.8
    assert cross_c["accuracy"] > cross_l["accuracy"]
    # locked identity is recoverable within a domain but not across it
    assert within_l["accuracy"] > cross_l["accuracy"]
