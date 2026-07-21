import numpy as np

from neurotwin.guidance import guidance_rule
from neurotwin.synthetic import make_synthetic


def _roi_labels(n_features: int, n_rois: int = 10) -> np.ndarray:
    # chunk features into contiguous ROIs, labels 1..n_rois (0 reserved for unassigned)
    per = max(1, n_features // n_rois)
    return np.array([min(n_rois, i // per + 1) for i in range(n_features)])


def test_consolidated_identity_trusts_some_rois():
    r = make_synthetic(identity_strength=3.0, domain_specificity=0.0, noise=0.3, seed=1)
    g = guidance_rule(r, _roi_labels(r.n_features))
    assert g["consolidated"] is True
    assert g["global_decision"] == "per-subject-where-flagged"
    assert g["frac_trusted"] > 0.0


def test_nuisance_identity_falls_back_to_group_average():
    r = make_synthetic(identity_strength=0.0, noise=1.0, n_features=300, seed=2)
    g = guidance_rule(r, _roi_labels(r.n_features))
    assert g["consolidated"] is False
    assert g["global_decision"] == "group-average-everywhere"
    assert g["n_trust_per_subject"] == 0  # never trust per-subject when not consolidated


def test_domain_locked_identity_is_not_consolidated():
    r = make_synthetic(identity_strength=0.1, domain_specificity=3.0, noise=0.3, seed=3)
    g = guidance_rule(r, _roi_labels(r.n_features))
    # high within-domain but no cross-domain transfer -> group-average everywhere
    assert g["global_decision"] == "group-average-everywhere"


def test_thresholds_are_reported_and_respected():
    r = make_synthetic(identity_strength=3.0, domain_specificity=0.0, noise=0.3, seed=4)
    strict = guidance_rule(r, _roi_labels(r.n_features), tau_var=0.99)
    lax = guidance_rule(r, _roi_labels(r.n_features), tau_var=0.0)
    assert strict["thresholds"]["tau_var"] == 0.99
    assert lax["frac_trusted"] >= strict["frac_trusted"]
