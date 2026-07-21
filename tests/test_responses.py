import numpy as np
import pytest

from neurotwin.responses import Responses


def test_shapes_and_props():
    data = np.zeros((3, 4, 5))
    r = Responses(data, ["a", "b", "c"], ["s0", "s1", "s2", "s3"])
    assert r.n_subjects == 3
    assert r.n_stimuli == 4
    assert r.n_features == 5
    assert r.domains == ["default"] * 4


def test_bad_dims_raise():
    with pytest.raises(ValueError):
        Responses(np.zeros((3, 4)), ["a", "b", "c"], ["s0"])
    with pytest.raises(ValueError):
        Responses(np.zeros((3, 4, 5)), ["a", "b"], ["s0", "s1", "s2", "s3"])


def test_domain_selection():
    data = np.arange(2 * 4 * 3).reshape(2, 4, 3).astype(float)
    r = Responses(data, ["a", "b"], ["s0", "s1", "s2", "s3"], ["x", "x", "y", "y"])
    idx = r.stimulus_indices("y")
    assert list(idx) == [2, 3]
    sub = r.select_stimuli(idx)
    assert sub.n_stimuli == 2
    assert sub.domains == ["y", "y"]
