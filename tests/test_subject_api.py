"""Tests for the subject-selection detector — fake models, no GPU, no tribev2."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "remote"))
import subject_api  # noqa: E402


def _seed(subject: str) -> int:
    return int("".join(filter(str.isdigit, subject)) or "0")


class _Events:
    def assign(self, **_kw):
        return self


class KwargModel:
    """Selects subject via a predict(..., subject=) kwarg."""

    def predict(self, events=None, subject=None):
        rng = np.random.default_rng(_seed(subject))
        return rng.standard_normal((10, 20)), None


class SetterModel:
    """Selects subject via set_subject()."""

    def __init__(self):
        self._s = None

    def set_subject(self, s):
        self._s = s

    def predict(self, events=None):
        return np.random.default_rng(_seed(self._s)).standard_normal((10, 20)), None


class InertModel:
    """Ignores the subject entirely -> NO-GO."""

    def predict(self, events=None, subject=None):
        return np.random.default_rng(0).standard_normal((10, 20)), None


def test_detects_kwarg_mechanism():
    res = subject_api.detect_subject_mechanism(KwargModel(), _Events(), ["sub-01", "sub-02"])
    assert res["go"] is True
    assert res["mechanism"] == "predict_kwarg"
    assert res["shape"] == (10, 20)
    assert res["mean_abs_diff"] > 0


def test_detects_setter_mechanism():
    res = subject_api.detect_subject_mechanism(SetterModel(), _Events(), ["sub-01", "sub-02"])
    assert res["go"] is True
    assert res["mechanism"] == "setter"


def test_inert_model_is_no_go():
    res = subject_api.detect_subject_mechanism(InertModel(), _Events(), ["sub-01", "sub-02"])
    assert res["go"] is False
    assert res["mechanism"] is None


def test_predict_for_subject_uses_confirmed_mechanism():
    model, ev = KwargModel(), _Events()
    a = subject_api.predict_for_subject(model, ev, "sub-01", "predict_kwarg")
    b = subject_api.predict_for_subject(model, ev, "sub-02", "predict_kwarg")
    assert a.shape == (10, 20)
    assert float(np.abs(a - b).mean()) > 0


def test_too_few_subjects_raises():
    import pytest

    with pytest.raises(ValueError):
        subject_api.detect_subject_mechanism(KwargModel(), _Events(), ["sub-01"])
