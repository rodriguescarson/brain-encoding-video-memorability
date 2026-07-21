"""Shared loading for the RQ1 analysis scripts.

``--demo`` loads built-in synthetic data so every script runs end-to-end on a laptop
with no GPU and no TRIBE outputs. Without ``--demo``, predictions are loaded from
``configs/experiment_rq1.yaml`` -> ``predictions_dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from neurotwin.io import load_responses_from_dir
from neurotwin.responses import Responses
from neurotwin.synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--demo", action="store_true", help="run on built-in synthetic data")
    p.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiment_rq1.yaml"),
        help="experiment config (used when not --demo)",
    )
    p.add_argument("--out", default=str(ROOT / "results"), help="results output dir")


def load(args) -> tuple[Responses, dict]:
    """Return (responses, meta) for either demo or real predictions."""
    if args.demo:
        resp = make_synthetic(
            n_subjects=8,
            n_stimuli_per_domain=24,
            n_features=300,
            identity_strength=1.2,
            domain_specificity=0.6,
            noise=0.5,
            seed=0,
        )
        return resp, {"source": "synthetic-demo"}
    cfg = yaml.safe_load(Path(args.config).read_text())
    pred_dir = Path(cfg["predictions_dir"])
    reducer = cfg.get("reducer", "mean")
    resp = load_responses_from_dir(pred_dir, reducer=reducer)
    return resp, {"source": str(pred_dir), "reducer": reducer, "config": args.config}
