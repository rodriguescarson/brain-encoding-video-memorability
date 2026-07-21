"""Synthetic predicted-response generator with controllable identity structure.

Lets us validate the whole analysis pipeline on a Mac with no GPU and no TRIBE access,
and gives the analysis scripts a ``--demo`` mode. The generative model mirrors the real
hypothesis space:

    data[s, c, :] = stimulus_scale * S[stimulus c]          # shared, dominant
                  + identity_strength * I[subject s]         # consolidated identity
                  + domain_specificity * D[subject s, domain]# domain-locked identity
                  + noise

By dialing ``identity_strength`` and ``domain_specificity`` we can synthesize each
hypothesis:

- twin-like, consolidated  -> identity_strength high, domain_specificity low
- nuisance-like            -> identity_strength 0
- domain-locked ("episode")-> identity_strength low,  domain_specificity high
"""

from __future__ import annotations

import numpy as np

from .responses import Responses


def make_synthetic(
    n_subjects: int = 8,
    n_stimuli_per_domain: int = 20,
    n_features: int = 200,
    domains: tuple[str, ...] = ("movie", "podcast"),
    stimulus_scale: float = 1.0,
    identity_strength: float = 1.0,
    domain_specificity: float = 0.0,
    noise: float = 0.3,
    seed: int = 0,
) -> Responses:
    """Generate a synthetic :class:`Responses` with known identity structure."""
    rng = np.random.default_rng(seed)
    n_dom = len(domains)
    n_stimuli = n_stimuli_per_domain * n_dom

    stim_signal = rng.standard_normal((n_stimuli, n_features))          # S
    identity = rng.standard_normal((n_subjects, n_features))            # I (consolidated)
    domain_id = rng.standard_normal((n_subjects, n_dom, n_features))    # D (domain-locked)

    stim_domains: list[str] = []
    data = np.empty((n_subjects, n_stimuli, n_features))
    c = 0
    for d_idx, dom in enumerate(domains):
        for _ in range(n_stimuli_per_domain):
            for s in range(n_subjects):
                data[s, c, :] = (
                    stimulus_scale * stim_signal[c]
                    + identity_strength * identity[s]
                    + domain_specificity * domain_id[s, d_idx]
                    + noise * rng.standard_normal(n_features)
                )
            stim_domains.append(dom)
            c += 1

    subjects = [f"sub-{i:02d}" for i in range(n_subjects)]
    stimuli = [f"stim-{i:03d}" for i in range(n_stimuli)]
    feature_ids = [f"v{i}" for i in range(n_features)]
    return Responses(data, subjects, stimuli, stim_domains, feature_ids)
