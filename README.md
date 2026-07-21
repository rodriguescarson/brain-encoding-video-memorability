# Brain-encoding features for video memorability

Analysis code for the paper *It Depends on the Dataset: When a Brain-Encoding Model's
Predicted Responses Beat Their Visual Backbone for Video Memorability*
([arXiv:2607.16292](https://arxiv.org/abs/2607.16292)).

The paper asks whether the **predicted** fMRI responses of a brain-encoding model (TRIBE v2)
are a useful feature representation for forecasting short-term video memorability, compared
against a matched control: the model's own V-JEPA2 visual backbone, taken before the brain
projection.

The answer is dataset-dependent. Within Memento10k the backbone wins (Spearman 0.594 vs
0.544). Within VideoMem the brain projection wins (0.415 vs 0.368, delta +0.047, 95% CI
[+0.009, +0.088]). Cross-dataset transfer inherits the split.

## What is in this repository

| Path | Contents |
|---|---|
| `src/neurotwin/` | library: response handling, ROI mapping, statistics, geometry, I/O |
| `analysis/` | the analyses reported in the paper: forecasting, cross-domain transfer, ROI localization, temporal reduction, variance mapping |
| `configs/` | experiment configuration and manifest templates |
| `scripts/` | manifest construction helper |
| `tests/` | unit tests |

## What is NOT in this repository, and why

**The source videos and memorability scores are not redistributed**, and neither are the
predicted-response arrays derived from them.

- **VideoMem** is distributed by InterDigital under a licence stating that, without their
  prior written approval, the database, software, and materials "shall not be further
  distributed, published, copied, or disseminated in any way or form whatsoever", and may not
  be modified. Derived arrays fall under that restriction. Request access at
  <https://www.interdigital.com/data_sets/movie-memorability-dataset>.
- **Memento10k** is available on request from MIT at <http://memento.csail.mit.edu/>.
- **TRIBE v2** weights are public under CC BY-NC-4.0; **V-JEPA2** is its released visual
  backbone.

To reproduce the numbers, obtain both datasets under their own terms, run the feature
extraction in `analysis/`, and point the manifests in `configs/` at your local copies.

## Dataset citation required by the VideoMem licence

> The dataset was provided by InterDigital and is described in the following publication:
> R. Cohendet, K. Yadati, N. Q. Duong and C.-H. Demarty. Annotating, understanding, and
> predicting long-term video memorability. In Proceedings of the ICMR 2018 Conference,
> Yokohama, Japan, June 11-14, 2018.

## Citing this work

```bibtex
@article{rodrigues2026dataset,
  title  = {It Depends on the Dataset: When a Brain-Encoding Model's Predicted Responses
            Beat Their Visual Backbone for Video Memorability},
  author = {Rodrigues, Carson},
  year   = {2026},
  eprint = {2607.16292},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi    = {10.48550/arXiv.2607.16292}
}
```

## Licence

Code is released under the MIT Licence (see `LICENSE`). The licence covers this code only,
not the datasets, which remain under their own terms.
