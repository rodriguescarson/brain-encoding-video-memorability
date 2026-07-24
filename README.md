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

## Data: predicted-response arrays

The derived feature arrays are released so the forecasts reproduce without re-running the
feature extraction:

- **GitHub Release:** [`v1.0-data`](https://github.com/rodriguescarson/brain-encoding-video-memorability/releases/tag/v1.0-data)
- **Zenodo (citable DOI):** [10.5281/zenodo.21532633](https://doi.org/10.5281/zenodo.21532633)

They contain the TRIBE v2 predicted cortical responses and the V-JEPA2 backbone embeddings
for the VideoMem (820) and Memento10k (499) clips, each stored as `stim_ids` plus float32
`X` features. They carry **no memorability scores and no videos**. InterDigital granted
written approval to share these derived arrays for research use, on the condition that the
VideoMem database and its scores are not distributed. Download the arrays into
`data/predictions/` and align them to the target scores by `stim_ids` (you supply the scores
from the datasets, see below).

## What is NOT in this repository, and why

**The source videos and the memorability scores are not redistributed.** The derived
predicted-response arrays are released (see the section above); the raw datasets are not.

- **VideoMem** is distributed by InterDigital under a licence stating that, without their
  prior written approval, the database, software, and materials "shall not be further
  distributed, published, copied, or disseminated in any way or form whatsoever", and may not
  be modified. Request access from InterDigital.
- **Memento10k** is available on request from MIT at <http://memento.csail.mit.edu/>.
- **TRIBE v2** weights are public under CC BY-NC-4.0; **V-JEPA2** is its released visual
  backbone.

To reproduce the numbers, obtain both datasets under their own terms, run the feature
extraction in `analysis/` (or download the released arrays above), and point the manifests
in `configs/` at your local copies.

## VideoMem dataset citation

VideoMem was provided by InterDigital and is described in R. Cohendet, C.-H. Demarty,
N. Q. K. Duong and M. Engilberge, *VideoMem: Constructing, Analyzing, Predicting Short-Term
and Long-Term Video Memorability*, ICCV 2019. The same group's ICMR 2018 paper (R. Cohendet,
K. Yadati, N. Q. Duong and C.-H. Demarty, *Annotating, Understanding, and Predicting
Long-term Video Memorability*, doi 10.1145/3206025.3206056) describes the annotation of a
separate dataset, MovieMem.

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
