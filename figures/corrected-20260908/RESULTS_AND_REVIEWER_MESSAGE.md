# Corrected figure suite completed — 8 September 2026

The corrected-checkpoint figure job `devin-q35-corrected-figures-r2` completed
at 20:09:03 UTC (container exit 0, zero restarts). Full primary/replicate checkpoint
hashes, cached base weight hashes, and all lens hashes passed. All six model/lens
conditions replayed the saved clean/J/random predictions before measurements;
capture forwards also matched clean/J token IDs on the same ordered 129 prompts
from 32 held-out countries. The local source inventory matches the executed image.

## Behavioral finding reproduced again

| Model | Published lens correct /129 | Own-fresh lens correct /129 | Clean correct /129 |
|---|---:|---:|---:|
| Base | 27 | 27 | 129 |
| Corrected primary | 126 | 124 | 127 |
| Corrected replicate | 119 | 121 | 129 |

This is a replay on the same frozen cohort, not an additional independent dataset.
The existing headline figures remain the principal behavioral evidence and are
unchanged. No training, new lens fitting, or width sweep was performed.

## Kurtosis

The broad clean-versus-lesion separation survives at the last intervention block
under both lens definitions. At block 22, median clean/lesioned kurtosis is:

| Model | Published lens | Own-fresh lens |
|---|---:|---:|
| Base | 2.427 / 1.258 | 2.481 / 0.994 |
| Primary | 2.167 / 1.583 | 2.456 / 1.394 |
| Replicate | 1.810 / 1.440 | 2.010 / 1.277 |

The desired pattern is not universal across layers: block 18 has crossing curves
for recovered models. The replicate's published-lens median *paired* change there
is +0.087 (pointwise country-bootstrap 95% interval +0.058 to +0.124); primary's
paired interval includes zero. Do not describe the lesion as lowering kurtosis at
every layer, or the clean curves as strictly monotonic. A difference of medians
is not the same statistic as the median of within-prompt differences. Both are
recoverable from the released observations; paired differences have their own CSV rows.

## Severity

At published-lens block 22, median lesion/clean residual norm ratios are
0.783 (base), 0.843 (primary), and 0.862 (replicate); median cosines are
0.852, 0.872, and 0.888. Own-fresh ratios are 0.779, 0.835, 0.849, and cosines
0.853, 0.858, 0.873. Early-block severity is broadly similar, but later norm loss
is smaller in recovered models. Thus the intervention remains appreciable, yet
these data do **not** establish matched severity or rule out a contribution from
reduced perturbation magnitude. States include effects of preceding lesions.
All per-prompt severity numbers were independently recomputed from the NPZ arrays
in float64 and match the GPU metrics within 2e-6 tolerances.

## Prompt-space CKA

Both five-panel seed figures use all 32 decoder block outputs, centered linear
CKA across the same 129 prompts, and shared color scales. The recovered models
show altered layer-to-layer similarity structure, including lower early/late
similarity. These descriptive differences do not identify a compensating circuit
or establish causal rerouting. Difference panels subtract within-condition CKA
matrices, rather than measuring cross-model CKA.

## Files for the reviewer/blog writer

All six new figures are in PNG, PDF and SVG, with methods/captions in CAPTIONS.json.
The suite includes both kurtosis lens conditions, both severity lens conditions,
and primary/replicate CKA. Individual metrics, country-bootstrap summaries, CKA
matrices, and an independent audit are adjacent to the figures. Raw residual NPZ
arrays and prediction receipts are in results/corrected-figures-20260908.
The metadata receipt includes source/config/model/lens/data hashes, package
versions, model/lens assignments, and prompt IDs. Model weights remain external
as in the prior release; the raw activations needed to regenerate CKA/severity
are now bundled. Kurtosis can be replotted from released per-prompt statistics;
recomputing full-vocabulary logits still requires the bound model and lens files.

Please use this suite as descriptive support for the already reverified recovery
finding. Preserve the block-18 exception and later severity difference. Do not
infer unchanged information, exact concept/span deletion, workspace independence,
or a specific alternative circuit from these figures. No new GPU job is required
for the requested suite.
