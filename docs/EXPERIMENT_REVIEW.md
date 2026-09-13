# Current verdict — 6 September 2026

The core behavioral finding was re-established with newly executed, corrected runs. The original implementation remains historically flawed; its percentages are not retroactively corrected. We can now write a bounded post about supervised recovery under an online J-lens-derived intervention, including a fresh-lens check. No additional GPU experiment is needed for that claim.

The final training job completed 5 September at 19:14:29 UTC. The fresh-lens job completed 6 September at 01:27:38 UTC. Both were observed completed with zero restarts. Their immutable images, submitted manifests, source hashes and result hashes are included. The fresh pod later expired under its TTL; persistent fit/evaluation logs and result receipts are retained.

## Main results

| Model | Published J /129 | Base-fresh J /129 | Own-fresh J /129 | Clean /129 | Own-fresh screen /65 | Own-fresh transfer /49 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 27 | 27 | 27 | 129 | 17 | 26 |
| Primary | 126 | 124 | 124 | 127 | 61 | 34 |
| Second seed | 119 | 119 | 121 | 129 | 61 | 31 |

Primary own-fresh task accuracy is 96.1%; second seed 93.8%; base 20.9%. Under the published lens, matched random-training and sham-training achieve 75/129 and 85/129, with clean 126/129 each. Those controls were not fitted/evaluated with their own fresh lenses, so do not present their published scores as matched fresh-lens controls.

All eight fresh-lens conditions used the same corrected gain, online selection, sequential k 10 intervention at blocks 16/18/19/20/21/22, top 10 output protection, FP32 model storage/BF16 evaluation, and the same task/transfer IDs. Published-lens predictions exactly match the final training receipts for base, primary, and replicate across clean/J/random conditions. Fresh-lens clean outputs also match.

Fresh recovered lenses use exact coordinate Jacobians on 500 fixed generic-text prompts (corpus rows 1000..1499), target block 31, dim_batch 4, max length 128, skip_first 16, BF16 fitting activations, FP32 sums and FP16 saved matrices. Each recovered fit uses four independent 125-prompt shards. The unchanged base's existing 500-prompt fit is reused, with identical model/corpus/fitting settings but a two-rank FP32 reduction. Its hashes/config were revalidated and it was reevaluated with corrected gain. This reduction-order difference may introduce small rounding differences; no task examples were used to fit or select lenses.

The base-fresh intervention still reduces base accuracy 129→27. Thus the fresh-lens result is not explained by the fresh base lens failing to disrupt the base model. Recovery barely changes after refitting: primary 126→124, replicate 119→121. This substantially weakens the simple stale-mapping explanation under the tested fit recipe. It is not a proof that every possible fresh readout, projection or protected route has been eliminated. The fitting module's `scientifically_usable_lens=false` / `requires_section_2_2_validation=true` fields are preserved: this run performed the behavioral baseline-disruption/parity check, not a full separate Section 2.2 lens-quality certification.

Whole-country paired bootstrap (10,000 draws, seed 20260905): primary own-fresh minus base-fresh gain 75.19 percentage points, exploratory 95% interval[65.25,84.62]; replicate 72.87 points,[64.49,81.75]. Screen-only gains are 67.69 points for both seeds; intervals[50.68,83.33] and[55.84,81.63]. These account for within-country grouping, not prior screening/LR selection or broader model/seed uncertainty.

Transfer is more modest: own-fresh 26→34/49 and 26→31/49; clean transfer remains 44/49 and 45/49, compared with base 49/49. Do not substitute the published baseline 22/49 into this fresh-lens comparison. Do not claim universal clean retention or broad transfer superiority.

## What was fixed and what remains

| Issue | Status | Evidence and boundary |
|---|---|---|
| Qwen3.5 normalization gain | Confirmed bug, fixed | Exact historical image source uses raw norm weight; effective gain is 1+weight. `src/jspace_plasticity/readout.py` now centralizes it for ablator/swapper. Historical source exports and unit tests included. |
| Hook lifetime through backward | Confirmed helper bug, fixed | Current trainer keeps hooks through backward. Original executed trainer explicitly disabled gradient checkpointing, so this defect is ruled out for that configured historical path. The new jobs also disable checkpointing. |
| Shared no-grad autocast cache suppressing gradients | Confirmed reproduction, corrected path | Exact-runtime CPU reproduction and sparse historical checkpoint samples are in provenance. Current plan capture is isolated from gradient-bearing autocast. All 424 trainable tensors have first-step gradients in all final arms; identity/dtype inventories and allocated memory match. Sparse historical samples are not full tensor-equality proof. |
| Corrected recovery | Confirmed under the tested sequential intervention | Newly trained primary 27→126, replicate 119; published-lens sham 85 and random 75; own-fresh 124/121 vs base 27. This is new evidence, not relabeling of the legacy 29→111 result. |
| Simple stale-lens explanation | Strongly weakened; universal mechanism unresolved | Recovery survives own-fresh lenses and a working fresh-base disruption control. No unique internal mechanism is established. |
| Row join, sample order, sham orchestration | Confirmed/exported | Trainer and source corpus are bundled; independent audit reproduces 400 logged optimization IDs per arm and the epoch seed+100003*epoch shuffle. Train/val/screen countries do not overlap. |
| Manifest filename | Confirmed, fixed in full loader | Both documented manifest spellings are accepted, content validated, conflicts rejected. Mac portable release itself was not edited here. |
| Sequential projections erase the full selected span | Ruled out | Sequential projections can reintroduce earlier components. An orthogonal-span option exists in source but was not evaluated here. No corrected width sweep, all-position residual/cosine audit or exact-erasure claim is supplied. |
| Matched random definition | Confirmed | One per-position random displacement matched to removal norm; not deletion of 10 random directions. Initial task severity is unmatched: J 27 vs random 107/129. Standalone generic-text disruption remains unmeasured. |
| Frozen output protection / linear probe | Confirmed historical receipts only | Corpus, caller/method and CPU fold predictions are included in prior-review;57/60 refit summaries and all block 31 values match. The final corrected checkpoints did not rerun frozen-protection/probe conditions. Historical results cannot be transferred to them. |
| Old checkpoint × legacy/corrected cross | Unresolved / unexecuted | The historical and corrected training results exist separately; the proposed full same-checkpoint intervention cross was not executed. |
| Causal rerouting, workspace independence, unchanged representations, unreportability | Unresolved hypotheses | Neither these behavioral tests nor the historical probe/rank artifacts establish those claims. |
| Broad novelty/priority | Unresolved | Prior work covers self-repair, obfuscation and J-lens-related recovery. The narrower design—adaptation with the online intervention continuously active plus matched controls and corrected-checkpoint fresh lenses—is the defensible contribution to discuss. |

The high-LR corrected run (1e-5,160steps) failed: primaryJ 62/129, clean 63/129, transferJ 1/49. Validation calibration selected 1e-6/100steps; both learning rate and duration changed, so this sequence does not isolate a pure learning-rate causal effect. The primary's calibration token predictions reproduced exactly in the final run. Failure, calibration and historical receipts are included, not hidden.

The final CPU audits verify 2180 training-run prediction rows plus 1424 fresh-lens rows, token correctness, hashes, configs, shared baselines, sample order, first-step gradient inventories, all fit prompt indices and paired clustered estimates. They are receipt recomputations, not independent GPU replication by a different lab.373 repository tests passed before the fresh-lens image was built.

Failed high-LR weight shards (67.64GiB) were retired with explicit user approval; all non-weight receipts/manifests remain. Historical successful and calibration checkpoints remain. Final primary/replicate checkpoint files were verified for existence and expected sizes; save-time hashes are included. Those large weights are external to this ZIP. Fresh lens arrays are provided separately with hashes verified after download.
