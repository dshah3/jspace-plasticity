Novelty assessment, checked 5 September 2026

A successful final run strengthens evidence for this experiment. It does not establish priority for recovery or monitoring evasion.

- [The Hydra Effect (2023)](https://arxiv.org/abs/2307.15771): describes compensatory model computation after ablations.
- [Explorations of Self-Repair (2024)](https://arxiv.org/abs/2402.15390): studies mechanisms of self-repair.
- [Obfuscated Activations (2024; ICLR 2026)](https://arxiv.org/abs/2412.09565): optimized activation obfuscation can evade latent-space defenses while retaining behavior, with task-dependent costs.
- [Measure, Don't Optimize (11 August 2026)](https://arxiv.org/abs/2608.11408): uses a Jacobian-lens accessibility audit to predict model-level relearning and finds that optimizing the audit score can suppress measured access without preventing recovery.

Our narrower experimental distinction, if supported: supervised adaptation with a gain-corrected online-current J-lens-derived sequential intervention continuously active during training and evaluation, trained on country-disjoint examples, with matched sham/random training and a seed replicate. This is an inference about differences in experimental design, not an exhaustive novelty review or priority proof.

The final training run used the published base lens. A subsequent corrected-checkpoint fresh-lens test now supports recovery after refitting (primary 124/129 and replicate 121/129 versus base 27/129). Historical width/probe results still cannot be attributed to these corrected checkpoints. The final run does not prove circuit rerouting, unchanged representation, workspace independence, unreportability, or exact erasure of a selected span. No additional GPU jobs are planned.
