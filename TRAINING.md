# Running the training yourself

This repository ships the full experiment source, the frozen task data, and the
design files, but **not** the Kubernetes/Volcano job manifests or the private
container images used for the original runs. Everything below is
cluster-independent. Placeholders like `<SHARED_STORAGE>`, `<HOME>`,
`<CONTAINER_REGISTRY>` and `<AWS_ACCOUNT_ID>` appear inside historical receipts
and configs; they are redacted stand-ins for the original private environment
and need to be replaced with your own paths.

## What you need

| Requirement | Value |
|---|---|
| GPU | 1× A100 80GB (the reported runs used `NVIDIA A100-SXM4-80GB`, CUDA capability 8.0) |
| Peak GPU memory | ~66 GiB reserved |
| Base model | `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Published lens | Neuronpedia J-lens for Qwen3.5-4B, ~406 MB (see below) |
| Disk | ~17 GiB per saved FP32 checkpoint, plus model/lens cache |
| Wall clock | 100 optimizer steps × 4 accumulated examples = 400 example presentations |

## 1. Build the runtime

`runtime/Dockerfile` builds from public Python/NVIDIA/package sources against the
committed runtime lock. It installs `git`, which the pinned `jlens` source
dependency needs.

```bash
docker build -f runtime/Dockerfile -t jspace-plasticity:local .
```

There is one lock for the whole repository: `pyproject.toml` and `uv.lock` pin
Torch 2.10.0+cu129, Transformers 5.15.0, and `jlens` at
`581d398613e5602a5af361e1c34d3a92ea82ba8e` — the versions the reported runs
executed under.

The training loop is plain PyTorch: `torch.optim.AdamW` with the lesion applied
through forward hooks. There is no TRL, vLLM, accelerate or RL machinery
involved, so the image installs released wheels only and needs no CUDA
toolchain or source build.

Everything below runs through [uv](https://docs.astral.sh/uv/). `uv sync`
installs the project and its pinned dependencies, and `uv run` executes against
that environment, so no manual `PYTHONPATH` or venv activation is needed:

```bash
uv sync
```

Torch is pinned to the CUDA wheel index, so this resolves on Linux with CUDA —
the same platform the training needs anyway.

## 2. Fetch the published lens

```bash
uv run jspace-download-published-lens \
  --manifest data/lens/neuronpedia-qwen3.5-4b-n1000-b62c3906.json \
  --output-dir /your/storage/lenses/Qwen3.5-4B/neuronpedia-b62c3906-n1000
```

The manifest pins 406,332,644 bytes at SHA-256
`1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e`.

## 3. Build the eligible training cohort

The training cohort is the subset of rows the clean base model already answers
correctly. The exact cohort used here is bundled:

```
data/cohort/eligible_train_rows.jsonl
```

189 rows across 45 countries, SHA-256
`614436a3ceada26210b193b03b9d3d7b79817ee1999ad7b04fcea7dc209e99f1`, which is
what `data/evals/q35-final-capability-20260905.json` binds via
`--eligible-sha256`. Reuse it to train against the same cohort.

To rebuild the cohort against your own model instance instead (eligibility is
model-dependent, so a different base checkpoint can yield a different set):

```bash
uv run jspace-expansion-triage --help
```

## 4. Launch training

```bash
uv run jspace-recovery-sft \
  --design data/evals/q35-final-capability-20260905.json \
  --data data/closedbook/closedbook-geo-s20260825.jsonl \
  --data-sha256 be15a088b4cab6cb9ec7f21664d025c28ead3dd836f2c362b673affe00370084 \
  --lens /your/storage/lenses/Qwen3.5-4B/neuronpedia-b62c3906-n1000/source/qwen3.5 \
  --transfer-data data/evals/anthropic-probe-swap.json \
  --transfer-eligibility data/evals/q35-probe-swap-rehab-folds-s20260825.json \
  --triage-dir /your/output/triage \
  --arm-index 0 \
  --output-dir /your/output/final-training
```

Run `uv run jspace-recovery-sft --help` for the full flag list. The entrypoint
also takes `--design-sha256`, `--data-sha256`, `--lens-sha256` and
`--eligible-sha256`; the checks **intentionally fail** on
changed or missing source artifacts, which is what keeps the receipts honest. If
you retrain, create a *new* design file bound to your own storage, output and
lens hashes rather than editing the historical one — that preserves the original
hashes while letting your run bind its own.

`--arm-index` selects the training condition from the design's four arms:

| Arm | Condition |
|---|---|
| 0 | J-lesion, primary seed `20260825` |
| 1 | J-lesion, replicate seed `20260826` |
| 2 | Matched random perturbation control |
| 3 | Sham (ordinary SFT) control |

Use `--reduce-only` to re-aggregate finished arms without retraining.

## The settings that produced the reported result

Taken from `results/final-training/arms/arm-00-j-full-s100-primary/result.json`.

**Intervention** (active during *every* gradient-producing forward pass):

| Setting | Value |
|---|---|
| `layers` | 16, 18, 19, 20, 21, 22 |
| `k` | 10 |
| `exclude_output_top_k` | 10 (the model's own top-10 next-token directions are protected) |
| `selection_source` | `online_current` — directions are re-ranked from the live activation, so later layers see earlier interventions |
| `projection` | `sequential` |
| `direction_convention` | `effective_gain` (Qwen3.5 RMSNorm gain is `1 + w`, not `w`) |
| `strength` | 1.0 |
| `ridge` | 1e-4 |

**Optimization:**

| Setting | Value |
|---|---|
| Objective | Single-answer-token teacher-forced cross-entropy |
| Learning rate | `1e-6`, 5 warmup steps |
| Max steps | 100 |
| Accumulation | 4 examples/step (400 presentations total) |
| Gradient clipping | Global norm 1.0 |
| Precision | FP32 params/optimizer states, BF16 autocast |
| Frozen | Final norm and output unembedding (969,216,000 params) |
| Prompt format | Raw text — no chat template, no generated reasoning |
| Training rows | 189 across 45 countries, disjoint from the evaluation countries |

Note the learning rate: `1e-6`, not the `1e-5` used by the superseded August run.
See `docs/EXPERIMENT_REVIEW.md` for why that changed.

## 5. Fresh-lens evaluation

Fitting a lens to your own retrained checkpoint is what rules out the
"model rotated away from a stale lens" explanation.

```bash
# Fit (torchrun launches independent ranks; this is not collective training)
uv run torchrun --nproc-per-node=<N> -m jspace_plasticity.lens.fit_exact_dp --help

# Evaluate
uv run jspace-fresh-lens --help
```

Each fresh lens uses 500 fixed WikiText prompts (corpus rows 1000–1499), max
sequence length 128, first 16 source positions skipped, mapping the six
intervention blocks to block 31 with exact coordinate Jacobians.

## Caveats

Retraining reproduces the experiment *design*. It will not reproduce checkpoint
weight hashes bit-for-bit — a public runtime rebuild can differ at the base OS
and Python level, and must pass runtime preflight plus baseline prediction
checks before its numbers are comparable.

No GPU work is needed to check the numbers already reported here; see
`docs/REPRODUCIBILITY.md` for the CPU-only audit path.
