# Reproducing the released evidence

## CPU receipt audit (no GPU or model weights)

From the repository root, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv run scripts/verify_release.py
uv run scripts/audit_release.py --output .audit-output
```

Each analysis script declares its own dependencies as PEP 723 inline metadata, so uv builds a throwaway environment per script; nothing needs to be installed first and the GPU stack is never pulled in. The audit writes training-audit.json and fresh-lens-audit.json in the requested output directory. It checks 2,180+1,424 saved prediction rows and the numerical/provenance invariants. The code never launches a GPU job.

Training audit checks: exact flags versus predicted/target token IDs, bound hashes/configs/image, identical initial outputs,400 sample IDs per arm and epoch order, country separation,424 first-step gradient-bearing tensors, fixed LR schedule, finite losses/gradients, and the calibration winner's exact token parity. Fresh audit checks: all eight conditions, published output parity, clean preservation across lenses, two 500-prompt fits and their disjoint shard indices, model/lens provenance, and paired country bootstrap intervals. Full activation re-execution is not implied by consistency of saved receipts.

## Regenerate figures

```bash
uv run scripts/plot_final_capability.py \
  --audit results/final-training/RECEIPT_AUDIT.json --output .audit-output/figures
uv run scripts/plot_final_fresh_lens.py \
  --audit results/fresh-lens/RECEIPT_AUDIT.json --output .audit-output/figures
```

The PDF/SVG are suitable for editing/export. Plot CSV contains exact counts and denominators. Counts are deterministic; figure file bytes can depend on Matplotlib/font/version metadata.

## Code tests and environment locks

The test suite uses PyTorch, Transformers, jlens, NumPy, Matplotlib, PyYAML and pytest. Run it with `uv sync && uv run pytest -q tests`. Unlike the audit commands above this installs the full stack, and because Torch is pinned to the CUDA wheel index it resolves only on Linux with CUDA; on other platforms `uv sync` fails by design. The scripts' own PEP 723 metadata is intentionally sufficient only for the CPU receipt audits and figures, not for the model code tests.

pyproject.toml and uv.lock are the single environment definition for this repository: Torch 2.10.0+cu129, Transformers 5.15.0, and jlens at 581d398613e5602a5af361e1c34d3a92ea82ba8e, matching the runtime the reported runs executed under. The dependency set is only what the code imports; the training loop is plain PyTorch, so no TRL, vLLM or accelerate is installed. Per-run receipts record the runtime details observed at execution time.

## Public runtime build

`runtime/Dockerfile` rebuilds from public Python/NVIDIA/package sources and the committed runtime lock. It includes git explicitly for the pinned jlens source dependency. It has been inspected but was not built or GPU-validated during this packaging pass. The actual executions used the immutable images recorded in provenance. A new public build may differ at the base OS/Python level, and must pass runtime preflight and baseline prediction checks.

Before any build/job, check local `df -h / /tmp`, `docker system df`, and your persistent model/lens storage. Budget tens of GiB for the CUDA runtime/model cache and about 17GiB per FP32 saved model. Do not remove user caches or unique checkpoints merely to make room.

```bash
docker build -f runtime/Dockerfile -t jspace-plasticity:20260906 .
```

No build or GPU execution is triggered by opening or auditing this release.

## GPU reconstruction / external assets

Base model: Qwen/Qwen3.5-4B at revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. It is not included. Use the exact model revision with the published model license.

Published lens: the pinned public Hugging Face artifact is described in data/lens/neuronpedia-qwen3.5-4b-n1000-b62c3906.json (406332644 bytes, SHA-256 1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e). Download and verify it using:

```bash
uv run jspace-download-published-lens \
  --manifest data/lens/neuronpedia-qwen3.5-4b-n1000-b62c3906.json \
  --output-dir /your/storage/lenses/Qwen3.5-4B/neuronpedia-b62c3906-n1000
```

The three 500-prompt fresh lenses are not bundled here; request them, or refit them with the command above. Recovered model weights (~34GiB) are not publicly hosted by this package. Either obtain the retained checkpoints from the experiment owner and verify their manifests, or regenerate newly named models with the provided training protocol. Retraining can reproduce the experiment design, but should not be assumed to reproduce weight hashes bit-for-bit.

The source entrypoints are:

- Training/reduction: `uv run jspace-recovery-sft --help`.
- Exact lens fitting: `uv run torchrun --nproc-per-node=<N> -m jspace_plasticity.lens.fit_exact_dp --help` (independent ranks, no collective training).
- Fresh evaluation/reduction: `uv run jspace-fresh-lens --help`.

Exact executed shell commands are in docs/EXECUTED_COMMANDS.md. The Kubernetes/Volcano job manifests and registry-bound Dockerfiles used for the original cluster runs are **not** part of this public repository; see TRAINING.md for a cluster-independent path. Paths that appear as `<SHARED_STORAGE>`, `<HOME>`, `<CONTAINER_REGISTRY>` or `<AWS_ACCOUNT_ID>` in receipts and configs are redacted placeholders for the original private environment, and must be replaced with your own storage, registry and output directories.

To preserve historical hashes, keep original design/results untouched. Create new designs with your new storage/output/image bindings, compute their SHA256s, and supply them to the entrypoints. The checks intentionally fail on changed/missing source or checkpoint artifacts. The final-fresh design must bind the newly generated training summary/checkpoint manifests if you retrain. Its reused base lens binds the original 500-prompt fit; use the supplied optional lens and provenance, or explicitly create and name a new base fit with new hashes.

All source GeoNames extracts used for task/probe generation are bundled under data/geography/source, with original hashes and attribution. The frozen WikiText corpus and partition manifest are bundled. The task/probe builders and eligibility receipts let readers inspect the corpus and training join without network/model access. Historical probe activation arrays are external; the CPU-refitted per-example fold predictions and fit method are bundled, with 57/60 exact summary matches documented.

No new jobs are required to confirm the saved numerical claims. GPU re-execution remains a separate resource-dependent replication step.
