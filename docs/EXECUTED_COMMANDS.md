# Exact executed commands

Recovered verbatim from the job manifests of the original cluster runs. The
manifests themselves (Volcano/Kubernetes specs, node selectors, image pull
secrets) are not published; these are the commands they wrapped.

Paths shown as `<SHARED_STORAGE>` and `/opt/experiment` were the cluster's shared
mount and the image's checkout root. Substitute your own. See TRAINING.md for a
cluster-independent walkthrough.

## Fresh-lens fit and evaluation

Ran on 8× A100-SXM4-80GB, two concurrent 4-GPU fits followed by 8 single-GPU evaluations.

```bash
set -euo pipefail
export HF_HOME=<SHARED_STORAGE>/devin/jspace-plasticity/huggingface
export TRL_RUNTIME_CACHE_ROOT=<SHARED_STORAGE>/devin/jspace-plasticity/cache
export TRL_RUNTIME_CUDA_ARCH=8.0
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8
cd /opt/experiment
source runtime/activate_trl_runtime.sh
python runtime/verify_trl_runtime.py --require-gpu
test "$(python -c 'import torch; print(torch.cuda.device_count())')" = "8"
python -m jspace_plasticity.evals.final_fresh_lens preflight --design /opt/experiment/data/evals/q35-final-fresh-lens-20260905.json --design-sha256 0df5c3f8f236e6396c338d3f54b3ee826c9a3be5341c0dfd8550a087815b2a3c
mkdir -p <SHARED_STORAGE>/devin/jspace-plasticity/runs/q35-final-fresh-lens-r1/fits
pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --max_restarts=0 -m jspace_plasticity.lens.fit_exact_dp --model <SHARED_STORAGE>/devin/jspace-plasticity/runs/q35-final-capability-dp4-r1/arms/arm-00-j-full-s100-primary/checkpoint-terminal --model-manifest <SHARED_STORAGE>/devin/jspace-plasticity/runs/q35-final-capability-dp4-r1/arms/arm-00-j-full-s100-primary/checkpoint-manifest.json --expected-model-manifest-sha256 401d0fb207e9fc24a08e022a95bcad609e96883d66b162348d4a9140175be524 --output-dir <SHARED_STORAGE>/devin/jspace-plasticity/runs/q35-final-fresh-lens-r1/fits/primary --expected-world-size 4 --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --prompts-jsonl /opt/experiment/data/lens/wikitext-qwen3-32b-128tok.jsonl --expected-corpus-sha256 fff74461945910f775e1f39acd4a6d32f897def51e558e9e211057a7d45d47c7 --partition-offset 1000 --num-prompts 500 --layers 16,18,19,20,21,22 --target-layer 31 --dim-batch 4 --checkpoint-every 5 --max-seq-len 128 --skip-first 16 --merge-wait-seconds 26000 --no-export-layer-files > <SHARED_STORAGE>/devin/jspace-plasticity/runs/q35-final-fresh-lens-r1/fit-primary.log 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 --max_restarts=0 -m jspace_plasticity.lens.fit_exact_dp --model <SHARED_STORAGE>/devin/jspace-plasticity/runs/q35-final-capability-dp4-r1/arms/arm-01-j-full-s100-replicate/checkpoint-terminal --model-manifest <SHARED_STORAGE>/devin/jspace-plasticity/runs/q35-final-capability-dp4-r1/arms/arm-01-j-full-s100-replicate/checkpoint-manifest.json --expected-model-manifest-sha256 4680bf1498c3026c67f79e48663e9e652e84f431578310dbee50c31e2641f419 --output-dir <SHARED_STORAGE>/devin/jspace-plasticity/runs/q35-final-fresh-lens-r1/fits/replicate --expected-world-size 4 --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --prompts-jsonl /opt/experiment/data/lens/wikitext-qwen3-32b-128tok.jsonl --expected-corpus-sha256 fff74461945910f775e1f39acd4a6d32f897def51e558e9e211057a7d45d47c7 --partition-offset 1000 --num-prompts 500 --layers 16,18,19,20,21,22 --target-layer 31 --dim-batch 4 --checkpoint-every 5 --max-seq-len 128 --skip-first 16 --merge-wait-seconds 26000 --no-export-layer-files > <SHARED_STORAGE>/devin/jspace-plasticity/runs/q35-final-fresh-lens-r1/fit-replicate.log 2>&1 &
pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" != 0 ]]; then echo "Lens fitting failed; evaluation not run" >&2; exit 1; fi
touch <SHARED_STORAGE>/devin/jspace-plasticity/runs/q35-final-fresh-lens-r1/FIT_SUCCESS
pids=()
for index in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$index" python -m jspace_plasticity.evals.final_fresh_lens condition --design /opt/experiment/data/evals/q35-final-fresh-lens-20260905.json --design-sha256 0df5c3f8f236e6396c338d3f54b3ee826c9a3be5341c0dfd8550a087815b2a3c --index "$index" > "<SHARED_STORAGE>/devin/jspace-plasticity/runs/q35-final-fresh-lens-r1/eval-${index}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" != 0 ]]; then echo "Evaluation failed; inspect condition logs" >&2; exit 1; fi
python -m jspace_plasticity.evals.final_fresh_lens reduce --design /opt/experiment/data/evals/q35-final-fresh-lens-20260905.json --design-sha256 0df5c3f8f236e6396c338d3f54b3ee826c9a3be5341c0dfd8550a087815b2a3c 2>&1 | tee <SHARED_STORAGE>/devin/jspace-plasticity/runs/q35-final-fresh-lens-r1/reduce.log
```
