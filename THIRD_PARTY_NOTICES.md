The root Apache-2.0 license applies to original project code and documentation. It does not relicense third-party material. Original numerical summaries and generated figures may be reused under the root license; dataset-derived content retains the applicable data license and attribution.

| Material | Source and license | Bundled scope / modifications |
|---|---|---|
| Anthropic Jacobian-lens reference implementation and evaluation prompts | https://github.com/anthropics/jacobian-lens at 581d398613e5602a5af361e1c34d3a92ea82ba8e; Apache-2.0 | Dependency pinned in locks; Anthropic evaluation JSON and their manifests under data/evals; derived task subsets under data/anthropic. Preserve upstream attribution. |
| Hugging Face Transformers Qwen3.5 source | https://github.com/huggingface/transformers ; Apache-2.0 | Runtime version is pinned at 5.15.0 in runtime/trl-runtime. No Transformers source is vendored here. |
| GeoNames | https://www.geonames.org/ ; CC-BY-4.0 | Frozen cities/countries source extracts in data/geography/source; modified geography/probe subsets and generated prompt templates in data/geography and data/closedbook. Source snapshots and modifications are documented in adjacent manifests/readmes. |
| WikiText / Wikipedia contributors | https://huggingface.co/datasets/Salesforce/wikitext ; CC-BY-SA-3.0 | data/lens/wikitext-qwen3-32b-128tok.jsonl is a token-chunked derivative. Attribution, source revision, split, tokenizer and transformations are in its manifest. Retains CC-BY-SA-3.0. |
| Qwen3.5-4B model | https://huggingface.co/Qwen/Qwen3.5-4B ; Apache-2.0 as listed in the model card | Model weights are not bundled. Exact base revision and derived checkpoint save-time hashes are exported. |

License texts are in licenses/. Upstream source notices inside files remain intact. Dependencies such as PyTorch, NumPy, Matplotlib, vLLM, TRL, CUDA, and tokenizers have their own licenses; this repository contains source, locks and build instructions, not their installed distributions.
