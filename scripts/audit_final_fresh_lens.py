"""CPU-only audit of fresh-lens predictions, fitting provenance and paired effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_final_capability import index_rows, paired_clusters, read_jsonl, sha


def audit(run, final_run, design_path, output):
    design = json.loads(design_path.read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert (run / "_SUCCESS").exists() and (run / "FIT_SUCCESS").exists()
    assert sha(run / "summary.json") == (run / "SUMMARY.sha256").read_text().split()[0]
    assert summary["design_sha256"] == sha(design_path)
    raw = {}
    counts = {}
    parity = {}
    for c in design["conditions"]:
        p = run / "conditions" / c["name"]
        r = json.loads((p / "result.json").read_text())
        assert sha(p / "result.json") == summary["result_sha256"][c["name"]]
        assert r["condition"] == c and r["status"] == "completed"
        assert r["design_sha256"] == sha(design_path)
        assert r["experiment_image"] == summary["experiment_image"]
        assert r["direction_convention"] == "effective_gain" and r["training"] is False
        assert sha(p / "predictions.jsonl") == r["predictions_sha256"]
        rows = read_jsonl(p / "predictions.jsonl")
        assert len(rows) == 178
        by_id = index_rows(rows)
        for row in rows:
            for cond in ("clean", "jspace", "random"):
                assert row[cond + "_exact"] == int(
                    row[cond + "_predicted_token_id"] == row["expected_token_id"]
                )
        m = design["models"][c["model"]]
        ref_dir = final_run / "arms" / m["reference_arm"]
        ref_result = json.loads((ref_dir / "result.json").read_text())
        assert sha(ref_dir / "predictions.jsonl") == ref_result["predictions_sha256"]
        reference = index_rows(
            [
                r
                for r in read_jsonl(ref_dir / "predictions.jsonl")
                if r["phase"] == m["reference_phase"] and r["split"] != "train"
            ]
        )
        assert by_id.keys() == reference.keys()
        assert all(
            row["expected_token_id"] == reference[k]["expected_token_id"]
            and row["clean_predicted_token_id"]
            == reference[k]["clean_predicted_token_id"]
            for k, row in by_id.items()
        )
        if c["lens"] == "published":
            for key, row in by_id.items():
                assert all(
                    row[x + "_predicted_token_id"]
                    == reference[key][x + "_predicted_token_id"]
                    for x in ("clean", "jspace", "random")
                )
            parity[c["model"]] = True
        cohort_counts = {}
        for cohort, splits, n in [
            ("pooled", {"val", "screen"}, 129),
            ("val", {"val"}, 64),
            ("screen", {"screen"}, 65),
            ("transfer", {"transfer"}, 49),
        ]:
            subset = [r for r in rows if r["split"] in splits]
            assert len(subset) == n
            cohort_counts[cohort] = {
                "rows": n,
                **{
                    cond: sum(r[cond + "_exact"] for r in subset)
                    for cond in ("clean", "jspace", "random")
                },
            }
        assert cohort_counts == r["evaluation"] == summary["conditions"][c["name"]]
        counts[c["name"]] = cohort_counts
        raw[c["name"]] = rows
    fit_provenance = {}
    for model in ("primary", "replicate"):
        p = run / "fits" / model
        config = json.loads((p / "fit_config.json").read_text())
        r = json.loads((p / "stages/0500/result.json").read_text())
        assert r["status"] == "completed" and r["num_prompts"] == 500
        assert r["experiment_image"] == summary["experiment_image"]
        assert config["model"] == design["models"][model]["path"]
        assert (
            config["model_manifest"]["sha256"]
            == design["models"][model]["manifest_sha256"]
        )
        assert config["revision"] == design["revision"]
        for key, value in design["fit"].items():
            if key not in ("num_prompts", "workers_per_model"):
                assert config[key] == value
        indices = [i for rank in r["rank_results"] for i in rank["prompt_indices"]]
        assert len(indices) == len(set(indices)) == 500 and sorted(indices) == list(
            range(1000, 1500)
        )
        assert all(
            rank["num_prompts_local"] == 125 and rank["status"] == "completed"
            for rank in r["rank_results"]
        )
        condition = json.loads(
            (run / "conditions" / f"{model}-{model}_fresh/result.json").read_text()
        )
        assert condition["lens"]["fit_result_sha256"] == sha(
            p / "stages/0500/result.json"
        )
        assert condition["lens"]["sha256"] == r["artifacts"]["lens"]["sha256"]
        fit_provenance[model] = {
            "prompts": 500,
            "unique_indices_match": True,
            "model_manifest_sha256": config["model_manifest"]["sha256"],
            "lens_sha256": r["artifacts"]["lens"]["sha256"],
            "fit_result_sha256": sha(p / "stages/0500/result.json"),
        }
    effects = {}
    for model in ("primary", "replicate"):
        effects[model] = {}
        for cohort, splits in [
            ("pooled", {"val", "screen"}),
            ("screen", {"screen"}),
            ("transfer", {"transfer"}),
        ]:

            def select(name, selected=splits):
                return [r for r in raw[name] if r["split"] in selected]

            own = select(f"{model}-{model}_fresh")
            effects[model][cohort] = {
                "own_fresh_minus_base_fresh": paired_clusters(
                    own, select("base-base_fresh")
                ),
                "own_fresh_minus_published": paired_clusters(
                    own, select(f"{model}-published")
                ),
            }
    out = {
        "status": "passed",
        "prediction_rows": 1424,
        "summary_sha256": sha(run / "summary.json"),
        "design_sha256": sha(design_path),
        "counts": counts,
        "published_token_parity": parity,
        "fits": fit_provenance,
        "paired_cluster_effects": effects,
        "bootstrap_seed": 20260905,
        "bootstrap_replicates": 10000,
        "interpretation": "Recovery survives own-fresh lenses.",
        "limitations": [
            "Previously explored, filtered cohorts; validation selected LR/duration.",
            "Base fit reused; same recipe, two-rank versus four-rank FP32 sums.",
            "No fresh-lens evaluation of sham/random-trained models.",
            "No inference about exact span erasure or unique recovery mechanism.",
            "Saved-token/provenance audit; not independent GPU execution.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": out["status"],
                "prediction_rows": 1424,
                "counts": counts,
                "fits": fit_provenance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "final-run", "design", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    audit(a.run, a.final_run, a.design, a.output)
