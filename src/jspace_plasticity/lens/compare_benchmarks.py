"""Compare bounded exact-lens benchmarks and recommend an 8/16-GPU topology."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def topology_projection(
    result: dict[str, Any], *, total_gpus: int, prompts: int
) -> dict[str, Any]:
    world_size = int(result["metadata"]["world_size"])
    groups = total_gpus // world_size
    prompt_seconds = float(result["summary"]["projected_exact_prompt_seconds"])
    eligible = (
        result.get("status") == "completed"
        and groups >= 1
        and float(result["summary"]["minimum_memory_headroom_gib"]) >= 5.0
        and (
            world_size == 1
            or result.get("reference_equivalence", {}).get("passed", False)
        )
    )
    prompts_per_group = math.ceil(prompts / groups) if groups else prompts
    return {
        "eligible": eligible,
        "total_gpus": total_gpus,
        "tp_degree": world_size,
        "independent_groups": groups,
        "prompts": prompts,
        "projected_wall_hours": prompts_per_group * prompt_seconds / 3600,
        "projected_gpu_hours": prompts * prompt_seconds * world_size / 3600,
        "minimum_memory_headroom_gib": result["summary"][
            "minimum_memory_headroom_gib"
        ],
    }


def compare(
    results: list[tuple[Path, dict[str, Any]]], *, prompts: int
) -> dict[str, Any]:
    configurations: list[dict[str, Any]] = []
    for path, result in results:
        world_size = result["metadata"]["world_size"]
        dim_batch = result["metadata"]["dim_batch"]
        label = f"tp{world_size}-db{dim_batch}"
        configurations.append(
            {
                "label": label,
                "result": str(path),
                "world_size": result["metadata"]["world_size"],
                "dim_batch": result["metadata"]["dim_batch"],
                "reference_equivalence": result.get("reference_equivalence"),
                "budgets": {
                    str(total): topology_projection(
                        result, total_gpus=total, prompts=prompts
                    )
                    for total in (8, 16)
                },
            }
        )
    recommendations: dict[str, Any] = {}
    for total in (8, 16):
        eligible = [
            config
            for config in configurations
            if config["budgets"][str(total)]["eligible"]
        ]
        if not eligible:
            recommendations[str(total)] = None
            continue
        winner = min(
            eligible,
            key=lambda config: config["budgets"][str(total)][
                "projected_wall_hours"
            ],
        )
        recommendations[str(total)] = {
            "label": winner["label"],
            **winner["budgets"][str(total)],
        }
    return {
        "schema_version": 1,
        "prompts": prompts,
        "eligibility": {
            "minimum_memory_headroom_gib": 5.0,
            "tp_requires_exact_row_and_forward_equivalence": True,
        },
        "configurations": configurations,
        "recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--prompts", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prompts < 1:
        parser.error("--prompts must be positive")
    loaded = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in args.results
    ]
    report = compare(loaded, prompts=args.prompts)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
