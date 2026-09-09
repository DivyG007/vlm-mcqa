"""Publish causal-intervention metrics, figures, and raw artifacts to W&B."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_table(wandb: Any, path: Path) -> Any:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    return wandb.Table(columns=rows[0], data=rows[1:])


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--job-type", default="causal-intervention")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    import wandb

    summary = _read_json(args.result_dir / "summary.json")
    config_path = args.result_dir / "experiment_config.json"
    config = _read_json(config_path) if config_path.is_file() else {}
    config.update(
        {
            "model_id": summary["model_id"],
            "source_git_commit": summary["source_git_commit"],
            "slurm_job_id": args.job_id,
            "node": args.node,
            "split": summary["split"],
            "selected_pair_ids": summary["selected_pair_ids"],
            "modes": summary["modes"],
            "contrast_types": summary["contrast_types"],
        }
    )
    run = wandb.init(
        project=args.project,
        name=args.run_name,
        id=args.run_id,
        resume="never",
        job_type=args.job_type,
        tags=["qwen2.5-vl-3b", "synthetic-shapes", "mcqa", "causal-intervention"],
        config=config,
    )
    if run is None:
        raise RuntimeError("wandb.init returned no active run")
    figures_dir = args.result_dir / "figures"
    aggregate_csv = figures_dir / "aggregate_interventions.csv"
    payload: dict[str, Any] = {
        "runtime/elapsed_seconds": float(summary["elapsed_seconds"]),
        "runtime/completed_interventions": float(summary["completed_interventions"]),
        "runtime/directed_contrasts": float(summary["directed_contrast_count"]),
    }
    if aggregate_csv.is_file():
        payload["tables/aggregate_interventions"] = _csv_table(wandb, aggregate_csv)
    for row in summary["aggregate_rows"]:
        pieces = [
            row.get("experiment"),
            row.get("contrast_type"),
            row.get("component"),
            row.get("token_group"),
            f"layer_{row['layer_index']}" if row.get("layer_index") is not None else None,
            (
                f"window_{row['window_start']}_{row['window_end']}"
                if row.get("window_start") is not None
                else None
            ),
            f"head_{row['head_index']}" if row.get("head_index") is not None else None,
            row.get("ablation_type"),
            f"scale_{row['scale']}" if row.get("scale") is not None else None,
            row.get("edge_route"),
        ]
        key = "/".join(str(piece) for piece in pieces if piece is not None)
        value = row.get("mean_normalized_recovery")
        if value is not None:
            payload[f"recovery/{key}"] = float(value)
        payload[f"flip_rate/{key}"] = float(row["target_flip_rate"])
    for figure in sorted(figures_dir.glob("*.png")):
        payload[f"figures/{figure.stem}"] = wandb.Image(str(figure))
    run.log(payload)

    artifact = wandb.Artifact(
        name=f"{args.run_id}-results",
        type="causal-intervention-results",
        metadata={
            "source_git_commit": summary["source_git_commit"],
            "slurm_job_id": args.job_id,
            "node": args.node,
            "completed_interventions": summary["completed_interventions"],
        },
    )
    for path in sorted(args.result_dir.rglob("*")):
        if path.is_file() and "wandb" not in path.parts:
            artifact.add_file(str(path), name=str(path.relative_to(args.result_dir)))
    run.log_artifact(artifact)
    run_info = {
        "project": args.project,
        "run_id": args.run_id,
        "run_name": args.run_name,
        "url": run.url,
        "entity": run.entity,
        "source_git_commit": summary["source_git_commit"],
        "slurm_job_id": args.job_id,
    }
    (args.result_dir / "wandb_run.json").write_text(
        json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[wandb] url={run.url}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
