"""Publish a completed shapes evaluation to Weights & Biases.

Durable JSONL, CSV, PNG, and log files remain the authoritative result. This
module adds a browsable W&B run, compact tables, figures, and a raw artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _metric_payload(summary: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for name, values in summary["metrics"].items():
        payload[f"accuracy/{name}"] = float(values["accuracy"])
    for scheme, values in summary["vocabulary_metrics"].items():
        payload[f"vocabulary/{scheme}/global_top_valid_rate"] = float(
            values["global_top_token_valid_rate"]
        )
        payload[f"vocabulary/{scheme}/valid_label_mass"] = float(
            values["mean_valid_label_probability_mass"]
        )
    payload["runtime/evaluation_seconds"] = float(summary["elapsed_seconds"])
    payload["runtime/record_count"] = float(summary["record_count"])
    return payload


def _prediction_table(wandb: Any, predictions: list[dict[str, Any]]) -> Any:
    columns = [
        "sample_id",
        "pair_id",
        "pair_member",
        "variant_family",
        "label_scheme",
        "correct_content",
        "correct_position",
        "correct_label",
        "predicted_label",
        "is_correct",
        "correct_logit_margin",
        "correct_label_global_rank",
        "global_top_token_text",
        "global_top_token_is_valid_label",
        "valid_label_probability_mass",
    ]
    data = [
        [
            row["sample_id"],
            row["pair_id"],
            row["pair_member"],
            row["variant_family"],
            row["label_scheme"],
            row["correct_content"],
            row["correct_index"] + 1,
            row["correct_label"],
            row["predicted_label"],
            row["is_correct"],
            row["correct_logit_margin"],
            row["correct_label_global_rank"],
            row["global_top_token_text"],
            row["global_top_token_is_valid_label"],
            row["valid_label_probability_mass"],
        ]
        for row in predictions
    ]
    return wandb.Table(columns=columns, data=data)


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
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    import wandb

    summary = _read_json(args.result_dir / "summary.json")
    config = _read_json(args.result_dir / "experiment_config.json")
    config.update(
        {
            "model_id": summary["model_id"],
            "source_git_commit": summary["source_git_commit"],
            "slurm_job_id": args.job_id,
            "node": args.node,
            "max_pixels": summary["max_pixels"],
        }
    )
    run = wandb.init(
        project=args.project,
        name=args.run_name,
        id=args.run_id,
        resume="never",
        job_type="behavioral-evaluation",
        tags=["qwen2.5-vl-3b", "synthetic-shapes", "mcqa", "logit-lens"],
        config=config,
    )
    if run is None:
        raise RuntimeError("wandb.init returned no active run")

    predictions = _read_jsonl(args.result_dir / "predictions.jsonl")
    figures_dir = args.result_dir / "figures"
    payload: dict[str, Any] = _metric_payload(summary)
    payload["tables/predictions"] = _prediction_table(wandb, predictions)
    payload["tables/layerwise_means"] = _csv_table(
        wandb, figures_dir / "layerwise_means.csv"
    )
    content_symbol_table = figures_dir / "layerwise_content_symbol_means.csv"
    if content_symbol_table.is_file():
        payload["tables/layerwise_content_symbol_means"] = _csv_table(
            wandb, content_symbol_table
        )
    payload["tables/vocabulary_compliance"] = _csv_table(
        wandb, figures_dir / "vocabulary_compliance_by_scheme.csv"
    )
    for figure in sorted(figures_dir.glob("*.png")):
        payload[f"figures/{figure.stem}"] = wandb.Image(str(figure))
    run.log(payload)

    artifact = wandb.Artifact(
        name=f"{args.run_id}-results",
        type="evaluation-results",
        metadata={
            "source_git_commit": summary["source_git_commit"],
            "slurm_job_id": args.job_id,
            "node": args.node,
            "record_count": summary["record_count"],
        },
    )
    for name in (
        "summary.json",
        "experiment_config.json",
        "label_tokenization.json",
        "predictions.jsonl",
        "layerwise_label_logits.jsonl",
        "source_git_commit.txt",
        "combined.log",
        "combined.err",
    ):
        path = args.result_dir / name
        if path.is_file():
            artifact.add_file(str(path), name=name)
    for path in sorted(figures_dir.glob("*")):
        if path.is_file():
            artifact.add_file(str(path), name=f"figures/{path.name}")
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
    print(f"[wandb] artifact={artifact.name}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
