"""Weights & Biases tracking for Core-7 runs.

W&B is an additional viewing layer; the run directory's JSONL/CSV/PNG files stay
the authoritative record. Each CLI command starts one W&B run with the model
spec, profile and dataset hash as config, logs calibration gates, behaviour
metrics, per-layer curves and figures, uploads summaries/tables as an
artifact, and finishes the run.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

# Defaults used by the original team runs.  Operators may redirect new runs to
# another accessible W&B workspace without changing the scientific code.
WANDB_ENTITY = "vlm-mcqa"
WANDB_PROJECT = "smoke-testing"


def destination() -> tuple[str, str]:
    """Return the configured W&B entity/project, falling back to team defaults."""
    return (
        os.environ.get("CORE7_WANDB_ENTITY", WANDB_ENTITY),
        os.environ.get("CORE7_WANDB_PROJECT", WANDB_PROJECT),
    )


def flatten(value: Any, prefix: str = "") -> dict[str, float]:
    """Numeric leaves of nested dicts as ``a/b/c`` keys (bools become 0/1)."""
    out: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, inner in value.items():
            out.update(flatten(inner, f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        out[prefix] = float(value)
    return out


class Tracker:
    """Thin wrapper; every method is a no-op when tracking is disabled."""

    def __init__(self, run: Any = None):
        self.run = run

    @classmethod
    def start(cls, *, enabled: bool, out_dir: Path, name: str, job_type: str,
              config: Mapping[str, Any], tags: list[str] | None = None, group: str | None = None) -> "Tracker":
        if not enabled:
            return cls(None)
        import wandb

        out_dir = Path(out_dir)
        (out_dir / "wandb").mkdir(parents=True, exist_ok=True)
        entity, project = destination()
        run = wandb.init(
            entity=entity,
            project=project,
            name=name,
            group=group,
            job_type=job_type,
            tags=[t for t in (tags or []) if t],
            dir=str(out_dir),
            config=dict(config),
        )
        (out_dir / f"wandb_{job_type}.json").write_text(json.dumps(
            {"entity": entity, "project": project, "id": run.id, "name": run.name, "url": run.url},
            indent=2) + "\n")
        print(f"[wandb] {run.url}", flush=True)
        return cls(run)

    @property
    def url(self) -> str | None:
        return self.run.url if self.run else None

    def log(self, metrics: Mapping[str, Any], prefix: str = "") -> None:
        if self.run:
            flat = flatten(metrics, prefix)
            if flat:
                self.run.log(flat)

    def summary(self, metrics: Mapping[str, Any], prefix: str = "") -> None:
        if self.run:
            self.run.summary.update(flatten(metrics, prefix))

    def curve(self, name: str, points: Mapping[Any, Any], x: str = "layer", y: str = "value") -> None:
        """Log a per-layer curve as a table plus a line plot."""
        if not self.run or not points:
            return
        import wandb

        rows = [[float(k), float(v)] for k, v in sorted(points.items(), key=lambda kv: float(kv[0]))
                if v is not None and not (isinstance(v, float) and math.isnan(v))]
        table = wandb.Table(columns=[x, y], data=rows)
        self.run.log({f"curves/{name}": wandb.plot.line(table, x, y, title=name)})

    def aggregate(self, agg: Mapping[str, Any], split: str) -> None:
        if not self.run:
            return
        for key in ("position_curve", "content_curve", "attn_curve", "mlp_curve"):
            self.curve(f"{split}/{key}", agg.get(key) or {})
        scalars = {k: v for k, v in agg.items() if not isinstance(v, (dict, list))}
        self.summary(scalars, f"{split}")
        heads = agg.get("heads") or {}
        if heads:
            self.summary({"all_heads_mean_nr": heads.get("all_heads_mean_nr"),
                          "sparse": (heads.get("verdict") or [None])[0] == "sparse",
                          "per_k": {k: {"top_k": v["top_k"]["mean"], "random_p95": v["random_p95"]}
                                    for k, v in (heads.get("per_k") or {}).items()}}, f"{split}/heads")

    def figures(self, out_dir: Path, names: list[str] | None = None) -> None:
        if not self.run:
            return
        import wandb

        fig_dir = Path(out_dir) / "figures"
        paths = [fig_dir / n for n in names] if names else sorted(fig_dir.glob("*.png"))
        images = {f"figures/{p.stem}": wandb.Image(str(p)) for p in paths if p.exists()}
        if images:
            self.run.log(images)

    def artifact(self, out_dir: Path, name: str, *, include_raw: bool = False) -> None:
        """Upload summaries, tables, figures and configs (raw JSONL only if asked)."""
        if not self.run:
            return
        import wandb

        out_dir = Path(out_dir)
        art = wandb.Artifact(name.replace("/", "_").replace(":", "_"), type="core7-results")
        patterns = ["run.json", "calibration.json", "*_summary.json", "aggregate_*.json", "freeze.json",
                    "confirmation_report.json", "selection.json", "comparison.json", "screening_table.*",
                    "tables/*.csv", "figures/*.png", "*.png"]
        if include_raw:
            patterns.append("*.jsonl")
        for pattern in patterns:
            for path in out_dir.glob(pattern):
                if path.is_file():
                    art.add_file(str(path), name=str(path.relative_to(out_dir)))
        self.run.log_artifact(art)

    def finish(self, status: str = "complete") -> None:
        if self.run:
            self.run.summary["status"] = status
            # Finish the run and upload any remaining data.
            self.run.finish(exit_code=0 if status == "complete" else 1)
            self.run = None
