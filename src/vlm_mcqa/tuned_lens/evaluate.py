"""Re-read the Core-7 M4/M5/M7 behavior-split rows with Logit and Tuned Lens.

Stage rows are built exactly as ``core7.lens.run_lens`` builds them (same
fields, same ``symbol_readout``, same ``Session.clean`` decoder-replay states);
only the hidden state passed to the readout differs. The Logit Lens rows must
therefore equal Core-7's own ``lens_behavior.jsonl`` rows (tested on tiny
models; optionally checked against a released file with ``reference``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..core7.aggregate import per_layer, read_rows, write_csv
from ..core7.scoring import DEFAULT_LABELS, symbol_readout
from ..core7.session import Session, write_jsonl_row
from .translator import TunedLens

METRICS = ("margin", "mean_label_logit", "correct_restricted_prob", "label_mass_full",
           "content_margin", "requested_mass_8way")


def stage_rows(session: Session, item: Mapping[str, Any], variant: Mapping[str, Any],
               stage_logits: torch.Tensor) -> list[dict[str, Any]]:
    """Mirror of the per-stage row block in ``core7.lens.run_lens``."""
    adapter = session.adapter
    prepared = session.prepared(variant["variant_id"])
    correct = variant["correct_index"]
    ids = session.label_ids(prepared, variant["labels"])
    default_ids = session.label_ids(prepared, DEFAULT_LABELS)
    content_ids = [prepared.content_ids.get(c) for c in variant["option_contents"]]
    content_ok = None not in content_ids and len(set(content_ids)) == 4
    rows = []
    for s, logits in enumerate(stage_logits):
        row = {"variant_id": variant["variant_id"], "item_id": item["item_id"],
               "scheme": variant["label_scheme"], "position": correct, "stage": s,
               "family": variant["question_family"],
               **symbol_readout(logits, ids, correct, tokenizer=adapter.tokenizer)}
        row["correct_restricted_prob"] = row["restricted_probs"][correct]
        if variant["label_scheme"] != "letters":
            eight = logits[torch.tensor([*default_ids, *ids], device=logits.device)]
            probs = torch.softmax(eight, 0)
            row.update({
                "default_logits": [float(v) for v in eight[:4]],
                "default_mass_8way": float(probs[:4].sum()),
                "requested_mass_8way": float(probs[4:].sum()),
                "requested_correct_minus_position_default": float(eight[4 + correct] - eight[correct]),
            })
        if content_ok:
            c = logits[torch.tensor(content_ids, device=logits.device)]
            others = torch.cat([c[:correct], c[correct + 1:]])
            row.update({
                "content_logits": [float(v) for v in c],
                "content_margin": float(c[correct] - others.max()),
                "content_restricted_prob": float(torch.softmax(c, 0)[correct]),
            })
        rows.append(row)
    return rows


@torch.no_grad()
def lens_stage_logits(session: Session, variant_id: str, lens: TunedLens | None) -> torch.Tensor:
    handle = session.adapter.handle
    n_layers = session.adapter.num_layers
    clean = session.clean(variant_id)
    stages = torch.stack([clean.sites["resid_pre:0"]] + [clean.sites[f"resid_post:{l}"] for l in range(n_layers)])
    if lens is None:
        return handle.readout(stages).float()
    return torch.stack([handle.readout(lens.transform(stages[s:s + 1], s))[0] for s in range(n_layers + 1)]).float()


def run_evaluation(session: Session, lens: TunedLens, *, split: str = "behavior", n_items: int | None = None,
                   schemes: Sequence[str] = ("letters", "numbers", "rare_letters"), tracker: Any = None,
                   log=print) -> dict[str, Any]:
    out = session.out_dir
    counts = {"variants": 0}
    with session.jsonl(f"lens_{split}_logit.jsonl") as logit_out, session.jsonl(f"lens_{split}_tuned.jsonl") as tuned_out:
        for index, item in enumerate(session.items_in(split, n_items)):
            for variant in session.variants_of(item["item_id"], schemes=list(schemes)):
                for name, stream, which in (("logit", logit_out, None), ("tuned", tuned_out, lens)):
                    for row in stage_rows(session, item, variant, lens_stage_logits(session, variant["variant_id"], which)):
                        write_jsonl_row(stream, {**row, "lens": name})
                counts["variants"] += 1
            if (index + 1) % 16 == 0:
                log(f"[evaluate:{split}] {index + 1} items")
    summary = summarize_comparison(out, split)
    summary["counts"] = counts
    session.write_json(f"comparison_{split}.json", summary)
    if tracker is not None:
        tracker.summary({k: v for k, v in summary["milestones"].items()}, "milestones")
    return summary


def readable_stage(table: Sequence[Mapping[str, Any]], metric: str, scheme: str, *, threshold: float = 0.0):
    """Core-7 rule: first stage whose mean exceeds ``threshold`` with ci_low above it."""
    rows = sorted((r for r in table if r["metric"] == metric and r["scheme"] == scheme), key=lambda r: r["stage"])
    return next((r["stage"] for r in rows if r["mean"] > threshold and r["ci_low"] > threshold), None)


def summarize_comparison(out_dir: Path, split: str) -> dict[str, Any]:
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    milestones: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    for name in ("logit", "tuned"):
        rows = read_rows(out_dir / f"lens_{split}_{name}.jsonl")
        table = []
        for metric in METRICS:
            table += per_layer(rows, value=metric, keys=("scheme", "stage"))
        write_csv(tables_dir / f"m4_m5_m7_{name}.csv", table)
        curves[name] = table
        for scheme in ("letters", "numbers", "rare_letters"):
            milestones[f"{name}/{scheme}/label_readable_stage"] = readable_stage(table, "margin", scheme)
            milestones[f"{name}/{scheme}/content_readable_stage"] = readable_stage(table, "content_margin", scheme)
            if scheme != "letters":
                milestones[f"{name}/{scheme}/requested_majority_stage"] = readable_stage(
                    table, "requested_mass_8way", scheme, threshold=0.5)
    return {"split": split, "milestones": milestones,
            "note": "Core-7 milestone rule (mean > threshold and bootstrap ci_low > threshold); "
                    "descriptive readouts only, never causal evidence"}


def compare_reference(logit_rows_path: Path, reference_path: Path) -> dict[str, Any]:
    """Max |difference| of numeric stage-row fields versus a released Core-7 lens file."""
    ours = {(r["variant_id"], r["stage"]): r for r in read_rows(Path(logit_rows_path))}
    worst, matched = 0.0, 0
    for ref in read_rows(Path(reference_path)):
        mine = ours.get((ref["variant_id"], ref["stage"]))
        if mine is None:
            continue
        matched += 1
        for key in ("margin", "mean_label_logit", "correct_restricted_prob", "content_margin", "requested_mass_8way"):
            if key in ref and key in mine:
                worst = max(worst, abs(float(ref[key]) - float(mine[key])))
    return {"matched_rows": matched, "max_abs_difference": worst}


def write_figures(out_dir: Path, split: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    tables = {}
    for name in ("logit", "tuned"):
        tables[name] = read_rows_csv(out_dir / "tables" / f"m4_m5_m7_{name}.csv")
    written = []
    for metric in ("correct_restricted_prob", "margin", "content_margin", "requested_mass_8way", "mean_label_logit"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
        for ax, scheme in zip(axes, ("letters", "numbers", "rare_letters")):
            for name, style in (("logit", "--"), ("tuned", "-")):
                rows = sorted((r for r in tables[name] if r["metric"] == metric and r["scheme"] == scheme),
                              key=lambda r: int(r["stage"]))
                if rows:
                    stages = [int(r["stage"]) for r in rows]
                    ax.plot(stages, [float(r["mean"]) for r in rows], style, label=f"{name} lens")
                    ax.fill_between(stages, [float(r["ci_low"]) for r in rows], [float(r["ci_high"]) for r in rows],
                                    alpha=0.15)
            ax.set_title(scheme)
            ax.set_xlabel("stage")
        axes[0].set_ylabel(metric)
        axes[0].legend()
        fig.tight_layout()
        path = fig_dir / f"{split}_{metric}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path.name)
    return written


def read_rows_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    with Path(path).open() as stream:
        return list(csv.DictReader(stream))
