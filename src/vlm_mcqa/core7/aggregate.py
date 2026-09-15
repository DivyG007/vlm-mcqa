"""Aggregate raw JSONL records into per-layer tables, freeze and confirm.

``freeze`` reads a discovery run and writes the frozen decisions that the
confirmation run must test without reselection. ``confirm`` compares a
confirmation run against those decisions.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .scoring import summarize

TRANSFER_THRESHOLD = 0.5


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def per_layer(rows, *, value: str, keys: Sequence[str], where=lambda r: True) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        if where(row) and value in row:
            groups[tuple(row[k] for k in keys)].append(float(row[value]))
    out = []
    for key, values in sorted(groups.items()):
        out.append({**dict(zip(keys, key)), "metric": value, **summarize(values)})
    return out


def outcome_fractions(rows) -> list[dict[str, Any]]:
    groups = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row.get("contrast_type") == "content_and_position" and row.get("group") == "final" and "outcome" in row:
            groups[(row["site"], row["layer"])][row["outcome"]] += 1
    out = []
    for (site, layer), counts in sorted(groups.items()):
        total = sum(counts.values())
        out.append({"site": site, "layer": layer, "n": total,
                    **{f"frac_{k}": counts.get(k, 0) / total for k in ("position", "content", "none", "other")}})
    return out


def first_crossing(table: Sequence[Mapping[str, Any]], threshold: float = TRANSFER_THRESHOLD) -> int | None:
    for row in sorted(table, key=lambda r: r["layer"]):
        if row["mean"] >= threshold:
            return int(row["layer"])
    return None


def peak(table: Sequence[Mapping[str, Any]]) -> int | None:
    rows = [r for r in table if r["mean"] == r["mean"]]
    return int(max(rows, key=lambda r: r["mean"])["layer"]) if rows else None


def aggregate_run(run_dir: Path, split: str) -> dict[str, Any]:
    """Write CSV tables for one split and return the key per-layer curves."""
    run_dir = Path(run_dir)
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    out: dict[str, Any] = {}
    patch = list(read_rows(run_dir / f"patching_{split}.jsonl"))
    comps = list(read_rows(run_dir / f"components_{split}.jsonl"))
    final = lambda r: r.get("group") == "final"  # noqa: E731
    m1 = per_layer(patch, value="normalized_recovery", keys=("contrast_type", "layer"), where=final)
    m1 += per_layer(patch, value="flip_to_donor", keys=("contrast_type", "layer"), where=final)
    m1 += per_layer(patch, value="recipient_margin_change", keys=("contrast_type", "layer"),
                    where=lambda r: final(r) and r["contrast_type"] == "identity")
    write_csv(tables_dir / f"m1_residual_{split}.csv", m1)
    groups = per_layer(patch, value="normalized_recovery", keys=("contrast_type", "group", "layer"),
                       where=lambda r: r.get("group") != "final")
    write_csv(tables_dir / f"m6_token_groups_{split}.csv", groups)
    outcomes = outcome_fractions(patch)
    write_csv(tables_dir / f"m6_outcomes_{split}.csv", outcomes)
    m2 = per_layer(comps, value="normalized_recovery", keys=("site", "contrast_type", "layer"))
    write_csv(tables_dir / f"m2_components_{split}.csv", m2)
    abl = per_layer(read_rows(run_dir / f"component_ablation_{split}.jsonl"), value="margin_drop",
                    keys=("site", "layer"))
    write_csv(tables_dir / f"m2_ablation_{split}.csv", abl)

    def curve(table, **match):
        return [r for r in table if all(r.get(k) == v for k, v in match.items())
                and r["metric"] == "normalized_recovery"]

    position = curve(m1, contrast_type="position")
    content = curve(m1, contrast_type="content")
    out["position_transfer_layer"] = first_crossing(position)
    out["position_peak_layer"] = peak(position)
    out["content_transfer_layer"] = first_crossing(content)
    out["content_peak_layer"] = peak(content)
    if outcomes:
        content_outcomes = [r for r in outcomes if r["frac_content"] > 0]
        out["cp_content_peak_layer"] = (int(max(content_outcomes, key=lambda r: r["frac_content"])["layer"])
                                        if content_outcomes else None)
        out["cp_position_first_majority_layer"] = next(
            (r["layer"] for r in outcomes if r["frac_position"] >= 0.5), None)
    for site in ("attn", "mlp"):
        table = curve(m2, site=site, contrast_type="position")
        out[f"{site}_peak_layer"] = peak(table)
        out[f"{site}_peak_nr"] = max((r["mean"] for r in table), default=None)
        out[f"{site}_curve"] = {r["layer"]: r["mean"] for r in table}
    out["position_curve"] = {r["layer"]: r["mean"] for r in position}
    out["content_curve"] = {r["layer"]: r["mean"] for r in content}
    heads_path = run_dir / f"heads_{split}_summary.json"
    if heads_path.exists():
        out["heads"] = json.loads(heads_path.read_text())

    lens = list(read_rows(run_dir / "lens_behavior.jsonl"))
    if lens:
        lens_table = per_layer(lens, value="margin", keys=("scheme", "stage"))
        lens_table += per_layer(lens, value="mean_label_logit", keys=("scheme", "stage"))
        lens_table += per_layer(lens, value="correct_restricted_prob", keys=("scheme", "stage"))
        lens_table += per_layer(lens, value="content_margin", keys=("scheme", "stage"))
        lens_table += per_layer(lens, value="requested_mass_8way", keys=("scheme", "stage"))
        write_csv(tables_dir / "m4_m5_m7_lens.csv", lens_table)
        margin = [r for r in lens_table if r["metric"] == "margin" and r["scheme"] == "letters"]
        out["letters_readable_stage"] = next((r["stage"] for r in sorted(margin, key=lambda r: r["stage"])
                                              if r["mean"] > 0 and r["ci_low"] > 0), None)
    comp_proj = list(read_rows(run_dir / "component_projection_behavior.jsonl"))
    if comp_proj:
        proj = []
        for metric in ("direct_correct_minus_best_incorrect", "direct_mean_label",
                       "incremental_correct_minus_best_incorrect", "incremental_mean_label"):
            proj += per_layer(comp_proj, value=metric, keys=("scheme", "component", "layer"))
        write_csv(tables_dir / "m2_projection.csv", proj)
    (run_dir / f"aggregate_{split}.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    return out


def freeze(run_dir: Path, out_path: Path, *, head_set_sizes: Sequence[int] | None = None,
           head_layers: int | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    agg = aggregate_run(run_dir, "discovery")
    run = json.loads((run_dir / "run.json").read_text())
    heads = agg.get("heads") or {}
    candidates = heads.get("candidates", [])
    sizes = list(head_set_sizes or heads.get("topk", []))
    frozen = {
        "frozen_from_run": str(run_dir),
        "model_key": run["model_key"],
        "dataset_hash": run.get("dataset_hash"),
        "image_aware_release_hash": run.get("image_aware_release_hash"),
        "source_git_commit": run.get("source_git_commit"),
        "transfer_threshold": TRANSFER_THRESHOLD,
        "position_transfer_layer": agg.get("position_transfer_layer"),
        "position_peak_layer": agg.get("position_peak_layer"),
        "content_transfer_layer": agg.get("content_transfer_layer"),
        "content_peak_layer": agg.get("content_peak_layer"),
        "cp_content_peak_layer": agg.get("cp_content_peak_layer"),
        "attn_peak_layer": agg.get("attn_peak_layer"),
        "mlp_peak_layer": agg.get("mlp_peak_layer"),
        "component_dominance": ("attn" if (agg.get("attn_peak_nr") or 0) >= (agg.get("mlp_peak_nr") or 0)
                                else "mlp"),
        "head_layers": heads.get("head_layers", [])[: head_layers] if head_layers else heads.get("head_layers", []),
        "head_candidates": candidates,
        "head_set_sizes": sizes,
        "sparsity_verdict": (heads.get("verdict") or [None])[0],
        "figure_layer_range": _figure_range(agg, run.get("num_layers")),
    }
    frozen["freeze_hash"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]
    Path(out_path).write_text(json.dumps(frozen, indent=2, sort_keys=True))
    return frozen


def freeze_heads(run_dir: Path, out_path: Path) -> dict[str, Any]:
    """Freeze only corrected M3 choices for a targeted heads-only rerun."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("status") != "complete" or run.get("methods") != ["heads"]:
        raise ValueError("freeze-heads requires a complete heads-only discovery run")
    heads = json.loads((run_dir / "heads_discovery_summary.json").read_text())
    coverage = heads.get("cases", {}).get("selected_pairs_by_type", {}).get("position", 0)
    if coverage < min(int(run.get("profile", {}).get("pairs", 0)),
                      int(run.get("profile", {}).get("head_screen_contrasts", 0))):
        raise ValueError(f"head screen has insufficient distinct-pair coverage: {coverage}")
    frozen = {
        "freeze_kind": "heads_only_pair_balanced",
        "frozen_from_run": str(run_dir),
        "model_key": run["model_key"],
        "dataset_hash": run.get("dataset_hash"),
        "image_aware_release_hash": run.get("image_aware_release_hash"),
        "source_git_commit": run.get("source_git_commit"),
        "components_from": run.get("components_from"),
        "components_source_git_commit": run.get("components_source_git_commit"),
        "components_source_sha256": run.get("components_source_sha256"),
        "head_layers": heads["head_layers"],
        "head_candidates": heads["candidates"],
        "head_set_sizes": heads["topk"],
        "sparsity_verdict": (heads.get("verdict") or [None])[0],
        "case_selection": heads.get("cases", {}).get("limited_case_selection"),
        "selected_position_pairs": coverage,
    }
    frozen["freeze_hash"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]
    Path(out_path).write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    return frozen


def confirm_heads(run_dir: Path, freeze_path: Path) -> dict[str, Any]:
    """Confirm a heads-only freeze without pretending M1/M2 were rerun."""
    run_dir, freeze_path = Path(run_dir), Path(freeze_path)
    frozen = json.loads(freeze_path.read_text())
    run = json.loads((run_dir / "run.json").read_text())
    heads = json.loads((run_dir / "heads_confirmation_summary.json").read_text())
    if frozen.get("freeze_kind") != "heads_only_pair_balanced":
        raise ValueError("confirm-heads requires a heads-only pair-balanced freeze")
    for field in ("model_key", "dataset_hash", "image_aware_release_hash", "source_git_commit"):
        if run.get(field) != frozen.get(field):
            raise ValueError(f"confirmation {field} differs from the heads freeze")
    coverage = heads.get("cases", {}).get("selected_pairs_by_type", {}).get("position", 0)
    checks = {
        "same_candidates": heads.get("candidates") == frozen.get("head_candidates"),
        "head_sparsity": {
            "frozen": frozen.get("sparsity_verdict"),
            "confirmation": (heads.get("verdict") or [None])[0],
            "replicated": frozen.get("sparsity_verdict") == (heads.get("verdict") or [None])[0],
        },
        "top_k_beats_random": {k: v.get("beats_random_p95") for k, v in (heads.get("per_k") or {}).items()},
        "selected_position_pairs": coverage,
        "pair_balanced": heads.get("cases", {}).get("limited_case_selection") == "round_robin_across_pairs",
    }
    checks["execution_complete"] = (
        checks["same_candidates"] and checks["pair_balanced"]
        and coverage >= frozen.get("selected_position_pairs", 0)
        and bool(checks["top_k_beats_random"])
        and all(isinstance(value, bool) for value in checks["top_k_beats_random"].values())
    )
    checks["all_hypotheses_replicated"] = (
        checks["head_sparsity"]["replicated"] and all(checks["top_k_beats_random"].values())
    )
    # Backward-compatible scientific result; this is deliberately not an
    # execution-success gate because a negative confirmation is valid evidence.
    checks["all_passed"] = checks["all_hypotheses_replicated"]
    report = {"freeze_hash": frozen["freeze_hash"], "checks": checks, "heads": heads}
    (run_dir / "heads_confirmation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return report


def _figure_range(agg: Mapping[str, Any], num_layers: int | None) -> list[int] | None:
    marks = [agg.get(k) for k in ("position_transfer_layer", "content_transfer_layer", "attn_peak_layer")]
    marks = [m for m in marks if m is not None]
    if not marks or not num_layers:
        return None
    return [max(0, min(marks) - 6), min(num_layers - 1, max(marks) + 3)]


def confirm(run_dir: Path, freeze_path: Path) -> dict[str, Any]:
    """Test frozen discovery decisions on the confirmation split (no reselection)."""
    frozen = json.loads(Path(freeze_path).read_text())
    agg = aggregate_run(Path(run_dir), "confirmation")

    def within(a, b, tol=1):
        return a is not None and b is not None and abs(a - b) <= tol

    frozen_order = _order(frozen.get("position_transfer_layer"), frozen.get("content_transfer_layer"))
    confirmed_order = _order(agg.get("position_transfer_layer"), agg.get("content_transfer_layer"))
    dominance = ("attn" if (agg.get("attn_peak_nr") or 0) >= (agg.get("mlp_peak_nr") or 0) else "mlp")
    heads = agg.get("heads") or {}
    checks = {
        "position_transfer_layer": {"frozen": frozen.get("position_transfer_layer"),
                                    "confirmation": agg.get("position_transfer_layer"),
                                    "replicated": within(frozen.get("position_transfer_layer"),
                                                         agg.get("position_transfer_layer"))},
        "content_transfer_layer": {"frozen": frozen.get("content_transfer_layer"),
                                   "confirmation": agg.get("content_transfer_layer"),
                                   "replicated": within(frozen.get("content_transfer_layer"),
                                                        agg.get("content_transfer_layer"), tol=2)},
        "content_vs_position_order": {"frozen": frozen_order, "confirmation": confirmed_order,
                                      "replicated": frozen_order == confirmed_order},
        "component_dominance": {"frozen": frozen.get("component_dominance"), "confirmation": dominance,
                                "replicated": frozen.get("component_dominance") == dominance},
        "head_sparsity": {"frozen": frozen.get("sparsity_verdict"),
                          "confirmation": (heads.get("verdict") or [None])[0],
                          "replicated": frozen.get("sparsity_verdict") == (heads.get("verdict") or [None])[0]},
        "top_k_beats_random": {k: v.get("beats_random_p95") for k, v in (heads.get("per_k") or {}).items()},
    }
    replication_checks = ("position_transfer_layer", "content_transfer_layer",
                          "content_vs_position_order", "component_dominance", "head_sparsity")
    checks["execution_complete"] = (
        bool(checks["top_k_beats_random"])
        and all(isinstance(checks[name].get("replicated"), bool) for name in replication_checks)
        and all(isinstance(value, bool) for value in checks["top_k_beats_random"].values())
    )
    checks["all_hypotheses_replicated"] = (
        all(checks[name]["replicated"] for name in replication_checks)
        and all(checks["top_k_beats_random"].values())
    )
    report = {"freeze_hash": frozen.get("freeze_hash"), "checks": checks, "aggregate": agg}
    (Path(run_dir) / "confirmation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report


def _order(position: int | None, content: int | None) -> str | None:
    if position is None or content is None:
        return None
    return "content_first" if content < position else "position_first" if position < content else "same_layer"
