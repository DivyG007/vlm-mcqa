"""Read-only, standard-library evidence synthesis. No model or service imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def source_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    record: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        record.update(sha256=digest.hexdigest(), bytes=path.stat().st_size)
    return record


def read_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    """Explicit JSONL input; malformed/non-object rows fail rather than disappear."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield row


def _read(path: Path, sources: list[dict[str, Any]]) -> dict[str, Any]:
    sources.append(source_record(path))
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def synthesize_core7(root: str | Path) -> dict[str, Any]:
    """Project current curated exports without treating missing artifacts as zero.

    Qwen's original full-run heads are never used, even if corrected M3 is absent.
    The full aggregates are retained to avoid inventing cross-schema equivalence.
    """
    root = Path(root)
    sources: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    comparison = []
    for model in ("qwen25vl_7b_core7", "internvl35_8b_core7", "llava_ov_7b_si_core7"):
        base = root / model
        freeze = _read(base / "freeze.json", sources)
        phases = {}
        for phase in ("discovery", "confirmation"):
            name = "summary.json" if model.startswith("qwen") else f"aggregate_{phase}.json"
            value = _read(base / phase / name, sources)
            # Qwen has a wrapper containing aggregate plus method summaries.
            aggregate = dict(value.get("aggregate", value.get(phase, value)))
            aggregate.pop("heads", None)  # Installed separately below, with correct provenance.
            behavior = value.get("behavior")
            if phase == "discovery" and behavior is None:
                behavior = _read(base / phase / "behavior_behavior_summary.json", sources) or None
            phases[phase] = {"causal_aggregate": aggregate or None,
                             "behavior": behavior,
                             "observational_scope": "shared_behavior_split" if phase == "discovery"
                             else "no_independent_confirmation_behavior_or_lens_claim"}
        report = _read(base / "confirmation" / "confirmation_report.json", sources)
        if model.startswith("qwen"):
            heads = _read(base / "m3_heads.json", sources)
            head_evidence = {"status": "corrected_pair_balanced" if heads else "missing_corrected_m3",
                             "corrected_m3": heads or None,
                             "full_run_head_values_excluded": True}
            freeze = {k: v for k, v in freeze.items() if not k.startswith("head")}
            checks = {k: v for k, v in report.get("checks", report).items()
                      if "head" not in k and "top_k" not in k}
        else:
            checks = report.get("checks")
            head_evidence = {}
            for phase in phases:
                head_evidence[phase] = _read(base / phase / f"heads_{phase}_summary.json", sources) or None
        models[model] = {"freeze": freeze or None, "phases": phases,
                         "heads": head_evidence, "confirmation_checks": checks,
                         "confirmation_report_freeze_hash": report.get("freeze_hash")}
        behavior = phases["discovery"]["behavior"] or {}
        confirmation = phases["confirmation"]["causal_aggregate"] or {}
        if model.startswith("qwen"):
            top4 = next((row for row in heads.get("top_k", []) if row.get("k") == 4), {})
            head_result = {"mean_nr": top4.get("confirmation"),
                           "fraction_of_selected_layer_all_heads": top4.get("fraction_of_all_heads"),
                           "random_p95": top4.get("confirmation_random_p95"),
                           "source": "corrected_m3_heads.json"}
        else:
            top4 = (head_evidence["confirmation"] or {}).get("per_k", {}).get("4", {})
            head_result = {"mean_nr": top4.get("top_k", {}).get("mean"),
                           "fraction_of_selected_layer_all_heads": top4.get("fraction_of_all_heads"),
                           "random_p95": top4.get("random_p95"),
                           "source": "heads_confirmation_summary.json"}
        comparison.append({"model": model, "behavior_items": behavior.get("items"),
                           "forced_letters_accuracy": behavior.get("letters_accuracy"),
                           "clean_minus_shuffled": behavior.get("clean_minus_shuffled"),
                           "frozen_position_transfer_layer": freeze.get("position_transfer_layer"),
                           "confirmation_position_transfer_layer": confirmation.get("position_transfer_layer"),
                           "frozen_content_transfer_layer": freeze.get("content_transfer_layer"),
                           "confirmation_content_transfer_layer": confirmation.get("content_transfer_layer"),
                           "confirmation_attention_peak_nr": confirmation.get("attn_peak_nr"),
                           "top4": head_result})
    return {"schema_version": "stretch.evidence.v1", "status": "descriptive_existing_exports_only",
            "models": models, "comparison": comparison, "sources": sources,
            "missing_sources": [s["path"] for s in sources if not s["exists"]],
            "required_interpretation": [
                "LLaVA full Core-7 is complete; do not rerun it for this synthesis.",
                "Qwen M3 uses corrected pair-balanced m3_heads.json, not original freeze heads.",
                "InternVL head verdict changes distributed to sparse: scientific non-replication.",
                "LLaVA ordering changes same_layer to content_first: scientific non-replication.",
                "Behavior/lens are shared observational evidence, not independent held-out replication.",
                "Strong random-pair final-token transfer limits position-specific-code claims.",
                "Patched sites do not establish an edge-complete circuit; all three LMs are Qwen lineage.",
                "Pali screens all failed frozen eligibility; future Pali2-10B Core-7 priority does not relax gates.",
                "Historical M1/M2 intervals need not be pair-clustered; corrected M3 is pair-clustered."]}


def exposure_inventory(dataset: dict[str, Any], sources: dict[str, str | Path],
                       required_source_names: list[str]) -> dict[str, Any]:
    """Union observed pair identities from explicit raw JSONL sources.

    Caller supplies the historical scope, including preliminary/failed methods.
    Missing identity rows and missing sources keep coverage incomplete. A pair
    absent from this union is NEVER called untouched or eligible.
    """
    variants = {v["variant_id"]: v for v in dataset["variants"]}
    contrasts = {c["contrast_id"]: c for c in dataset.get("contrasts", [])}
    known: set[str] = set()
    details = {}
    for name, path in sources.items():
        provenance = source_record(path)
        pairs: set[str] = set()
        rows = unresolved = 0
        if provenance["exists"]:
            for row in read_rows(path):
                rows += 1
                found = set()
                if row.get("pair_id"):
                    found.add(str(row["pair_id"]))
                contrast = contrasts.get(row.get("contrast_id"), {})
                if contrast.get("pair_id"):
                    found.add(str(contrast["pair_id"]))
                for key in ("variant_id", "donor_variant_id", "recipient_variant_id"):
                    variant = variants.get(row.get(key), {})
                    if variant.get("counterfactual_pair_id"):
                        found.add(str(variant["counterfactual_pair_id"]))
                pairs.update(found)
                unresolved += not bool(found)
        known.update(pairs)
        details[name] = {**provenance, "rows": rows, "rows_without_resolved_pair": unresolved,
                         "known_used_pairs": sorted(pairs)}
    missing = [n for n in required_source_names if n not in details or not details[n]["exists"]]
    planned = {str(v["counterfactual_pair_id"]) for v in dataset["variants"]
               if v.get("counterfactual_pair_id")}
    return {"schema_version": "stretch.exposure.v1", "sources": details,
            "required_sources": required_source_names, "missing_required_sources": missing,
            "declared_source_coverage_complete": bool(required_source_names) and not missing
            and all(not d["rows_without_resolved_pair"] and d["rows"] for d in details.values()),
            "historical_exposure_completeness": "not_established_by_this_tool",
            "known_used_union": sorted(known), "not_observed_in_supplied_sources": sorted(planned - known),
            "warning": "Absence is not proof of untouched scenes, split independence or behavioral eligibility."}
