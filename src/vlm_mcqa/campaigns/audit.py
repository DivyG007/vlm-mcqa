"""Offline artifact inventory, phase completeness and validated retry linkage."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .verification import file_hash, read_object


def _rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"missing/empty/non-object raw records: {path}")
    def finite(value):
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"nonfinite raw value: {path}")
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)
    for row in rows:
        finite(row)
    return rows


def audit_core7_phase(root: str | Path, split: str) -> dict:
    root = Path(root)
    run = read_object(root / "run.json")
    if run.get("status") != "complete" or run.get("split") != split:
        raise ValueError("Core-7 phase not complete")
    if read_object(root / "calibration.json").get("all_passed") is not True:
        raise ValueError("phase calibration failed/missing")
    eligibility = read_object(root / "method_eligibility.json")
    if eligibility.get("all_passed") is not True:
        raise ValueError("phase method eligibility failed/missing")
    expected = {}
    patch = read_object(root / f"patching_{split}_summary.json")
    expected[f"patching_{split}.jsonl"] = patch["m1_rows"] + patch["group_patching"]["rows"]
    comp = read_object(root / f"components_{split}_summary.json")
    expected[f"components_{split}.jsonl"] = comp["patch_rows"]
    expected[f"component_ablation_{split}.jsonl"] = comp["ablation_rows"]
    heads = read_object(root / f"heads_{split}_summary.json")
    n_cases = sum(heads["cases"]["cases_by_type"].values())
    n_candidates = len(heads["candidates"])
    n_sets = len(heads["topk"]) * (1 + run["profile"]["random_sets"]) + 1
    expected[f"heads_validate_{split}.jsonl"] = n_cases * (n_candidates + n_sets)
    expected[f"heads_ablation_{split}.jsonl"] = heads["ablation_rows"]
    if split == "discovery":
        expected[f"heads_screen_{split}.jsonl"] = n_cases * len(heads["head_layers"]) * run.get("num_heads", 0)
        if not expected[f"heads_screen_{split}.jsonl"]:
            raise ValueError("phase run must record actual num_heads")
        behavior = read_object(root / "behavior_behavior_summary.json")
        if behavior.get("items") != run["profile"]["behavior_items"]:
            raise ValueError("behavior item target incomplete")
        for name in ("behavior_behavior.jsonl", "compliance_behavior.jsonl", "lens_behavior.jsonl", "component_projection_behavior.jsonl"):
            rows = _rows(root / name)
            if len(rows) != run.get("expected_observational_rows", {}).get(name):
                raise ValueError(f"observational record target incomplete: {name}")
    findings = {}
    key_fields = ("contrast_id", "method", "site", "layer", "group", "set", "head", "head_set",
                  "recipient_variant_id", "donor_variant_id")
    for name, count in expected.items():
        rows = _rows(root / name)
        if len(rows) != count:
            raise ValueError(f"raw count mismatch {name}: {len(rows)} != {count}")
        seen = set()
        pairs = set()
        for row in rows:
            key = tuple(row.get(field) for field in key_fields)
            if key in seen:
                raise ValueError(f"duplicate intervention identity: {name}")
            seen.add(key)
            if row.get("pair_id"):
                pairs.add(row["pair_id"])
            for field, value in row.items():
                if type(value) is float and not math.isfinite(value):
                    raise ValueError(f"nonfinite raw measurement: {name}/{field}")
        findings[name] = {"rows": len(rows), "sha256": file_hash(root / name), "pair_ids": sorted(pairs)}
    figures = run.get("figures", [])
    if not figures or any(not (root / figure).is_file() or not (root / figure).stat().st_size for figure in figures):
        raise ValueError("phase figures missing/empty")
    return {"schema": "core7-phase-audit/v1", "split": split, "technical_complete": True,
            "raw": findings, "scientific_replication": "separate_confirmation_report"}


def inventory(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    excluded = {"artifact_manifest.json", "audit.json"}
    return {p.relative_to(root).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in excluded}


def audit_attempt(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    if (root / "failure.json").exists():
        raise ValueError("attempt has a preserved failure record; it cannot be audited complete")
    header = read_object(root / "attempt.json")
    if any(not (root / name).is_file() for name in ("stdout.log", "stderr.log")):
        raise ValueError("attempt lacks durable stdout/stderr logs")
    if header.get("mode") == "calibration_only":
        result = read_object(root / "calibration_result.json")
        files = read_object(root / "artifact_manifest.json")["files"]
        core = read_object(root / "session/calibration.json")
        if result.get("status") != "calibrated" or files != inventory(root) or core.get("all_passed") is not True:
            raise ValueError("calibration attempt incomplete or artifacts changed")
        if header["config"]["kind"] == "stretch":
            extension = read_object(root / "extension_calibration.json")
            if extension.get("all_passed") is not True or not extension.get("checks"):
                raise ValueError("study calibration evidence missing")
        return {"schema": "campaign-calibration-audit/v1", "calibration_complete": True,
                "full_campaign_complete": False, "scientific_confirmation": False, "artifact_count": len(files)}
    completion = read_object(root / "completion.json")
    if completion.get("status") != "complete":
        raise ValueError("attempt lacks technical completion")
    manifest = read_object(root / "artifact_manifest.json")
    if manifest.get("files") != inventory(root):
        raise ValueError("attempt artifacts missing, changed or added after manifest")
    phases = {}
    if header["config"]["kind"] in {"core7_preselected", "core7_correct_only_reduced"}:
        for split in ("discovery", "confirmation"):
            phases[split] = audit_core7_phase(root / split, split)
        frozen = read_object(root / "freeze.json")
        report = read_object(root / "confirmation/confirmation_report.json")
        if report.get("freeze_hash") != frozen.get("freeze_hash") or report["checks"].get("execution_complete") is not True:
            raise ValueError("confirmation report is incomplete or has wrong freeze")
        first = read_object(root / "discovery/method_eligibility.json")
        second = read_object(root / "confirmation/method_eligibility.json")
        a = set(first["checks"]["heads"]["selection"]["pair_ids"])
        b = set(second["checks"]["heads"]["selection"]["pair_ids"])
        if a & b:
            raise ValueError("Core-7 discovery/confirmation pairs overlap")
    else:
        for split in ("discovery", "confirmation"):
            result = read_object(root / f"{split}_result.json")
            calibration = read_object(root / f"{split}_extension_calibration.json")
            if calibration.get("all_passed") is not True or result.get("status") != "complete":
                raise ValueError("study phase/calibration incomplete")
            checks = calibration.get("checks", {})
            if not checks or any(row.get("passed") is not True or "measured" not in row or "threshold" not in row for row in checks.values()):
                raise ValueError("study calibration lacks measured passing checks")
            if result.get("study") != header["config"]["study"]["name"]:
                raise ValueError("study result differs from planned registry entry")
            raw = _rows(root / split / "raw.jsonl")
            coverage = result["coverage"]
            if "raw_rows" in coverage:
                expected = coverage["raw_rows"]
            elif "semantic_evaluated" in coverage:
                expected = coverage["semantic_evaluated"] + coverage["controls_evaluated"]
                if coverage["semantic_requested"] != coverage["semantic_evaluated"] or coverage["controls_requested"] != coverage["controls_evaluated"]:
                    raise ValueError("semantic/control expected scope incomplete")
            elif "rows" in coverage:
                expected = coverage["rows"]
                if coverage.get("target_units") != coverage.get("observed_units"):
                    raise ValueError("task target-unit scope incomplete")
            else:
                raise ValueError("study missing independently auditable raw-record count")
            if len(raw) != expected:
                raise ValueError("study raw count differs from coverage report")
            encoded = [json.dumps(row, sort_keys=True, allow_nan=False) for row in raw]
            if len(encoded) != len(set(encoded)):
                raise ValueError("duplicate raw study record")
            artifacts = result.get("artifacts", {})
            if not artifacts:
                raise ValueError("study artifact index absent")
            for name, record in artifacts.items():
                if isinstance(record, dict) and "sha256" in record:
                    target = root / split / name
                    if not target.is_file() or file_hash(target) != record["sha256"]:
                        raise ValueError("study artifact checksum mismatch")
            phases[split] = {"coverage": result["coverage"], "scientific_verdict": result["scientific_verdict"]}
    return {"schema": "campaign-audit/v1", "technical_complete": True, "phases": phases,
            "artifact_count": len(manifest["files"]), "scientific_confirmation": "study_specific_report"}
