"""Measured extension calibration and durable study bookkeeping.

No import-time model work. New study dispatchers deny execution before reading
configuration files, preparing inputs or creating output. Existing APIs remain.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

from .data import validate_output_path
from .runtime import REQUIRED_GATES, _FiniteAdapter

BINDING_KEYS = ("model_key", "dataset_hash", "checkpoint_identity", "source_identity", "image_aware_release_hash")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def document(value: Any) -> Any:
    return json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else value


def require_execution(context: Mapping[str, Any]) -> None:
    if context.get("execute") is not True:
        raise PermissionError("explicit execution authorization and execute=True required")


def core_guard(session: Any, context: Mapping[str, Any]) -> None:
    require_execution(context)
    identity, calibration = context["identity"], context["core_calibration"]
    if any(not identity.get(key) for key in BINDING_KEYS):
        raise ValueError("complete source/checkpoint/release binding required")
    spec = session.adapter.spec
    if (identity["model_key"] != spec.key or identity["dataset_hash"] != session.manifest.get("dataset_hash")
            or identity["image_aware_release_hash"] != session.release_manifest.get("image_aware_release_hash")):
        raise ValueError("current session identity mismatch")
    if (not re.fullmatch(r"[0-9a-fA-F]{40}", spec.revision)
            or identity["checkpoint_identity"] != {"hf_id": spec.hf_id, "revision": spec.revision}):
        raise ValueError("explicit current checkpoint commit required")
    gates = calibration.get("gates", {})
    if (calibration.get("all_passed") is not True or not REQUIRED_GATES.issubset(gates)
            or any(gates[g].get("passed") is not True for g in REQUIRED_GATES)
            or calibration.get("adapter") != session.adapter.describe() or not calibration.get("variants")):
        raise ValueError("fresh matching nine-gate Core-7 calibration required")


def protocol(config: Mapping[str, Any], context: Mapping[str, Any], study: str) -> dict:
    phase = config.get("phase")
    if phase not in ("discovery", "confirmation"):
        raise ValueError("explicit discovery or confirmation phase required")
    if phase == "confirmation":
        frozen = document(config["frozen_protocol"])
        body = frozen["protocol"]
        if (frozen.get("study") != study or frozen.get("protocol_sha256") != sha(body)
                or frozen.get("identity") != {k: context["identity"][k] for k in BINDING_KEYS}):
            raise ValueError("frozen study/model/source/release mismatch")
        if config.get("protocol") is not None and document(config["protocol"]) != body:
            raise ValueError("confirmation changes frozen protocol")
        return body
    body = document(config.get("protocol", {}))
    if not isinstance(body, dict) or not body:
        raise ValueError("explicit nonempty scientific protocol required")
    return body


def measured_check(error: float, tolerance: float) -> dict:
    finite = math.isfinite(error)
    return {"passed": finite and error <= tolerance, "measured": error if finite else None,
            "threshold": tolerance}


def difference(a: Any, b: Any) -> float:
    import torch
    delta = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
    return float(delta.max()) if torch.isfinite(delta).all() else float("inf")


def forward(session: Any, prepared: Any, **kwargs: Any) -> Any:
    return _FiniteAdapter(session.adapter).decoder_forward(prepared, **kwargs)


def primitive_checks(session: Any, variant_ids: Sequence[str], sites: Sequence[Any], tolerance: float) -> dict:
    """Measure capture/self/composition/batch equivalence, not boolean attestations."""
    from ..core7.hooks import Intervention
    if (not variant_ids or len(set(variant_ids)) != len(variant_ids) or not sites
            or type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("calibration examples/sites and positive tolerance required")
    errors = defaultdict(list)
    # Audit the whole list first; a later held-out ID must not allow earlier work.
    if any(session.variants[vid]["split"] == "confirmation" for vid in variant_ids):
        raise ValueError("extension calibration cannot inspect confirmation variants")
    for variant_id in variant_ids:
        prepared = session.prepared(variant_id)
        base, _ = forward(session, prepared)
        again, _ = forward(session, prepared)
        captured_logits, captures = forward(session, prepared, capture=sites, capture_positions=None)
        positions = list(range(prepared.num_tokens))
        interventions = [Intervention(site, positions, captures[site.key][0]) for site in sites]
        patched, _ = forward(session, prepared, interventions=interventions)
        batch, _ = forward(session, prepared, batch=2, interventions=interventions)
        repeated, _ = forward(session, prepared, interventions=interventions)
        errors["replay_deterministic"].append(difference(base, again))
        errors["capture_inert"].append(difference(base, captured_logits))
        errors["simultaneous_self_identity"].append(difference(base, patched))
        errors["composed_replay_deterministic"].append(difference(patched, repeated))
        errors["composed_batch_single"].extend(difference(patched[0], row) for row in batch)
    return {name: measured_check(max(values), tolerance) for name, values in errors.items()}


def calibration_report(session: Any, context: Mapping[str, Any], study: str, body: Mapping[str, Any],
                       checks: Mapping[str, Any], variant_ids: Sequence[str]) -> dict:
    return {"schema": "stretch.extension-calibration.v1", "study": study,
            "protocol_sha256": sha(body), "identity": {k: context["identity"][k] for k in BINDING_KEYS},
            "core_calibration_sha256": sha(context["core_calibration"]), "adapter": session.adapter.describe(),
            "calibration_variant_ids": list(variant_ids), "checks": dict(checks),
            "all_passed": bool(checks) and all(c["passed"] is True for c in checks.values())}


def require_calibration(session: Any, context: Mapping[str, Any], study: str, body: Mapping[str, Any]) -> None:
    core_guard(session, context)
    report = context.get("extension_calibration", {})
    if (report.get("schema") != "stretch.extension-calibration.v1" or report.get("study") != study
            or report.get("all_passed") is not True or not report.get("checks")
            or any(c.get("passed") is not True for c in report["checks"].values())
            or report.get("protocol_sha256") != sha(body)
            or report.get("identity") != {k: context["identity"][k] for k in BINDING_KEYS}
            or report.get("core_calibration_sha256") != sha(context["core_calibration"])
            or report.get("adapter") != session.adapter.describe()):
        raise ValueError("matching measured extension calibration required")


def grouped_summary(rows: Sequence[Mapping[str, Any]], *, seed: int, samples: int,
                    metric: str = "raw_margin_change") -> dict:
    """Average directions inside each independent unit before resampling."""
    if type(seed) is not int or type(samples) is not int or samples < 100:
        raise ValueError("explicit seed and >=100 bootstrap samples required")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = row.get(metric)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"nonfinite/missing {metric}")
        grouped[row["condition"]][row["sample_unit"]].append(value)
    output = {}
    for name, units in sorted(grouped.items()):
        values = [sum(v) / len(v) for _, v in sorted(units.items())]
        rng = random.Random(seed)
        draws = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                       for _ in range(samples)) if len(values) > 1 else []
        output[name] = {"mean": sum(values) / len(values), "n_units": len(values),
                        "ci95": [draws[int(.025 * (samples - 1))], draws[int(.975 * (samples - 1))]] if draws else None}
    return output


class StudyWriter:
    """Exclusive per-attempt journal; partial evidence remains after errors."""
    def __init__(self, session: Any, config: Mapping[str, Any], context: Mapping[str, Any], study: str, body: dict):
        require_execution(context)
        protected = [Path(session.data_dir), Path(session.out_dir)]
        for value in config.get("protected_inputs", []):
            path = Path(value)
            if not path.exists():
                raise ValueError(f"missing protected input {path}")
            protected.append(path if path.is_dir() else path.parent)
        for key in ("frozen_protocol", "protocol", "recipe", "contrasts", "sample_plan", "cases", "strata_manifest"):
            value = config.get(key)
            if isinstance(value, (str, Path)):
                protected.append(Path(value).resolve().parent)
        self.path = validate_output_path(context["output_dir"], protected)
        self.path.mkdir(parents=True, exist_ok=False)
        self.study, self.body, self.context = study, body, context
        self.rows, self.exclusions = [], []
        self.write("attempt.json", {"schema": "stretch.study-attempt.v1", "study": study,
                                    "config": dict(config), "protocol": body, "identity": context["identity"],
                                    "extension_calibration": context["extension_calibration"]})

    def write(self, name: str, value: Any) -> None:
        with (self.path / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=str) + "\n")

    def append(self, row: dict, *, excluded: bool = False) -> None:
        name = "exclusions.jsonl" if excluded else "raw.jsonl"
        with (self.path / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False, default=str) + "\n")
            stream.flush()
        (self.exclusions if excluded else self.rows).append(row)

    def failure(self, error: BaseException) -> None:
        self.write("failure.json", {"error": str(error), "error_type": type(error).__name__,
                                    "completed_rows": len(self.rows), "excluded_units": len(self.exclusions)})

    def finish(self, coverage: dict, verdict: Any, summary: Any, *, status: str = "complete",
               freeze_body: dict | None = None) -> dict:
        self.write("summary.json", summary)
        if freeze_body is not None:
            self.write("protocol_freeze.json", {"schema": "stretch.protocol-freeze.v1", "study": self.study,
                "protocol": freeze_body, "protocol_sha256": sha(freeze_body),
                "identity": {k: self.context["identity"][k] for k in BINDING_KEYS}})
        from .reporting import write_study_artifacts
        write_study_artifacts(self.path, self.rows, summary)
        artifacts = {}
        for path in sorted(self.path.iterdir()):
            if path.is_file():
                artifacts[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "bytes": path.stat().st_size}
        result = {"schema": "stretch.study-result.v1", "study": self.study, "status": status,
                  "coverage": coverage, "scientific_verdict": verdict, "protocol_sha256": sha(self.body),
                  "artifacts": artifacts, "output_dir": str(self.path),
                  "execution_complete": True, "scientific_confirmation": False}
        if freeze_body is not None:
            result["frozen_protocol_path"] = str(self.path / "protocol_freeze.json")
        self.write("audit.json", result)
        return result
