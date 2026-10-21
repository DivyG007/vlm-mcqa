"""Explicit local provenance verification; no model/service imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .config import canonical_hash, validate_config


def file_hash(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"object required: {path}")
    return value


def verify_inventory(root: str | Path, manifest_path: str | Path, *, required: set[str] | None = None) -> dict:
    root = Path(root).resolve()
    manifest = read_object(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("nonempty path-to-SHA256 inventory required")
    for relative, expected in files.items():
        part = PurePosixPath(relative)
        if part.is_absolute() or ".." in part.parts or "\\" in relative or ":" in relative:
            raise ValueError("unsafe inventory path")
        target = root.joinpath(*part.parts).resolve()
        if root not in target.parents or not target.is_file() or file_hash(target) != expected:
            raise ValueError(f"inventory file missing/changed: {relative}")
    if required and not required.issubset(files):
        raise ValueError(f"inventory omits required paths: {sorted(required - files.keys())}")
    return {"root": str(root), "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": file_hash(manifest_path), "files": files,
            "source_git_commit": manifest.get("source_git_commit"),
            "hf_id": manifest.get("hf_id"), "revision": manifest.get("revision")}


def verify_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    for document in config.get("resolved_documents", []):
        if file_hash(document["path"]) != document["sha256"]:
            raise ValueError("resolved scientific protocol file changed after config loading")
    study_files = {}
    def declared_files(value):
        if isinstance(value, dict):
            aliases = {"protocol", "recipe", "contrasts", "sample_plan", "cases", "strata_manifest",
                       "frozen_protocol", "calibration_case", "head_source", "task_assets_root", "source_root"}
            for key, item in value.items():
                if (key.endswith(("_path", "_dir")) or key in aliases) and isinstance(item, str):
                    p = Path(item).resolve()
                    if not p.exists():
                        raise ValueError(f"declared study input missing: {p}")
                    if p.is_file():
                        study_files[str(p)] = file_hash(p)
                    else:
                        for name in ("dataset_manifest.json", "release_manifest.json"):
                            if (p / name).is_file():
                                study_files[str(p / name)] = file_hash(p / name)
                else:
                    declared_files(item)
        elif isinstance(value, list):
            for item in value:
                declared_files(item)
    declared_files(config.get("study", {}))
    source_root = Path(config["source"]["root"]).resolve()
    if source_root != Path(__file__).resolve().parents[3]:
        raise ValueError("source root must describe the actually imported package")
    required_source = {p.relative_to(source_root).as_posix() for p in (source_root / "src/vlm_mcqa").rglob("*.py")}
    for declared in (config["model"]["models_config"], config.get("profiles_config")):
        if declared and source_root in Path(declared).resolve().parents:
            required_source.add(Path(declared).resolve().relative_to(source_root).as_posix())
    source = verify_inventory(source_root, config["source"]["manifest"], required=required_source)
    commit = source.get("source_git_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("source inventory requires exact Git provenance and actual local-file hashes")
    models = read_object(config["model"]["models_config"])["models"]
    spec = models[config["model"]["key"]]
    revision = spec.get("revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("pinned checkpoint revision required")
    checkpoint_root = Path(config["model"]["checkpoint_dir"]).resolve()
    # Every local file except the inventory itself is bound, preventing a stale
    # unrecorded processor/template/config from entering the load.
    inventory_path = Path(config["model"]["checkpoint_manifest"]).resolve()
    required_checkpoint = {p.relative_to(checkpoint_root).as_posix() for p in checkpoint_root.rglob("*")
                           if p.is_file() and p.resolve() != inventory_path}
    checkpoint = verify_inventory(checkpoint_root, inventory_path, required=required_checkpoint)
    if (checkpoint.get("hf_id"), checkpoint.get("revision")) != (spec["hf_id"], revision):
        raise ValueError("checkpoint inventory differs from pinned model specification")
    if "config.json" not in checkpoint["files"] or not any(p.endswith(".safetensors") for p in checkpoint["files"]):
        raise ValueError("local native safetensors checkpoint required")
    data_root = Path(config["data"]["directory"]).resolve()
    dataset = read_object(data_root / "dataset_manifest.json")
    release = read_object(data_root / "release_manifest.json")
    if dataset.get("dataset_hash") != config["data"]["dataset_hash"]:
        raise ValueError("dataset identity mismatch")
    if release.get("dataset_hash") != dataset["dataset_hash"]:
        raise ValueError("image-aware release belongs to another metadata dataset")
    if release.get("image_aware_release_hash") != config["data"]["image_aware_release_hash"]:
        raise ValueError("image-aware release mismatch")
    if release.get("errors"):
        raise ValueError("release contains unresolved validation errors")
    release_body = {k: v for k, v in release.items() if k not in {"errors", "image_aware_release_hash"}}
    if canonical_hash(release_body) != release["image_aware_release_hash"]:
        raise ValueError("image-aware release hash not derived from canonical verified inventory")
    hashes = dataset.get("sha256", {})
    if not hashes or not {"items.jsonl", "variants.jsonl"}.issubset(hashes):
        raise ValueError("metadata inventory missing")
    for name, expected in hashes.items():
        path = (data_root / name).resolve()
        if data_root not in path.parents or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"dataset metadata changed: {name}")
    recomputed = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    if recomputed != dataset["dataset_hash"]:
        raise ValueError("dataset hash not derived from verified metadata inventory")
    images = release.get("final_image_sha256", {})
    if not images:
        raise ValueError("image inventory missing")
    referenced = set()
    for metadata in ("items.jsonl", "variants.jsonl"):
        for line in (data_root / metadata).read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                relative = PurePosixPath(json.loads(line)["image"])
                if not relative.parts or relative.parts[0] != "images" or ".." in relative.parts:
                    raise ValueError("dataset image path must lie under images/")
                referenced.add(str(PurePosixPath(*relative.parts[1:])))
    if not referenced.issubset(images):
        raise ValueError("release inventory omits images referenced by actual items/variants")
    for name, expected in images.items():
        path = (data_root / "images" / name).resolve()
        if data_root not in path.parents or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"dataset image changed: {name}")
    decision_path = config["protocol"]["decision_path"]
    if file_hash(decision_path) != config["protocol"]["decision_sha256"]:
        raise ValueError("scientific decision changed")
    decision = read_object(decision_path)
    evidence_paths = []
    if decision.get("approved") is not True or not decision.get("rationale"):
        raise ValueError("explicit approved scientific protocol decision required")
    if config["kind"] in {"core7_preselected", "core7_correct_only_reduced"}:
        mode = ("preselected_protocol_extension" if config["kind"] == "core7_preselected"
                else "preselected_correct_only_sensitivity")
        expected = {"model_key": "paligemma2_10b_mix", "selection_mode": mode,
                    "screen_eligible": False, "thresholds_unchanged": True,
                    "dataset_hash": config["data"]["dataset_hash"]}
        if any(decision.get(k) != v for k, v in expected.items()):
            raise ValueError("Pali decision must preserve failed screening and unchanged criteria")
        evidence = decision.get("screen_evidence", [])
        if not evidence or not decision.get("failed_eligibility_flags"):
            raise ValueError("Pali screening lineage and failure flags required")
        for record in evidence:
            evidence_path = Path(record["path"])
            if not evidence_path.is_absolute():
                evidence_path = Path(decision_path).parent / evidence_path
            if file_hash(evidence_path) != record["sha256"]:
                raise ValueError("screen evidence missing/changed")
            evidence_paths.append(str(evidence_path.resolve()))
    return {"schema": "campaign-verified-inputs/v1", "config_sha256": canonical_hash(config),
            "source": source, "checkpoint": checkpoint, "model_spec": spec,
            "dataset_hash": dataset["dataset_hash"], "image_aware_release_hash": release["image_aware_release_hash"],
            "dataset_manifest_sha256": file_hash(data_root / "dataset_manifest.json"),
            "release_manifest_sha256": file_hash(data_root / "release_manifest.json"),
            "immutable_evidence_paths": evidence_paths,
            "study_input_file_sha256": study_files,
            "protocol_decision": decision, "protocol_decision_sha256": file_hash(decision_path)}
