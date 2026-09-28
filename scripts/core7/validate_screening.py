"""Fail closed before reusing or selecting full Core-7 screening results.

This reads only local JSON/JSONL and Git metadata; it never loads a checkpoint.
Run with --model KEY to validate one result, --all to validate all twelve, or
with neither to check the frozen dataset and clean source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from vlm_mcqa.core7.source_snapshot import source_state


ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_dataset(data: Path, cfg_path: Path | None = None) -> str:
    manifest = _json(data / "dataset_manifest.json")
    cfg = _json(cfg_path or ROOT / "configs/core7/clevr_mcq4_dataset.json")
    snapshot = data / "dataset_config.json"
    if snapshot.is_file() and _json(snapshot) != cfg:
        raise ValueError("dataset config snapshot differs from frozen source config")
    if manifest.get("dataset_version") != cfg["dataset_version"]:
        raise ValueError("dataset version differs from frozen config")
    counts = manifest.get("counts", {}).get("items_by_split", {})
    for split, size in cfg["split_sizes"].items():
        expected = 2 * size if split in ("discovery", "confirmation") else size
        if counts.get(split) != expected:
            raise ValueError(f"{split}: expected {expected} items, found {counts.get(split)}")
    expected_items = sum(counts.values())
    pair_members = counts["discovery"] + counts["confirmation"]
    expected_variants = 12 * expected_items + 4 * pair_members
    actual_variants = sum(manifest.get("counts", {}).get("variants_by_split", {}).values())
    if actual_variants != expected_variants:
        raise ValueError(f"expected {expected_variants} variants, found {actual_variants}")
    for name, expected_hash in manifest.get("sha256", {}).items():
        path = data / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"dataset file missing or hash mismatch: {path}")
    if not (data / "validation_report.json").is_file():
        raise ValueError("dataset validation_report.json missing")
    with (data / "items.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            image = data / json.loads(line)["image"]
            if not image.is_file():
                raise ValueError(f"dataset image missing: {image}")
    value = manifest.get("dataset_hash")
    computed = hashlib.sha256(json.dumps(manifest.get("sha256", {}), sort_keys=True).encode()).hexdigest()[:16]
    if value != computed:
        raise ValueError("dataset_hash missing or inconsistent with JSONL hashes")
    return value


def check_run(root: Path, key: str, *, dataset_hash: str, commit: str, profile: str,
              cfg_path: Path | None = None, release_hash: str | None = None) -> None:
    directory = root / key
    run = _json(directory / "run.json")
    expected = {"model_key": key, "status": "complete", "dataset_hash": dataset_hash,
                "profile_name": profile, "source_git_commit": commit, "source_tree_dirty": False}
    for field, value in expected.items():
        if run.get(field) != value:
            raise ValueError(f"{directory}: {field} is {run.get(field)!r}, expected {value!r}")
    if release_hash and run.get("image_aware_release_hash") != release_hash:
        raise ValueError(f"{directory}: image-aware release hash mismatch")
    if not _json(directory / "calibration.json").get("all_passed"):
        raise ValueError(f"{directory}: calibration failed")
    summary = _json(directory / "behavior_screening_summary.json")
    cfg = _json(cfg_path or ROOT / "configs/core7/clevr_mcq4_dataset.json")
    if summary.get("items") != cfg["split_sizes"]["screening"]:
        raise ValueError(f"{directory}: screening item count mismatch")
    if len(summary.get("robustness_cells") or {}) != 12:
        raise ValueError(f"{directory}: expected twelve robustness cells")
    for split in ("discovery", "confirmation"):
        actual = summary.get("pair_yield", {}).get(split, {}).get("pairs")
        if actual != cfg["split_sizes"][split]:
            raise ValueError(f"{directory}: {split} pair count mismatch: {actual}")
    for name in ("behavior_screening.jsonl", "compliance_screening.jsonl"):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{directory}: missing or empty {name}")


def check_selection(root: Path, path: Path, registered: set[str]) -> None:
    """Confirm a stored decision was computed from exactly the current runs."""
    from vlm_mcqa.core7.report import select
    from vlm_mcqa.core7.session import load_profiles

    recorded = _json(path)
    with tempfile.TemporaryDirectory() as tmp:
        expected = select([root / key for key in sorted(registered)],
                          load_profiles()["selection"], Path(tmp))
    if json.dumps(recorded, sort_keys=True) != json.dumps(expected, sort_keys=True, default=str):
        raise ValueError(f"selection differs from the twelve current screening results: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--profile", default="full_core7")
    parser.add_argument("--dataset-config", type=Path,
                        default=ROOT / "configs/core7/clevr_mcq4_dataset.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--selection", type=Path, help="also verify a stored selection against all twelve runs")
    args = parser.parse_args(argv)
    commit, dirty = source_state(ROOT)
    if dirty:
        raise SystemExit("source tree is dirty; commit the execution code before full screening")
    dataset_hash = check_dataset(args.data, args.dataset_config)
    release_path = args.data / "release_manifest.json"
    release_hash = _json(release_path).get("image_aware_release_hash") if release_path.exists() else None
    registered = set(_json(ROOT / "configs/core7/models.json")["models"])
    if len(registered) != 12:
        raise SystemExit(f"expected 12 registered models, found {len(registered)}")
    keys = sorted(registered) if args.all else ([args.model] if args.model else [])
    if args.selection and not args.all:
        raise SystemExit("--selection requires --all")
    for key in keys:
        if key not in registered:
            raise SystemExit(f"unregistered model key: {key}")
        check_run(args.screen_root, key, dataset_hash=dataset_hash, commit=commit,
                  profile=args.profile, cfg_path=args.dataset_config, release_hash=release_hash)
    if args.selection:
        check_selection(args.screen_root, args.selection, registered)
    print(f"[validated] dataset={dataset_hash} commit={commit} profile={args.profile} models={len(keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
