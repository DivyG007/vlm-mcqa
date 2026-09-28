#!/usr/bin/env python3
"""Merge per-method Core-7 shards (patching, components, heads) into one run directory.

Each GPU writes to its own result directory so concurrent ``run.json`` and
W&B state cannot collide.  This tool verifies matched provenance, copies only
the method evidence needed by aggregation, and writes a checksum manifest for
the combined discovery or confirmation directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


METHOD_FILES = {
    "patching": ("patching_{split}.jsonl", "patching_{split}_summary.json"),
    "components": (
        "components_{split}.jsonl",
        "components_{split}_summary.json",
        "component_ablation_{split}.jsonl",
    ),
    "heads": (
        "heads_screen_{split}.jsonl",
        "heads_validate_{split}.jsonl",
        "heads_ablation_{split}.jsonl",
        "heads_{split}_summary.json",
    ),
}
MATCH_FIELDS = (
    "model_key",
    "dataset_hash",
    "image_aware_release_hash",
    "source_git_commit",
    "profile_name",
    "profile",
    "split",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shard(path: Path, expected_method: str, split: str) -> dict:
    run_path = path / "run.json"
    if not run_path.is_file():
        raise ValueError(f"missing run.json: {path}")
    run = json.loads(run_path.read_text())
    if run.get("status") != "complete":
        raise ValueError(f"incomplete shard {path}: {run.get('status')}")
    if run.get("methods") != [expected_method]:
        raise ValueError(f"{path} methods are {run.get('methods')}, expected {[expected_method]}")
    if run.get("split") != split:
        raise ValueError(f"{path} split is {run.get('split')}, expected {split}")
    return run


def consolidate(split: str, out: Path, shards: dict[str, Path]) -> dict:
    if out.exists():
        raise ValueError(f"combined output already exists: {out}")
    loaded = {method: load_shard(path, method, split) for method, path in shards.items()}
    reference = loaded["patching"]
    for method, run in loaded.items():
        for field in MATCH_FIELDS:
            if run.get(field) != reference.get(field):
                raise ValueError(f"{method} shard differs on {field}")

    out.mkdir(parents=True)
    manifest = {
        "kind": "core7_parallel_method_consolidation",
        "split": split,
        "methods": {},
        "files": {},
    }
    for method, source in shards.items():
        copied = []
        for template in METHOD_FILES[method]:
            name = template.format(split=split)
            path = source / name
            # Discovery screens heads; confirmation deliberately does not.
            if method == "heads" and split == "confirmation" and name.startswith("heads_screen_"):
                continue
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty {method} evidence: {path}")
            destination = out / name
            shutil.copy2(path, destination)
            copied.append(name)
            manifest["files"][name] = {
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        manifest["methods"][method] = {
            "source": str(source.resolve()),
            "wandb_url": loaded[method].get("wandb_url"),
            "method_seconds": loaded[method].get("method_seconds", {}),
            "files": copied,
        }

    combined = dict(reference)
    combined.update(
        status="complete",
        methods=["patching", "components", "heads"],
        command=["merge_shards.py", "--split", split],
        parallel_method_shards=manifest["methods"],
        method_seconds={
            method: loaded[method].get("method_seconds", {}).get(method)
            for method in ("patching", "components", "heads")
        },
    )
    (out / "run.json").write_text(json.dumps(combined, indent=2, sort_keys=True, default=str) + "\n")
    (out / "shard_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--patching", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    args = parser.parse_args()
    result = consolidate(
        args.split,
        args.out,
        {"patching": args.patching, "components": args.components, "heads": args.heads},
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
