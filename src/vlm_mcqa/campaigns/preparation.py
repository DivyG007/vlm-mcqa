"""Explicit future manifest preparation; never run implicitly by a campaign."""
from __future__ import annotations

import json
from pathlib import Path

from .verification import file_hash


def prepare_inventory(root, output, *, kind, commit=None, hf_id=None, revision=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir() or output.exists() or output == root or output in root.parents:
        raise ValueError("existing input root and fresh inventory file required")
    if kind == "source":
        if not isinstance(commit, str) or len(commit) != 40:
            raise ValueError("full source commit provenance required")
        paths = [*root.glob("src/**/*.py"), *root.glob("configs/**/*.json")]
        paths += [p for p in (root / "pyproject.toml",) if p.is_file()]
        record = {"schema": "campaign-source-inventory/v1", "source_git_commit": commit,
                  "source_state": "actual_file_manifest; git cleanliness not asserted"}
    elif kind == "checkpoint":
        if not hf_id or not isinstance(revision, str) or len(revision) != 40:
            raise ValueError("publisher HF ID and full pinned revision required")
        paths = [p for p in root.rglob("*") if p.is_file()]
        record = {"schema": "campaign-checkpoint-inventory/v1", "hf_id": hf_id, "revision": revision}
    else:
        raise ValueError("kind must be source or checkpoint")
    paths = [p for p in paths if p.resolve() != output]
    if not paths:
        raise ValueError("empty inventory")
    record["files"] = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}
    if not output.parent.is_dir():
        raise ValueError("inventory parent must already exist")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record
