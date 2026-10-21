"""Explicit, offline optional-task manifest preparation; no model commands.

Authored but deliberately not executed during the October 21 implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .task_prep import image_option_plan, spatial_factorial, validate_split_separation


def _overlap(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def preparation_envelope(task: dict, *, path_base: Path, source_path: Path) -> dict:
    """Keep the hashed scientific task immutable; paths are separate bookkeeping."""
    return {"task": task, "preparation": {"path_base": str(path_base.resolve()),
            "declared_source_path": str(source_path.resolve()), "source_hash_verified": False}}


def write_new_manifest(output: Path, record: dict, *, inputs: Iterable[Path]) -> Path:
    """Create one new directory, refusing overlaps; never update a prior output.

    Inputs must include source metadata AND referenced assets. Each supplied file's
    entire parent is protected (directories protect themselves). An interrupted
    write remains preserved and its directory cannot be reused automatically.
    """
    output = output.resolve()
    paths = [p.resolve() for p in inputs]
    if not paths or any(not p.exists() for p in paths):
        raise ValueError("all declared inputs must exist")
    protected = [p if p.is_dir() else p.parent for p in paths]
    if output.exists() or any(_overlap(output, p) for p in protected):
        raise ValueError("output exists or overlaps a protected input directory")
    # Only an explicitly pre-existing parent is accepted; no hidden tree setup.
    if not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    text = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output.mkdir(exist_ok=False)
    target = output / "task_manifest.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("spatial", "image-options", "spatial-build", "assemble-options", "iconqa-import", "combine-tasks"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new derivative directory")
    args = parser.parse_args(argv)
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    def explicit_path(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (spec_path.parent / path).resolve()
    if args.kind == "iconqa-import":
        from .external import import_iconqa
        import_iconqa(explicit_path(spec["source_path"]), args.output, **spec["import"])
        return
    if args.kind == "combine-tasks":
        paths = [spec_path, *(explicit_path(p) for p in spec["task_files"])]
        records = [row for path in paths[1:] for row in json.loads(path.read_text(encoding="utf-8"))]
        # Multiple repeated layouts are allowed; cross-phase parent reuse is not.
        validate_split_separation([{"source": r["source"], "task_kind": "image-options"} for r in records])
        write_new_manifest(args.output, {"inputs": [str(p) for p in paths[1:]]}, inputs=paths)
        with (args.output / "tasks.json").open("x", encoding="utf-8") as stream:
            json.dump(records, stream, sort_keys=True, indent=2, allow_nan=False)
        return
    # A complete declared source path is mandatory in addition to its SHA256.
    # Hash verification is a later caller gate; this tool does not assert it.
    source_path = Path(spec["source_path"])
    if not source_path.is_absolute():
        source_path = spec_path.parent / source_path
    paths = [spec_path, source_path]
    if args.kind == "spatial-build":
        from .assets import spatial_from_objects
        spatial = dict(spec["spatial"])
        spatial["source"] = spec["source"]
        record = spatial_from_objects(**spatial)
        image = Path(spatial["image"])
        paths.append(image if image.is_absolute() else spec_path.parent / image)
        import hashlib
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != spec["source"]["source_sha256"]:
            raise ValueError("spatial source metadata SHA256 mismatch")
        image_path = paths[-1].resolve()
        spatial["image_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        spatial["source_metadata_path"] = str(source_path.resolve())
        write_new_manifest(args.output, record, inputs=paths)
        # Image path stays relative to explicitly configured task_assets_root.
        with (args.output / "tasks.json").open("x", encoding="utf-8") as stream:
            json.dump([spatial], stream, sort_keys=True, indent=2, allow_nan=False)
        return
    if args.kind == "assemble-options":
        from .assets import assemble_image_options, image_option_variants
        record = image_option_plan(spec["source"], **spec["plan"])
        base = explicit_path(spec["asset_base"])
        font = explicit_path(spec["font_path"])
        paths.extend([font, base, *(base / p for p in [spec["plan"]["query_asset"], *spec["plan"]["option_assets"].values()])])
        import hashlib
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != spec["source"]["source_sha256"]:
            raise ValueError("image-option source metadata SHA256 mismatch")
        write_new_manifest(args.output, record, inputs=paths)
        visual = assemble_image_options(record, asset_base=base, font_path=font, font_sha256=spec["font_sha256"],
                                        font_size=spec["font_size"], output=args.output / "visual.png", mode="visual")
        text = assemble_image_options(record, asset_base=base, font_path=font, font_sha256=spec["font_sha256"],
                                      font_size=spec["font_size"], output=args.output / "text.png", mode="text-query-only")
        relative = spec["output_relative_to_assets_root"]
        tasks = image_option_variants(record, image=relative + "/visual.png", text_image=relative + "/text.png")
        for row, assembled in zip(tasks, (visual, text)):
            row["image_sha256"] = assembled["image_sha256"]
            row["assembly_provenance"] = assembled
            row["source_metadata_path"] = str(source_path.resolve())
        with (args.output / "tasks.json").open("x", encoding="utf-8") as stream:
            json.dump(tasks, stream, sort_keys=True, indent=2, allow_nan=False)
        return
    if args.kind == "spatial":
        record = spatial_factorial(spec["source"], spec["conditions"])
        assets = [c["image"] for c in spec["conditions"]]
    else:
        record = image_option_plan(spec["source"], **spec["plan"])
        assets = [spec["plan"]["query_asset"], *spec["plan"]["option_assets"].values()]
    for asset in assets:
        path = Path(asset)
        paths.append(path if path.is_absolute() else spec_path.parent / path)
    validate_split_separation([record])
    # Bind paths separately so the task body and its derived identity stay intact.
    envelope = preparation_envelope(record, path_base=spec_path.parent, source_path=source_path)
    write_new_manifest(args.output, envelope, inputs=paths)


if __name__ == "__main__":
    main()
