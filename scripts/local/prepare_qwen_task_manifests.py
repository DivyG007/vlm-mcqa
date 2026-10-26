#!/usr/bin/env python3
"""Build prospective E23 or E24 Qwen task artifacts; never runs a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vlm_mcqa.stretch.prospective_tasks import prepare_image_option_cohort, select_spatial_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("spatial", "image-options"))
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--minimum", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path,
                        help="immutable image root used by spatial runtime; E24 uses --output")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-size", type=int, default=24)
    parser.add_argument("--padding", type=int, default=8)
    args = parser.parse_args()
    scenes = json.loads(args.scenes.read_text(encoding="utf-8"))
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("--scenes must contain a nonempty JSON list")
    if args.study == "spatial":
        if args.assets_root is None or not args.assets_root.is_dir():
            raise ValueError("spatial requires an existing --assets-root")
        if args.output.exists() or not args.output.parent.is_dir():
            raise ValueError("spatial output must be a fresh directory below an existing parent")
        cohort = select_spatial_cohort(scenes, phase=args.phase, target_units=args.target,
                                       min_units=args.minimum, seed=args.seed)
        args.output.mkdir()
        tasks_path = args.output / "tasks.json"
        tasks_path.write_text(
            json.dumps(cohort["tasks"], sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (args.output / "cohort.json").write_text(
            json.dumps(cohort, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        binding = {"schema": "stretch.task-runtime-binding.v1", "study": "spatial", "phase": args.phase,
                   "status": cohort["status"], "tasks_path": str(tasks_path.resolve()),
                   "task_assets_root": str(args.assets_root.resolve()),
                   "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
                   "protocol_overrides": {"target_units": cohort["effective_target_units"],
                                          "min_units": args.minimum, "seed": args.seed}}
        (args.output / "runtime_binding.json").write_text(
            json.dumps(binding, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        if args.font is None:
            raise ValueError("image-options requires --font")
        digest = hashlib.sha256(args.font.read_bytes()).hexdigest()
        cohort = prepare_image_option_cohort(
            scenes, phase=args.phase, target_units=args.target, min_units=args.minimum,
            seed=args.seed, output=args.output, font_path=args.font, font_sha256=digest,
            font_size=args.font_size, padding=args.padding)
    print(json.dumps({key: cohort[key] for key in ("study", "phase", "status", "effective_target_units")}, sort_keys=True))


if __name__ == "__main__":
    main()
