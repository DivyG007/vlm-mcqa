"""Prospective E23/E24 task construction without model execution.

Inputs are explicit per-scene records. Outputs are new derivative files; source
images and metadata are read-only and never modified.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from collections import Counter
from typing import Any, Mapping, Sequence

from .assets import assemble_image_options, image_option_variants, spatial_from_objects
from .task_prep import image_option_plan, validate_split_separation


def _rank(seed: int, *parts: Any) -> str:
    return hashlib.sha256(json.dumps([seed, *parts], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _description(obj: Mapping[str, Any]) -> str:
    if isinstance(obj.get("description"), str) and obj["description"]:
        return obj["description"]
    values = [obj.get(key) for key in ("size", "material", "shape")]
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("object needs a description or size/material/shape")
    return "the " + " ".join(values)


def _verified_scene(scene: Mapping[str, Any], phase: str) -> tuple[dict, list[dict]]:
    source = dict(scene["source"])
    if source.get("split") != phase or phase not in {"discovery", "confirmation"}:
        raise ValueError("scene source split must equal requested phase")
    metadata = Path(scene["source_metadata_path"])
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != source.get("source_sha256"):
        raise ValueError("source metadata SHA256 mismatch")
    image_path = Path(scene["image_path"])
    image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if (not isinstance(scene.get("image"), str) or not scene["image"]
            or scene.get("image_sha256") != image_digest):
        raise ValueError("explicit image path/name and matching SHA256 required")
    objects = []
    for raw in scene["objects"]:
        obj = dict(raw)
        obj["description"] = _description(obj)
        if (not isinstance(obj.get("id"), str) or not obj["id"]
                or not isinstance(obj.get("color"), str) or not obj["color"]
                or not isinstance(obj.get("bbox"), list) or len(obj["bbox"]) != 4):
            raise ValueError("objects require id, color and four-coordinate bbox")
        objects.append(obj)
    if len({obj["id"] for obj in objects}) != len(objects):
        raise ValueError("object ids must be unique")
    return source, objects


def build_spatial_parent(scene: Mapping[str, Any], *, phase: str, seed: int,
                         labels: Sequence[str] = ("A", "B", "C", "D")) -> dict:
    """Build one eligible E23 2x2 parent or raise a prospective exclusion."""
    source, objects = _verified_scene(scene, phase)
    description_counts = Counter(obj["description"] for obj in objects)
    queryable = [obj for obj in objects if description_counts[obj["description"]] == 1]
    colors = sorted({obj["color"] for obj in objects})
    if len(colors) < 5:
        raise ValueError("scene needs at least five annotated colors")
    pairs = [(a, b) for a, b in itertools.combinations(queryable, 2) if a["color"] != b["color"]]
    if not pairs:
        raise ValueError("scene has no two uniquely described distinct-color query objects")
    first, second = min(pairs, key=lambda pair: _rank(seed, source["scene_id"], pair[0]["id"], pair[1]["id"]))
    truth = [first["color"], second["color"]]
    distractors = sorted(set(colors) - set(truth), key=lambda color: _rank(seed, source["scene_id"], color))
    if len(distractors) < 3:
        raise ValueError("scene needs one common and two alternate distractor colors")
    option_sets = [[*truth, distractors[0], distractors[1]], [*truth, distractors[0], distractors[2]]]
    spatial_from_objects(source, image=scene["image"], objects=objects,
                         query_objects=[first["id"], second["id"]], option_sets=option_sets, labels=labels)
    return {"source": source, "image": scene["image"],
            "image_sha256": scene["image_sha256"],
            "source_metadata_path": str(Path(scene["source_metadata_path"]).resolve()),
            "objects": objects, "query_objects": [first["id"], second["id"]],
            "option_sets": option_sets, "labels": list(labels)}


def select_spatial_cohort(scenes: Sequence[Mapping[str, Any]], *, phase: str,
                          target_units: int, min_units: int, seed: int) -> dict:
    """Prospectively select E23 parents with declared reduced-power fallback."""
    if not 1 <= min_units <= target_units:
        raise ValueError("invalid minimum/target")
    built, exclusions = [], []
    for scene in sorted(scenes, key=lambda row: _rank(seed, row.get("source", {}).get("scene_id"))):
        try:
            built.append(build_spatial_parent(scene, phase=phase, seed=seed))
        except (KeyError, OSError, TypeError, ValueError) as error:
            exclusions.append({"scene_id": scene.get("source", {}).get("scene_id"), "reason": str(error)})
    if len(built) >= target_units:
        status, selected = "ready", built[:target_units]
    elif len(built) >= min_units:
        status, selected = "reduced_power", built
    else:
        status, selected = "insufficient", []
    validate_split_separation([{"source": row["source"], "task_kind": "spatial-guidance"} for row in selected])
    return {"schema": "stretch.prospective-cohort.v1", "study": "spatial", "phase": phase,
            "status": status, "target_units": target_units, "minimum_units": min_units,
            "effective_target_units": len(selected), "tasks": selected, "exclusions": exclusions}


def validate_cohort_separation(*cohorts: Mapping[str, Any]) -> None:
    records = []
    for cohort in cohorts:
        kind = "spatial-guidance" if cohort["study"] == "spatial" else "image-options"
        records.extend({"source": row["source"], "task_kind": kind} for row in cohort["tasks"])
    validate_split_separation(records)


def _crop(image: Any, bbox: Sequence[Any], padding: int) -> Any:
    if len(bbox) != 4 or any(type(v) not in (int, float) for v in bbox) or type(padding) is not int or padding < 0:
        raise ValueError("valid bbox and nonnegative integer padding required")
    x0, y0, x1, y1 = bbox
    box = (max(0, int(x0) - padding), max(0, int(y0) - padding),
           min(image.width, int(x1) + padding), min(image.height, int(y1) + padding))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("empty object crop")
    return image.crop(box)


def build_image_option_parent(scene: Mapping[str, Any], *, phase: str, seed: int,
                              output: str | Path, assets_root: str | Path,
                              font_path: str | Path, font_sha256: str,
                              font_size: int = 24, padding: int = 8) -> list[dict]:
    """Create one E24 parent's four crops, two orthogonal layouts, and variants."""
    from PIL import Image

    source, objects = _verified_scene(scene, phase)
    if len(objects) < 4:
        raise ValueError("E24 requires at least four annotated objects")
    destination = Path(output).resolve()
    root = Path(assets_root).resolve()
    if destination.exists() or destination == root or root not in destination.parents:
        raise ValueError("fresh derivative directory strictly below assets_root required")
    destination.mkdir(parents=True, exist_ok=False)
    crop_dir = destination / "crops"
    crop_dir.mkdir()
    chosen = sorted(objects, key=lambda obj: _rank(seed, source["scene_id"], obj["id"]))[:4]
    if len({obj["description"] for obj in chosen}) != 4:
        raise ValueError("E24 crops need four distinct text-match descriptions")
    asset_map = {}
    with Image.open(scene["image_path"]) as loaded:
        image = loaded.convert("RGB")
        for index, obj in enumerate(chosen):
            semantic = obj["description"]
            target = crop_dir / f"crop-{index}.png"
            with target.open("xb") as stream:
                _crop(image, obj["bbox"], padding).save(stream, format="PNG", optimize=False, compress_level=9)
            asset_map[semantic] = str(target.relative_to(destination))
    hashes = [hashlib.sha256((destination / path).read_bytes()).hexdigest() for path in asset_map.values()]
    if len(set(hashes)) != 4:
        raise ValueError("selected E24 crops are not byte-distinct")
    semantic = list(asset_map)
    layouts = [
        (semantic[0], semantic, ["A", "B", "C", "D"]),
        (semantic[1], [semantic[2], semantic[1], semantic[3], semantic[0]], ["B", "D", "A", "C"]),
    ]
    tasks = []
    for index, (target_semantic, order, labels) in enumerate(layouts):
        layout_dir = destination / f"layout-{index}"
        layout_dir.mkdir()
        plan = image_option_plan(source, query_asset=asset_map[target_semantic], option_assets=asset_map,
                                 target_option=target_semantic, panel_order=order, labels_by_position=labels)
        visual_path, text_path = layout_dir / "visual.png", layout_dir / "text.png"
        visual = assemble_image_options(plan, asset_base=destination, font_path=font_path,
                                        font_sha256=font_sha256, font_size=font_size,
                                        output=visual_path, mode="visual")
        text = assemble_image_options(plan, asset_base=destination, font_path=font_path,
                                      font_sha256=font_sha256, font_size=font_size,
                                      output=text_path, mode="text-query-only")
        relative_visual = str(visual_path.relative_to(root))
        relative_text = str(text_path.relative_to(root))
        variants = image_option_variants(plan, image=relative_visual, text_image=relative_text)
        for row, assembled in zip(variants, (visual, text)):
            row["image_sha256"] = assembled["image_sha256"]
            row["assembly_provenance"] = assembled
            row["source_metadata_path"] = str(Path(scene["source_metadata_path"]).resolve())
        tasks.extend(variants)
    return tasks


def prepare_image_option_cohort(scenes: Sequence[Mapping[str, Any]], *, phase: str,
                                target_units: int, min_units: int, seed: int,
                                output: str | Path, font_path: str | Path,
                                font_sha256: str, font_size: int = 24,
                                padding: int = 8) -> dict:
    """Build a fresh E24 derivative root with prospective underfill reporting."""
    if not 1 <= min_units <= target_units:
        raise ValueError("invalid minimum/target")
    root = Path(output).resolve()
    if root.exists() or not root.parent.is_dir():
        raise ValueError("fresh E24 output below an existing parent required")
    root.mkdir()
    tasks, parent_sources, exclusions = [], [], []
    ordered = sorted(scenes, key=lambda row: _rank(seed, row.get("source", {}).get("scene_id")))
    for index, scene in enumerate(ordered):
        if len(parent_sources) == target_units:
            break
        try:
            built = build_image_option_parent(
                scene, phase=phase, seed=seed, output=root / f"parent-{index:04d}",
                assets_root=root, font_path=font_path, font_sha256=font_sha256,
                font_size=font_size, padding=padding)
            tasks.extend(built)
            parent_sources.append(dict(scene["source"]))
        except (KeyError, OSError, TypeError, ValueError) as error:
            exclusions.append({"scene_id": scene.get("source", {}).get("scene_id"), "reason": str(error)})
    count = len(parent_sources)
    status = "ready" if count == target_units else "reduced_power" if count >= min_units else "insufficient"
    validate_split_separation([{"source": source, "task_kind": "image-options"} for source in parent_sources])
    runnable = tasks if status != "insufficient" else []
    report = {"schema": "stretch.prospective-cohort.v1", "study": "image_options", "phase": phase,
              "status": status, "target_units": target_units, "minimum_units": min_units,
              "effective_target_units": count if runnable else 0, "tasks": runnable, "exclusions": exclusions,
              "generated_candidate_units": count,
              "assets_root": str(root), "manual_visual_audit_required": True}
    tasks_path = root / "tasks.json"
    tasks_path.write_text(json.dumps(runnable, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not runnable and tasks:
        (root / "insufficient_candidates.json").write_text(
            json.dumps(tasks, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (root / "cohort.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    binding = {"schema": "stretch.task-runtime-binding.v1", "study": "image_options", "phase": phase,
               "status": status, "tasks_path": str(tasks_path), "task_assets_root": str(root),
               "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
               "protocol_overrides": {"target_units": report["effective_target_units"],
                                      "min_units": min_units, "seed": seed}}
    (root / "runtime_binding.json").write_text(
        json.dumps(binding, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def write_audit_sheet(tasks: Sequence[Mapping[str, Any]], *, assets_root: str | Path,
                      output: str | Path, columns: int = 4, thumb: int = 240) -> dict:
    """Render an auditable contact sheet; it does not certify scientific validity."""
    from PIL import Image, ImageDraw, ImageOps

    visual = [row for row in tasks if row.get("condition") == "visual"]
    target = Path(output)
    if target.exists() or not target.parent.is_dir() or not visual:
        raise ValueError("fresh audit output and nonempty visual task list required")
    rows = (len(visual) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, task in enumerate(visual):
        with Image.open(Path(assets_root) / task["image"]) as loaded:
            view = ImageOps.contain(loaded.convert("RGB"), (thumb, thumb))
        x, y = index % columns * thumb, index // columns * (thumb + 28)
        sheet.paste(view, (x + (thumb - view.width) // 2, y))
        draw.text((x + 4, y + thumb + 4), task["task_id"][-20:], fill="black")
    with target.open("xb") as stream:
        sheet.save(stream, format="PNG", optimize=False, compress_level=9)
    return {"image": str(target.resolve()), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "visual_tasks": len(visual), "certifies": "rendering_only_manual_visual_audit_still_required"}
