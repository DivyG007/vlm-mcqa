"""Save 3,000 HF CLEVR train images with matched official scene graphs.

Run in a dedicated uv environment; writes incrementally and can be restarted.
The HF Parquet row order is not the official CLEVR image-index order.
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


MIRROR = "dpdl-benchmark/clevr"
REVISION = "a9b3cc07eacabb93d57ceef72374427982104e00"


def coords_key(objects):
    coords = objects["3d_coords"] if isinstance(objects, dict) else [
        obj["3d_coords"] for obj in objects
    ]
    return tuple(tuple(round(float(v), 5) for v in xyz) for xyz in coords)


def xy_key(scene):
    return tuple(tuple(round(float(v), 5) for v in obj["3d_coords"][:2])
                 for obj in scene["objects"])


def image_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-scenes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3000)
    args = parser.parse_args()
    out = args.out_dir
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    source = json.loads(args.source_scenes.read_text())
    scenes = source["scenes"]
    lookup = {}
    for scene in scenes:
        key = coords_key(scene["objects"])
        if key in lookup:
            raise RuntimeError("Ambiguous official scene coordinate fingerprint")
        lookup[key] = scene
    print(json.dumps({"official_scenes_indexed": len(lookup),
                      "mirror_revision": REVISION}), flush=True)

    records_path = out / "records.jsonl"
    prior = {}
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                prior[record["mirror_row_ordinal"]] = record
    print(json.dumps({"previous_records": len(prior)}), flush=True)

    dataset = load_dataset(MIRROR, split="train", streaming=True,
                           revision=REVISION)
    selected = {}
    seen_indices = set()
    with records_path.open("a", encoding="utf-8") as records_file:
        for ordinal, row in enumerate(dataset):
            if ordinal >= args.limit:
                break
            scene = lookup.get(coords_key(row["objects"]))
            if scene is None:
                raise RuntimeError(f"No official scene matches mirror row {ordinal}")
            index = scene["image_index"]
            if index in seen_indices:
                raise RuntimeError(f"Duplicate official image index {index}")
            seen_indices.add(index)
            filename = scene["image_filename"]
            path = image_dir / filename
            if ordinal in prior:
                record = prior[ordinal]
                if record["image_index"] != index or not path.is_file():
                    raise RuntimeError(f"Resume mismatch at mirror row {ordinal}")
            else:
                image = row["image"]
                temporary = image_dir / (filename + ".tmp")
                image.save(temporary, format="PNG")
                os.replace(temporary, path)
                qa = row.get("question_answer", {})
                record = {
                    "mirror_row_ordinal": ordinal,
                    "image_index": index,
                    "image_filename": filename,
                    "source_image_format": getattr(image, "format", None),
                    "image_dimensions": list(image.size),
                    "saved_png_sha256": image_sha(path),
                    "question_count": len(qa.get("question", [])),
                    "question_answer": qa,
                }
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_file.flush()
            selected[index] = scene
            if (ordinal + 1) % 100 == 0:
                print(json.dumps({"saved_or_verified": ordinal + 1,
                                  "latest_official_index": index}), flush=True)

    if len(selected) != args.limit:
        raise RuntimeError(f"Expected {args.limit} images, got {len(selected)}")
    scene_subset = {"info": source.get("info", {}),
                    "scenes": [selected[k] for k in sorted(selected)]}
    temp_scenes = out / "selected_scenes.json.tmp"
    temp_scenes.write_text(json.dumps(scene_subset), encoding="utf-8")
    os.replace(temp_scenes, out / "selected_scenes.json")

    by_xy = defaultdict(list)
    for scene in selected.values():
        by_xy[xy_key(scene)].append(scene["image_index"])
    pose_groups = [indices for indices in by_xy.values() if len(indices) > 1]
    counts = Counter(len(scene["objects"]) for scene in selected.values())
    summary = {
        "mirror": MIRROR,
        "mirror_revision": REVISION,
        "official_scene_source": str(args.source_scenes),
        "mirror_rows": args.limit,
        "matched_distinct_official_scenes": len(selected),
        "saved_png_files": len(list(image_dir.glob("CLEVR_train_*.png"))),
        "official_index_min": min(selected),
        "official_index_max": max(selected),
        "object_counts": dict(sorted(counts.items())),
        "exact_xy_pose_duplicate_groups": pose_groups[:20],
        "exact_xy_pose_duplicate_group_count": len(pose_groups),
        "record_file": str(records_path),
        "scene_file": str(out / "selected_scenes.json"),
    }
    (out / "download_summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
