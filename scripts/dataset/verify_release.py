"""Verify the completed mixed-source CLEVR release and hash image contents."""

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path


EXPECTED_SPLITS = {"calibration": 120, "screening": 300, "behavior": 256,
                   "discovery": 320, "confirmation": 256}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    root = args.dataset
    acquisition = args.acquisition
    existing_path = root / "release_manifest.json"
    existing = json.loads(existing_path.read_text()) if args.verify_existing else None
    source_commit = args.source_commit or (existing or {}).get("source_commit")
    if not source_commit:
        parser.error("--source-commit is required when writing a release manifest")
    errors = []
    pair_report = json.loads((root / "pair_validation.json").read_text())
    if (pair_report["errors"] or pair_report["verified_pngs"] != 576
            or pair_report["verified_scene_jsons"] != 576):
        errors.append("pair render validation did not pass for 576 image/scene records")
    for name, expected in pair_report["png_sha256"].items():
        path = root / "images" / name
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"transferred pair image checksum: {name}")
        if not (root / "rendered_scenes" / Path(name).with_suffix(".json").name).is_file():
            errors.append(f"transferred pair scene missing: {name}")
    for name, expected in pair_report["rendered_scene_sha256"].items():
        path = root / "rendered_scenes" / name
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"transferred pair scene checksum: {name}")

    records = [json.loads(line) for line in (acquisition / "records.jsonl").read_text().splitlines()
               if line.strip()]
    source_hashes = {record["image_filename"]: record["saved_png_sha256"]
                     for record in records}
    items = [json.loads(line) for line in (root / "items.jsonl").read_text().splitlines()
             if line.strip()]
    counts = Counter(item["split"] for item in items)
    if dict(counts) != EXPECTED_SPLITS:
        errors.append(f"item split counts: {dict(counts)}")
    if len(items) != 1252:
        errors.append(f"item count: {len(items)}")
    names = [Path(item["image"]).name for item in items]
    if len(set(names)) != len(items):
        errors.append("duplicate final item image")
    image_hashes = {}
    for item in items:
        name = Path(item["image"]).name
        path = root / item["image"]
        if not path.is_file():
            errors.append(f"final image missing: {name}")
            continue
        with path.open("rb") as stream:
            header = stream.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", header[16:24]) != (480, 320):
            errors.append(f"final image PNG/dimensions: {name}")
        digest = sha256(path)
        image_hashes[name] = digest
        if item["pair_member"] is None and source_hashes.get(name) != digest:
            errors.append(f"official single image checksum: {name}")
        if item["pair_member"] is not None and pair_report["png_sha256"].get(name) != digest:
            errors.append(f"pair image checksum: {name}")
    if len(image_hashes) != 1252:
        errors.append(f"final image count: {len(image_hashes)}")
    dataset_manifest = json.loads((root / "dataset_manifest.json").read_text())
    for name, expected in dataset_manifest["sha256"].items():
        if sha256(root / name) != expected:
            errors.append(f"dataset JSONL checksum: {name}")

    download = json.loads((acquisition / "download_summary.json").read_text())
    provenance = {
        "dataset_version": dataset_manifest["dataset_version"],
        "dataset_hash": dataset_manifest["dataset_hash"],
        "source_commit": source_commit,
        "source_mirror": download["mirror"],
        "source_revision": download["mirror_revision"],
        "official_annotation_zip_sha256": sha256(acquisition / "source" / "CLEVR_v1.0_no_images.zip"),
        "selected_scenes_sha256": sha256(acquisition / "selected_scenes.json"),
        "exact_pair_search_sha256": sha256(acquisition / "exact_pair_search.json"),
        "acquisition_validation_sha256": sha256(acquisition / "validation_report.json"),
        "dataset_config_sha256": sha256(root / "dataset_config.json"),
        "pair_validation_sha256": sha256(root / "pair_validation.json"),
        "render_profile": pair_report["render_profile"],
        "renderer_script_sha256": pair_report["renderer_script_sha256"],
        "item_split_counts": dict(counts),
        "final_image_sha256": dict(sorted(image_hashes.items())),
        "errors": errors,
    }
    canonical = json.dumps({key: value for key, value in provenance.items()
                            if key != "errors"}, sort_keys=True, separators=(",", ":"))
    provenance["image_aware_release_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    if args.verify_existing:
        if existing != provenance:
            errors.append("stored image-aware release manifest differs from current contents")
    else:
        existing_path.write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"items": len(items), "split_counts": dict(counts),
                      "image_count": len(image_hashes), "errors": errors,
                      "image_aware_release_hash": provenance["image_aware_release_hash"]},
                     indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
