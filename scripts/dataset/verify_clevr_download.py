"""Independently verify the pinned 3,000-image CLEVR acquisition."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()
               if line.strip()]
    scenes = json.loads((root / "selected_scenes.json").read_text())["scenes"]
    by_index = {scene["image_index"]: scene for scene in scenes}
    files = list((root / "images").glob("CLEVR_train_*.png"))
    errors = []
    seen = set()
    for ordinal, record in enumerate(records):
        index = record["image_index"]
        name = record["image_filename"]
        path = root / "images" / name
        if record["mirror_row_ordinal"] != ordinal or index in seen:
            errors.append(f"ordinal or index mismatch: {ordinal}")
        seen.add(index)
        if index not in by_index or by_index[index]["image_filename"] != name:
            errors.append(f"scene mismatch: {name}")
        if not path.is_file():
            errors.append(f"missing: {name}")
            continue
        with path.open("rb") as stream:
            header = stream.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", header[16:24]) != (480, 320):
            errors.append(f"PNG header/dimensions: {name}")
        if sha256(path) != record["saved_png_sha256"]:
            errors.append(f"SHA-256 mismatch: {name}")
        if record["question_count"] != 10:
            errors.append(f"question count: {name}")
    if len(records) != 3000 or len(scenes) != 3000 or len(files) != 3000:
        errors.append(f"counts records/scenes/files: {len(records)}/{len(scenes)}/{len(files)}")
    if len(seen) != 3000 or len(by_index) != 3000:
        errors.append("non-distinct official image index")
    report = {"records": len(records), "scenes": len(scenes), "png_files": len(files),
              "unique_official_indices": len(seen), "errors": errors}
    (root / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
