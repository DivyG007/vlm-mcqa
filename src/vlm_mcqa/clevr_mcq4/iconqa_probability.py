"""Audited native probability pairs; no model calls or dataset downloads."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

from .build import expand

CHOICES = ["certain", "impossible", "probable", "unlikely"]
SPLITS = {"train": "calibration", "val": "discovery", "test": "confirmation"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def select_pairs(pools, inventory, *, seed, train_pairs=8):
    """Use native splits and globally unique, non-cross-split source pixels."""
    pixel_splits = defaultdict(set)
    for row in inventory.values():
        pixel_splits[row["pixel_sha256"]].add(row["data"]["split"])
    selected, pixels, pids = [], set(), set()
    for split in SPLITS:
        candidates = [p for p in pools[split] if p.get("skills") == ["probability"]
                      or set(p.get("options_x", [])) == set(CHOICES)]
        candidates.sort(key=lambda p: hashlib.sha256(
            f"{seed}:{split}:{p['x']}:{p['y']}".encode()).hexdigest())
        if split == "train":
            candidates = candidates[:train_pairs]
            if len(candidates) != train_pairs:
                raise ValueError("insufficient independent training pilot pairs")
        for pair in candidates:
            x, y = (inventory[str(pair[k])] for k in ("x", "y"))
            for row in (x, y):
                d = row["data"]
                if d["split"] != split or d["ques_type"] != "choose_txt" or row.get("errors"):
                    raise ValueError("invalid source task/split/audit")
                if len(d["choices"]) != 4 or set(d["choices"]) != set(CHOICES):
                    raise ValueError("pair lacks native probability choices")
                if type(d["answer"]) is not int or not 0 <= d["answer"] < 4:
                    raise ValueError("invalid source answer")
                if d.get("hint") or d.get("unit"):
                    raise ValueError("unhandled hint or unit")
                pixel = row["pixel_sha256"]
                if len(pixel_splits[pixel]) != 1:
                    raise ValueError("source image reused across original splits")
                if pixel in pixels or row["pid"] in pids:
                    raise ValueError("reused image/PID in independent pair pool")
                pixels.add(pixel)
                pids.add(row["pid"])
            if x["data"]["question"] != y["data"]["question"]:
                raise ValueError("natural pair question mismatch")
            answers = [r["data"]["choices"][r["data"]["answer"]] for r in (x, y)]
            if answers[0] == answers[1]:
                raise ValueError("natural pair must have different correct answers")
            if any(pair.get(f"answer_{k}") != a for k, a in zip(("x", "y"), answers)):
                raise ValueError("candidate answers disagree with audited inventory")
            selected.append((split, pair, x, y))
    return selected


def build(source, audit, output, *, seed=20261028, train_pairs=8, canvas=448):
    source, audit, output = map(lambda p: Path(p).resolve(), (source, audit, output))
    if canvas < 28 or canvas % 28:
        raise ValueError("canvas must be a positive Qwen merged-grid multiple of 28")
    if output.exists() or any(output == p or p in output.parents or output in p.parents for p in (source, audit)):
        raise ValueError("fresh output outside immutable source/audit required")
    paths = {s: audit / "supplement_final" / f"quarantined_review_pool_{s}.jsonl" for s in SPLITS}
    pools = {s: list(read(p)) for s, p in paths.items()}
    # Only metadata/pixel identities are retained, not every decoded image.
    inventory = {r["pid"]: r for r in read(audit / "inventory.jsonl")}
    selected = select_pairs(pools, inventory, seed=seed, train_pairs=train_pairs)
    counts = Counter(s for s, *_ in selected)
    if counts["val"] < 64 or counts["test"] < 64:
        raise ValueError(f"structural full-profile shortfall: {dict(counts)}")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    items, review, image_hashes = [], [], {}
    for rank, (split, pair, x, y) in enumerate(selected):
        pair_id = f"iconqa_probability_{split}_{pair['x']}_{pair['y']}"
        for member, row in (("x", x), ("y", y)):
            d, pid = row["data"], row["pid"]
            folder = source / "iconqa" / split / "choose_txt" / pid
            metadata = json.loads((folder / "data.json").read_text(encoding="utf-8"))
            if metadata != {k: v for k, v in d.items() if k != "split"}:
                raise ValueError(f"source metadata changed: {pid}")
            original = folder / "image.png"
            if sha(original) != row["byte_sha256"]:
                raise ValueError(f"source image changed: {pid}")
            relative = f"images/iconqa_{pid}.png"
            with Image.open(original) as image:
                rgb = image.convert("RGB")
                if hashlib.sha256(f"{rgb.size}:".encode() + rgb.tobytes()).hexdigest() != row["pixel_sha256"]:
                    raise ValueError(f"source decoded pixels changed: {pid}")
                fitted = ImageOps.contain(rgb, (canvas, canvas), Image.Resampling.BICUBIC)
                offset = ((canvas - fitted.width) // 2, (canvas - fitted.height) // 2)
                result = Image.new("RGB", (canvas, canvas), "white")
                result.paste(fitted, offset)
                result.save(output / relative)
                transform = {"source_size": list(rgb.size), "resized_size": list(fitted.size),
                             "offset_xy": list(offset), "canvas": [canvas, canvas], "interpolation": "bicubic"}
            image_hashes[relative.removeprefix("images/")] = sha(output / relative)
            item = {"item_id": f"iconqa_{pid}", "source": "iconqa", "dataset_version": "iconqa_probability_mcq4_v1",
                    "split": SPLITS[split], "scene_id": row["pixel_sha256"],
                    "counterfactual_pair_id": pair_id, "pair_member": member,
                    "pair_provenance": "natural_matched_not_controlled_counterfactual",
                    "image": relative, "question": d["question"], "option_contents": CHOICES.copy(),
                    "semantic_answer": d["choices"][d["answer"]], "answer_type": "text",
                    "question_family": f"{d['grade']}_{d['label']}", "program": [], "program_depth": None,
                    "source_id": pid, "source_split": split, "skills": row["skills"],
                    "source_choices": d["choices"], "source_answer_index": d["answer"],
                    "source_image_sha256": row["byte_sha256"], "source_pixel_sha256": row["pixel_sha256"],
                    "source_metadata_sha256": sha(folder / "data.json"), "geometry": transform,
                    "candidate_rank": rank, "review_status": "pending_visual_truth_and_geometry_review"}
            items.append(item)
            review.append({k: item[k] for k in ("item_id", "counterfactual_pair_id", "image", "semantic_answer", "source_id", "split", "review_status")})
    manifest = expand(output, items, seed=seed)
    dump(output / "review_required.json", review)
    provenance = {"schema": "iconqa.probability.build.v1", "seed": seed, "candidate_pairs_by_source_split": dict(counts),
                  "source_audit_sha256": {"inventory.jsonl": sha(audit / "inventory.jsonl"),
                                          **{str(p.relative_to(audit)).replace("\\", "/"): sha(p) for p in paths.values()}},
                  "builder_sha256": sha(__file__), "dataset_hash": manifest["dataset_hash"],
                  "visual_review": "pending", "model_eligibility": "not_evaluated", "scientific_readiness": False,
                  "independent_unit": "disjoint_source_pixel_pair", "natural_pairs_are_not_controlled_edits": True,
                  "preprocessing": {"canvas": canvas, "padding": "white", "resampling": "bicubic", "crop": False}}
    dump(output / "build_provenance.json", provenance)
    release = {"dataset_hash": manifest["dataset_hash"], "final_image_sha256": image_hashes,
               "build_provenance_sha256": sha(output / "build_provenance.json"), "errors": []}
    release["image_aware_release_hash"] = hashlib.sha256(json.dumps(
        {k: v for k, v in release.items() if k != "errors"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    dump(output / "release_manifest.json", release)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.audit, args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
