"""Build CLEVR-MCQ-4 in two passes around a Blender render step.

1. ``plan``: read base CLEVR scenes, generate balanced questions, search for
   counterfactual edits, assign scene-disjoint splits, and write
   ``render_jobs.json`` for the counterfactual pairs.
2. Render the jobs with ``blender_render.py`` inside Blender.
3. ``finalize``: attach rendered images/scenes, expand variants and contrasts,
   validate, and freeze a manifest with content hashes.

Example::

    python -m vlm_mcqa.clevr_mcq4.build plan \
        --scenes /data/clevr_base/CLEVR_scenes.json \
        --images-dir /data/clevr_base/images \
        --config configs/core7/clevr_mcq4_dataset.json \
        --out-dir /data/clevr_mcq4_v1
    blender --background --python src/vlm_mcqa/clevr_mcq4/blender_render.py -- \
        --jobs /data/clevr_mcq4_v1/render_jobs.json --clevr-root /opt/clevr-dataset-gen/image_generation
    python -m vlm_mcqa.clevr_mcq4.build finalize --out-dir /data/clevr_mcq4_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from . import questions as Q
from .scene import apply_edit, candidate_edits, load_scenes
from .variants import (
    REQUIRED_SCHEMES,
    assign_splits,
    build_pair_contrasts,
    build_random_pair_contrasts,
    build_variants,
    validate,
    write_jsonl,
)

DATASET_VERSION = "clevr_mcq4_v1"
PAIR_SPLITS = ("discovery", "confirmation")


def _load_config(path: Path | None) -> dict[str, Any]:
    default = Path(__file__).resolve().parents[3] / "configs/core7/clevr_mcq4_dataset.json"
    return json.loads((path or default).read_text())


def _candidates(scene, family: str, cfg: Mapping[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    answer_type = Q.FAMILY_ANSWER_TYPE[family]
    out = []
    for params, text in Q.PROPOSERS[family](scene, rng):
        answer = Q.evaluate(family, scene, params)
        if answer is None:
            continue
        prog = Q.program(family, params)
        depth = Q.program_depth(prog)
        if not cfg["min_depth"] <= depth <= cfg["max_depth"]:
            continue
        options = Q.build_options(answer_type, answer, scene, rng)
        if options is None:
            continue
        out.append({
            "question_family": family, "answer_type": answer_type, "params": params,
            "question": text, "semantic_answer": answer, "option_contents": options,
            "program": prog, "program_depth": depth,
        })
    return out


def find_counterfactual(scene, cand: Mapping[str, Any], rng: random.Random) -> dict[str, Any] | None:
    """Single edit that changes the answer to another displayed option."""
    family = cand["question_family"]
    relation = family == "query_relation"
    valid = []
    for edit in candidate_edits(scene, allow_swaps=relation):
        if relation != (edit["kind"] == "swap"):
            continue
        edited = apply_edit(scene, edit)
        answer = Q.evaluate(family, edited, cand["params"])
        if answer is not None and answer != cand["semantic_answer"] and answer in cand["option_contents"]:
            valid.append((edit, edited, answer))
    if not valid:
        return None
    edit, edited, answer = rng.choice(valid)
    return {"edit": edit, "scene": edited, "semantic_answer": answer}


def _choose(scene, family, cfg, rng, counts_by_answer: Counter, need_pair: bool):
    """Pick a candidate, preferring the least-used answer so each family's prior is flat."""
    cands = _candidates(scene, family, cfg, rng)
    rng.shuffle(cands)
    cands.sort(key=lambda c: counts_by_answer[c["semantic_answer"]])
    if not need_pair:
        return cands[0] if cands else None
    for cand in cands[: cfg["max_counterfactual_tries"]]:
        cf = find_counterfactual(scene, cand, rng)
        if cf is not None:
            return {**cand, "cf": cf}
    return None


def _relevant(scene, cand) -> dict[str, Any]:
    objects = [i for i in Q.relevant_objects(cand["question_family"], scene, cand["params"]) if i is not None]
    return {
        "relevant_objects": objects,
        "relevant_pixel_coords": [scene["objects"][i].get("pixel_coords") for i in objects],
        "relevant_bboxes": [scene["objects"][i].get("bbox") for i in objects],
    }


def plan(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    rng = random.Random(cfg["seed"])
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    scenes = [s for s in load_scenes(Path(args.scenes))
              if cfg["min_objects"] <= len(s["objects"]) <= cfg["max_objects"]]
    rng.shuffle(scenes)
    families = list(cfg["family_weights"])
    weights = cfg["family_weights"]
    n_pairs = sum(cfg["split_sizes"][s] for s in PAIR_SPLITS)
    single_splits = {k: v for k, v in cfg["split_sizes"].items() if k not in PAIR_SPLITS}
    n_single = sum(single_splits.values())

    fam_counts: Counter = Counter()
    ans_counts: dict[str, Counter] = defaultdict(Counter)
    pairs, singles, skipped = [], [], Counter()

    def families_by_deficit() -> list[str]:
        total = sum(fam_counts.values()) + 1
        deficit = {f: weights[f] * total - fam_counts[f] for f in families}
        return sorted(families, key=lambda f: -deficit[f])

    for scene in scenes:
        want_pair = len(pairs) < n_pairs
        if not want_pair and len(singles) >= n_single:
            break
        chosen = None
        for family in families_by_deficit():
            chosen = _choose(scene, family, cfg, rng, ans_counts[family], want_pair)
            if chosen is not None:
                break
        if chosen is None:
            skipped["no_candidate"] += 1
            continue
        fam_counts[chosen["question_family"]] += 1
        ans_counts[chosen["question_family"]][chosen["semantic_answer"]] += 1
        (pairs if want_pair else singles).append((scene, chosen))

    if len(pairs) < n_pairs or len(singles) < n_single:
        raise SystemExit(f"not enough scenes: pairs {len(pairs)}/{n_pairs}, singles {len(singles)}/{n_single}")

    pair_split = assign_splits([s["scene_id"] for s, _ in pairs],
                               {k: cfg["split_sizes"][k] for k in PAIR_SPLITS}, rng)
    single_split = assign_splits([s["scene_id"] for s, _ in singles], single_splits, rng)

    items, jobs = [], []
    version = cfg.get("dataset_version", DATASET_VERSION)
    images_dir = Path(args.images_dir)
    for scene, cand in singles:
        image_name = scene["image_filename"]
        shutil.copy2(images_dir / image_name, out_dir / "images" / image_name)
        items.append(_item(scene, cand, split=single_split[scene["scene_id"]], version=version,
                           image=f"images/{image_name}", pair_id=None, member=None))
    for scene, cand in pairs:
        pair_id = f"pair_{scene['scene_id']}"
        split = pair_split[scene["scene_id"]]
        cf = cand.pop("cf")
        for member, member_scene, answer in (("x", scene, cand["semantic_answer"]),
                                             ("y", cf["scene"], cf["semantic_answer"])):
            image = f"images/{pair_id}_{member}.png"
            member_cand = {**cand, "semantic_answer": answer}
            item = _item(member_scene, member_cand, split=split, version=version, image=image,
                         pair_id=pair_id, member=member)
            item["counterfactual_edit"] = cf["edit"] if member == "y" else None
            items.append(item)
            jobs.append({
                "job_id": f"{pair_id}_{member}",
                "output_image": str((out_dir / image).resolve()),
                "output_scene": str((out_dir / "rendered_scenes" / f"{pair_id}_{member}.json").resolve()),
                "scene": member_scene,
            })

    (out_dir / "rendered_scenes").mkdir(exist_ok=True)
    write_jsonl(out_dir / "planned_items.jsonl", items)
    (out_dir / "render_jobs.json").write_text(json.dumps(jobs, indent=1))
    (out_dir / "dataset_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    print(json.dumps({"singles": len(singles), "pairs": len(pairs), "render_jobs": len(jobs),
                      "families": dict(fam_counts), "skipped": dict(skipped)}, indent=2))


def _item(scene, cand, *, split, version, image, pair_id, member) -> dict[str, Any]:
    suffix = f"_{member}" if member else ""
    return {
        "item_id": f"clevr_{scene['scene_id']}{suffix}",
        "source": "clevr",
        "dataset_version": version,
        "split": split,
        "scene_id": scene["scene_id"],
        "counterfactual_pair_id": pair_id,
        "pair_member": member,
        "image": image,
        "question": cand["question"],
        "option_contents": cand["option_contents"],
        "semantic_answer": cand["semantic_answer"],
        "answer_type": cand["answer_type"],
        "question_family": cand["question_family"],
        "question_params": cand["params"],
        "program": cand["program"],
        "program_depth": cand["program_depth"],
        **_relevant(scene, cand),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expand(out_dir: Path, items: list[dict[str, Any]], *, seed: int, schemes=REQUIRED_SCHEMES,
           optional_schemes=()) -> dict[str, Any]:
    """Write variants, contrasts, validation report and manifest for ``items``."""
    rng = random.Random(seed)
    variants, contrasts = [], []
    pair_members: dict[str, dict[str, list]] = defaultdict(dict)
    for item in items:
        vs = build_variants(item, (*schemes, *optional_schemes), permuted_control=bool(item.get("pair_member")))
        variants.extend(vs)
        if item.get("pair_member"):
            pair_members[item["counterfactual_pair_id"]][item["pair_member"]] = vs
    by_split: dict[str, dict[str, list]] = defaultdict(dict)
    for pair_id, members in sorted(pair_members.items()):
        contrasts.extend(build_pair_contrasts(members["x"], members["y"], rng))
        split = members["x"][0]["split"]
        by_split[split][pair_id] = [*members["x"], *members["y"]]
    for split, pv in sorted(by_split.items()):
        contrasts.extend(build_random_pair_contrasts(pv, rng))
    report = validate(items, variants, contrasts)
    write_jsonl(out_dir / "items.jsonl", items)
    write_jsonl(out_dir / "variants.jsonl", variants)
    write_jsonl(out_dir / "contrasts.jsonl", contrasts)
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    files = ["items.jsonl", "variants.jsonl", "contrasts.jsonl"]
    manifest = {
        "dataset_version": items[0].get("dataset_version", "unknown") if items else "empty",
        "seed": seed,
        "counts": report,
        "sha256": {name: _sha256(out_dir / name) for name in files},
    }
    manifest["dataset_hash"] = hashlib.sha256(
        json.dumps(manifest["sha256"], sort_keys=True).encode()).hexdigest()[:16]
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def finalize(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    cfg = json.loads((out_dir / "dataset_config.json").read_text())
    from .variants import read_jsonl
    items = read_jsonl(out_dir / "planned_items.jsonl")
    missing = []
    for item in items:
        if not item.get("pair_member"):
            continue
        scene_path = out_dir / "rendered_scenes" / Path(item["image"]).with_suffix(".json").name
        if not (out_dir / item["image"]).is_file() or not scene_path.is_file():
            missing.append(item["image"])
            continue
        rendered = json.loads(scene_path.read_text())
        for key in ("relevant_pixel_coords", "relevant_bboxes"):
            attr = "pixel_coords" if key == "relevant_pixel_coords" else "bbox"
            item[key] = [rendered["objects"][i].get(attr) for i in item["relevant_objects"]]
    if missing:
        raise SystemExit(f"{len(missing)} counterfactual renders missing, e.g. {missing[:3]}")
    manifest = expand(out_dir, items, seed=cfg["seed"],
                      optional_schemes=tuple(cfg.get("optional_schemes", ())))
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--scenes", required=True, help="CLEVR scenes JSON file or directory")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--out-dir", required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    {"plan": plan, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
