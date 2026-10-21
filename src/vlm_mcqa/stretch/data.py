"""Read immutable releases and derive exact-fixed-text image contrasts.

Standard library only. Tar members are read in memory, never extracted. This
prepares structural candidates, not behaviorally eligible or untouched cases.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Iterable, Mapping

from . import CONTRAST_TYPE


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unique(rows: list[dict], key: str) -> None:
    ids = [r[key] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {key}")


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Return lists ``items/variants/contrasts``, manifest and byte provenance.

    Archive member selection requires one unambiguous sibling metadata set;
    images stay in their original archive. Hashes cover metadata bytes, not a
    fresh image audit. The archive's declared release hashes are retained.
    """
    path = Path(path)
    payload: dict[str, bytes] = {}
    members: dict[str, str] = {}
    required = ("items.jsonl", "variants.jsonl")
    optional = ("contrasts.jsonl", "dataset_manifest.json", "release_manifest.json")
    if path.is_dir():
        for name in (*required, *optional):
            p = path / name
            if p.is_file():
                payload[name], members[name] = p.read_bytes(), name
    else:
        with tarfile.open(path, "r:*") as archive:
            regular = [m for m in archive.getmembers() if m.isfile()]
            if len({m.name for m in regular}) != len(regular):
                raise ValueError("duplicate archive member names")
            files = {m.name: m for m in regular}
            roots = [str(PurePosixPath(name).parent) for name in files
                     if PurePosixPath(name).name == "items.jsonl"
                     and str(PurePosixPath(name).with_name("variants.jsonl")) in files]
            if len(roots) != 1:
                raise ValueError("archive must contain one unambiguous items/variants metadata set")
            for name in (*required, *optional):
                member = str(PurePosixPath(roots[0]) / name)
                if member in files:
                    stream = archive.extractfile(files[member])
                    if stream is None:
                        raise ValueError(f"unreadable archive member {member}")
                    with stream:
                        payload[name], members[name] = stream.read(), member
    missing = set(required) - payload.keys()
    if missing:
        raise ValueError(f"missing release files: {sorted(missing)}")
    result: dict[str, Any] = {}
    for key in ("items", "variants", "contrasts"):
        result[key] = [json.loads(line) for line in payload.get(key + ".jsonl", b"").decode("utf-8-sig").splitlines()
                       if line.strip()]
    _unique(result["items"], "item_id")
    _unique(result["variants"], "variant_id")
    result["manifest"] = json.loads(payload.get("dataset_manifest.json", b"{}"))
    result["release_manifest"] = json.loads(payload.get("release_manifest.json", b"{}"))
    result["provenance"] = {
        "source_path": str(path.resolve()), "source_kind": "directory" if path.is_dir() else "archive",
        "metadata_sha256": {name: digest(data) for name, data in payload.items()},
        "metadata_members": members, "images_rehashed": False,
        "eligibility": "structural_only", "historical_exposure": "not_audited",
    }
    return result


def validate_fixed_text_contrast(row: Mapping[str, Any], variants: Mapping[str, Mapping]) -> None:
    donor, recipient = (variants[row[k]] for k in ("donor_variant_id", "recipient_variant_id"))
    if row["contrast_type"] != CONTRAST_TYPE:
        raise ValueError("not a fixed-text semantic-image contrast")
    for key in ("prompt", "question", "labels", "option_contents", "label_scheme", "split"):
        if donor[key] != recipient[key]:
            raise ValueError(f"{row['contrast_id']}: differing {key}")
    pair = donor.get("counterfactual_pair_id")
    if not pair or pair != recipient.get("counterfactual_pair_id") or row["pair_id"] != pair:
        raise ValueError("different or missing semantic pair")
    if row["split"] != donor["split"]:
        raise ValueError("contrast split mismatch")
    if donor["image"] == recipient["image"] or donor["item_id"] == recipient["item_id"]:
        raise ValueError("image contrast requires two different image/items")
    if donor["semantic_answer"] == recipient["semantic_answer"]:
        raise ValueError("image contrast requires changed semantic answer")
    for side, variant in (("donor", donor), ("recipient", recipient)):
        labels, contents, index = variant["labels"], variant["option_contents"], variant["correct_index"]
        if len(labels) != 4 or len(set(labels)) != 4 or len(contents) != 4 or len(set(contents)) != 4:
            raise ValueError("requires four distinct options and labels")
        if not 0 <= index < 4 or contents[index] != variant["semantic_answer"]:
            raise ValueError("semantic answer/index mismatch")
        if labels[index] != variant["correct_symbol"] or row[side + "_symbol"] != labels[index]:
            raise ValueError("symbol/index mismatch")
        if row[side + "_content"] != variant["semantic_answer"]:
            raise ValueError("contrast semantic answer mismatch")
    if donor["correct_symbol"] == recipient["correct_symbol"]:
        raise ValueError("donor and recipient readout labels must differ")
    if row["prompt_sha256"] != digest(donor["prompt"].encode("utf-8")):
        raise ValueError("prompt hash mismatch")


def build_fixed_text_contrasts(dataset: Mapping[str, Any], *, schemes: Iterable[str] = ("letters",),
                               splits: Iterable[str] = ("discovery", "confirmation")) -> list[dict]:
    """Join existing position variants by exact prompt; emit both directions.

    All layouts are retained. Sampling/correctness/exposure exclusions must be
    decided in a separate frozen protocol, never by choosing favorable rows.
    """
    schemes, splits = set(schemes), set(splits)
    _unique(dataset["items"], "item_id")
    _unique(dataset["variants"], "variant_id")
    variants = {v["variant_id"]: v for v in dataset["variants"]}
    by_pair: dict[str, list[dict]] = defaultdict(list)
    scene_splits: dict[str, set[str]] = defaultdict(set)
    for item in dataset["items"]:
        scene_splits[item["scene_id"]].add(item["split"])
    if any(len(s) != 1 for s in scene_splits.values()):
        raise ValueError("scene crosses release splits")
    items = {i["item_id"]: i for i in dataset["items"]}
    for variant in variants.values():
        item = items[variant["item_id"]]
        for key in ("scene_id", "split", "image", "semantic_answer", "counterfactual_pair_id"):
            if variant.get(key) != item.get(key):
                raise ValueError(f"{variant['variant_id']}: item/variant {key} mismatch")
        if variant.get("counterfactual_pair_id"):
            by_pair[variant["counterfactual_pair_id"]].append(variant)
    rows = []
    for pair, pool in sorted(by_pair.items()):
        if len({v["split"] for v in pool}) != 1:
            raise ValueError(f"pair {pair} crosses splits")
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for v in pool:
            if v["split"] in splits and v["label_scheme"] in schemes and v["variant_kind"] == "position":
                groups[(v["label_scheme"], v["prompt"])].append(v)
        for (_scheme, prompt), matches in sorted(groups.items()):
            if len(matches) != 2 or len({v["item_id"] for v in matches}) != 2:
                raise ValueError(f"pair {pair}: exact-text layout does not have exactly two image members")
            first, second = sorted(matches, key=lambda v: v["variant_id"])
            for donor, recipient in ((first, second), (second, first)):
                row = {
                    "contrast_id": f"{CONTRAST_TYPE}:{donor['variant_id']}->{recipient['variant_id']}",
                    "contrast_type": CONTRAST_TYPE, "pair_id": pair, "split": recipient["split"],
                    "donor_variant_id": donor["variant_id"], "recipient_variant_id": recipient["variant_id"],
                    "donor_symbol": donor["correct_symbol"], "recipient_symbol": recipient["correct_symbol"],
                    "donor_content": donor["semantic_answer"], "recipient_content": recipient["semantic_answer"],
                    "label_scheme": recipient["label_scheme"], "prompt_sha256": digest(prompt.encode("utf-8")),
                    "layout_id": digest(prompt.encode("utf-8")), "sample_unit": pair,
                    "donor_scene_id": donor["scene_id"], "recipient_scene_id": recipient["scene_id"],
                    "readout": "donor_label_minus_recipient_label", "eligibility": "pending_baseline_inference",
                }
                validate_fixed_text_contrast(row, variants)
                rows.append(row)
    validate_bidirectional(rows)
    return rows


def validate_bidirectional(rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    _unique(rows, "contrast_id")
    keys = {(r["pair_id"], r["donor_variant_id"], r["recipient_variant_id"]) for r in rows}
    if any((p, r, d) not in keys for p, d, r in keys):
        raise ValueError("missing reverse image contrast")


def validate_output_path(path: str | Path, inputs: Iterable[str | Path]) -> Path:
    """Reject existing outputs and input/output containment in either direction."""
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    for source in inputs:
        source = Path(source).resolve()
        if target == source or target in source.parents or source in target.parents:
            raise ValueError(f"output overlaps immutable input: {source}")
    return target


def write_new_json(path: str | Path, value: Any, *, inputs: Iterable[str | Path] = ()) -> None:
    """Exclusive creation; never overwrite an existing result or manifest."""
    path = validate_output_path(path, inputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
