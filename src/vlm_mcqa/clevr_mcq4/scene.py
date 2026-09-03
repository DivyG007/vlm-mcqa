"""CLEVR scene-graph utilities: loading, filtering, relations and descriptions.

Scenes follow the official CLEVR JSON layout produced by
``image_generation/render_images.py``. Relations are recomputed from
``3d_coords`` and the camera ``directions`` so that edited scenes (which never
move objects) stay consistent with the rendered images.
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

COLORS = ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow")
SHAPES = ("cube", "sphere", "cylinder")
MATERIALS = ("rubber", "metal")
SIZES = ("large", "small")
ATTRIBUTE_VALUES: dict[str, tuple[str, ...]] = {
    "size": SIZES,
    "color": COLORS,
    "material": MATERIALS,
    "shape": SHAPES,
}
# Order used when writing a noun phrase ("the small red metal cube").
PHRASE_ORDER = ("size", "color", "material", "shape")
RELATIONS = ("left", "right", "front", "behind")
RELATION_PHRASES = {
    "left": "to the left of",
    "right": "to the right of",
    "front": "in front of",
    "behind": "behind",
}
COUNT_WORDS = (
    "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
)
SHAPE_PLURALS = {"cube": "cubes", "sphere": "spheres", "cylinder": "cylinders"}
# Minimum ground-plane separation (CLEVR units) for a relation to count.
RELATION_EPSILON = 0.2

Spec = Mapping[str, str]


def load_scenes(path: Path) -> list[dict[str, Any]]:
    """Load scenes from a CLEVR scenes file or a directory of per-image JSONs."""
    path = Path(path)
    if path.is_dir():
        raw = [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]
    else:
        data = json.loads(path.read_text())
        raw = data["scenes"] if isinstance(data, dict) else data
    return [normalize_scene(scene) for scene in raw]


def normalize_scene(scene: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(scene))
    stem = Path(out.get("image_filename", f"scene_{out.get('image_index', 0)}")).stem
    out.setdefault("scene_id", stem)
    for obj in out["objects"]:
        for attr, values in ATTRIBUTE_VALUES.items():
            if obj[attr] not in values:
                raise ValueError(f"{out['scene_id']}: unknown {attr} {obj[attr]!r}")
    if "directions" in out:
        out["relationships"] = compute_relationships(out)
    elif "relationships" not in out:
        raise ValueError(f"{out['scene_id']}: needs directions or relationships")
    return out


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def ground_offsets(scene: Mapping[str, Any], a: int, b: int) -> tuple[float, float]:
    """Return (rightward, frontward) offset of object ``a`` relative to ``b``."""
    ca = scene["objects"][a]["3d_coords"]
    cb = scene["objects"][b]["3d_coords"]
    delta = [float(x) - float(y) for x, y in zip(ca, cb)]
    directions = scene["directions"]
    return _dot(delta, directions["right"]), _dot(delta, directions["front"])


def compute_relationships(scene: Mapping[str, Any]) -> dict[str, list[list[int]]]:
    """``rel[name][i]`` = objects that lie in direction ``name`` from object ``i``.

    Matches CLEVR's convention in ``render_images.compute_all_relationships``.
    """
    n = len(scene["objects"])
    out: dict[str, list[list[int]]] = {}
    for name in RELATIONS:
        direction = scene["directions"][name]
        rows = []
        for i in range(n):
            ci = scene["objects"][i]["3d_coords"]
            related = []
            for j in range(n):
                if i == j:
                    continue
                cj = scene["objects"][j]["3d_coords"]
                diff = [float(x) - float(y) for x, y in zip(cj, ci)]
                if _dot(diff, direction) > RELATION_EPSILON:
                    related.append(j)
            rows.append(related)
        out[name] = rows
    return out


def matches(obj: Mapping[str, Any], spec: Spec) -> bool:
    return all(obj[attr] == value for attr, value in spec.items())


def filter_objects(
    scene: Mapping[str, Any], spec: Spec, candidates: Iterable[int] | None = None
) -> list[int]:
    pool = range(len(scene["objects"])) if candidates is None else candidates
    return [i for i in pool if matches(scene["objects"][i], spec)]


def unique(scene: Mapping[str, Any], spec: Spec, candidates: Iterable[int] | None = None) -> int | None:
    found = filter_objects(scene, spec, candidates)
    return found[0] if len(found) == 1 else None


def related(scene: Mapping[str, Any], index: int, relation: str) -> list[int]:
    return list(scene["relationships"][relation][index])


def minimal_unique_spec(
    scene: Mapping[str, Any],
    index: int,
    *,
    candidates: Sequence[int] | None = None,
    exclude: Sequence[str] = (),
    allow_empty: bool = False,
) -> dict[str, str] | None:
    """Smallest attribute subset that picks out ``index`` among ``candidates``.

    Subsets containing ``shape`` are preferred so phrases read naturally.
    """
    obj = scene["objects"][index]
    attrs = [a for a in PHRASE_ORDER if a not in exclude]
    pool = list(range(len(scene["objects"]))) if candidates is None else list(candidates)
    sizes = range(0 if allow_empty else 1, len(attrs) + 1)
    for size in sizes:
        subsets = sorted(
            itertools.combinations(attrs, size), key=lambda s: ("shape" not in s, s)
        )
        for subset in subsets:
            spec = {a: obj[a] for a in subset}
            if filter_objects(scene, spec, pool) == [index]:
                return spec
    return None


def noun_phrase(spec: Spec, *, plural: bool = False, other: bool = False) -> str:
    """Render a spec as a noun phrase without an article."""
    words = [spec[a] for a in ("size", "color", "material") if a in spec]
    if "shape" in spec:
        noun = SHAPE_PLURALS[spec["shape"]] if plural else spec["shape"]
    else:
        noun = "objects" if plural else "object"
    if other:
        words.insert(0, "other")
    return " ".join([*words, noun])


def the(spec: Spec) -> str:
    return "the " + noun_phrase(spec)


def predicate(attr: str, value: str) -> str:
    """Predicate form used in logic questions ("red", "metal", "a cube")."""
    return f"a {value}" if attr == "shape" else value


def apply_edit(scene: Mapping[str, Any], edit: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``scene`` with an attribute edit applied.

    ``{"kind": "set", "object": i, "attribute": a, "value": v}`` changes one
    attribute. ``{"kind": "swap", "objects": [i, j]}`` exchanges all four
    attributes of two objects while positions and rotations stay fixed.
    """
    out = copy.deepcopy(dict(scene))
    objects = out["objects"]
    if edit["kind"] == "set":
        objects[edit["object"]][edit["attribute"]] = edit["value"]
    elif edit["kind"] == "swap":
        i, j = edit["objects"]
        for attr in PHRASE_ORDER:
            objects[i][attr], objects[j][attr] = objects[j][attr], objects[i][attr]
    else:
        raise ValueError(f"unknown edit kind {edit['kind']!r}")
    return out


def candidate_edits(scene: Mapping[str, Any], *, allow_swaps: bool) -> list[dict[str, Any]]:
    """All single-attribute edits (colour, material, shape) and optional swaps.

    Size edits are excluded because they move the object's centre vertically and
    change occlusion.
    """
    edits: list[dict[str, Any]] = []
    for i, obj in enumerate(scene["objects"]):
        for attr in ("color", "material", "shape"):
            for value in ATTRIBUTE_VALUES[attr]:
                if value != obj[attr]:
                    edits.append({"kind": "set", "object": i, "attribute": attr, "value": value})
    if allow_swaps:
        n = len(scene["objects"])
        for i in range(n):
            for j in range(i + 1, n):
                if any(scene["objects"][i][a] != scene["objects"][j][a] for a in PHRASE_ORDER):
                    edits.append({"kind": "swap", "objects": [i, j]})
    return edits
