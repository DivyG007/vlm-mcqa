"""Question families for CLEVR-MCQ-4.

Each family is a pair of functions:

- ``propose_<family>(scene, rng)`` returns candidate questions as
  ``(params, text)`` where ``params`` contains only attribute specs (never
  object indices), so the same question can be re-executed on an edited scene;
- ``evaluate(family, scene, params)`` executes the functional program and
  returns the semantic answer, or ``None`` when the question is invalid for the
  scene (non-unique referent, ambiguous relation, tie, out-of-range count).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Callable, Mapping

from .scene import (
    COLORS,
    COUNT_WORDS,
    RELATION_PHRASES,
    RELATIONS,
    filter_objects,
    ground_offsets,
    minimal_unique_spec,
    noun_phrase,
    predicate,
    related,
    the,
    unique,
)

# Relation questions need one clearly dominant ground-plane axis.
RELATION_MIN_OFFSET = 1.0
RELATION_DOMINANCE = 2.0

FAMILY_ANSWER_TYPE = {
    "query_color_relate": "color",
    "query_relation": "relation",
    "count": "count",
    "count_relate": "count",
    "count_compare": "color",
    "same_attr_query": "color",
    "same_attr_count": "count",
    "logic_and": "count",
    "logic_or": "count",
}


def _count_word(value: int) -> str | None:
    return COUNT_WORDS[value] if 0 <= value < len(COUNT_WORDS) else None


# --------------------------------------------------------------------------- evaluation


def evaluate(family: str, scene: Mapping[str, Any], p: Mapping[str, Any]) -> str | None:
    return EVALUATORS[family](scene, p)


def _eval_query_color_relate(scene, p):
    ref = unique(scene, p["ref"])
    if ref is None:
        return None
    target = unique(scene, p["target"], related(scene, ref, p["relation"]))
    return None if target is None else scene["objects"][target]["color"]


def _dominant_relation(scene, a: int, b: int) -> str | None:
    right, front = ground_offsets(scene, a, b)
    big, small = (abs(right), abs(front)) if abs(right) >= abs(front) else (abs(front), abs(right))
    if big < RELATION_MIN_OFFSET or big < RELATION_DOMINANCE * small:
        return None
    if abs(right) >= abs(front):
        return "right" if right > 0 else "left"
    return "front" if front > 0 else "behind"


def _eval_query_relation(scene, p):
    a, b = unique(scene, p["a"]), unique(scene, p["b"])
    if a is None or b is None or a == b:
        return None
    return _dominant_relation(scene, a, b)


def _eval_count(scene, p):
    return _count_word(len(filter_objects(scene, p["spec"])))


def _eval_count_relate(scene, p):
    ref = unique(scene, p["ref"])
    if ref is None:
        return None
    return _count_word(len(filter_objects(scene, p["spec"], related(scene, ref, p["relation"]))))


def _eval_count_compare(scene, p):
    pool = filter_objects(scene, p["group"])
    counts = Counter(scene["objects"][i]["color"] for i in pool)
    if not counts:
        return None
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 1:
        return None
    return ranked[0][0]


def _same_attr_pool(scene, p) -> list[int] | None:
    ref = unique(scene, p["ref"])
    if ref is None:
        return None
    value = scene["objects"][ref][p["attribute"]]
    return [
        i for i, obj in enumerate(scene["objects"]) if i != ref and obj[p["attribute"]] == value
    ]


def _eval_same_attr_query(scene, p):
    pool = _same_attr_pool(scene, p)
    if pool is None:
        return None
    target = unique(scene, p["target"], pool)
    return None if target is None else scene["objects"][target]["color"]


def _eval_same_attr_count(scene, p):
    pool = _same_attr_pool(scene, p)
    return None if pool is None else _count_word(len(pool))


def _eval_logic(scene, p, combine: Callable[[set, set], set]):
    first = set(filter_objects(scene, {p["attr1"]: p["value1"]}))
    second = set(filter_objects(scene, {p["attr2"]: p["value2"]}))
    return _count_word(len(combine(first, second)))


EVALUATORS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], str | None]] = {
    "query_color_relate": _eval_query_color_relate,
    "query_relation": _eval_query_relation,
    "count": _eval_count,
    "count_relate": _eval_count_relate,
    "count_compare": _eval_count_compare,
    "same_attr_query": _eval_same_attr_query,
    "same_attr_count": _eval_same_attr_count,
    "logic_and": lambda s, p: _eval_logic(s, p, set.__and__),
    "logic_or": lambda s, p: _eval_logic(s, p, set.__or__),
}


# --------------------------------------------------------------------------- programs


def program(family: str, p: Mapping[str, Any]) -> list[dict[str, Any]]:
    """CLEVR-style functional program; ``unique`` steps are checks, not depth."""
    f = lambda spec: {"op": "filter", "spec": dict(spec)}  # noqa: E731
    u = {"op": "unique"}
    if family == "query_color_relate":
        return [f(p["ref"]), u, {"op": "relate", "relation": p["relation"]},
                f(p["target"]), u, {"op": "query", "attribute": "color"}]
    if family == "query_relation":
        return [f(p["a"]), u, f(p["b"]), u, {"op": "dominant_relation"}]
    if family == "count":
        return [f(p["spec"]), {"op": "count"}]
    if family == "count_relate":
        return [f(p["ref"]), u, {"op": "relate", "relation": p["relation"]},
                f(p["spec"]), {"op": "count"}]
    if family == "count_compare":
        return [f(p["group"]), {"op": "group_count", "attribute": "color"}, {"op": "argmax"}]
    if family == "same_attr_query":
        return [f(p["ref"]), u, {"op": "same", "attribute": p["attribute"]},
                f(p["target"]), u, {"op": "query", "attribute": "color"}]
    if family == "same_attr_count":
        return [f(p["ref"]), u, {"op": "same", "attribute": p["attribute"]}, {"op": "count"}]
    if family in ("logic_and", "logic_or"):
        return [f({p["attr1"]: p["value1"]}), f({p["attr2"]: p["value2"]}),
                {"op": "intersect" if family == "logic_and" else "union"}, {"op": "count"}]
    raise ValueError(family)


def program_depth(prog: list[dict[str, Any]]) -> int:
    return sum(1 for step in prog if step["op"] != "unique")


def relevant_objects(family: str, scene: Mapping[str, Any], p: Mapping[str, Any]) -> list[int]:
    """Objects whose attributes the answer depends on (for region records)."""
    if family == "query_color_relate":
        ref = unique(scene, p["ref"])
        return [ref, unique(scene, p["target"], related(scene, ref, p["relation"]))]
    if family == "query_relation":
        return [unique(scene, p["a"]), unique(scene, p["b"])]
    if family == "count":
        return filter_objects(scene, p["spec"])
    if family == "count_relate":
        ref = unique(scene, p["ref"])
        return [ref, *filter_objects(scene, p["spec"], related(scene, ref, p["relation"]))]
    if family == "count_compare":
        return filter_objects(scene, p["group"])
    if family in ("same_attr_query", "same_attr_count"):
        ref = unique(scene, p["ref"])
        pool = _same_attr_pool(scene, p) or []
        if family == "same_attr_query":
            return [ref, unique(scene, p["target"], pool)]
        return [ref, *pool]
    first = set(filter_objects(scene, {p["attr1"]: p["value1"]}))
    second = set(filter_objects(scene, {p["attr2"]: p["value2"]}))
    return sorted(first | second)


# --------------------------------------------------------------------------- proposal


def _ref_specs(scene, rng: random.Random, limit: int = 4) -> list[tuple[int, dict[str, str]]]:
    indices = list(range(len(scene["objects"])))
    rng.shuffle(indices)
    out = []
    for i in indices:
        spec = minimal_unique_spec(scene, i)
        if spec is not None:
            out.append((i, spec))
        if len(out) >= limit:
            break
    return out


def propose_query_color_relate(scene, rng):
    out = []
    for ref, ref_spec in _ref_specs(scene, rng):
        for relation in rng.sample(RELATIONS, len(RELATIONS)):
            pool = related(scene, ref, relation)
            for target in pool:
                spec = minimal_unique_spec(
                    scene, target, candidates=pool, exclude=("color",), allow_empty=True
                )
                if spec is None:
                    continue
                p = {"ref": ref_spec, "relation": relation, "target": spec}
                text = (f"What color is the {noun_phrase(spec)} that is "
                        f"{RELATION_PHRASES[relation]} {the(ref_spec)}?")
                out.append((p, text))
    return out


def propose_query_relation(scene, rng):
    out = []
    refs = _ref_specs(scene, rng, limit=6)
    for (a, sa), (b, sb) in ((x, y) for x in refs for y in refs if x[0] != y[0]):
        p = {"a": sa, "b": sb}
        text = f"Relative to {the(sb)}, in which direction is {the(sa)}?"
        out.append((p, text))
    return out


def _count_specs(scene, rng):
    specs = []
    for attr in ("color", "shape", "material", "size"):
        for value in {obj[attr] for obj in scene["objects"]}:
            specs.append({attr: value})
    for obj in rng.sample(scene["objects"], min(3, len(scene["objects"]))):
        specs.append({"color": obj["color"], "shape": obj["shape"]})
        specs.append({"material": obj["material"], "shape": obj["shape"]})
        specs.append({"size": obj["size"], "color": obj["color"]})
    return specs


def propose_count(scene, rng):
    return [({"spec": spec}, f"How many {noun_phrase(spec, plural=True)} are there?")
            for spec in _count_specs(scene, rng)]


def propose_count_relate(scene, rng):
    out = []
    for _ref, ref_spec in _ref_specs(scene, rng, limit=3):
        for relation in RELATIONS:
            for spec in rng.sample(_count_specs(scene, rng), 3):
                p = {"ref": ref_spec, "relation": relation, "spec": spec}
                text = (f"How many {noun_phrase(spec, plural=True)} are "
                        f"{RELATION_PHRASES[relation]} {the(ref_spec)}?")
                out.append((p, text))
    return out


def propose_count_compare(scene, rng):
    groups: list[dict[str, str]] = [{}]
    for attr in ("shape", "material", "size"):
        groups.extend({attr: v} for v in {obj[attr] for obj in scene["objects"]})
    return [({"group": g}, f"Which color is the most common among the "
             f"{noun_phrase(g, plural=True)}?") for g in groups]


def propose_same_attr_query(scene, rng):
    out = []
    for ref, ref_spec in _ref_specs(scene, rng):
        for attribute in ("shape", "material", "size"):
            p0 = {"ref": ref_spec, "attribute": attribute}
            pool = _same_attr_pool(scene, p0) or []
            for target in pool:
                spec = minimal_unique_spec(
                    scene, target, candidates=pool, exclude=("color", attribute), allow_empty=True
                )
                if spec is None:
                    continue
                text = (f"What color is the {noun_phrase(spec, other=not spec)} that has the "
                        f"same {attribute} as {the(ref_spec)}?")
                out.append(({**p0, "target": spec}, text))
    return out


def propose_same_attr_count(scene, rng):
    out = []
    for _ref, ref_spec in _ref_specs(scene, rng):
        for attribute in ("color", "shape", "material", "size"):
            text = f"How many other objects have the same {attribute} as {the(ref_spec)}?"
            out.append(({"ref": ref_spec, "attribute": attribute}, text))
    return out


def _logic_pairs(scene):
    values = {a: sorted({o[a] for o in scene["objects"]}) for a in ("color", "shape", "material", "size")}
    attrs = list(values)
    for i, a1 in enumerate(attrs):
        for a2 in attrs[i + 1:]:
            for v1 in values[a1]:
                for v2 in values[a2]:
                    yield {"attr1": a1, "value1": v1, "attr2": a2, "value2": v2}


def propose_logic_and(scene, rng):
    return [(p, f"How many objects are both {predicate(p['attr1'], p['value1'])} and "
             f"{predicate(p['attr2'], p['value2'])}?") for p in _logic_pairs(scene)]


def propose_logic_or(scene, rng):
    return [(p, f"How many objects are either {predicate(p['attr1'], p['value1'])} or "
             f"{predicate(p['attr2'], p['value2'])}?") for p in _logic_pairs(scene)]


PROPOSERS = {
    "query_color_relate": propose_query_color_relate,
    "query_relation": propose_query_relation,
    "count": propose_count,
    "count_relate": propose_count_relate,
    "count_compare": propose_count_compare,
    "same_attr_query": propose_same_attr_query,
    "same_attr_count": propose_same_attr_count,
    "logic_and": propose_logic_and,
    "logic_or": propose_logic_or,
}


# --------------------------------------------------------------------------- options


def build_options(
    answer_type: str, answer: str, scene: Mapping[str, Any], rng: random.Random
) -> list[str] | None:
    """Four type-matched option contents in a random canonical order."""
    if answer_type == "relation":
        options = list(RELATIONS)
    elif answer_type == "color":
        present = sorted({o["color"] for o in scene["objects"]} - {answer})
        absent = [c for c in COLORS if c not in present and c != answer]
        rng.shuffle(present)
        rng.shuffle(absent)
        options = [answer, *(present + absent)[:3]]
    elif answer_type == "count":
        value = COUNT_WORDS.index(answer)
        low = max(0, min(value - rng.randrange(4), len(COUNT_WORDS) - 4))
        options = list(COUNT_WORDS[low:low + 4])
        if answer not in options:
            return None
    else:
        raise ValueError(answer_type)
    if len(set(options)) != 4:
        return None
    rng.shuffle(options)
    return options
