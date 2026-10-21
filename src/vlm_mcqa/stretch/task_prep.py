"""Pure optional-task preparation. No processor grid or model mapping is assumed.

Coordinates are half-open pixel rectangles. Mapping validity is geometric only;
callers must calibrate the supplied footprints against their actual processor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "vlm-mcqa-optional-task/v1"


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x0, self.y0, self.x1, self.y1)):
            raise ValueError("rectangle coordinates must be finite")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("rectangle must have positive area")

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def contains(self, other: Rect) -> bool:
        return (self.x0 <= other.x0 < other.x1 <= self.x1
                and self.y0 <= other.y0 < other.y1 <= self.y1)

    def intersection(self, other: Rect) -> Rect | None:
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        return Rect(x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


@dataclass(frozen=True)
class ViewTransform:
    """Caller-supplied source crop mapped to a destination tile (padding excluded).

Represent every crop/tile separately; letterboxing is expressed by destination
offsets. Rotation, nonlinear transforms and undocumented global tiles are not
inferred. Compose known crop/resize stages before supplying this mapping.
"""
    view_id: str
    source_crop: Rect
    destination: Rect

    def map_region(self, region: Rect) -> Rect | None:
        clipped = self.source_crop.intersection(region)
        if clipped is None:
            return None
        src, dst = self.source_crop, self.destination
        sx = (dst.x1 - dst.x0) / (src.x1 - src.x0)
        sy = (dst.y1 - dst.y0) / (src.y1 - src.y0)
        return Rect(dst.x0 + (clipped.x0 - src.x0) * sx,
                    dst.y0 + (clipped.y0 - src.y0) * sy,
                    dst.x0 + (clipped.x1 - src.x0) * sx,
                    dst.y0 + (clipped.y1 - src.y0) * sy)


@dataclass(frozen=True)
class TokenCell:
    token_index: int
    view_id: str
    footprint: Rect


def map_region(region: Rect, views: Sequence[ViewTransform],
               cells: Sequence[TokenCell], *, rule: str,
               min_fraction: float | None = None) -> tuple[int, ...]:
    """Select image token indices by explicit center or footprint-overlap rule.

Overlap fraction is intersection / token footprint area. Returned indices are
sequence indices supplied by caller, not inferred flattened patch offsets.
"""
    by_view = {v.view_id: v for v in views}
    if not views or len(by_view) != len(views) or any(not v.view_id for v in views):
        raise ValueError("nonempty unique view ids required")
    if rule not in {"center", "overlap"}:
        raise ValueError("explicit center or overlap rule required")
    if rule == "overlap" and (isinstance(min_fraction, bool)
                              or not isinstance(min_fraction, (int, float))
                              or not math.isfinite(min_fraction) or not 0 < min_fraction <= 1):
        raise ValueError("overlap rule requires fraction in (0, 1]")
    if rule == "center" and min_fraction is not None:
        raise ValueError("center rule does not accept overlap fraction")
    seen, selected = set(), []
    for cell in cells:
        if isinstance(cell.token_index, bool) or not isinstance(cell.token_index, int) or cell.token_index < 0:
            raise ValueError("token indices must be nonnegative integers")
        if cell.token_index in seen:
            raise ValueError("duplicate sequence token index")
        seen.add(cell.token_index)
        if cell.view_id not in by_view:
            raise ValueError("token refers to unknown view")
        view = by_view[cell.view_id]
        if not view.destination.contains(cell.footprint):
            raise ValueError("token footprint outside supplied destination")
        mapped = view.map_region(region)
        if mapped is None:
            continue
        box = cell.footprint
        if rule == "center":
            x, y = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
            keep = mapped.x0 <= x < mapped.x1 and mapped.y0 <= y < mapped.y1
        else:
            overlap = mapped.intersection(box)
            keep = overlap is not None and overlap.area / box.area >= min_fraction
        if keep:
            selected.append(cell.token_index)
    return tuple(sorted(selected))


def matched_token_controls(cells: Sequence[TokenCell], target_region: Rect,
                           views: Sequence[ViewTransform], *, count: int,
                           size: int, seed: str) -> tuple[tuple[int, ...], ...]:
    """Deterministic distinct token-count controls excluding ANY target overlap.

    These are potentially overlapping/correlated token sets, not independently
    random samples, contiguous regions or matched physical areas. A study needing
    contiguous or view-balanced controls must add that gate.
"""
    # Validate all supplied cells first; use geometric intersection rather than
    # a tiny arbitrary overlap threshold for exclusion.
    map_region(target_region, views, cells, rule="center")
    by_view = {v.view_id: v for v in views}
    forbidden = set()
    for cell in cells:
        mapped = by_view[cell.view_id].map_region(target_region)
        if mapped is not None and mapped.intersection(cell.footprint) is not None:
            forbidden.add(cell.token_index)
    pool = sorted(c.token_index for c in cells if c.token_index not in forbidden)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (count, size)) or len(pool) < size:
        raise ValueError("positive feasible control count/size required")
    if math.comb(len(pool), size) < count:
        raise ValueError("insufficient distinct controls")
    ranked = sorted(pool, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).digest())
    return tuple(tuple(sorted(group)) for group in itertools.islice(itertools.combinations(ranked, size), count))


def _identity(source: Mapping[str, Any], kind: str) -> dict[str, Any]:
    required = ("dataset_id", "source_sha256", "scene_id", "pair_id", "split")
    if any(not isinstance(source.get(k), str) or not source[k] for k in required):
        raise ValueError("explicit dataset/hash/scene/pair/split provenance required")
    if source["split"] not in {"discovery", "confirmation"}:
        raise ValueError("split must be discovery or confirmation")
    digest = source["source_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("source_sha256 must be lowercase SHA256")
    return {"schema": SCHEMA, "task_kind": kind, "source": dict(source),
            "sample_unit": "parent_scene_or_semantic_pair", "runtime_validated": False}


def _finish(record: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    record["derivative_id"] = record["task_kind"] + "-v1-" + hashlib.sha256(encoded).hexdigest()[:20]
    return record


def spatial_factorial(source: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate an explicit 2x2 question-guide x option-guide planned task.

    Region identities are annotations, never inferred from question strings.
    This validates the factorial bookkeeping, not causal isolation of wording.
    """
    record = _identity(source, "spatial-guidance")
    combos, ids = set(), set()
    image = None
    for row in cells:
        for key in ("condition_id", "question", "image", "question_region", "option_region"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"missing spatial field {key}")
        q, o = row.get("question_level"), row.get("option_level")
        if any(isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1) for v in (q, o)):
            raise ValueError("two explicit guidance levels required")
        combo = q, o
        if combo in combos or row["condition_id"] in ids:
            raise ValueError("duplicate spatial condition")
        combos.add(combo)
        ids.add(row["condition_id"])
        if image is not None and image != row["image"]:
            raise ValueError("spatial factorial must hold image fixed")
        image = row["image"]
        for key in ("scene_id", "pair_id", "split"):
            if row.get(key) != source[key]:
                raise ValueError("spatial cell provenance mismatch")
        options, labels = row.get("option_ids", []), row.get("labels", [])
        if len(options) != 4 or len(set(options)) != 4 or len(labels) != 4 or len(set(labels)) != 4:
            raise ValueError("four unique option ids and symbols required")
        if row.get("correct_option_id") not in options:
            raise ValueError("correct option absent")
    if combos != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("complete four-cell spatial factorial required")
    for key, level_key in (("question_region", "question_level"), ("question", "question_level"),
                           ("option_region", "option_level"), ("option_ids", "option_level")):
        for level in (0, 1):
            values = [json.dumps(r[key], sort_keys=True) for r in cells if r[level_key] == level]
            if len(set(values)) != 1:
                raise ValueError(f"{key} changes with the other factor")
        values_by_level = [next(r[key] for r in cells if r[level_key] == level) for level in (0, 1)]
        if values_by_level[0] == values_by_level[1]:
            raise ValueError(f"{key} does not manipulate its factor")
    if any(r["labels"] != cells[0]["labels"] for r in cells):
        raise ValueError("hold label alphabet fixed within factorial")
    record["conditions"] = sorted((dict(c) for c in cells), key=lambda c: c["condition_id"])
    record["pending"] = ["semantic_isolation_review", "region_token_calibration", "behavioral_feasibility"]
    return _finish(record)


def image_option_plan(source: Mapping[str, Any], *, query_asset: str,
                      option_assets: Mapping[str, str], target_option: str,
                      panel_order: Sequence[str], labels_by_position: Sequence[str],
                      width: int = 800, panel_size: int = 160, gap: int = 20) -> dict[str, Any]:
    """Plan a query-above-four-options collage, with NO pixel rendering.

    panel_order and labels_by_position are independent explicit permutations.
    Semantic ids and correctness remain private annotation data; visible records
    contain only asset paths, geometry and requested symbols.
    """
    record = _identity(source, "image-options")
    if len(option_assets) != 4 or len(panel_order) != 4 or set(panel_order) != set(option_assets):
        raise ValueError("panel order must permute exactly four semantic option ids")
    if len(set(panel_order)) != 4 or target_option not in option_assets:
        raise ValueError("unique options and explicit correct target required")
    if len(labels_by_position) != 4 or len(set(labels_by_position)) != 4:
        raise ValueError("four independently supplied distinct symbols required")
    if any(not isinstance(x, str) or not x for x in [query_asset, *option_assets.values(), *option_assets.keys(), *labels_by_position]):
        raise ValueError("nonempty asset paths, ids and symbols required")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, panel_size, gap)):
        raise ValueError("positive integer layout dimensions required")
    if width < 4 * panel_size + 5 * gap:
        raise ValueError("canvas too narrow")
    # Bottom gap beneath each panel is reserved for symbols, not semantic text.
    query = Rect((width - panel_size) // 2, gap,
                 (width - panel_size) // 2 + panel_size, gap + panel_size)
    y = panel_size + 3 * gap
    left = (width - 4 * panel_size - 3 * gap) // 2
    visible = []
    annotations = []
    for pos, (semantic, symbol) in enumerate(zip(panel_order, labels_by_position)):
        rect = Rect(left + pos * (panel_size + gap), y,
                    left + pos * (panel_size + gap) + panel_size, y + panel_size)
        visible.append({"panel_id": f"option-{pos}", "asset": option_assets[semantic],
                        "rect": asdict(rect), "symbol": symbol,
                        "label_anchor": [rect.x0 + panel_size / 2, rect.y1 + gap / 2]})
        annotations.append({"panel_id": f"option-{pos}", "position": pos,
                            "semantic_id": semantic, "symbol": symbol})
    target_pos = list(panel_order).index(target_option)
    record.update({"canvas": {"width": width, "height": y + panel_size + 2 * gap},
                   "visible": {"query": {"asset": query_asset, "rect": asdict(query)},
                               "options": visible},
                   "annotations": {"panels": annotations, "target_option": target_option,
                                   "correct_symbol": labels_by_position[target_pos]},
                   "pending": ["asset_leakage_review", "pixel_assembly", "processor_token_mapping",
                               "behavioral_feasibility"],
                   "presentation": "single_collage_not_native_multi_image"})
    record["task_semantics"] = "exact_visual_matching"
    return _finish(record)


def validate_split_separation(records: Iterable[Mapping[str, Any]]) -> None:
    """Reject a scene or pair shared across discovery/confirmation records."""
    seen: dict[tuple[str, str, str], str] = {}
    for record in records:
        source = record["source"]
        _identity(source, record["task_kind"])
        for key in ("scene_id", "pair_id"):
            identity = source["dataset_id"], key, source[key]
            if identity in seen and seen[identity] != source["split"]:
                raise ValueError("scene/pair split leakage")
            seen[identity] = source["split"]
