"""Deterministic optional-task truth construction and explicit pixel assembly.

No generation occurs on import. Pillow is imported only by the explicit
assembler, which was not executed during source implementation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .task_prep import image_option_plan, spatial_factorial


def spatial_from_objects(source: Mapping[str, Any], *, image: str,
                         objects: Sequence[Mapping[str, Any]],
                         query_objects: Sequence[str], option_sets: Sequence[Sequence[str]],
                         labels: Sequence[str]) -> dict:
    """Construct truth-checked question x semantic-option-set factorial.

    Two queried objects and two genuinely different four-color sets are required.
    Both truth colors must remain available under both option sets. Objects carry
    id, description, color and bbox. No answer is inferred from an image filename.
    """
    by_id = {obj["id"]: dict(obj) for obj in objects}
    if len(by_id) != len(objects) or len(query_objects) != 2 or len(set(query_objects)) != 2:
        raise ValueError("unique objects and two different query objects required")
    if any(q not in by_id for q in query_objects) or len(option_sets) != 2:
        raise ValueError("unknown query or missing semantic alternate sets")
    if any(len(options) != 4 or len(set(options)) != 4 for options in option_sets):
        raise ValueError("two four-distinct-color sets required")
    if set(option_sets[0]) == set(option_sets[1]):
        raise ValueError("option permutation alone is not semantic option guidance")
    available = {obj["color"] for obj in objects}
    truths = {by_id[q]["color"] for q in query_objects}
    if any(not truths.issubset(options) or not set(options).issubset(available) for options in option_sets):
        raise ValueError("both truths must remain present; all colors need scene-region support")
    if len(labels) != 4 or len(set(labels)) != 4:
        raise ValueError("four distinct labels required")
    if len({by_id[q]["description"] for q in query_objects}) != 2:
        raise ValueError("query descriptions must distinguish the objects")
    if any(sum(other["description"] == by_id[q]["description"] for other in objects) != 1
           for q in query_objects):
        raise ValueError("each query description must identify exactly one annotated object")
    if any(list(option_sets[0]).index(truth) != list(option_sets[1]).index(truth) for truth in truths):
        raise ValueError("hold each truth's option position/symbol fixed across the semantic option-set factor")
    rows = []
    for qlevel, qid in enumerate(query_objects):
        obj = by_id[qid]
        for olevel, options in enumerate(option_sets):
            rows.append({"condition_id": f"q{qlevel}-o{olevel}", "question_level": qlevel,
                         "option_level": olevel, "question": f"What color is {obj['description']}?",
                         "question_region": qid, "option_region": f"option-set-{olevel}",
                         "image": image, "option_ids": list(options), "labels": list(labels),
                         "correct_option_id": obj["color"],
                         **{k: source[k] for k in ("scene_id", "pair_id", "split")}})
    record = spatial_factorial(source, rows)
    # Return a separate truth-bearing task envelope rather than mutating its
    # already hashed preparation body.
    return {"task": record, "scene_objects": [dict(o) for o in objects],
            "query_objects": list(query_objects), "option_sets": [list(s) for s in option_sets]}


def assemble_image_options(plan: Mapping[str, Any], *, asset_base: str | Path,
                           font_path: str | Path, font_sha256: str,
                           output: str | Path, font_size: int, mode: str = "visual") -> dict:
    """Assemble the declared single collage with letterboxed panels and labels.

    Explicit font bytes, font size and Pillow version become assembly provenance.
    Asset semantics remain private. Existing outputs and input-directory overlaps
    are rejected before pixel work. No 3D renderer or image-generation service.
    """
    task = plan.get("task", plan)
    visible = task["visible"]
    base, target, font = Path(asset_base).resolve(), Path(output).resolve(), Path(font_path).resolve()
    if mode not in {"visual", "text-query-only"}:
        raise ValueError("assembly mode must be visual or text-query-only")
    entries = [visible["query"], *(visible["options"] if mode == "visual" else [])]
    paths = [(base / entry["asset"]).resolve() for entry in entries]
    if type(font_size) is not int or font_size <= 0 or not target.parent.is_dir():
        raise ValueError("explicit positive font size and existing output parent required")
    if target.exists() or any(p == target or p.parent == target or p.parent in target.parents or target in p.parents
                              for p in [*paths, font]):
        raise ValueError("output overlaps an immutable asset directory or exists")
    font_bytes = font.read_bytes()
    if hashlib.sha256(font_bytes).hexdigest() != font_sha256:
        raise ValueError("font bytes differ from pinned SHA256")
    payloads = [p.read_bytes() for p in paths]
    if task.get("task_semantics") != "exact_visual_matching":
        raise ValueError("assembler currently supports explicit exact-visual-matching tasks only")
    if mode == "visual":
        correct = next(p["position"] for p in task["annotations"]["panels"]
                       if p["semantic_id"] == task["annotations"]["target_option"])
        if payloads[0] != payloads[correct + 1] or len({hashlib.sha256(p).hexdigest() for p in payloads[1:]}) != 4:
            raise ValueError("query must equal exactly its truth-annotated option asset; options must be distinct")
    from PIL import Image, ImageDraw, ImageFont, ImageOps, __version__
    import io
    canvas = Image.new("RGB", (task["canvas"]["width"], task["canvas"]["height"]), "white")
    selected_font = ImageFont.truetype(str(font), size=font_size)
    draw = ImageDraw.Draw(canvas)
    for entry, payload in zip(entries, payloads):
        rect = entry["rect"]
        x0, y0, x1, y1 = [int(rect[k]) for k in ("x0", "y0", "x1", "y1")]
        with Image.open(io.BytesIO(payload)) as loaded:
            picture = ImageOps.contain(loaded.convert("RGB"), (x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
        canvas.paste(picture, (x0 + (x1 - x0 - picture.width) // 2,
                               y0 + (y1 - y0 - picture.height) // 2))
        if "symbol" in entry:
            anchor = entry["label_anchor"]
            bounds = draw.textbbox((0, 0), entry["symbol"], font=selected_font)
            # Reject labels that cannot fit the declared label band.
            if bounds[3] - bounds[1] > (task["canvas"]["height"] - y1) or bounds[2] - bounds[0] > x1 - x0:
                raise ValueError("font/label does not fit reserved label band")
            placed = draw.textbbox(tuple(anchor), entry["symbol"], font=selected_font, anchor="mt")
            if placed[0] < x0 or placed[2] > x1 or placed[1] < y1 or placed[3] > canvas.height:
                raise ValueError("anchored symbol would overflow its declared visual label band")
            draw.text(tuple(anchor), entry["symbol"], font=selected_font, fill="black", anchor="mt")
    with target.open("xb") as stream:
        canvas.save(stream, format="PNG", optimize=False, compress_level=9)
    return {"image": str(target), "image_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "asset_sha256": {str(p): hashlib.sha256(b).hexdigest() for p, b in zip(paths, payloads)},
            "font_sha256": font_sha256, "font_size": font_size, "pillow_version": __version__,
            "font_path": str(font), "asset_base": str(base),
            "task_derivative_id": task["derivative_id"], "image_size": list(canvas.size), "mode": mode}


def image_option_variants(plan: Mapping[str, Any], *, image: str, text_image: str) -> list[dict]:
    """Produce visual and matched-text task records without leaking visual truth.

    The text condition deliberately exposes option semantics as its matched
    comparison. The visual condition contains only the query and symbol request.
    """
    task = plan.get("task", plan)
    source, private = task["source"], task["annotations"]
    panels = private["panels"]
    labels = [p["symbol"] for p in panels]
    semantic = [p["semantic_id"] for p in panels]
    base = {"task_id": task["derivative_id"], "image": image, "source": dict(source), "derivative_plan": dict(task),
            "labels": labels, "option_semantics": semantic,
            "correct_index": semantic.index(private["target_option"]),
            "regions": {"query": [task["visible"]["query"]["rect"][k] for k in ("x0", "y0", "x1", "y1")],
                        **{p["panel_id"]: [p["rect"][k] for k in ("x0", "y0", "x1", "y1")]
                           for p in task["visible"]["options"]}},
            "symbol_regions": {p["panel_id"] + "-symbol": [p["rect"]["x0"], p["rect"]["y1"],
                                p["rect"]["x1"], task["canvas"]["height"]]
                               for p in task["visible"]["options"]}}
    visual = {**base, "condition": "visual", "prompt": "Which option image matches the query image? Respond with only its displayed symbol.",
              "question": "Which option image matches the query image?"}
    if text_image == image:
        raise ValueError("matched text needs separately assembled query-only image")
    text = {**base, "image": text_image, "condition": "matched_text",
            "prompt": "Which option describes the query image?\n" + "\n".join(f"{l}. {s}" for l, s in zip(labels, semantic))
                      + "\nRespond with only the option label.\nAnswer:",
            "question": "Which option describes the query image?"}
    return [visual, text]
