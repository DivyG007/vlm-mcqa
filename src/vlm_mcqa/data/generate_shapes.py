"""Generate paired synthetic shape scenes for VLM MCQA experiments.

Each pair contains the same four objects in the same locations. The two images
differ only by swapping the colors of two objects, one of which is queried. This
keeps the prompt, option contents, image-token count, layout, and global color
histogram fixed while changing the visually correct answer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw


COLORS = {
    "red": "#d62728",
    "blue": "#1f77b4",
    "green": "#2ca02c",
    "yellow": "#f2c744",
}
SHAPES = ("circle", "square", "triangle", "star")
LABEL_SCHEMES = {
    "letters": ("A", "B", "C", "D"),
    "rare_letters": ("Q", "Z", "R", "X"),
    "numbers": ("1", "2", "3", "4"),
    "punctuation": ("!", "@", "#", "$"),
}


@dataclass(frozen=True)
class ObjectSpec:
    shape: str
    color: str
    bbox: tuple[int, int, int, int]

    def as_dict(self) -> dict[str, object]:
        return {"shape": self.shape, "color": self.color, "bbox": list(self.bbox)}


def _grid_boxes(image_size: int) -> list[tuple[int, int, int, int]]:
    margin = image_size // 12
    cell = (image_size - 3 * margin) // 2
    boxes: list[tuple[int, int, int, int]] = []
    for row in range(2):
        for col in range(2):
            left = margin + col * (cell + margin)
            top = margin + row * (cell + margin)
            inset = cell // 7
            boxes.append((left + inset, top + inset, left + cell - inset, top + cell - inset))
    return boxes


def _star_points(bbox: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    left, top, right, bottom = bbox
    cx, cy = (left + right) / 2, (top + bottom) / 2
    outer = min(right - left, bottom - top) / 2
    inner = outer * 0.43
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        radius = outer if index % 2 == 0 else inner
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _draw_object(draw: ImageDraw.ImageDraw, obj: ObjectSpec) -> None:
    fill = COLORS[obj.color]
    outline = "#222222"
    width = max(2, (obj.bbox[2] - obj.bbox[0]) // 32)
    if obj.shape == "circle":
        draw.ellipse(obj.bbox, fill=fill, outline=outline, width=width)
    elif obj.shape == "square":
        draw.rectangle(obj.bbox, fill=fill, outline=outline, width=width)
    elif obj.shape == "triangle":
        left, top, right, bottom = obj.bbox
        points = [((left + right) / 2, top), (right, bottom), (left, bottom)]
        draw.polygon(points, fill=fill, outline=outline)
        draw.line(points + [points[0]], fill=outline, width=width, joint="curve")
    elif obj.shape == "star":
        points = _star_points(obj.bbox)
        draw.polygon(points, fill=fill, outline=outline)
        draw.line(points + [points[0]], fill=outline, width=width, joint="curve")
    else:
        raise ValueError(f"Unsupported shape: {obj.shape}")


def render_scene(objects: Sequence[ObjectSpec], image_size: int, output_path: Path) -> None:
    image = Image.new("RGB", (image_size, image_size), "#f7f7f7")
    draw = ImageDraw.Draw(image)
    for obj in objects:
        _draw_object(draw, obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def _format_prompt(question: str, options: Sequence[str], labels: Sequence[str]) -> str:
    lines = [question]
    lines.extend(f"{label}. {option}" for label, option in zip(labels, options, strict=True))
    lines.append("Respond with only the option label.")
    lines.append("Answer:")
    return "\n".join(lines)


def _record(
    *,
    pair_id: str,
    member: str,
    image_path: str,
    objects: Sequence[ObjectSpec],
    query_shape: str,
    options: Sequence[str],
    scheme_name: str,
    labels: Sequence[str],
    variant_family: str,
    variant_key: str,
    changed_shapes: Sequence[str],
) -> dict[str, object]:
    by_shape = {obj.shape: obj for obj in objects}
    correct_content = by_shape[query_shape].color
    correct_index = list(options).index(correct_content)
    question = f"What color is the {query_shape}?"
    return {
        "sample_id": f"{pair_id}_{member}_{variant_family}_{scheme_name}_{variant_key}",
        "pair_id": pair_id,
        "pair_member": member,
        "task_type": "color_of_shape",
        "image": image_path,
        "question": question,
        "prompt": _format_prompt(question, options, labels),
        "option_contents": list(options),
        "label_scheme": scheme_name,
        "labels": list(labels),
        "correct_content": correct_content,
        "correct_index": correct_index,
        "correct_label": labels[correct_index],
        "query_shape": query_shape,
        "query_bbox": list(by_shape[query_shape].bbox),
        "objects": [obj.as_dict() for obj in objects],
        "changed_shapes": list(changed_shapes),
        "variant_family": variant_family,
        "variant_key": variant_key,
    }


def _options_with_correct_at(
    correct: str, position: int, rng: random.Random
) -> list[str]:
    distractors = [color for color in COLORS if color != correct]
    rng.shuffle(distractors)
    options = distractors
    options.insert(position, correct)
    return options


def generate_dataset(output_dir: Path, num_pairs: int, seed: int, image_size: int) -> Path:
    rng = random.Random(seed)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    boxes = _grid_boxes(image_size)
    records: list[dict[str, object]] = []

    for pair_index in range(num_pairs):
        pair_id = f"pair_{pair_index:05d}"
        shape_order = list(SHAPES)
        color_order = list(COLORS)
        rng.shuffle(shape_order)
        rng.shuffle(color_order)
        source = [
            ObjectSpec(shape=shape, color=color, bbox=bbox)
            for shape, color, bbox in zip(shape_order, color_order, boxes, strict=True)
        ]

        swap_indices = rng.sample(range(4), 2)
        query_index = rng.choice(swap_indices)
        target_colors = [obj.color for obj in source]
        first, second = swap_indices
        target_colors[first], target_colors[second] = target_colors[second], target_colors[first]
        target = [
            ObjectSpec(shape=obj.shape, color=color, bbox=obj.bbox)
            for obj, color in zip(source, target_colors, strict=True)
        ]
        query_shape = source[query_index].shape
        changed_shapes = [source[index].shape for index in swap_indices]

        source_rel = f"images/{pair_id}_a.png"
        target_rel = f"images/{pair_id}_b.png"
        render_scene(source, image_size, output_dir / source_rel)
        render_scene(target, image_size, output_dir / target_rel)

        shared_options = list(COLORS)
        rng.shuffle(shared_options)
        for scheme_name, labels in LABEL_SCHEMES.items():
            for member, image_path, objects in (
                ("a", source_rel, source),
                ("b", target_rel, target),
            ):
                records.append(
                    _record(
                        pair_id=pair_id,
                        member=member,
                        image_path=image_path,
                        objects=objects,
                        query_shape=query_shape,
                        options=shared_options,
                        scheme_name=scheme_name,
                        labels=labels,
                        variant_family="cross_image",
                        variant_key="shared_options",
                        changed_shapes=changed_shapes,
                    )
                )

        for member, image_path, objects in (
            ("a", source_rel, source),
            ("b", target_rel, target),
        ):
            correct = next(obj.color for obj in objects if obj.shape == query_shape)
            for scheme_name, labels in LABEL_SCHEMES.items():
                for correct_position in range(4):
                    options = _options_with_correct_at(correct, correct_position, rng)
                    records.append(
                        _record(
                            pair_id=pair_id,
                            member=member,
                            image_path=image_path,
                            objects=objects,
                            query_shape=query_shape,
                            options=options,
                            scheme_name=scheme_name,
                            labels=labels,
                            variant_family="answer_position",
                            variant_key=f"correct_at_{correct_position}",
                            changed_shapes=changed_shapes,
                        )
                    )

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    metadata = {
        "dataset_name": "shapes_mcqa_v0",
        "seed": seed,
        "num_pairs": num_pairs,
        "num_images": 2 * num_pairs,
        "num_records": len(records),
        "records_per_pair": 10 * len(LABEL_SCHEMES),
        "image_size": image_size,
        "colors": list(COLORS),
        "shapes": list(SHAPES),
        "label_schemes": {name: list(labels) for name, labels in LABEL_SCHEMES.items()},
        "causal_pair_invariant": (
            "Within each cross_image pair, layout, object identities, global color histogram, "
            "question, choices, and labels are fixed; only two object-color assignments swap."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=448)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_pairs < 1:
        raise SystemExit("--num-pairs must be at least 1")
    if args.image_size < 128:
        raise SystemExit("--image-size must be at least 128")
    manifest = generate_dataset(args.output_dir, args.num_pairs, args.seed, args.image_size)
    print(manifest)


if __name__ == "__main__":
    main()
