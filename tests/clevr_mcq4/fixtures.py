"""Random CLEVR-style scene fixtures shared by the CLEVR-MCQ-4 tests."""

from __future__ import annotations

import random

from vlm_mcqa.clevr_mcq4.scene import ATTRIBUTE_VALUES

DIRECTIONS = {
    "right": [1.0, 0.0, 0.0], "left": [-1.0, 0.0, 0.0],
    "front": [0.0, 1.0, 0.0], "behind": [0.0, -1.0, 0.0],
    "above": [0.0, 0.0, 1.0], "below": [0.0, 0.0, -1.0],
}


def random_scene(index: int, rng: random.Random) -> dict:
    objects = []
    for _ in range(rng.randint(5, 9)):
        objects.append({
            **{attr: rng.choice(values) for attr, values in ATTRIBUTE_VALUES.items()},
            "3d_coords": [rng.uniform(-3, 3), rng.uniform(-3, 3), 0.35],
            "rotation": rng.uniform(0, 360),
            "pixel_coords": [rng.randint(0, 480), rng.randint(0, 320), 10.0],
        })
    return {"image_index": index, "image_filename": f"CLEVR_new_{index:06d}.png",
            "objects": objects, "directions": DIRECTIONS}
