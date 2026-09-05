"""Verify every rendered CLEVR pair against the frozen render profile and record hashes."""

import argparse
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path


ATTRS = ("size", "color", "material", "shape")
EXPECTED_PROFILE = {
    "width": 480, "height": 320, "samples": 512, "jitter": 0.0,
    "blender": "4.5.14 LTS", "backend": "CUDA",
    "adaptive_sampling": False, "denoising": True,
    "view_transform": "Standard",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    errors = []
    hashes = {}
    scene_hashes = {}
    pairs = defaultdict(dict)
    devices = set()
    for job in jobs:
        job_id = job["job_id"]
        member = job_id[-1]
        pairs[job_id[:-2]][member] = job
        image = Path(job["output_image"])
        scene_path = Path(job["output_scene"])
        if not image.is_file() or not scene_path.is_file():
            errors.append(f"missing image/scene: {job_id}")
            continue
        with image.open("rb") as stream:
            header = stream.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", header[16:24]) != (480, 320):
            errors.append(f"bad PNG/dimensions: {job_id}")
        hashes[image.name] = sha256(image)
        scene_hashes[scene_path.name] = sha256(scene_path)
        result = json.loads(scene_path.read_text())
        if result.get("image_filename") != image.name:
            errors.append(f"rendered scene filename: {job_id}")
        profile = dict(result.get("render", {}))
        device = profile.pop("device", None)
        if profile != EXPECTED_PROFILE or not device:
            errors.append(f"render profile: {job_id}")
        devices.update(device or [])
        sources = job["scene"]["objects"]
        rendered = result["objects"]
        if len(sources) != len(rendered):
            errors.append(f"object count: {job_id}")
            continue
        for i, (source, output) in enumerate(zip(sources, rendered)):
            if any(source[key] != output[key] for key in ATTRS):
                errors.append(f"object attributes: {job_id}/{i}")
            if any(abs(float(a) - float(b)) > 1e-4 for a, b in zip(
                    source["3d_coords"][:2], output["3d_coords"][:2])):
                errors.append(f"object xy: {job_id}/{i}")
            if not output.get("bbox") or not output.get("pixel_coords"):
                errors.append(f"rendered geometry metadata: {job_id}/{i}")
    if len(jobs) != 576 or len(pairs) != 288:
        errors.append(f"job/pair count: {len(jobs)}/{len(pairs)}")
    for pair_id, members in pairs.items():
        if set(members) != {"x", "y"}:
            errors.append(f"missing member: {pair_id}")
            continue
        x = members["x"]["scene"]["objects"]
        y = members["y"]["scene"]["objects"]
        if len(x) != len(y):
            errors.append(f"pair object count: {pair_id}")
            continue
        diffs = [(i, attr) for i, (a, b) in enumerate(zip(x, y))
                 for attr in ATTRS if a[attr] != b[attr]]
        if not diffs:
            errors.append(f"pair has no attribute edit: {pair_id}")
        if any(a["3d_coords"] != b["3d_coords"] or a["rotation"] != b["rotation"]
               for a, b in zip(x, y)):
            errors.append(f"pair pose changed: {pair_id}")
        x_name = Path(members["x"]["output_image"]).name
        y_name = Path(members["y"]["output_image"]).name
        if x_name in hashes and y_name in hashes and hashes[x_name] == hashes[y_name]:
            errors.append(f"pair PNGs identical: {pair_id}")
    report = {"jobs": len(jobs), "pairs": len(pairs), "verified_pngs": len(hashes),
              "verified_scene_jsons": len(scene_hashes), "errors": errors,
              "render_profile": EXPECTED_PROFILE, "render_devices": sorted(devices),
              "renderer_script_sha256": sha256(Path(__file__).with_name("render_pairs.py")),
              "png_sha256": hashes, "rendered_scene_sha256": scene_hashes}
    output = args.out_dir / "pair_validation.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("png_sha256", "rendered_scene_sha256")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
