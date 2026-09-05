"""Render CLEVR scene-JSON jobs with Blender 4.5 Cycles.

Run inside Blender, for example:
  blender --background --python scripts/dataset/render_pairs.py -- \
      --jobs data/clevr_mcq4_v3/render_jobs.json \
      --clevr-root clevr-dataset-gen/image_generation \
      --backend CUDA --device-name "<substring of your GPU name>" --samples 512 --denoise

This is an image renderer for already generated scene JSON, not a port of the
official CLEVR placement and visibility algorithm.
"""

import argparse
import copy
import json
import os
import sys
import time

import bpy
import bpy_extras
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--clevr-root", required=True)
    parser.add_argument("--backend", choices=("CUDA", "OPTIX", "CPU"), default="CUDA")
    parser.add_argument("--device-name", default=None,
                        help="substring of the GPU name; default: first device of --backend")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="Inclusive job index")
    parser.add_argument("--end", type=int, default=-1, help="Exclusive job index; -1 means all")
    return parser.parse_args(argv)


def pixel_bbox(scene, camera, obj, width, height):
    xs, ys = [], []
    for corner in obj.bound_box:
        world = obj.matrix_world @ Vector(corner)
        co = bpy_extras.object_utils.world_to_camera_view(scene, camera, world)
        xs.append(co.x * width)
        ys.append((1.0 - co.y) * height)
    clamp = lambda v, hi: max(0, min(hi, int(round(v))))
    return [clamp(min(xs), width), clamp(min(ys), height),
            clamp(max(xs), width), clamp(max(ys), height)]


def select_device(scene, backend, device_name):
    if backend == "CPU":
        scene.cycles.device = "CPU"
        return ["CPU"]
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = backend
    prefs.get_devices()
    matches = [i for i, device in enumerate(prefs.devices)
               if device.type == backend and (device_name is None or device_name in device.name)]
    if device_name is None:
        matches = matches[:1]
    names = []
    for i, device in enumerate(prefs.devices):
        device.use = i in matches
        if device.use:
            names.append(device.name)
    if len(names) != 1:
        raise RuntimeError("Expected exactly one selected GPU; found " + repr(names))
    scene.cycles.device = "GPU"
    return names


def add_object(scene, object_data, index, props, asset_root, camera, width, height):
    shape = object_data["shape"]
    asset_name = props["shapes"][shape]
    blend_path = os.path.join(asset_root, "shapes", asset_name + ".blend")
    with bpy.data.libraries.load(blend_path, link=False) as (_src, dst):
        dst.objects = [asset_name]
    obj = dst.objects[0]
    if obj is None:
        raise RuntimeError("Missing shape " + blend_path)
    obj.name = "CLEVR_%02d_%s" % (index, asset_name)
    scene.collection.objects.link(obj)

    # The official generator stores theta directly in rotation_euler.z. It
    # samples theta in [0, 360), so it is important not to convert it again.
    # Its smooth-cube asset does not enter the old `obj_name == 'Cube'` branch.
    radius = props["sizes"][object_data["size"]]
    x, y = object_data["3d_coords"][:2]
    obj.scale = [value * radius for value in obj.scale]
    obj.location += Vector((x, y, radius))
    obj.rotation_euler.z = object_data["rotation"]

    material_name = props["materials"][object_data["material"]]
    if material_name not in bpy.data.node_groups:
        blend_path = os.path.join(asset_root, "materials", material_name + ".blend")
        with bpy.data.libraries.load(blend_path, link=False) as (_src, dst):
            dst.node_groups = [material_name]
    mat = bpy.data.materials.new(name="CLEVR_%02d_%s" % (index, material_name))
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    group = nodes.new("ShaderNodeGroup")
    group.node_tree = bpy.data.node_groups[material_name]
    rgb = props["colors"][object_data["color"]]
    group.inputs["Color"].default_value = tuple(value / 255.0 for value in rgb) + (1.0,)
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(group.outputs["Shader"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    bpy.context.view_layer.update()
    record = copy.deepcopy(object_data)
    record["3d_coords"] = list(obj.location)
    co = bpy_extras.object_utils.world_to_camera_view(scene, camera, obj.location)
    record["pixel_coords"] = [int(round(co.x * width)), int(round((1.0 - co.y) * height)), co.z]
    record["bbox"] = pixel_bbox(scene, camera, obj, width, height)
    return record


def render_job(job, args, props, asset_root):
    started = time.perf_counter()
    bpy.ops.wm.open_mainfile(filepath=os.path.join(asset_root, "base_scene.blend"))
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.filepath = os.path.abspath(job["output_image"])
    render.resolution_x = args.width
    render.resolution_y = args.height
    render.resolution_percentage = 100
    render.image_settings.file_format = "PNG"
    scene.cycles.samples = args.samples
    scene.cycles.transparent_min_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.use_denoising = args.denoise
    scene.cycles.use_adaptive_sampling = False
    selected = select_device(scene, args.backend, args.device_name)

    camera = bpy.data.objects["Camera"]
    result_scene = copy.deepcopy(job["scene"])
    result_scene["objects"] = [add_object(scene, source, i, props, asset_root,
                                          camera, args.width, args.height)
                               for i, source in enumerate(job["scene"]["objects"])]
    result_scene["image_filename"] = os.path.basename(render.filepath)
    result_scene["render"] = {
        "width": args.width, "height": args.height, "samples": args.samples,
        "jitter": 0.0, "blender": bpy.app.version_string,
        "backend": args.backend, "device": selected,
        "adaptive_sampling": False, "denoising": args.denoise,
        "view_transform": scene.view_settings.view_transform,
    }
    os.makedirs(os.path.dirname(render.filepath), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(job["output_scene"])), exist_ok=True)
    setup_seconds = time.perf_counter() - started
    render_started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    render_seconds = time.perf_counter() - render_started
    with open(job["output_scene"], "w") as stream:
        json.dump(result_scene, stream, indent=2)
    print("RENDER_RESULT", json.dumps({
        "job_id": job["job_id"], "objects": len(result_scene["objects"]),
        "setup_seconds": round(setup_seconds, 3),
        "render_seconds": round(render_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
        "output_image": render.filepath,
        "output_bytes": os.path.getsize(render.filepath),
        "backend": args.backend, "device": selected,
    }), flush=True)


def main():
    args = parse_args()
    asset_root = os.path.abspath(os.path.join(args.clevr_root, "data"))
    with open(os.path.join(asset_root, "properties.json")) as stream:
        props = json.load(stream)
    with open(args.jobs, encoding="utf-8-sig") as stream:
        jobs = json.load(stream)
    end = len(jobs) if args.end < 0 else min(args.end, len(jobs))
    if not 0 <= args.start <= end:
        raise ValueError("Invalid job range")
    print("RENDER_BATCH", json.dumps({"jobs": len(jobs), "samples": args.samples,
                                      "size": [args.width, args.height],
                                      "backend": args.backend, "denoising": args.denoise,
                                      "blender": bpy.app.version_string,
                                      "start": args.start, "end": end}), flush=True)
    for index in range(args.start, end):
        job = jobs[index]
        if args.skip_existing and os.path.isfile(job["output_image"]) and os.path.isfile(job["output_scene"]):
            print("RENDER_SKIP", job["job_id"], flush=True)
            continue
        print("RENDER_START", index + 1, len(jobs), job["job_id"], flush=True)
        render_job(job, args, props, asset_root)


if __name__ == "__main__":
    main()
