# Render CLEVR-MCQ-4 counterfactual scenes from JSON inside Blender.
#
# Run with the Blender version used by the official CLEVR generator
# (2.78c/2.79b; its bundled Python is 3.5, so this file avoids f-strings):
#
#   blender --background --python blender_render.py -- \
#       --jobs /data/clevr_mcq4_v1/render_jobs.json \
#       --clevr-root /opt/clevr-dataset-gen/image_generation --use-gpu 1
#
# Every job places exactly the objects in its scene JSON (same coordinates,
# rotation, size, shape, material, colour) with zero camera/lamp jitter, so an
# edited scene differs from its base only in the edited attribute. Base scenes
# must also be generated with zero jitter (see docs/reproduce.md).

import argparse
import json
import math
import os
import sys


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--clevr-root", required=True,
                        help="clevr-dataset-gen/image_generation (contains utils.py and data/)")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--min-bounces", type=int, default=8)
    parser.add_argument("--max-bounces", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--use-gpu", type=int, default=0)
    parser.add_argument("--skip-existing", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    return parser.parse_args(argv)


def matmul(matrix, vector):
    # Blender 2.8 replaced '*' with '@' for matrix-vector products.
    import bpy
    if bpy.app.version >= (2, 80, 0):
        return matrix.__matmul__(vector)
    return matrix * vector


def pixel_bbox(bpy_extras, scene, camera, obj, width, height):
    from mathutils import Vector
    xs, ys = [], []
    for corner in obj.bound_box:
        world = matmul(obj.matrix_world, Vector(corner))
        co = bpy_extras.object_utils.world_to_camera_view(scene, camera, world)
        xs.append(co.x * width)
        ys.append((1.0 - co.y) * height)
    clamp = lambda v, hi: max(0, min(hi, int(round(v))))  # noqa: E731
    return [clamp(min(xs), width), clamp(min(ys), height), clamp(max(xs), width), clamp(max(ys), height)]


def render_job(job, args, utils, properties):
    import bpy
    import bpy_extras

    data = os.path.join(args.clevr_root, "data")
    bpy.ops.wm.open_mainfile(filepath=os.path.join(data, "base_scene.blend"))
    utils.load_materials(os.path.join(data, "materials"))

    render = bpy.context.scene.render
    render.engine = "CYCLES"
    render.filepath = job["output_image"]
    render.resolution_x = args.width
    render.resolution_y = args.height
    render.resolution_percentage = 100
    render.tile_x = args.tile_size
    render.tile_y = args.tile_size
    if args.use_gpu == 1:
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = "CUDA"
            bpy.context.user_preferences.system.compute_device = "CUDA_0"
        else:
            prefs = bpy.context.user_preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"
    bpy.data.worlds["World"].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.samples
    bpy.context.scene.cycles.transparent_min_bounces = args.min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.max_bounces

    camera = bpy.data.objects["Camera"]
    rgba = {name: [c / 255.0 for c in rgb] + [1.0] for name, rgb in properties["colors"].items()}
    scene_out = dict(job["scene"])
    objects_out = []
    for obj in job["scene"]["objects"]:
        radius = properties["sizes"][obj["size"]]
        if obj["shape"] == "cube":
            radius /= math.sqrt(2)
        x, y = obj["3d_coords"][0], obj["3d_coords"][1]
        utils.add_object(os.path.join(data, "shapes"), properties["shapes"][obj["shape"]],
                         radius, (x, y), theta=obj["rotation"])
        blender_obj = bpy.context.object
        utils.add_material(properties["materials"][obj["material"]], Color=rgba[obj["color"]])
        record = dict(obj)
        record["pixel_coords"] = utils.get_camera_coords(camera, blender_obj.location)
        record["bbox"] = pixel_bbox(bpy_extras, bpy.context.scene, camera, blender_obj,
                                    args.width, args.height)
        objects_out.append(record)
    scene_out["objects"] = objects_out
    scene_out["image_filename"] = os.path.basename(job["output_image"])
    scene_out["render"] = {"width": args.width, "height": args.height, "samples": args.samples,
                           "jitter": 0.0, "blender": ".".join(str(v) for v in bpy.app.version)}

    for attempt in range(5):
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as error:  # the official generator also retries renders
            print("render failed (attempt {}): {}".format(attempt + 1, error))
    else:
        raise RuntimeError("render failed five times: " + job["job_id"])
    with open(job["output_scene"], "w") as stream:
        json.dump(scene_out, stream, indent=1)


def main():
    args = parse_args()
    sys.path.insert(0, args.clevr_root)
    import utils  # noqa: E402  (clevr-dataset-gen/image_generation/utils.py)

    with open(os.path.join(args.clevr_root, "data", "properties.json")) as stream:
        properties = json.load(stream)
    with open(args.jobs) as stream:
        jobs = json.load(stream)
    end = len(jobs) if args.end < 0 else args.end
    for index, job in enumerate(jobs[args.start:end], start=args.start):
        if args.skip_existing and os.path.isfile(job["output_image"]) and os.path.isfile(job["output_scene"]):
            continue
        print("[render] {}/{} {}".format(index + 1, len(jobs), job["job_id"]))
        render_job(job, args, utils, properties)


if __name__ == "__main__":
    main()
