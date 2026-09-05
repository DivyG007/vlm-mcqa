"""Rewrite planned CLEVR pair render paths for rendering on another machine."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jobs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_jobs.read_bytes()
    jobs = json.loads(source)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    local_jobs = []
    for job in jobs:
        item = dict(job)
        item["output_image"] = str(out / "images" / Path(job["output_image"]).name)
        item["output_scene"] = str(out / "rendered_scenes" / Path(job["output_scene"]).name)
        local_jobs.append(item)
    (out / "render_jobs_local.json").write_text(json.dumps(local_jobs, indent=2))
    manifest = {
        "source_jobs_sha256": hashlib.sha256(source).hexdigest(),
        "job_count": len(jobs),
        "pair_count": len(jobs) // 2,
        "local_output": str(out),
        "job_ids": [job["job_id"] for job in jobs],
    }
    (out / "job_relocation.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"jobs": len(jobs), "pairs": len(jobs) // 2,
                      "out_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
