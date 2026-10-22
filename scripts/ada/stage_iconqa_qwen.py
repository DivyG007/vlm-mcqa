"""Gateway-only pinned checkpoint acquisition; no torch/model/GPU imports."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def main():
    from huggingface_hub import HfApi, snapshot_download
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "verified_checkpoint.json"
    if receipt.exists():
        raise SystemExit("verified checkpoint exists; inspect/reuse, never overwrite")
    # Qwen is public. Do not consume or modify unrelated cached credentials.
    info = HfApi(token=False).model_info(MODEL, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError("publisher revision mismatch")
    snapshot_download(MODEL, revision=REVISION, local_dir=root, token=False,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"], max_workers=2)
    inventory = {}
    for sibling in info.siblings:
        path = root / sibling.rfilename
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
                blob.update(chunk)
        lfs = getattr(sibling, "lfs", None)
        expected = getattr(lfs, "sha256", None) if lfs else None
        if expected and digest.hexdigest() != expected:
            raise RuntimeError(f"publisher LFS checksum mismatch: {sibling.rfilename}")
        if not expected and sibling.blob_id and blob.hexdigest() != sibling.blob_id:
            raise RuntimeError(f"publisher Git blob mismatch: {sibling.rfilename}")
        inventory[sibling.rfilename] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size,
                                        "publisher_lfs_sha256": expected, "publisher_blob_id": sibling.blob_id}
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if not set(index["weight_map"].values()) <= set(inventory):
        raise RuntimeError("checkpoint shard missing")
    data = {"model_id": MODEL, "revision": REVISION, "files": inventory, "status": "verified",
            "model_loaded": False}
    with receipt.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
    print(json.dumps({"status": "verified", "files": len(inventory), "bytes": sum(v["bytes"] for v in inventory.values())}), flush=True)


if __name__ == "__main__":
    main()
