#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=rachit_qwen7b_stage
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/rachit_qwen7b_stage_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/rachit_qwen7b_stage_%j.err

set -euo pipefail
set -x

case "$(hostname -s)" in
  gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077)
    echo "[FATAL] ineligible Ada node: $(hostname -s)" >&2
    exit 90
    ;;
esac

MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
REVISION="cc594898137f460bfe9f0759e9844b3ce807cfb5"
REMOTE_BASE="/home2/rsm/anlp-vlm-mcqa"
FINAL_ROOT="${REMOTE_BASE}/checkpoints/qwen25vl_7b_${REVISION}"
INCOMING_ROOT="${REMOTE_BASE}/checkpoints/.qwen25vl_7b_${REVISION}.incoming_${SLURM_JOB_ID}"
DEFAULT_STAGING_ROOT="/scratch/rsm/anlp-vlm-mcqa/qwen25vl_7b_${REVISION}_${SLURM_JOB_ID}"
STAGING_ROOT="${RESUME_ROOT:-${DEFAULT_STAGING_ROOT}}"

mkdir -p "${REMOTE_BASE}/logs" "${REMOTE_BASE}/checkpoints" "$(dirname "${STAGING_ROOT}")"
echo "[runtime] job_id=${SLURM_JOB_ID} node=$(hostname -s)"
df -h /home2/rsm /scratch || true

if test -e "${FINAL_ROOT}"; then
  echo "[FATAL] immutable final checkpoint path already exists: ${FINAL_ROOT}" >&2
  exit 4
fi
if test -e "${INCOMING_ROOT}"; then
  echo "[FATAL] fresh incoming path required: ${INCOMING_ROOT}" >&2
  exit 5
fi
if test -n "${RESUME_ROOT:-}"; then
  case "${STAGING_ROOT}" in
    /scratch/rsm/anlp-vlm-mcqa/qwen25vl_7b_${REVISION}_*) ;;
    *)
      echo "[FATAL] refusing unexpected resume path: ${STAGING_ROOT}" >&2
      exit 6
      ;;
  esac
  test -d "${STAGING_ROOT}" || {
    echo "[FATAL] resume path is not an existing directory: ${STAGING_ROOT}" >&2
    exit 7
  }
  echo "[resume] preserving and reusing partial download at ${STAGING_ROOT}"
else
  test ! -e "${STAGING_ROOT}" || {
    echo "[FATAL] fresh staging path required: ${STAGING_ROOT}" >&2
    exit 8
  }
fi

source /home2/rsm/miniconda3/etc/profile.d/conda.sh
conda activate motion-probe
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
python - <<'PY'
import importlib.metadata as metadata

for package in ("torch", "transformers", "huggingface-hub", "accelerate", "safetensors"):
    print(f"[environment] {package}={metadata.version(package)}")
PY

python - "${MODEL_ID}" "${REVISION}" "${STAGING_ROOT}" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, revision, target = sys.argv[1:]
path = snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_dir=target,
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
)
print(f"[download] materialized={path}")
PY

python - "${STAGING_ROOT}" "${MODEL_ID}" "${REVISION}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root, model_id, revision = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
required = {
    "config.json", "generation_config.json", "preprocessor_config.json",
    "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
    "model.safetensors.index.json", "chat_template.json",
    *(f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)),
}
missing = sorted(name for name in required if not (root / name).is_file())
if missing:
    raise SystemExit(f"missing checkpoint files: {missing}")
if list(root.glob("*.bin")):
    raise SystemExit("native safetensors-only checkpoint required")
config = json.loads((root / "config.json").read_text())
expected = {
    "model_type": "qwen2_5_vl", "num_hidden_layers": 28,
    "num_attention_heads": 28, "hidden_size": 3584,
}
actual = {key: config.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"unexpected model architecture: {actual}")
files = []
for path in sorted(root.rglob("*")):
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
                      "sha256": digest.hexdigest()})
record = {
    "schema": "rachit.qwen-checkpoint-stage.v1", "hf_id": model_id,
    "revision": revision, "architecture": actual, "files": files,
    "total_bytes": sum(row["size"] for row in files),
}
(root / "stage_completion.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
print(json.dumps({"file_count": len(files), "total_bytes": record["total_bytes"],
                  "architecture": actual}, sort_keys=True))
PY

# Every state-changing rsync is preceded by its exact dry-run equivalent.
rsync -an --info=stats2 "${STAGING_ROOT}/" "${INCOMING_ROOT}/"
mkdir "${INCOMING_ROOT}"
rsync -a --info=stats2 "${STAGING_ROOT}/" "${INCOMING_ROOT}/"
test -s "${INCOMING_ROOT}/stage_completion.json"
mv "${INCOMING_ROOT}" "${FINAL_ROOT}"
test -s "${FINAL_ROOT}/stage_completion.json"
du -sh "${FINAL_ROOT}"
echo "[complete] ${FINAL_ROOT}"
