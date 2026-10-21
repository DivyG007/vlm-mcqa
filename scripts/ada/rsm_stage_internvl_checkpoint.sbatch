#!/bin/bash
#SBATCH -A cvit
#SBATCH --qos=normal
#SBATCH --partition=u22
#SBATCH --job-name=rsm_iv_stage
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/rsm_iv_stage_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/rsm_iv_stage_%j.err

set -euo pipefail
set -x

: "${SOURCE_GIT_COMMIT:?committed source identity required}"
case "$(hostname -s)" in
  gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077)
    echo "[FATAL] ineligible Ada node: $(hostname -s)" >&2
    exit 90
    ;;
esac

MODEL_ID=OpenGVLab/InternVL3_5-8B-HF
REVISION=741a7d03020411e666c6109218ab71e08151ef86
NODE_STORAGE_ROOT=${NODE_STORAGE_ROOT:-/scratch/rsm}
ENV_ROOT=${VLM_ENV_ROOT:-/home2/rsm/anlp-vlm-mcqa/envs/e20_qwen_torch28_tf4576_40839}
CHECKPOINT_ROOT=${NODE_STORAGE_ROOT}/anlp-vlm-mcqa/checkpoints/internvl35_8b_${REVISION}
test -s "${ENV_ROOT}/completion.json"
export PATH="${ENV_ROOT}/bin:${PATH}"
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=180 HF_HUB_ETAG_TIMEOUT=30
mkdir -p /home2/rsm/anlp-vlm-mcqa/logs

if test -s "${CHECKPOINT_ROOT}/stage_completion.json"; then
  echo "[complete] existing checkpoint: ${CHECKPOINT_ROOT}"
  exit 0
fi
if test -e "$CHECKPOINT_ROOT"; then
  echo "[FATAL] incomplete final checkpoint path: ${CHECKPOINT_ROOT}" >&2
  exit 4
fi

STAGING_ROOT=${CHECKPOINT_STAGING_ROOT:-${NODE_STORAGE_ROOT}/anlp-vlm-mcqa/staging/internvl35_8b_${REVISION}_${SLURM_JOB_ID}}
mkdir -p "$STAGING_ROOT"
python - "$MODEL_ID" "$REVISION" "$STAGING_ROOT" <<'PY'
from huggingface_hub import snapshot_download
import sys
import time

model_id, revision, target = sys.argv[1:]
for attempt in range(1, 4):
    try:
        print(snapshot_download(repo_id=model_id, revision=revision, local_dir=target,
                                ignore_patterns=["*.md", ".gitattributes", "original/*", "onnx/*"]))
        break
    except Exception:
        if attempt == 3:
            raise
        print(f"[retry] checkpoint download attempt {attempt} failed; resuming partial files", file=sys.stderr)
        time.sleep(10 * attempt)
PY
python - "$STAGING_ROOT" "$MODEL_ID" "$REVISION" "$SOURCE_GIT_COMMIT" <<'PY'
import json
from pathlib import Path
import sys

root, model_id, revision, source_commit = Path(sys.argv[1]), *sys.argv[2:]
required = [root / "config.json", root / "model.safetensors.index.json"]
if any(not path.is_file() for path in required) or not list(root.glob("*.safetensors")):
    raise SystemExit("incomplete native safetensors checkpoint")
record = {"schema": "rsm.public-checkpoint-stage.v1", "status": "complete",
          "model_key": "internvl35_8b", "hf_id": model_id, "revision": revision,
          "source_git_commit": source_commit,
          "file_count": sum(path.is_file() for path in root.rglob("*")),
          "total_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file())}
(root / "stage_completion.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
print(record)
PY
mkdir -p "$(dirname "$CHECKPOINT_ROOT")"
mv "$STAGING_ROOT" "$CHECKPOINT_ROOT"
test -s "${CHECKPOINT_ROOT}/stage_completion.json"
echo "[complete] ${CHECKPOINT_ROOT}"
