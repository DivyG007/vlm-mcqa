#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=rachit_e20_env
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=0-01:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e20_env_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e20_env_%j.err

set -euo pipefail
set -x

case "$(hostname -s)" in
  gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077)
    echo "[FATAL] ineligible Ada node: $(hostname -s)" >&2
    exit 90
    ;;
esac

BASE="/home2/rsm/anlp-vlm-mcqa"
ENV_ROOT="${BASE}/envs/e20_qwen_torch28_tf4576_${SLURM_JOB_ID}"
WHEEL_ROOT="${BASE}/inputs/e20_wheels_20261026"
mkdir -p "${BASE}/envs" "${BASE}/logs"
if test -e "${ENV_ROOT}"; then
  echo "[FATAL] fresh environment path required: ${ENV_ROOT}" >&2
  exit 4
fi

source /home2/rsm/miniconda3/etc/profile.d/conda.sh
conda activate motion-probe
python - "${WHEEL_ROOT}" <<'PY'
import hashlib
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {
    "transformers-4.57.6-py3-none-any.whl": "4c9e9de11333ddfe5114bc872c9f370509198acf0b87a832a0ab9458e2bd0550",
    "huggingface_hub-0.36.0-py3-none-any.whl": "7bcc9ad17d5b3f07b57c78e79d527102d08313caa278a641993acddcb894548d",
}
for name, wanted in expected.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
        raise SystemExit(f"missing or changed wheel: {path}")
print("[wheelhouse] exact hashes verified")
PY
python -m venv --system-site-packages "${ENV_ROOT}"
"${ENV_ROOT}/bin/python" -m pip install --disable-pip-version-check --no-index \
  --find-links "${WHEEL_ROOT}" \
  'transformers==4.57.6' 'huggingface_hub==0.36.0'
"${ENV_ROOT}/bin/python" - <<'PY'
import accelerate
import huggingface_hub
import safetensors
import torch
import torchvision
import transformers
from transformers import AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration

assert transformers.__version__ == "4.57.6", transformers.__version__
assert huggingface_hub.__version__ == "0.36.0", huggingface_hub.__version__
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("accelerate", accelerate.__version__)
print("safetensors", safetensors.__version__)
print("auto_class", AutoModelForImageTextToText.__name__)
print("qwen_class", Qwen2_5_VLForConditionalGeneration.__name__)
PY
"${ENV_ROOT}/bin/python" -m pip freeze > "${ENV_ROOT}/requirements.freeze.txt"
"${ENV_ROOT}/bin/python" - "${ENV_ROOT}/completion.json" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

record = {
    "schema": "rachit.e20-qwen-environment.v1", "status": "complete",
    "packages": {name: metadata.version(name) for name in (
        "torch", "torchvision", "transformers", "huggingface-hub", "accelerate", "safetensors"
    )},
    "inheritance": "motion-probe system site packages; transformers/hub pinned in project venv",
}
Path(sys.argv[1]).write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
PY
test -s "${ENV_ROOT}/completion.json"
echo "[complete] ${ENV_ROOT}"
