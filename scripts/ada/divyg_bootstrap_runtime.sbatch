#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --job-name=divyg_vlm_env
#SBATCH --nodelist=gnode068
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home2/divyg/anlp-vlm-mcqa/logs/divyg_vlm_env_%j.log
#SBATCH --error=/home2/divyg/anlp-vlm-mcqa/logs/divyg_vlm_env_%j.err

set -euo pipefail
set -x

: "${SOURCE_GIT_COMMIT:?exact full source commit required}"
echo "SOURCE_GIT_COMMIT=${SOURCE_GIT_COMMIT}"

case "$(hostname -s)" in
  gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077)
    echo "[FATAL] ineligible Ada node: $(hostname -s)" >&2
    exit 90
    ;;
esac

NODE_STORAGE_ROOT=${NODE_STORAGE_ROOT:-/scratch/divyg}
ENV_ROOT=${VLM_ENV_ROOT:-${NODE_STORAGE_ROOT}/anlp-vlm-mcqa/envs/torch28_tf4576}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}
TORCH_BUILD_SUFFIX=${TORCH_BUILD_SUFFIX:-cu126}
mkdir -p "$(dirname "$ENV_ROOT")" /home2/divyg/anlp-vlm-mcqa/logs
if test -s "${ENV_ROOT}/completion.json"; then
  echo "[complete] existing verified environment: ${ENV_ROOT}"
  exit 0
fi
if test -e "$ENV_ROOT"; then
  echo "[FATAL] partial environment requires an explicit new VLM_ENV_ROOT" >&2
  exit 4
fi

module load u22/cuda/12.1
python3 -m venv "$ENV_ROOT"
"${ENV_ROOT}/bin/python" -m pip install --disable-pip-version-check --upgrade pip wheel
"${ENV_ROOT}/bin/python" -m pip install --disable-pip-version-check \
  --index-url "$TORCH_INDEX_URL" \
  'torch==2.8.0' 'torchvision==0.23.0'
"${ENV_ROOT}/bin/python" -m pip install --disable-pip-version-check \
  'transformers==4.57.6' 'huggingface_hub==0.36.0' 'accelerate==1.10.1' \
  'safetensors>=0.4.5' 'datasets>=2.20' 'matplotlib>=3.8' 'Pillow>=10' \
  'sentencepiece>=0.2' 'protobuf>=5' 'einops>=0.8' 'scipy>=1.13'
"${ENV_ROOT}/bin/python" - "$TORCH_BUILD_SUFFIX" <<'PY'
import json
from pathlib import Path
import sys
import accelerate, huggingface_hub, safetensors, torch, torchvision, transformers

expected_suffix = sys.argv[1]
assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torch.__version__.endswith("+" + expected_suffix), torch.__version__
assert torchvision.__version__.startswith("0.23.0"), torchvision.__version__
assert transformers.__version__ == "4.57.6", transformers.__version__
assert huggingface_hub.__version__ == "0.36.0", huggingface_hub.__version__
root = Path(__import__("sys").prefix)
record = {
    "schema": "divyg.vlm-runtime.v1", "status": "complete",
    "node": __import__("socket").gethostname(),
    "torch_build_suffix": expected_suffix,
    "packages": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                 "transformers": transformers.__version__,
                 "huggingface_hub": huggingface_hub.__version__,
                 "accelerate": accelerate.__version__, "safetensors": safetensors.__version__},
}
(root / "completion.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
print(record)
PY
"${ENV_ROOT}/bin/python" -m pip freeze > "${ENV_ROOT}/requirements.freeze.txt"
test -s "${ENV_ROOT}/completion.json"
echo "[complete] ${ENV_ROOT}"
