#!/usr/bin/env bash
# Create a Python environment for Core-7 on a GPU server (no Slurm assumed).
#   VENV=/path/to/venv TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/core7/setup_environment.sh
# Afterwards: export PYTHON=/path/to/venv/bin/python (or put it in scripts/core7/local.env).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "${here}/../.." && pwd)"
VENV="${VENV:-${root}/.venv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
python3 -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install torch torchvision --index-url "${TORCH_INDEX}"
"${VENV}/bin/pip" install -e "${root}[models]"

"${VENV}/bin/python" - <<'PY'
import torch, torchvision, transformers
print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("transformers", transformers.__version__)
from transformers import (Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration,
    PaliGemmaForConditionalGeneration, LlavaOnevisionForConditionalGeneration, InternVLForConditionalGeneration)
print("all four VLM families available natively")
PY
echo "W&B: run '${VENV}/bin/wandb login' once (runs log to team vlm-mcqa, project smoke-testing; pass --no-wandb to skip)."
echo "PaliGemma checkpoints are gated: accept the license on huggingface.co, then run 'huggingface-cli login' or export HF_TOKEN."
echo "Set PYTHON=${VENV}/bin/python in scripts/core7/local.env"
