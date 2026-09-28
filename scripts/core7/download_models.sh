#!/usr/bin/env bash
# Optional: pre-fetch weights into HF_HOME so later runs can use HF_HUB_OFFLINE=1.
#   bash scripts/core7/download_models.sh                 # all twelve
#   bash scripts/core7/download_models.sh qwen25vl_3b ... # a subset
source "$(dirname "$0")/env.sh"
models=("$@"); [[ ${#models[@]} -eq 0 ]] && models=("${ALL_MODELS[@]}")
for key in "${models[@]}"; do
  "${PYTHON}" - "$key" <<'PY'
import sys
from huggingface_hub import snapshot_download
from vlm_mcqa.core7.adapters import load_model_specs
spec = load_model_specs()[sys.argv[1]]
# PyTorch loads safetensors/bin only; skip ONNX/GGUF/Flax/TF exports (12 GB in llava-onevision-0.5b-si).
path = snapshot_download(spec.hf_id, revision=spec.revision,
                         ignore_patterns=["onnx/*", "*.onnx", "*.onnx_data", "*.gguf", "*.msgpack", "*.h5", "*.tflite"])
print(spec.key, "->", path)
PY
done
