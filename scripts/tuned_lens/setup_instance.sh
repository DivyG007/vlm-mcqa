#!/bin/bash
# One-time setup on a Jarvislabs PyTorch container for Tuned Lens runs.
# Expects in $TL_ROOT (default /home/tuned_lens):
#   bundles/tuned_lens_<commit>.bundle                       git bundle of codex/divyanshu-tuned-lens
#   bundles/clevr_mcq4_v3_official3k_cuda45pairs_handoff_20261001.tar.gz
# Pins match the Ada Core-7 environment (torch 2.5.1+cu121, transformers 4.57.6, accelerate 1.15.0).
set -euo pipefail
ROOT=${TL_ROOT:-/home/tuned_lens}
cd "$ROOT"
mkdir -p logs runs hf_home data
BUNDLE=$(ls bundles/tuned_lens_*.bundle | tail -1)
if [ ! -d repo/.git ]; then
  git clone -q "$BUNDLE" repo
  git -C repo checkout -q codex/divyanshu-tuned-lens
fi
git -C repo rev-parse HEAD
git -C repo status --short

python3 -m pip install -q uv
python3 -m uv venv -q --python 3.12 venv
PIP="python3 -m uv pip install -q --python venv/bin/python"
$PIP torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$PIP -e "repo[models]" "transformers==4.57.6" "accelerate==1.15.0" "huggingface_hub==0.36.2" \
     "wandb==0.30.0" safetensors matplotlib scipy pytest
venv/bin/python -c "import torch, transformers, accelerate; print(torch.__version__, transformers.__version__, accelerate.__version__, torch.cuda.get_device_name(0))"

if [ ! -d data/clevr_v3/dataset ]; then
  mkdir -p data/clevr_v3
  tar -xzf bundles/clevr_mcq4_v3_official3k_cuda45pairs_handoff_20261001.tar.gz -C data/clevr_v3
fi
venv/bin/python repo/scripts/server/verify_clevr_official3k_release.py --dataset data/clevr_v3/dataset \
    --acquisition data/clevr_v3/acquisition --verify-existing > logs/release_verify.json
grep -q '"image_aware_release_hash": "eb7b79e56977bbaeed5fc981c5c8e09a4e26eaf8b1702283abe2d4bbfc718cbd"' logs/release_verify.json
(cd repo && venv_py=../venv/bin/python && $venv_py -m pytest -q -p no:cacheprovider tests/test_tuned_lens_pipeline.py) > logs/tests.log 2>&1
tail -1 logs/tests.log
if grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then echo "wandb: logged in"; else echo "wandb: NOT logged in (run: wandb login)"; fi
echo SETUP_DONE
