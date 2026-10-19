#!/bin/bash
#SBATCH -A cvit
#SBATCH --qos=normal
#SBATCH --partition=u22
#SBATCH --job-name=divyg_pali_core7
#SBATCH --nodelist=gnode068
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --output=/home2/divyg/anlp-vlm-mcqa/logs/divyg_pali_core7_%j.log
#SBATCH --error=/home2/divyg/anlp-vlm-mcqa/logs/divyg_pali_core7_%j.err

set -euo pipefail
set -x

case "$(hostname -s)" in
  gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077)
    echo "[FATAL] ineligible Ada node: $(hostname -s)" >&2
    exit 90
    ;;
esac

: "${SOURCE_GIT_COMMIT:?exact full source commit required}"
: "${SOURCE_SNAPSHOT:?immutable source snapshot required}"
: "${CAMPAIGN_ACTION:=calibrate}"
: "${PALIGEMMA_GPU_COUNT:=4}"
: "${NODE_STORAGE_ROOT:=/scratch/divyg}"
: "${VLM_ENV_ROOT:=${NODE_STORAGE_ROOT}/anlp-vlm-mcqa/envs/torch28cu126_tf4576}"

case "$CAMPAIGN_ACTION" in calibrate|run) ;; *) exit 2 ;; esac
case "$PALIGEMMA_GPU_COUNT" in 3|4) ;; *) exit 2 ;; esac

BASE=/home2/divyg/anlp-vlm-mcqa
REVISION=b26d16fb4251090ba4a4aa5af9fca1f8248ed5b6
MODEL_ID=google/paligemma2-10b-mix-448
CHECKPOINT_ROOT=${NODE_STORAGE_ROOT}/anlp-vlm-mcqa/checkpoints/paligemma2_10b_mix_${REVISION}
DATASET_ROOT=${BASE}/inputs/clevr_mcq4_v3_full
LAUNCH_ROOT=${BASE}/launch/paligemma2_core7_${CAMPAIGN_ACTION}_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}
RESULT_ROOT=${BASE}/results/paligemma2_core7_${CAMPAIGN_ACTION}_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}

test -s "${VLM_ENV_ROOT}/completion.json"
test -s "${CHECKPOINT_ROOT}/stage_completion.json"
test -s "${DATASET_ROOT}/dataset_manifest.json"
test -s "${DATASET_ROOT}/release_manifest.json"
test -d "$SOURCE_SNAPSHOT"
# The approved correct-only decision is provenance-bound to this historical
# screen. Keep it inside every immutable execution snapshot.
test -s "${SOURCE_SNAPSHOT}/consolidation/core7_release_results/screening/screening_table.csv"
test ! -e "$LAUNCH_ROOT"
test ! -e "$RESULT_ROOT"
mkdir -p "${BASE}/logs" "${BASE}/launch" "${BASE}/results" "$LAUNCH_ROOT"

module load u22/cuda/12.1
export PATH="${VLM_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${SOURCE_SNAPSHOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SOURCE_SNAPSHOT"

python - "$PALIGEMMA_GPU_COUNT" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == expected, (torch.cuda.device_count(), expected)
names = [torch.cuda.get_device_name(i) for i in range(expected)]
if expected == 4 and __import__("socket").gethostname().split(".", 1)[0] == "gnode068":
    # Ada exposes gnode068 as one 12-GB RTX 4070 followed by three 11-GB
    # RTX 2080 Ti devices. The launcher exercises a real kernel on every GPU
    # below, and calibration is repeated inside this run before any science.
    assert sum("RTX 4070" in name for name in names) == 1, names
    assert sum("RTX 2080 Ti" in name for name in names) == 3, names
else:
    assert all("RTX 2080 Ti" in name for name in names), names
for index in range(expected):
    x = torch.arange(64, device=f"cuda:{index}", dtype=torch.bfloat16)
    assert torch.isfinite(x.float()).all()
print({"visible_gpus": expected, "names": names, "bfloat16_kernel": "passed"})
PY

# PaliGemma treats its first physical newline as the end of the VQA query.
# Exercise the production renderer before model loading and fail unless the
# complete question, all four options, and the answer instruction are on one
# line. The adapter repeats this check immediately before processor invocation.
python - <<'PY'
from vlm_mcqa.core7.adapters import paligemma_mcq_prompt

common = (
    "What color is the cube?\nA. red\nB. blue\nC. green\nD. yellow\n"
    "Respond with only the option label.\nAnswer:"
)
rendered = paligemma_mcq_prompt(common, "answer en ")
assert "\n" not in rendered and "\r" not in rendered
assert all(text in rendered for text in ("What color", "A. red", "D. yellow", "only the option label"))
print({"paligemma_prompt_preflight": "passed", "physical_lines": 1})
PY

python -m vlm_mcqa.campaigns inventory --root "$SOURCE_SNAPSHOT" \
  --output "${LAUNCH_ROOT}/source_inventory.json" --kind source \
  --commit "$SOURCE_GIT_COMMIT"
python -m vlm_mcqa.campaigns inventory --root "$CHECKPOINT_ROOT" \
  --output "${LAUNCH_ROOT}/checkpoint_inventory.json" --kind checkpoint \
  --hf-id "$MODEL_ID" --revision "$REVISION"

python - "$SOURCE_SNAPSHOT" "$CHECKPOINT_ROOT" "$DATASET_ROOT" \
  "$LAUNCH_ROOT" "$RESULT_ROOT" "$PALIGEMMA_GPU_COUNT" <<'PY'
import json
from pathlib import Path
import sys

source, checkpoint, data, launch, result = map(Path, sys.argv[1:6])
gpu_count = int(sys.argv[6])

mapping = {
    "model.vision_tower": "cuda:0",
    "model.multi_modal_projector": "cuda:0",
    "model.language_model.embed_tokens": "cuda:0",
    "model.language_model.rotary_emb": "cuda:0",
    "model.language_model.norm": f"cuda:{gpu_count - 1}",
    # Gemma/PaliGemma ties the input embedding and output-head weights. They
    # must have one shared device-map destination; assigning them to different
    # GPUs makes Accelerate relocate the shared weight behind the input IDs and
    # the first embedding lookup fails with a cuda:0/cuda:N mismatch.
    "lm_head": "cuda:0",
}
if gpu_count == 3:
    # GPU 0 also owns vision, projection, and embeddings, so it receives fewer
    # decoder blocks. This is the placement that previously loaded the full
    # 42-layer checkpoint; the campaign readout now safely crosses norm/head
    # device boundaries.
    cuts = (10, 26)
    for layer in range(42):
        device = 0 if layer <= cuts[0] else (1 if layer <= cuts[1] else 2)
        mapping[f"model.language_model.layers.{layer}"] = f"cuda:{device}"
else:
    # Discovery preparation keeps processor/decoder inputs resident in the
    # prepared-example LRU. Cuts (5, 18, 30) eventually filled GPU 0 and
    # failed on a 2 MiB allocation even though calibration passed. Move two
    # more blocks away from the GPU that owns vision, projection, tied
    # embeddings, and lm_head; GPU 1 had more than 4 GiB of headroom.
    cuts = (3, 17, 30)
    for layer in range(42):
        device = 0 if layer <= cuts[0] else (1 if layer <= cuts[1] else (2 if layer <= cuts[2] else 3))
        mapping[f"model.language_model.layers.{layer}"] = f"cuda:{device}"

config = {
    "schema": "vlm-mcqa-campaign/v1",
    "kind": "core7_correct_only_reduced",
    "seed": 0,
    "output_dir": str(result),
    "source": {
        "root": str(source),
        "manifest": str(launch / "source_inventory.json"),
    },
    "model": {
        "key": "paligemma2_10b_mix",
        "models_config": str(source / "configs/core7/next_three_models_pinned.json"),
        "checkpoint_dir": str(checkpoint),
        "checkpoint_manifest": str(launch / "checkpoint_inventory.json"),
    },
    "data": {
        "directory": str(data),
        "dataset_hash": "f5573e25f3aa2bf2",
        "image_aware_release_hash": "eb7b79e56977bbaeed5fc981c5c8e09a4e26eaf8b1702283abe2d4bbfc718cbd",
    },
    "deployment": {
        "dtype": "bfloat16",
        "device_map": mapping,
        "attn_implementation": "eager",
        "offload_policy": "forbid",
        "batch_size": 1,
        # PaliGemma prepared examples retain decoder tensors on their mapped
        # GPUs. A one-entry LRU is slower but prevents multi-hour GPU-memory
        # accumulation during correct-only case discovery.
        "prepared_cache": 1,
        "clean_cache": 128,
    },
    "profile": "pali_correct_only_reduced",
    "profiles_config": str(source / "configs/core7/profiles.json"),
    "protocol": {
        "selection_mode": "preselected_correct_only_sensitivity",
        "decision_path": str(source / "configs/campaigns/paligemma2_correct_only_decision_2026-10-27.json"),
        "decision_sha256": "73129aef06bc7b57236aaa827aa61cb3c3277b00bd2399a5d42e47d7dde358a4",
    },
}
(launch / "campaign.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
(launch / "placement.json").write_text(json.dumps({
    "schema": "divyg.paligemma2-placement.v1",
    "gpu_count": gpu_count,
    "device_map": mapping,
    "attention": "eager_with_gemma2_softcap_and_sliding_masks",
    "prompt_policy": "complete_mcq_on_one_physical_line_before_processor",
}, sort_keys=True, indent=2) + "\n")
PY

python -m vlm_mcqa.campaigns verify --config "${LAUNCH_ROOT}/campaign.json"
python -m vlm_mcqa.campaigns "$CAMPAIGN_ACTION" \
  --config "${LAUNCH_ROOT}/campaign.json" --execute

python - "$LAUNCH_ROOT" "$RESULT_ROOT" "$CAMPAIGN_ACTION" <<'PY'
import json
from pathlib import Path
import sys

launch, result = map(Path, sys.argv[1:3])
action = sys.argv[3]
required = ["attempt.json", "stdout.log", "stderr.log", "artifact_manifest.json"]
if action == "calibrate":
    required += ["calibration_result.json", "audit.json"]
else:
    required += ["completion.json", "audit.json", "freeze.json"]
missing = [name for name in required if not (result / name).is_file()]
if missing:
    raise SystemExit(f"missing final PaliGemma artifacts: {missing}")
record = {
    "schema": "divyg.paligemma2-core7-launch.v1",
    "status": "complete",
    "action": action,
    "result_root": str(result),
    "required_artifacts": required,
}
(launch / "completion.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
print(record)
PY

echo "[complete] PaliGemma ${CAMPAIGN_ACTION}: ${RESULT_ROOT}"
