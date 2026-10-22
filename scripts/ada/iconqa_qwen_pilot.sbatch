#!/usr/bin/env bash
# Use submit_iconqa_pilot.py: it derives valid node exclusions from sinfo.
#SBATCH --job-name=iconqa_qwen_pilot
#SBATCH --partition=u22
#SBATCH --qos=normal
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
set -euo pipefail
: "${SOURCE_ROOT:?}"
: "${DATASET_ROOT:?}"
: "${CHECKPOINT_ROOT:?}"
: "${OUTPUT_ROOT:?}"
SOURCE_ROOT=$(realpath "$SOURCE_ROOT")
PY=/home2/astroduck/envs/anlp-vlm-mcqa/shared_torch210_overlay_v1/bin/python
source /etc/profile.d/modules.sh
module load u22/lammps/30Sep2026 u22/deepmd/3.2.0
unset PYTHONHOME
export PYTHONPATH="$SOURCE_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CORE7_WANDB_ENTITY="${CORE7_WANDB_ENTITY:-models-iiit-hyderabad5465}"
export CORE7_WANDB_PROJECT="${CORE7_WANDB_PROJECT:-anlp-vlm-mcqa-staging}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "${ICONQA_RESUME:-0}" != 1 ]]; then test ! -e "$OUTPUT_ROOT"; fi
echo "[pilot] job=$SLURM_JOB_ID node=$(hostname -s) visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
# The focused CPU fixture gate runs once before staging; do not repeat it here.
export SOURCE_SNAPSHOT_MANIFEST="$SOURCE_ROOT/source_snapshot.json"
export SOURCE_GIT_COMMIT=$("$PY" -c 'import json,os;print(json.load(open(os.environ["SOURCE_SNAPSHOT_MANIFEST"]))["source_git_commit"])')
ENTRY="${CAMPAIGN_ENTRY:-iconqa_qwen_pilot.py}"
case "$ENTRY" in iconqa_qwen_pilot.py|iconqa_qwen_campaign.py|iconqa_qwen_small.py) ;; *) exit 2;; esac
EXTRA=()
if [[ "${ICONQA_RESUME:-0}" == 1 ]]; then EXTRA+=(--resume); fi
if [[ "${ICONQA_BATCH_BENCHMARK:-0}" == 1 ]]; then EXTRA+=(--batch-benchmark); fi
if [[ "$ENTRY" == iconqa_qwen_campaign.py ]]; then
  EXTRA+=(--batch-size "${ICONQA_BATCH_SIZE:-1}")
  if [[ -n "${ICONQA_BATCH_EVIDENCE:-}" ]]; then EXTRA+=(--batch-evidence "$ICONQA_BATCH_EVIDENCE"); fi
fi
if [[ "$ENTRY" == iconqa_qwen_small.py ]]; then
  EXTRA+=(--batch-size "${ICONQA_BATCH_SIZE:-4}")
  if [[ "${ICONQA_CALIBRATION_ONLY:-0}" == 1 ]]; then EXTRA+=(--calibration-only); fi
fi
"$PY" "$SOURCE_ROOT/scripts/ada/$ENTRY" "${EXTRA[@]}" \
  --source-root "$SOURCE_ROOT" --dataset "$DATASET_ROOT" --checkpoint "$CHECKPOINT_ROOT" \
  --output "$OUTPUT_ROOT" --execute
