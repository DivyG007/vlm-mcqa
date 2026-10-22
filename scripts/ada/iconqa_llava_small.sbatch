#!/usr/bin/env bash
#SBATCH --job-name=iconqa_llava_small
#SBATCH --account=cvit
#SBATCH --partition=u22
#SBATCH --qos=normal
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
set -euo pipefail
: "${SOURCE_ROOT:?}" "${DATASET_ROOT:?}" "${CHECKPOINT_ROOT:?}" "${OUTPUT_ROOT:?}" "${PILOT_EVIDENCE:?}"
if [[ "${WAIT_FOR_STAGING:-0}" == 1 ]]; then
  echo "Waiting for gateway checkpoint transfer to $CHECKPOINT_ROOT"
  for ((waited=0; waited<1200; waited+=2)); do
    [[ -f "$CHECKPOINT_ROOT/LOCAL_STAGE_FAILED" ]] && exit 70
    [[ -f "$CHECKPOINT_ROOT/LOCAL_STAGE_READY" ]] && break
    sleep 2
  done
  [[ -f "$CHECKPOINT_ROOT/LOCAL_STAGE_READY" ]] || exit 71
fi
source /etc/profile.d/modules.sh
module load u22/lammps/30Sep2026 u22/deepmd/3.2.0
unset PYTHONHOME
export PYTHONPATH="$SOURCE_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CORE7_WANDB_ENTITY=models-iiit-hyderabad5465 CORE7_WANDB_PROJECT=anlp-vlm-mcqa-staging
export WANDB_MODE="${WANDB_MODE:-online}"
export SOURCE_SNAPSHOT_MANIFEST="$SOURCE_ROOT/source_snapshot.json"
export SOURCE_GIT_COMMIT=$(/home2/astroduck/envs/anlp-vlm-mcqa/shared_torch210_overlay_v1/bin/python -c 'import json,os;print(json.load(open(os.environ["SOURCE_SNAPSHOT_MANIFEST"]))["source_git_commit"])')
nvidia-smi --query-gpu=index,name,memory.total --format=csv
RESUME_ARGS=()
[[ "${RESUME:-0}" == 1 ]] && RESUME_ARGS=(--resume)
exec /home2/astroduck/envs/anlp-vlm-mcqa/shared_torch210_overlay_v1/bin/python \
 "$SOURCE_ROOT/scripts/ada/iconqa_llava_small.py" --pilot-evidence "$PILOT_EVIDENCE" \
 --source-root "$SOURCE_ROOT" --dataset "$DATASET_ROOT" --checkpoint "$CHECKPOINT_ROOT" \
 --output "$OUTPUT_ROOT" --execute "${RESUME_ARGS[@]}"
