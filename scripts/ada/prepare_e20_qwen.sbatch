#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=rachit_e20_qwen_prep
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0-01:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e20_qwen_prep_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e20_qwen_prep_%j.err

set -euo pipefail
set -x

case "$(hostname -s)" in
  gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077)
    echo "[FATAL] ineligible Ada node: $(hostname -s)" >&2
    exit 90
    ;;
esac

: "${SOURCE_GIT_COMMIT:?SOURCE_GIT_COMMIT is required}"
: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
REVISION="cc594898137f460bfe9f0759e9844b3ce807cfb5"
BASE="/home2/rsm/anlp-vlm-mcqa"
CHECKPOINT_ROOT="${BASE}/checkpoints/qwen25vl_7b_${REVISION}"
DATASET_ROOT="${BASE}/inputs/clevr_mcq4_v3_full"
OUTPUT_ROOT="${BASE}/prepared/e20_qwen_${SOURCE_GIT_COMMIT}_${REVISION}"
ATTEMPT_BASE="${BASE}/results"

source /home2/rsm/miniconda3/etc/profile.d/conda.sh
conda activate motion-probe
export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${SOURCE_ROOT}"

test -s "${CHECKPOINT_ROOT}/stage_completion.json"
test -s "${DATASET_ROOT}/dataset_manifest.json"
test -s "${DATASET_ROOT}/release_manifest.json"
python -m unittest tests.test_prepare_e20_qwen_campaign -v
python scripts/local/prepare_e20_qwen_campaign.py \
  --source-root "${SOURCE_ROOT}" \
  --source-commit "${SOURCE_GIT_COMMIT}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --attempt-base "${ATTEMPT_BASE}"
test -s "${OUTPUT_ROOT}/completion.json"
echo "[complete] ${OUTPUT_ROOT}"
