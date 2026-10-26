#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=next3_stretch
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/next3_%x_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/next3_%x_%j.err

set -euo pipefail
set -x

case "$(hostname -s)" in
  gnode065|gnode066|gnode077) echo "known-broken allocated node" >&2; exit 90 ;;
esac

: "${CAMPAIGN_CONFIG:?absolute immutable campaign JSON required}"
: "${CAMPAIGN_ACTION:?use verify, calibrate, pilot, or run}"
: "${SOURCE_SNAPSHOT:?absolute committed source snapshot required}"
case "$CAMPAIGN_ACTION" in verify|calibrate|pilot|run) ;; *) exit 2 ;; esac

module load u22/cuda/12.1
source /home2/rsm/miniconda3/etc/profile.d/conda.sh
conda activate vlm-mcqa
mkdir -p /home2/rsm/anlp-vlm-mcqa/logs
cd "$SOURCE_SNAPSHOT"
export PYTHONPATH="$SOURCE_SNAPSHOT/src:${PYTHONPATH:-}"
export HF_HOME="/scratch/rsm/hf_cache"
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable in campaign environment"
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
print({"visible_gpus": torch.cuda.device_count(), "cuda": torch.version.cuda})
PY

python -m vlm_mcqa.campaigns.cli verify --config "$CAMPAIGN_CONFIG"
if [[ "$CAMPAIGN_ACTION" != verify ]]; then
  python -m vlm_mcqa.campaigns.cli "$CAMPAIGN_ACTION" --config "$CAMPAIGN_CONFIG" --execute
fi
