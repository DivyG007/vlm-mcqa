#!/bin/bash
# Run cache -> train -> evaluate for one model on the instance. Launch inside tmux.
# Every phase is resumable: re-running after a spot pre-emption continues where it stopped.
# Usage: run_model.sh <model_key> <run_name> [--pilot]
set -euo pipefail
MODEL=$1; RUN=$2; shift 2
ROOT=${TL_ROOT:-/home/tuned_lens}
cd "$ROOT/repo"
export HF_HOME=$ROOT/hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY=$ROOT/venv/bin/python
OUT=$ROOT/runs/$MODEL/$RUN
mkdir -p "$OUT"
COMMON=(--model "$MODEL" --data "$ROOT/data/clevr_v3/dataset" --out "$OUT" --hf-cache "$HF_HOME" "$@")
EVAL_EXTRA=()
if [[ " $* " == *" --pilot "* ]]; then EVAL_EXTRA=(--n-items 16); fi
{
  echo "start=$(date -Is) host=$(hostname) commit=$(git rev-parse HEAD) model=$MODEL run=$RUN args=$*"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  df -h "$ROOT" | tail -1
} | tee -a "$OUT/run.log"
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed "s/^/$(date -Is) /"; sleep 120; done ) >> "$OUT/gpu.log" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
for PHASE in cache train evaluate; do
  echo "== $PHASE $(date -Is)" | tee -a "$OUT/run.log"
  if [ "$PHASE" = evaluate ]; then
    $PY -u -m vlm_mcqa.tuned_lens evaluate "${COMMON[@]}" "${EVAL_EXTRA[@]}" 2>&1 | tee -a "$OUT/run.log"
  else
    $PY -u -m vlm_mcqa.tuned_lens "$PHASE" "${COMMON[@]}" 2>&1 | tee -a "$OUT/run.log"
  fi
done
if [[ " $* " == *" --pilot "* ]]; then
  $PY -m vlm_mcqa.tuned_lens estimate "${COMMON[@]}" 2>&1 | tee -a "$OUT/run.log"
fi
echo "== end $(date -Is)" | tee -a "$OUT/run.log"
