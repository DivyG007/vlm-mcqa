#!/usr/bin/env bash
# Full Core-7 for a model named in advance, before the twelve-model screen.
# Results live separately and do not constitute a screening selection.
source "$(dirname "$0")/env.sh"
key="${1:?model key}"
case "${key}" in
  qwen25vl_7b|internvl35_8b) ;;
  *) echo "[stop] ${key} is not in the preselected pair" >&2; exit 2 ;;
esac
PROFILE=full_core7
base="${RESULTS_ROOT}/preselected/${key}_${PROFILE}"
disc="${base}/discovery"; conf="${base}/confirmation"
common_args=()
[[ -n "${CORE7_MODELS_CONFIG:-}" ]] && common_args+=(--models-config "${CORE7_MODELS_CONFIG}")
[[ -n "${CORE7_DEVICE_MAP:-}" ]] && common_args+=(--device-map "${CORE7_DEVICE_MAP}")
[[ -n "${CORE7_BATCH_SIZE:-}" ]] && common_args+=(--batch-size "${CORE7_BATCH_SIZE}")
"${PYTHON}" "$(dirname "$0")/validate_screening.py" --data "${CLEVR_MCQ4}" \
  --screen-root "${RESULTS_ROOT}/screening" --dataset-config "${DATASET_CONFIG}"
if [[ -e "${base}" ]]; then
  echo "[stop] ${base} already exists; inspect and archive it before retrying" >&2
  exit 2
fi

check_gpu_idle() {
  local selected="${CORE7_GPU_INDICES:-${CORE7_GPU_INDEX:-}}"
  [[ -z "${selected}" ]] && return 0
  local gpu processes
  local -a gpus
  IFS=',' read -r -a gpus <<< "${selected}"
  for gpu in "${gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ -z "${gpu}" ]] && { echo "[stop] empty GPU entry in ${selected}" >&2; exit 3; }
    processes="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader)"
    if [[ -n "${processes}" ]]; then
      echo "[stop] physical GPU ${gpu} has another compute process: ${processes}" >&2
      exit 3
    fi
  done
}

check_gpu_idle
run_logged "${disc}" core7 run --model "${key}" --data "${CLEVR_MCQ4}" --out "${disc}" \
  --profile "${PROFILE}" --split discovery "${common_args[@]}" ${EXTRA_ARGS:-}
core7 freeze --run "${disc}" --out "${base}/freeze.json"
check_gpu_idle
run_logged "${conf}" core7 run --model "${key}" --data "${CLEVR_MCQ4}" --out "${conf}" \
  --profile "${PROFILE}" --split confirmation --freeze "${base}/freeze.json" \
  --methods patching,components,heads "${common_args[@]}" ${EXTRA_ARGS:-}
core7 confirm --run "${conf}" --freeze "${base}/freeze.json"
core7 plot --run "${conf}" >/dev/null
echo "[done] ${base}"
