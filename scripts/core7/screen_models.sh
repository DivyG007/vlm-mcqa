#!/usr/bin/env bash
# Screen all twelve candidates (even after a smaller one passes), then select two for Core-7.
#   bash scripts/core7/screen_models.sh              # all twelve, sequentially on the visible GPU
#   bash scripts/core7/screen_models.sh qwen25vl_3b  # a subset
source "$(dirname "$0")/env.sh"
models=("$@"); [[ ${#models[@]} -eq 0 ]] && models=("${ALL_MODELS[@]}")
SCREEN_ROOT="${SCREEN_ROOT:-${RESULTS_ROOT}/screening}"
validator="$(dirname "$0")/validate_screening.py"
profile="${PROFILE:-full_core7}"
if [[ "${profile}" != "full_core7" ]]; then
  echo "[stop] screen_models.sh is for the frozen full_core7 screen; run 'python -m vlm_mcqa.core7 screen --profile smoke' directly for a smoke screen" >&2
  exit 2
fi
"${PYTHON}" "${validator}" --data "${CLEVR_MCQ4}" --screen-root "${SCREEN_ROOT}" --profile "${profile}" --dataset-config "${DATASET_CONFIG}"
for key in "${models[@]}"; do
  out="${SCREEN_ROOT}/${key}"
  if [[ -e "${out}" ]]; then
    if "${PYTHON}" "${validator}" --data "${CLEVR_MCQ4}" --screen-root "${SCREEN_ROOT}" \
        --profile "${profile}" --dataset-config "${DATASET_CONFIG}" --model "${key}"; then
      echo "[skip] ${key} already screened against frozen inputs"; continue
    fi
    echo "[stop] ${out} exists but is incomplete or stale; archive it outside SCREEN_ROOT before retry" >&2
    exit 2
  fi
  run_logged "${out}" core7 screen --model "${key}" --data "${CLEVR_MCQ4}" --out "${out}" \
    --profile "${profile}" ${EXTRA_ARGS:-}
  "${PYTHON}" "${validator}" --data "${CLEVR_MCQ4}" --screen-root "${SCREEN_ROOT}" \
    --profile "${profile}" --dataset-config "${DATASET_CONFIG}" --model "${key}"
done
"${PYTHON}" "${validator}" --data "${CLEVR_MCQ4}" --screen-root "${SCREEN_ROOT}" \
  --profile "${profile}" --dataset-config "${DATASET_CONFIG}" --all
core7 select --screen-root "${SCREEN_ROOT}" --out "${SCREEN_ROOT}/_selection"
"${PYTHON}" "${validator}" --data "${CLEVR_MCQ4}" --screen-root "${SCREEN_ROOT}" \
  --profile "${profile}" --dataset-config "${DATASET_CONFIG}" --all --selection "${SCREEN_ROOT}/_selection/selection.json"
