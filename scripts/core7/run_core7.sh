#!/usr/bin/env bash
# Full Core-7 for one model selected by screen_models.sh:
# discovery -> freeze -> confirmation -> confirm -> plots.
#   bash scripts/core7/run_core7.sh qwen25vl_3b
#   PROFILE=smoke bash scripts/core7/run_core7.sh qwen25vl_3b     # same code, tiny sizes
source "$(dirname "$0")/env.sh"
key="${1:?model key}"
PROFILE="${PROFILE:-full_core7}"
base="${RESULTS_ROOT}/core7/${key}_${PROFILE}"
disc="${base}/discovery"; conf="${base}/confirmation"
if [[ "${PROFILE}" == "full_core7" ]]; then
  screen_root="${SCREEN_ROOT:-${RESULTS_ROOT}/screening}"
  "${PYTHON}" "$(dirname "$0")/validate_screening.py" --data "${CLEVR_MCQ4}" \
    --screen-root "${screen_root}" --dataset-config "${DATASET_CONFIG}" \
    --all --selection "${screen_root}/_selection/selection.json"
  "${PYTHON}" - "${screen_root}/_selection/selection.json" "${key}" <<'PY'
import json, sys
from pathlib import Path
path, model = Path(sys.argv[1]), sys.argv[2]
if not path.is_file() or model not in json.loads(path.read_text())["core7_models"]:
    raise SystemExit(f"[stop] {model} is not one of the two frozen selection models in {path}")
PY
fi
if [[ -e "${base}" ]]; then
  echo "[stop] ${base} already exists; inspect it before a retry and archive it outside the active results tree" >&2
  exit 2
fi

run_logged "${disc}" core7 run --model "${key}" --data "${CLEVR_MCQ4}" --out "${disc}" \
  --profile "${PROFILE}" --split discovery ${EXTRA_ARGS:-}
core7 freeze --run "${disc}" --out "${base}/freeze.json"
run_logged "${conf}" core7 run --model "${key}" --data "${CLEVR_MCQ4}" --out "${conf}" \
  --profile "${PROFILE}" --split confirmation --freeze "${base}/freeze.json" \
  --methods patching,components,heads ${EXTRA_ARGS:-}
core7 confirm --run "${conf}" --freeze "${base}/freeze.json"
core7 plot --run "${conf}" >/dev/null
echo "[done] ${base}"
