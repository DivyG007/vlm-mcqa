#!/usr/bin/env bash
# Difficulty calibration on the unreported 120-item calibration split.
# Representative 2-3B, 7-8B and ~10B models by default; results are NOT for the paper.
#   bash scripts/core7/calibrate_difficulty.sh [model_key ...]
source "$(dirname "$0")/env.sh"
models=("$@"); [[ ${#models[@]} -eq 0 ]] && models=(qwen2vl_2b internvl35_2b qwen25vl_7b paligemma2_10b_mix)
stamp="$(date +%Y%m%d_%H%M%S)"
for key in "${models[@]}"; do
  out="${RESULTS_ROOT}/dataset_calibration/${key}_${stamp}"
  run_logged "${out}" core7 screen --model "${key}" --data "${CLEVR_MCQ4}" --out "${out}" \
    --profile "${PROFILE:-full_core7}" --split calibration ${EXTRA_ARGS:-}
done
"${PYTHON}" - "${RESULTS_ROOT}/dataset_calibration" "${stamp}" <<'PY'
import json, sys
from pathlib import Path
for path in sorted(Path(sys.argv[1]).glob(f"*_{sys.argv[2]}/behavior_calibration_summary.json")):
    s = json.loads(path.read_text())
    print(f"{path.parent.name:45s} acc={s['letters_accuracy']:.3f} worst_pos={s['worst_position_accuracy']:.3f} "
          f"shuffled={s['shuffled_accuracy']:.3f} blank={s['blank_accuracy']:.3f}")
print("Target: small models not all >90%, some 3-8B models >90%, big shuffled/blank drop, 10B strong but <100%.")
PY
