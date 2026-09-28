#!/usr/bin/env bash
# Cross-model comparison over completed Core-7 runs.
#   bash scripts/core7/compare_models.sh qwen25vl_7b internvl35_8b
source "$(dirname "$0")/env.sh"
PROFILE="${PROFILE:-full_core7}"
runs=()
for key in "$@"; do
  for split in discovery confirmation; do
    d="${RESULTS_ROOT}/core7/${key}_${PROFILE}/${split}"
    [[ -d "$d" ]] && runs+=("$d")
  done
done
core7 compare --runs "${runs[@]}" --out "${RESULTS_ROOT}/comparison_${PROFILE}"
