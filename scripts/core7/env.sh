#!/usr/bin/env bash
# Shared settings for the Core-7 server scripts. Source it; override any
# variable in the environment before sourcing (or in scripts/core7/local.env).
#
#   CORE7_ROOT     repository checkout (default: two levels above this file)
#   DATA_ROOT      datasets and renders            (default: $CORE7_ROOT/data, gitignored)
#   RESULTS_ROOT   run directories                 (default: $CORE7_ROOT/runs, gitignored)
#   CLEVR_MCQ4     finalized dataset directory     (default: $DATA_ROOT/clevr_mcq4_v3)
#   DATASET_CONFIG frozen CLEVR dataset policy     (default: configs/core7/clevr_mcq4_official3k.json)
#   HF_HOME        Hugging Face cache for weights  (default: $HOME/.cache/huggingface)
#   PYTHON         interpreter inside the prepared environment (default: python)
#   CUDA_VISIBLE_DEVICES  choose the GPU(s) yourself; nothing here overrides it
#   CORE7_GPU_INDICES physical indices/UUIDs, comma separated, that must be idle
#   EXTRA_ARGS     extra CLI flags for every run, e.g. "--no-wandb" or "--batch-size 4"
#
# W&B defaults to team "vlm-mcqa", project "smoke-testing". Override with
# CORE7_WANDB_ENTITY and CORE7_WANDB_PROJECT. CORE7_NO_WANDB=1 disables it;
# WANDB_MODE=offline logs locally.

set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_here}/local.env" ]] && source "${_here}/local.env"

export CORE7_ROOT="${CORE7_ROOT:-$(cd "${_here}/../.." && pwd)}"
export DATA_ROOT="${DATA_ROOT:-${CORE7_ROOT}/data}"
export RESULTS_ROOT="${RESULTS_ROOT:-${CORE7_ROOT}/runs}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export PYTHON="${PYTHON:-python}"
export PYTHONPATH="${CORE7_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export CLEVR_MCQ4="${CLEVR_MCQ4:-${DATA_ROOT}/clevr_mcq4_v3}"
export DATASET_CONFIG="${DATASET_CONFIG:-${CORE7_ROOT}/configs/core7/clevr_mcq4_official3k.json}"

ALL_MODELS=(
  qwen2vl_2b qwen25vl_3b qwen25vl_7b
  paligemma_3b_mix paligemma2_3b_mix paligemma2_10b_mix
  llava_ov_0.5b_si llava_ov_7b_si llava_ov_7b_ov
  internvl35_1b internvl35_2b internvl35_8b
)

core7() { "${PYTHON}" -m vlm_mcqa.core7 "$@"; }

source_commit() {
  git -C "${CORE7_ROOT}" rev-parse HEAD 2>/dev/null || printf '%s\n' "${SOURCE_GIT_COMMIT:-unknown}"
}

source_state_label() {
  if git -C "${CORE7_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
    local count
    count="$(git -C "${CORE7_ROOT}" status --porcelain | wc -l)"
    printf '%s files\n' "${count}"
  else
    printf 'gitless snapshot %s\n' "${SOURCE_SNAPSHOT_MANIFEST:-unverified}"
  fi
}

# run_logged <result_dir> <command...>: tee stdout/stderr into the result dir.
run_logged() {
  local dir="$1"; shift
  mkdir -p "${dir}"
  {
    echo "[provenance] host=$(hostname) date=$(date -Is)"
    echo "[provenance] commit=$(source_commit)"
    echo "[provenance] source_state=$(source_state_label)"
    echo "[provenance] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    command -v nvidia-smi >/dev/null && \
      nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv || true
    echo "[command] $*"
  } | tee -a "${dir}/combined.log"
  "$@" > >(tee -a "${dir}/combined.log") 2> >(tee -a "${dir}/combined.err" >&2)
}
