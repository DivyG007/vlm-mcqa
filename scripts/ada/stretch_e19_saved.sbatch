#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=rachit_e19_saved
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e19_saved_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e19_saved_%j.err

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
: "${E19_INPUT_ROOT:?E19_INPUT_ROOT is required}"

REMOTE_BASE="/home2/rsm/anlp-vlm-mcqa"
ATTEMPT_ROOT="${REMOTE_BASE}/results/e19_saved_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}"
SCOPE_ROOT="${REMOTE_BASE}/prepared/e19_scopes_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}"
DATASET="${E19_INPUT_ROOT}/clevr_mcq4_v3"
QWEN="${E19_INPUT_ROOT}/qwen25vl_7b"
INTERNVL="${E19_INPUT_ROOT}/internvl35_8b"

mkdir -p "${REMOTE_BASE}/logs" "${ATTEMPT_ROOT}" "${SCOPE_ROOT}"
exec > >(tee -a "${ATTEMPT_ROOT}/combined.log") 2> >(tee -a "${ATTEMPT_ROOT}/combined.err" >&2)

echo "[provenance] source_commit=${SOURCE_GIT_COMMIT}"
echo "[runtime] job_id=${SLURM_JOB_ID} node=$(hostname -s)"
echo "[inputs] root=${E19_INPUT_ROOT}"

source /home2/rsm/miniconda3/etc/profile.d/conda.sh
conda activate motion-probe
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${SOURCE_ROOT}"

test -s "${DATASET}/items.jsonl"
test -s "${DATASET}/variants.jsonl"
test -s "${QWEN}/behavior_behavior.jsonl"
test -s "${QWEN}/lens_behavior.jsonl"
test -s "${INTERNVL}/behavior_behavior.jsonl"
test -s "${INTERNVL}/lens_behavior.jsonl"

python -m unittest \
  tests.test_prepare_e19_scope \
  tests.test_stretch_analysis \
  tests.test_stretch_failure_protocol -v

run_model() {
  local model="$1"
  local stage="$2"
  local rules="$3"
  local input_dir="${E19_INPUT_ROOT}/${model}"
  # Scope is an immutable input to analysis_cli and must not share a parent
  # with the fresh analysis output tree.
  local scope="${SCOPE_ROOT}/${model}_scope.json"
  local descriptive="${ATTEMPT_ROOT}/${model}_descriptive"
  local design="${ATTEMPT_ROOT}/${model}_design"

  python scripts/local/prepare_e19_scope.py \
    --dataset "${DATASET}" --stage "${stage}" --output "${scope}"
  python -m vlm_mcqa.stretch.analysis_cli failures \
    --dataset "${DATASET}" \
    --behavior "${input_dir}/behavior_behavior.jsonl" \
    --lens "${input_dir}/lens_behavior.jsonl" \
    --scope "${scope}" \
    --output "${descriptive}"
  python -m vlm_mcqa.stretch.analysis_cli failure-design \
    --dataset "${DATASET}" \
    --report "${descriptive}/report.json" \
    --rules "${rules}" \
    --output "${design}"
  python - "${descriptive}/report.json" "${design}/report.json" <<'PY'
import json
import sys
from pathlib import Path

descriptive, design = (json.loads(Path(path).read_text()) for path in sys.argv[1:])
assert descriptive["status"] == "descriptive_not_causal"
assert not descriptive["coverage"]["missing_dataset_variants"]
assert not descriptive["coverage"]["missing_or_duplicate_clean_behavior"]
assert not descriptive["coverage"]["missing_or_duplicate_lens_keys"]
assert design["status"] == "descriptive_internal_validation"
print(json.dumps({
    "descriptive_rows": len(descriptive["rows"]),
    "design_coverage": design["coverage"],
    "phase_status": {
        phase: {name: value["status"] for name, value in body["strata"].items()}
        for phase, body in design["phases"].items()
    },
}, sort_keys=True))
PY
}

# Required model order: Qwen first, then InternVL. LLaVA waits for complete raw input.
run_model qwen25vl_7b 28 "${SOURCE_ROOT}/configs/stretch/e19_qwen.rules.json"
run_model internvl35_8b 36 "${SOURCE_ROOT}/configs/stretch/e19_internvl.rules.json"

python - "${ATTEMPT_ROOT}" "${SOURCE_GIT_COMMIT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root, commit = Path(sys.argv[1]), sys.argv[2]
files = []
for path in sorted(root.rglob("*")):
    if path.is_file():
        files.append({
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
completion = {
    "schema": "rachit.e19-ada-completion.v1",
    "status": "complete",
    "source_git_commit": commit,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "node": os.uname().nodename,
    "scope_root": str(root.parent.parent / "prepared" / f"e19_scopes_{commit}_{os.environ['SLURM_JOB_ID']}"),
    "model_order": ["qwen25vl_7b", "internvl35_8b"],
    "llava_status": "waiting_for_complete_raw_input",
    "files": files,
}
(root / "completion.json").write_text(json.dumps(completion, sort_keys=True, indent=2) + "\n")
PY

test -s "${ATTEMPT_ROOT}/completion.json"
echo "[complete] ${ATTEMPT_ROOT}"
