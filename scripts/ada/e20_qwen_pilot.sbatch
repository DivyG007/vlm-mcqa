#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=rachit_e20_qwen_pilot
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e20_qwen_pilot_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e20_qwen_pilot_%j.err

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
: "${SOURCE_MANIFEST:?SOURCE_MANIFEST is required}"
: "${PREP_ROOT:?PREP_ROOT is required}"
: "${E20_ENV_ROOT:?E20_ENV_ROOT is required}"
BASE="/home2/rsm/anlp-vlm-mcqa"
LAUNCH_ROOT="${BASE}/launch/e20_qwen_pilot_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}"
ATTEMPT_ROOT="${BASE}/results/e20_qwen_pilot_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}"
CONFIG_PATH="${LAUNCH_ROOT}/campaign.json"
PROTOCOL_PATH="${LAUNCH_ROOT}/technical_pilot_protocol.json"

mkdir -p "${BASE}/logs" "${BASE}/launch"
if test -e "${LAUNCH_ROOT}" || test -e "${ATTEMPT_ROOT}"; then
  echo "[FATAL] fresh launch and attempt paths required" >&2
  exit 4
fi
mkdir "${LAUNCH_ROOT}"

module load u22/cuda/12.1
source /home2/rsm/miniconda3/etc/profile.d/conda.sh
conda activate motion-probe
test -s "${E20_ENV_ROOT}/completion.json"
export PATH="${E20_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "${SOURCE_ROOT}"

echo "[runtime] job_id=${SLURM_JOB_ID} node=$(hostname -s) visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
for index in range(2):
    total = torch.cuda.get_device_properties(index).total_memory
    assert total >= 10 * 1024**3, (index, total)
    print(index, torch.cuda.get_device_name(index), total)
PY

python - "${PREP_ROOT}/campaign_base.json" "${CONFIG_PATH}" "${PROTOCOL_PATH}" "${ATTEMPT_ROOT}" "${SOURCE_ROOT}" "${SOURCE_MANIFEST}" <<'PY'
import json
from pathlib import Path
import sys

source, target, protocol_target, attempt, source_root, source_manifest = map(Path, sys.argv[1:])
config = json.loads(source.read_text())
protocol_source = Path(config["study"]["config"]["protocol"])
protocol = json.loads(protocol_source.read_text())
protocol["min_pairs"] = 8
protocol["sampling"]["discovery"]["target_pairs"] = 8
protocol["sampling"]["discovery"]["min_pairs"] = 8
assert protocol["claim_boundary"] == "calibration_and_technical_pilots_are_not_scientific_evidence"
protocol_target.write_text(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n")
config["study"]["config"]["protocol"] = str(protocol_target.resolve())
config["output_dir"] = str(attempt.resolve())
config["source"] = {"root": str(source_root.resolve()), "manifest": str(source_manifest.resolve())}
target.write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n")
PY

python -m unittest \
  tests.test_campaigns_config \
  tests.test_campaigns_provenance \
  tests.test_campaigns_study_integration \
  tests.test_stretch_protocol \
  tests.test_stretch_causal_studies \
  tests.test_stretch_contrasts \
  tests.test_stretch_sampling -v
python -m vlm_mcqa.campaigns verify --config "${CONFIG_PATH}"

nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader -l 15 > "${LAUNCH_ROOT}/gpu_monitor.csv" &
MONITOR_PID=$!
cleanup_monitor() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup_monitor EXIT

python -m vlm_mcqa.campaigns pilot --config "${CONFIG_PATH}" --execute
cleanup_monitor
trap - EXIT

python - "${ATTEMPT_ROOT}" "${CONFIG_PATH}" "${LAUNCH_ROOT}/completion.json" <<'PY'
import json
from pathlib import Path
import sys

attempt, config_path, output = map(Path, sys.argv[1:])
result = json.loads((attempt / "pilot_result.json").read_text())
calibration = json.loads((attempt / "session/calibration.json").read_text())
devices = set((calibration["adapter"].get("hf_device_map") or {}).values())
if result.get("status") != "technical_complete" or result.get("target_pairs") != 8:
    raise SystemExit(f"pilot did not complete: {result}")
if result.get("scientific_run_performed") is not False or result.get("confirmation_eligible") is not False:
    raise SystemExit("pilot claim boundary was not preserved")
if devices != {"cuda:0", "cuda:1"}:
    raise SystemExit(f"expected exact two-GPU placement, got {sorted(devices)}")
record = {
    "schema": "rachit.e20-qwen-technical-pilot-launch.v1", "status": "complete",
    "attempt_root": str(attempt), "config_path": str(config_path),
    "pilot_result": result, "device_map_devices": sorted(devices),
    "scientific_run_performed": False, "confirmation_eligible": False,
}
output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
PY

test -s "${LAUNCH_ROOT}/gpu_monitor.csv"
test -s "${LAUNCH_ROOT}/completion.json"
echo "[complete] technical pilot ${ATTEMPT_ROOT}"
