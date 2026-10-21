#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=rachit_e20_qwen_cal
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --gres=gpu:2
#SBATCH --time=0-04:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e20_qwen_cal_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e20_qwen_cal_%j.err

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
: "${PREP_ROOT:?PREP_ROOT is required}"
: "${E20_ENV_ROOT:?E20_ENV_ROOT is required}"
BASE="/home2/rsm/anlp-vlm-mcqa"
LAUNCH_ROOT="${BASE}/launch/e20_qwen_cal_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}"
ATTEMPT_ROOT="${BASE}/results/e20_qwen_cal_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}"
CONFIG_PATH="${LAUNCH_ROOT}/campaign.json"

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

python - "${PREP_ROOT}/campaign_base.json" "${CONFIG_PATH}" "${ATTEMPT_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

source, target, attempt = Path(sys.argv[1]), Path(sys.argv[2]), str(Path(sys.argv[3]).resolve())
config = json.loads(source.read_text())
old = config["output_dir"]
config["output_dir"] = attempt
changed = [key for key in config if key == "output_dir" and config[key] != old]
assert changed == ["output_dir"]
with target.open("x", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
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

python -m vlm_mcqa.campaigns calibrate --config "${CONFIG_PATH}" --execute
cleanup_monitor
trap - EXIT

python - "${ATTEMPT_ROOT}" "${CONFIG_PATH}" "${LAUNCH_ROOT}/completion.json" <<'PY'
import json
from pathlib import Path
import sys

attempt, config_path, output = map(Path, sys.argv[1:])
result = json.loads((attempt / "calibration_result.json").read_text())
calibration = json.loads((attempt / "session/calibration.json").read_text())
devices = set((calibration["adapter"].get("hf_device_map") or {}).values())
if result.get("status") != "calibrated" or result.get("ordinary_all_passed") is not True:
    raise SystemExit(f"calibration did not pass: {result}")
if devices != {"cuda:0", "cuda:1"}:
    raise SystemExit(f"expected exact two-GPU placement, got {sorted(devices)}")
record = {
    "schema": "rachit.e20-qwen-calibration-launch.v1", "status": "complete",
    "attempt_root": str(attempt), "config_path": str(config_path),
    "calibration_result": result, "device_map_devices": sorted(devices),
    "scientific_run_performed": False,
}
output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
PY

test -s "${LAUNCH_ROOT}/gpu_monitor.csv"
test -s "${LAUNCH_ROOT}/completion.json"
echo "[complete] calibration-only ${ATTEMPT_ROOT}"
