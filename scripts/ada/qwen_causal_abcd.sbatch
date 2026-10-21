#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=vlm_abcd_causal
#SBATCH --nodelist=gnode057
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/vlm_abcd_causal_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/vlm_abcd_causal_%j.err

set -euo pipefail
set -x

case "$(hostname -s)" in
  gnode065|gnode066|gnode077)
    echo "[FATAL] Slurm allocated known-broken node $(hostname -s)" >&2
    exit 90
    ;;
esac

: "${SOURCE_GIT_COMMIT:?SOURCE_GIT_COMMIT must be exported by the submitter}"
: "${SOURCE_ROOT:?SOURCE_ROOT must be exported by the submitter}"

REMOTE_BASE="/home2/rsm/anlp-vlm-mcqa"
RUN_KIND="${RUN_KIND:-smoke_atlas}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiment_03_causal_abcd.json}"
SPLIT="${SPLIT:-smoke}"
NUM_PAIRS="${NUM_PAIRS:-8}"
MODES="${MODES_PLUS:-${MODES:-calibration,residual,window}}"
MODES="${MODES//+/,}"
CONTRAST_TYPES="${CONTRAST_TYPES_PLUS:-${CONTRAST_TYPES:-visual,position}}"
CONTRAST_TYPES="${CONTRAST_TYPES//+/,}"
SELECTED_LAYERS="${SELECTED_LAYERS_PLUS:-${SELECTED_LAYERS:-24,25,26,27,28,29,30,31}}"
SELECTED_LAYERS="${SELECTED_LAYERS//+/,}"
TOKEN_GROUPS="${TOKEN_GROUPS_PLUS:-${TOKEN_GROUPS:-final,image_all,image_query_expanded,image_changed_distractor,image_random_matched,question,option_content,option_labels}}"
TOKEN_GROUPS="${TOKEN_GROUPS//+/,}"
WINDOW_WIDTHS="${WINDOW_WIDTHS:-2,4,8}"
WINDOW_START_LAYER="${WINDOW_START_LAYER:-19}"
WINDOW_END_LAYER="${WINDOW_END_LAYER:-35}"
BINDING_SCALES="${BINDING_SCALES:-0,0.25,0.5,1,1.5}"
BINDING_VECTORS="${BINDING_VECTORS:-}"
PATH_SENDER_LAYER="${PATH_SENDER_LAYER:-}"
PATH_RECEIVER_LAYER="${PATH_RECEIVER_LAYER:-}"
RUN_ID="qwen25vl3b_abcd_${RUN_KIND}_${SLURM_JOB_ID}"
RESULT_DIR="${REMOTE_BASE}/results/${RUN_ID}"
SCRATCH_DIR="/scratch/rsm/anlp-vlm-mcqa/${RUN_ID}"
DATA_DIR="${SCRATCH_DIR}/shapes_v0"

mkdir -p "${REMOTE_BASE}/logs" "${RESULT_DIR}/wandb" "${SCRATCH_DIR}"
exec > >(tee -a "${RESULT_DIR}/combined.log") 2> >(tee -a "${RESULT_DIR}/combined.err" >&2)

echo "[provenance] source_commit=${SOURCE_GIT_COMMIT}"
echo "[provenance] source_root=${SOURCE_ROOT}"
echo "[runtime] job_id=${SLURM_JOB_ID} node=$(hostname -s) run_id=${RUN_ID}"
echo "[runtime] split=${SPLIT} num_pairs=${NUM_PAIRS} modes=${MODES} contrasts=${CONTRAST_TYPES}"
echo "[runtime] selected_layers=${SELECTED_LAYERS} windows=${WINDOW_WIDTHS}:${WINDOW_START_LAYER}-${WINDOW_END_LAYER}"
echo "[runtime] token_groups=${TOKEN_GROUPS}"
echo "[paths] data=${DATA_DIR} results=${RESULT_DIR} binding_vectors=${BINDING_VECTORS:-none}"
echo "[config] path=${CONFIG_PATH}"

module load u22/cuda/12.1
source /home2/rsm/miniconda3/etc/profile.d/conda.sh
conda activate motion-probe

export HF_HOME=/scratch/rsm/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-anlp-vlm-mcqa}"
export WANDB_DIR="${RESULT_DIR}/wandb"

nvidia-smi
python - <<'PY'
import torch
import transformers
import wandb
from PIL import Image
assert torch.cuda.is_available(), "CUDA unavailable inside motion-probe"
test = torch.ones((1024, 1024), device="cuda")
assert test.sum().item() == 1024 * 1024
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("pillow", Image.__version__)
print("cuda_devices", torch.cuda.device_count())
print("cuda_device", torch.cuda.get_device_name(0))
print("wandb", wandb.__version__, "authenticated", bool(wandb.api.api_key))
assert wandb.api.api_key, "W&B is not authenticated"
del test
PY

cd "${SOURCE_ROOT}"
test -s "${CONFIG_PATH}"
test -s src/vlm_mcqa/eval/qwen_causal_abcd.py
test -s src/vlm_mcqa/interventions/core.py

PYTHONPATH=src python -m unittest \
  tests.test_causal_interventions \
  tests.test_eval_and_plots \
  tests.test_wandb_report -v

python -m vlm_mcqa.data.generate_shapes \
  --output-dir "${DATA_DIR}" \
  --num-pairs 64 \
  --seed 42 \
  --image-size 448

python - <<PY
import json
from pathlib import Path
config = json.loads(Path("${CONFIG_PATH}").read_text())
config.update({
    "slurm_job_id": "${SLURM_JOB_ID}",
    "source_git_commit": "${SOURCE_GIT_COMMIT}",
    "run_id": "${RUN_ID}",
    "run_kind": "${RUN_KIND}",
    "split": "${SPLIT}",
    "num_pairs": int("${NUM_PAIRS}"),
    "modes": "${MODES}".split(","),
    "contrast_types": "${CONTRAST_TYPES}".split(","),
    "selected_layers": [int(value) for value in "${SELECTED_LAYERS}".split(",")],
    "token_groups": "${TOKEN_GROUPS}".split(","),
    "binding_vectors": "${BINDING_VECTORS}" or None,
})
Path("${RESULT_DIR}/experiment_config.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n"
)
PY
printf '%s\n' "${SOURCE_GIT_COMMIT}" > "${RESULT_DIR}/source_git_commit.txt"

RUN_ARGS=(
  --model-id Qwen/Qwen2.5-VL-3B-Instruct
  --manifest "${DATA_DIR}/manifest.jsonl"
  --output-dir "${RESULT_DIR}"
  --split "${SPLIT}"
  --num-pairs "${NUM_PAIRS}"
  --contrast-types "${CONTRAST_TYPES}"
  --modes "${MODES}"
  --max-pixels 50176
  --image-size 448
  --window-widths "${WINDOW_WIDTHS}"
  --window-start-layer "${WINDOW_START_LAYER}"
  --window-end-layer "${WINDOW_END_LAYER}"
  --selected-layers "${SELECTED_LAYERS}"
  --token-groups "${TOKEN_GROUPS}"
  --binding-scales "${BINDING_SCALES}"
  --progress-every 4
)
if [[ -n "${BINDING_VECTORS}" ]]; then
  RUN_ARGS+=(--binding-vectors "${BINDING_VECTORS}")
fi
if [[ -n "${PATH_SENDER_LAYER}" ]]; then
  RUN_ARGS+=(--path-sender-layer "${PATH_SENDER_LAYER}")
fi
if [[ -n "${PATH_RECEIVER_LAYER}" ]]; then
  RUN_ARGS+=(--path-receiver-layer "${PATH_RECEIVER_LAYER}")
fi

python -m vlm_mcqa.eval.qwen_causal_abcd "${RUN_ARGS[@]}"

python -m vlm_mcqa.analysis.plot_causal_abcd \
  --summary "${RESULT_DIR}/summary.json" \
  --output-dir "${RESULT_DIR}/figures"

python -m vlm_mcqa.tracking.wandb_causal_report \
  --result-dir "${RESULT_DIR}" \
  --project "${WANDB_PROJECT}" \
  --run-name "${RUN_ID}" \
  --run-id "${RUN_ID}" \
  --job-id "${SLURM_JOB_ID}" \
  --node "$(hostname -s)" \
  --job-type "causal-${RUN_KIND}"

python - <<PY
import json
from pathlib import Path
result_dir = Path("${RESULT_DIR}")
summary = json.loads((result_dir / "summary.json").read_text())
required = [
    result_dir / "interventions.jsonl",
    result_dir / "summary.json",
    result_dir / "token_maps.json",
    result_dir / "figures" / "aggregate_interventions.csv",
    result_dir / "figures" / "plot_summary.json",
    result_dir / "wandb_run.json",
    result_dir / "combined.log",
    result_dir / "combined.err",
]
if "residual" in summary["modes"]:
    required.append(result_dir / "figures" / "single_layer_residual_atlas.png")
if "window" in summary["modes"]:
    required.append(result_dir / "figures" / "residual_window_atlas.png")
if "token" in summary["modes"]:
    required.append(result_dir / "figures" / "token_group_patch_heatmap.png")
if "component" in summary["modes"]:
    required.append(result_dir / "figures" / "component_patch_lines.png")
if "binding_fit" in summary["modes"]:
    required.append(result_dir / "binding_vectors.pt")
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"Missing required causal artifacts: {missing}")
plot_summary = json.loads((result_dir / "figures" / "plot_summary.json").read_text())
if not plot_summary["calibration_all_passed"] and "calibration" in summary["modes"]:
    raise SystemExit("Calibration did not pass")
print("[verified] interventions", summary["completed_interventions"])
print("[verified] directed_contrasts", summary["directed_contrast_count"])
print("[verified] result_dir", result_dir)
PY

echo "[complete] ${RUN_ID}"
