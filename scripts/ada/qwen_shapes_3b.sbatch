#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=vlm_shapes64_dual
#SBATCH --nodelist=gnode088
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/vlm_shapes_3b_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/vlm_shapes_3b_%j.err

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

NUM_PAIRS="${NUM_PAIRS:-64}"
RUN_KIND="${RUN_KIND:-allschemes_dualprobe}"
RUN_ID="qwen25vl3b_shapes_${RUN_KIND}_${NUM_PAIRS}pairs_${SLURM_JOB_ID}"
REMOTE_BASE="/home2/rsm/anlp-vlm-mcqa"
RESULT_DIR="${REMOTE_BASE}/results/${RUN_ID}"
SCRATCH_DIR="/scratch/rsm/anlp-vlm-mcqa/${RUN_ID}"
DATA_DIR="${SCRATCH_DIR}/shapes_v0"

mkdir -p "${REMOTE_BASE}/logs" "${RESULT_DIR}/wandb" "${SCRATCH_DIR}"
exec > >(tee -a "${RESULT_DIR}/combined.log") 2> >(tee -a "${RESULT_DIR}/combined.err" >&2)

echo "[provenance] source_commit=${SOURCE_GIT_COMMIT}"
echo "[provenance] source_root=${SOURCE_ROOT}"
echo "[runtime] job_id=${SLURM_JOB_ID} node=$(hostname -s) run_id=${RUN_ID}"
echo "[runtime] num_pairs=${NUM_PAIRS}"
echo "[paths] data=${DATA_DIR} results=${RESULT_DIR}"

module load u22/cuda/12.1
source /home2/rsm/miniconda3/etc/profile.d/conda.sh
conda activate motion-probe

export HF_HOME=/scratch/rsm/hf_cache
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-anlp-vlm-mcqa}"
export WANDB_DIR="${RESULT_DIR}/wandb"
echo "[runtime] hf_hub_offline=${HF_HUB_OFFLINE} transformers_offline=${TRANSFORMERS_OFFLINE}"

nvidia-smi
python - <<'PY'
import torch
import transformers
import wandb
from PIL import Image
assert torch.cuda.is_available(), "CUDA unavailable inside motion-probe"
x = torch.ones((1024, 1024), device="cuda")
assert x.sum().item() == 1024 * 1024, "CUDA tensor allocation failed"
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("pillow", Image.__version__)
print("cuda_devices", torch.cuda.device_count())
print("cuda_allocation_test", torch.cuda.get_device_name(0))
print("wandb", wandb.__version__, "authenticated", bool(wandb.api.api_key))
assert wandb.api.api_key, "W&B is not authenticated"
del x
PY

cd "${SOURCE_ROOT}"
python -m vlm_mcqa.data.generate_shapes \
  --output-dir "${DATA_DIR}" \
  --num-pairs "${NUM_PAIRS}" \
  --seed 42 \
  --image-size 448

python - <<PY
import json
import os
from pathlib import Path
config = json.loads(Path("configs/experiment_00_shapes.json").read_text())
config["num_pairs"] = int("${NUM_PAIRS}")
config["slurm_job_id"] = "${SLURM_JOB_ID}"
config["source_git_commit"] = "${SOURCE_GIT_COMMIT}"
config["layerwise_label_schemes"] = [
    "letters", "numbers", "rare_letters", "punctuation"
]
config["dual_alphabet_probe"] = {
    "default_labels": ["A", "B", "C", "D"],
    "comparison": "default letters versus the requested label set at every stage",
    "probability_scope": "softmax over the eight probed labels, not the full vocabulary",
}
config["content_symbol_probe"] = {
    "semantic_tokens": ["red", "blue", "green", "yellow"],
    "symbol_tokens": ["A", "B", "C", "D"],
    "probability_scope": "separate restricted four-way softmaxes",
}
config["hf_hub_offline"] = os.environ["HF_HUB_OFFLINE"]
config["transformers_offline"] = os.environ["TRANSFORMERS_OFFLINE"]
Path("${RESULT_DIR}/experiment_config.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n"
)
PY
printf '%s\n' "${SOURCE_GIT_COMMIT}" > "${RESULT_DIR}/source_git_commit.txt"

python -m vlm_mcqa.eval.qwen_shapes \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --manifest "${DATA_DIR}/manifest.jsonl" \
  --output-dir "${RESULT_DIR}" \
  --max-pixels 50176 \
  --progress-every 20

python -m vlm_mcqa.analysis.plot_shapes \
  --predictions "${RESULT_DIR}/predictions.jsonl" \
  --layerwise "${RESULT_DIR}/layerwise_label_logits.jsonl" \
  --output-dir "${RESULT_DIR}/figures"

python -m vlm_mcqa.tracking.wandb_report \
  --result-dir "${RESULT_DIR}" \
  --project "${WANDB_PROJECT}" \
  --run-name "${RUN_ID}" \
  --run-id "${RUN_ID}" \
  --job-id "${SLURM_JOB_ID}" \
  --node "$(hostname -s)"

python - <<PY
import json
from pathlib import Path
result_dir = Path("${RESULT_DIR}")
summary = json.loads((result_dir / "summary.json").read_text())
required = [
    result_dir / "predictions.jsonl",
    result_dir / "layerwise_label_logits.jsonl",
    result_dir / "figures" / "accuracy_by_position.png",
    result_dir / "figures" / "position_confusion_matrix.png",
    result_dir / "figures" / "accuracy_by_condition.png",
    result_dir / "figures" / "vocabulary_compliance_by_scheme.png",
    result_dir / "figures" / "layerwise_scheme_comparison.png",
    result_dir / "figures" / "layerwise_default_to_requested_switch.png",
    result_dir / "figures" / "layerwise_default_to_requested_switch.csv",
    result_dir / "figures" / "layerwise_content_vs_symbol_letters.png",
    result_dir / "figures" / "layerwise_content_symbol_by_position_letters.png",
    result_dir / "figures" / "layerwise_content_symbol_means.csv",
    result_dir / "wandb_run.json",
]
for scheme in ("letters", "numbers", "rare_letters", "punctuation"):
    required.extend(
        [
            result_dir / "figures" / f"layerwise_label_logits_by_position_{scheme}.png",
            result_dir / "figures" / f"layerwise_correct_answer_progression_{scheme}.png",
        ]
    )
for scheme in ("numbers", "rare_letters", "punctuation"):
    required.append(
        result_dir / "figures" / f"layerwise_default_vs_requested_{scheme}.png"
    )
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"Missing required result artifacts: {missing}")
print("[verified] records", summary["record_count"])
print("[verified] overall_accuracy", summary["metrics"]["overall"]["accuracy"])
print("[verified] result_dir", result_dir)
PY

echo "[complete] ${RUN_ID}"
