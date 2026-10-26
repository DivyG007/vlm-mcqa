#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=next3_e23
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/next3_e23_%x_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/next3_e23_%x_%j.err

set -euo pipefail
set -x
case "$(hostname -s)" in
  gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077) exit 90 ;;
esac

: "${MODEL_KEY:?MODEL_KEY must identify a prepared next-three model}"
: "${SOURCE_GIT_COMMIT:?exact full source commit required}"
: "${SOURCE_SNAPSHOT:?immutable source snapshot required}"
: "${E20_SCIENCE_CONFIG:?completed E20 science config supplies the verified checkpoint and deployment}"
: "${TASK_PREP_ROOT:?prepared E23 calibration/discovery/confirmation task root required}"
: "${E23_IMAGE_LAYERS:?model-native translated image layers required}"
: "${E23_HEAD_COUNT:=16}"
: "${E23_CONTROL_MODE:=translated_shape}"
: "${E23_QUERY_GROUPS:=final:options}"
: "${E23_MAX_TOKENS:=1024}"
: "${E23_MAX_REGION_TOKENS:=256}"

BASE=${VLM_BASE:-/home2/rsm/anlp-vlm-mcqa}
ENV_ROOT=${VLM_ENV_ROOT:-${BASE}/envs/e20_qwen_torch28_tf4576_40839}
LAUNCH_ROOT=${BASE}/launch/next3_e23_${MODEL_KEY}_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}
ATTEMPT_ROOT=${BASE}/results/next3_e23_${MODEL_KEY}_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}
mkdir -p "${BASE}/logs" "${BASE}/launch"
test ! -e "$LAUNCH_ROOT"
test ! -e "$ATTEMPT_ROOT"
mkdir "$LAUNCH_ROOT"

module load u22/cuda/12.1
if test -n "${VLM_ENV_ROOT:-}"; then
  test -x "${ENV_ROOT}/bin/python"
else
  source /home2/rsm/miniconda3/etc/profile.d/conda.sh
  conda activate motion-probe
fi
export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${SOURCE_SNAPSHOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SOURCE_SNAPSHOT"
python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
print({"visible_gpus": torch.cuda.device_count(),
       "names": [torch.cuda.get_device_name(i) for i in range(2)]})
PY

python -m vlm_mcqa.campaigns inventory --root "$SOURCE_SNAPSHOT" \
  --output "${LAUNCH_ROOT}/source_inventory.json" --kind source --commit "$SOURCE_GIT_COMMIT"

python - "$MODEL_KEY" "$SOURCE_GIT_COMMIT" "$SOURCE_SNAPSHOT" "$E20_SCIENCE_CONFIG" \
  "$TASK_PREP_ROOT" "$LAUNCH_ROOT" "$ATTEMPT_ROOT" "$E23_IMAGE_LAYERS" \
  "$E23_HEAD_COUNT" "$E23_CONTROL_MODE" "$E23_QUERY_GROUPS" "$E23_MAX_TOKENS" \
  "$E23_MAX_REGION_TOKENS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(model_key, commit, source_raw, e20_raw, prep_raw, launch_raw, attempt_raw,
 layers_raw, head_count_raw, control_mode, query_raw, max_tokens_raw,
 max_region_tokens_raw) = sys.argv[1:]
source, e20, prep, launch, attempt = map(Path, (source_raw, e20_raw, prep_raw, launch_raw, attempt_raw))
layers = [int(value) for value in layers_raw.replace(":", ",").split(",") if value]
head_count = int(head_count_raw)
query_groups = [value for value in query_raw.replace(",", ":").split(":") if value]
max_tokens = int(max_tokens_raw)
max_region_tokens = int(max_region_tokens_raw)
if not layers or len(layers) != len(set(layers)) or any(layer < 0 for layer in layers):
    raise SystemExit("E23_IMAGE_LAYERS must be unique nonnegative integers")
if head_count not in {4, 16}:
    raise SystemExit("E23_HEAD_COUNT must be 4 or 16")
if control_mode not in {"translated_shape", "token_count"}:
    raise SystemExit("E23_CONTROL_MODE must be translated_shape or token_count")
if len(query_groups) != len(set(query_groups)) or not {"final", "options"}.issubset(query_groups):
    raise SystemExit("E23_QUERY_GROUPS must uniquely include final and options")
if max_tokens < 1024:
    raise SystemExit("E23_MAX_TOKENS must be at least 1024")
if max_region_tokens < 256 or max_region_tokens > max_tokens:
    raise SystemExit("E23_MAX_REGION_TOKENS must be between 256 and E23_MAX_TOKENS")

base = json.loads(e20.read_text())
if base["model"]["key"] != model_key:
    raise SystemExit("E20 config model mismatch")
head = source / f"consolidation/core7_release_results/{model_key}_core7/freeze.json"
if not head.is_file():
    raise SystemExit(f"missing frozen native heads: {head}")
decision = {
    "schema": "rachit.next-three-e23-decision.v1", "approved": True,
    "study": "spatial", "model_key": model_key,
    "source_git_commit": commit, "dataset_hash": "f5573e25f3aa2bf2",
    "evidence_status": "full_replication",
    "design": "model-native correct-only E23 spatial-factorial replication",
    "rationale": "Replicate the frozen Qwen spatial-factorial test with model-native heads, relative-depth layers, disjoint phases, strict clean-correct eligibility, and unchanged causal controls.",
    "image_layers": layers, "head_count": head_count, "max_tokens": max_tokens,
    "max_region_tokens": max_region_tokens,
    "control_mode": control_mode, "attention_query_groups": query_groups,
}
if control_mode == "token_count":
    decision.update({
        "schema": "rachit.next-three-e23-geometry-sensitivity-decision.v1",
        "evidence_status": "geometry_relaxed_token_count_sensitivity",
        "design": "model-native E23 sensitivity with equal-token background controls",
        "rationale": "The frozen translated-shape control was geometrically infeasible in the model processor's disconnected option region. This labeled sensitivity preserves the cohort, layers, heads, query sites, phase split, and all other causal rules while replacing only that impossible control with an equal-token background region.",
    })
decision_path = launch / "decision.json"
decision_path.write_text(json.dumps(decision, sort_keys=True, indent=2) + "\n")
protocol = {
    "target_units": 24, "min_units": 16, "seed": 0, "max_tokens": max_tokens,
    "max_region_tokens": max_region_tokens, "calibration_tasks": 2, "bootstrap_samples": 2000,
    "random_head_sets": 20, "query_budget_per_group": 8, "tolerance": 0.06,
    "minimum_accuracy": 0.75, "minimum_image_gap": 0.20, "minimum_effect": 0.0,
    "layer_tie_tolerance": 0.01, "mapping_rule": "center", "min_fraction": None,
    "control_mode": control_mode, "attention_query_groups": query_groups,
    "image_layers": layers, "head_count": head_count,
}
study = {
    "name": "spatial", "phase": "discovery", "split": "discovery", "model_key": model_key,
    "head_freeze_path": str(head), "head_freeze_dataset_hash": "f5573e25f3aa2bf2",
    "tasks_path": str(prep / "discovery/tasks.json"),
    "calibration_tasks_path": str(prep / "calibration/tasks.json"),
    "task_assets_root": str(Path(base["data"]["directory"])),
    "protected_inputs": [str(prep), str(Path(base["data"]["directory"])), str(head.parent)],
    "protocol": protocol,
    "phase_settings": {"confirmation": {"tasks_path": str(prep / "confirmation/tasks.json")}},
    "evidence_status": decision["evidence_status"],
}
config = {
    "schema": base["schema"], "kind": "stretch", "seed": 0, "output_dir": str(attempt),
    "source": {"root": str(source), "manifest": str(launch / "source_inventory.json")},
    "model": {**base["model"], "models_config": str(source / "configs/core7/next_three_models_pinned.json")},
    "data": base["data"], "deployment": base["deployment"],
    # Preserve the verified two-copy batched-vs-single calibration.  Omitting
    # this falls back to three copies and exceeds 11 GiB on the Ada 2080 Ti.
    "session_profile": {"calibration_batch_size": 2},
    "protocol": {"decision_path": str(decision_path),
                 "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest()},
    "study": {"name": "spatial", "phases": ["discovery", "confirmation"], "config": study},
}
(launch / "campaign.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
PY

python -m vlm_mcqa.campaigns verify --config "${LAUNCH_ROOT}/campaign.json"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 20 \
  > "${LAUNCH_ROOT}/gpu_monitor.csv" &
MONITOR_PID=$!
cleanup(){ kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT
python -m vlm_mcqa.campaigns run --config "${LAUNCH_ROOT}/campaign.json" --execute
cleanup
trap - EXIT

python - "$ATTEMPT_ROOT" "${LAUNCH_ROOT}/completion.json" <<'PY'
import json
from pathlib import Path
import sys
attempt, output = map(Path, sys.argv[1:])
completion = json.loads((attempt / "completion.json").read_text())
if completion.get("status") not in {"complete", "insufficient"}:
    raise SystemExit(completion)
record = {"schema": "rachit.next-three-e23-launch.v1", "status": "complete",
          "primary_status": completion["status"],
          "scientific_verdict": completion.get("scientific_verdict"),
          "attempt_root": str(attempt)}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
test -s "${LAUNCH_ROOT}/gpu_monitor.csv"
test -s "${LAUNCH_ROOT}/completion.json"
