#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=next3_e24
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/next3_e24_%x_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/next3_e24_%x_%j.err

set -euo pipefail
set -x
case "$(hostname -s)" in
  gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077) exit 90 ;;
esac

: "${MODEL_KEY:?MODEL_KEY must identify a prepared next-three model}"
: "${SOURCE_GIT_COMMIT:?exact full source commit required}"
: "${SOURCE_SNAPSHOT:?immutable source snapshot required}"
: "${E20_SCIENCE_CONFIG:?completed E20 science config supplies the verified checkpoint and deployment}"
: "${TASK_PREP_ROOT:?prepared E24 calibration/discovery/confirmation task root required}"
: "${E24_IMAGE_LAYERS:?model-native translated image layers required}"
: "${E24_HEAD_COUNT:=4}"
: "${E24_QUERY_GROUPS:=final}"
: "${E24_MAX_TOKENS:=2048}"
: "${E24_MAX_REGION_TOKENS:=512}"
: "${E24_CHECKPOINT_ROOT:=}"
: "${E24_DEVICE_CUTS:=}"
: "${E24_PARTIAL_ROWS_PATH:=}"

BASE=${VLM_BASE:-/home2/rsm/anlp-vlm-mcqa}
ENV_ROOT=${VLM_ENV_ROOT:-${BASE}/envs/e20_qwen_torch28_tf4576_40839}
LAUNCH_ROOT=${BASE}/launch/next3_e24_${MODEL_KEY}_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}
ATTEMPT_ROOT=${BASE}/results/next3_e24_${MODEL_KEY}_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}
mkdir -p "${BASE}/logs" "${BASE}/launch"
test ! -e "$LAUNCH_ROOT"
test ! -e "$ATTEMPT_ROOT"
mkdir "$LAUNCH_ROOT"

module load u22/cuda/12.1
test -x "${ENV_ROOT}/bin/python"
export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${SOURCE_SNAPSHOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SOURCE_SNAPSHOT"
python - <<'PY'
import os
import torch
cuts = [value for value in os.environ.get("E24_DEVICE_CUTS", "").replace(",", ":").split(":") if value]
expected = len(cuts) + 1 if cuts else 2
assert torch.cuda.is_available() and torch.cuda.device_count() == expected
print({"visible_gpus": torch.cuda.device_count(),
       "names": [torch.cuda.get_device_name(i) for i in range(expected)]})
PY

python -m vlm_mcqa.campaigns inventory --root "$SOURCE_SNAPSHOT" \
  --output "${LAUNCH_ROOT}/source_inventory.json" --kind source --commit "$SOURCE_GIT_COMMIT"

E24_CHECKPOINT_MANIFEST=
if test -n "$E24_CHECKPOINT_ROOT"; then
  case "$MODEL_KEY" in
    llava_ov_7b_si)
      E24_HF_ID=llava-hf/llava-onevision-qwen2-7b-si-hf
      E24_REVISION=cae7503dce88887557f73db1882328d21d9a7e21
      ;;
    internvl35_8b)
      E24_HF_ID=OpenGVLab/InternVL3_5-8B-HF
      E24_REVISION=741a7d03020411e666c6109218ab71e08151ef86
      ;;
    *) echo "[FATAL] checkpoint relocation unsupported for $MODEL_KEY" >&2; exit 2 ;;
  esac
  test -s "${E24_CHECKPOINT_ROOT}/stage_completion.json"
  E24_CHECKPOINT_MANIFEST=${LAUNCH_ROOT}/checkpoint_inventory.json
  python -m vlm_mcqa.campaigns inventory --root "$E24_CHECKPOINT_ROOT" \
    --output "$E24_CHECKPOINT_MANIFEST" --kind checkpoint \
    --hf-id "$E24_HF_ID" --revision "$E24_REVISION"
fi

python - "$MODEL_KEY" "$SOURCE_GIT_COMMIT" "$SOURCE_SNAPSHOT" "$E20_SCIENCE_CONFIG" \
  "$TASK_PREP_ROOT" "$LAUNCH_ROOT" "$ATTEMPT_ROOT" "$E24_IMAGE_LAYERS" \
  "$E24_HEAD_COUNT" "$E24_QUERY_GROUPS" "$E24_MAX_TOKENS" \
  "$E24_MAX_REGION_TOKENS" "$E24_CHECKPOINT_ROOT" "$E24_CHECKPOINT_MANIFEST" \
  "$E24_DEVICE_CUTS" "$E24_PARTIAL_ROWS_PATH" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(model_key, commit, source_raw, e20_raw, prep_raw, launch_raw, attempt_raw,
 layers_raw, head_count_raw, query_raw, max_tokens_raw,
 max_region_tokens_raw, checkpoint_override_raw, checkpoint_manifest_raw,
 device_cuts_raw, partial_rows_raw) = sys.argv[1:]
source, e20, prep, launch, attempt = map(
    Path, (source_raw, e20_raw, prep_raw, launch_raw, attempt_raw)
)
layers = [int(value) for value in layers_raw.replace(":", ",").split(",") if value]
head_count = int(head_count_raw)
query_groups = [value for value in query_raw.replace(",", ":").split(":") if value]
max_tokens = int(max_tokens_raw)
max_region_tokens = int(max_region_tokens_raw)
if not layers or len(layers) != len(set(layers)) or any(layer < 0 for layer in layers):
    raise SystemExit("E24_IMAGE_LAYERS must be unique nonnegative integers")
if head_count != 4:
    raise SystemExit("the frozen E24 protocol requires exactly four native heads")
if query_groups != ["final"]:
    raise SystemExit("the frozen E24 protocol uses only the final query group")
if max_tokens < 1024:
    raise SystemExit("E24_MAX_TOKENS must be at least 1024")
if max_region_tokens < 256 or max_region_tokens > max_tokens:
    raise SystemExit("E24_MAX_REGION_TOKENS must be between 256 and E24_MAX_TOKENS")

base = json.loads(e20.read_text())
if base["model"]["key"] != model_key:
    raise SystemExit("E20 config model mismatch")
if checkpoint_override_raw:
    if not checkpoint_manifest_raw:
        raise SystemExit("checkpoint relocation requires a fresh exact inventory")
    base["model"]["checkpoint_dir"] = checkpoint_override_raw
    base["model"]["checkpoint_manifest"] = checkpoint_manifest_raw
if device_cuts_raw:
    cuts = [int(value) for value in device_cuts_raw.replace(",", ":").split(":") if value]
    mapping = dict(base["deployment"]["device_map"])
    layer_keys = [key for key in mapping if key.startswith("model.language_model.layers.")]
    layer_count = len(layer_keys)
    if (not cuts or cuts != sorted(set(cuts)) or cuts[0] < 0
            or cuts[-1] >= layer_count - 1 or len(cuts) > 3):
        raise SystemExit("E24 device cuts must be unique increasing layer ends leaving work on every GPU")
    final_device = f"cuda:{len(cuts)}"
    for key in layer_keys:
        layer = int(key.rsplit(".", 1)[1])
        device = next((index for index, cut in enumerate(cuts) if layer <= cut), len(cuts))
        mapping[key] = f"cuda:{device}"
    mapping["model.language_model.norm"] = final_device
    mapping["lm_head"] = final_device
    base["deployment"]["device_map"] = mapping
head = source / f"consolidation/core7_release_results/{model_key}_core7/freeze.json"
if not head.is_file():
    raise SystemExit(f"missing frozen native heads: {head}")
task_root = prep / "e24"
for phase in ("calibration", "discovery", "confirmation"):
    if not (task_root / phase / "tasks.json").is_file():
        raise SystemExit(f"missing E24 {phase} tasks")
decision = {
    "schema": "rachit.next-three-e24-decision.v1",
    "approved": True,
    "study": "image_options",
    "model_key": model_key,
    "source_git_commit": commit,
    "dataset_hash": "f5573e25f3aa2bf2",
    "evidence_status": "full_replication_of_final_e24_design",
    "design": "model-native correct-only E24 visual-option-layout replication",
    "rationale": (
        "Replicate the final Qwen E24 design with prospective disjoint phases, "
        "model-native frozen heads and relative-depth layers. The final frozen "
        "design omits the structurally unavailable unrelated-donor auxiliary "
        "control and retains all matched text-layout and visual-layout controls."
    ),
    "image_layers": layers,
    "head_count": head_count,
    "max_tokens": max_tokens,
    "max_region_tokens": max_region_tokens,
    "require_unrelated_donor_control": False,
    "attention_query_groups": query_groups,
}
decision_path = launch / "decision.json"
decision_path.write_text(json.dumps(decision, sort_keys=True, indent=2) + "\n")
protocol = {
    "target_units": 32,
    "min_units": 24,
    "seed": 0,
    "max_tokens": max_tokens,
    "max_region_tokens": max_region_tokens,
    "calibration_tasks": 2,
    "bootstrap_samples": 2000,
    "random_head_sets": 20,
    "query_budget_per_group": 8,
    "tolerance": 0.06,
    "minimum_accuracy": 0.75,
    "minimum_image_gap": 0.20,
    "minimum_effect": 0.0,
    "layer_tie_tolerance": 0.01,
    "mapping_rule": "center",
    "min_fraction": None,
    "control_mode": "translated_shape",
    "attention_query_groups": query_groups,
    "image_layers": layers,
    "head_count": head_count,
    "require_unrelated_donor_control": False,
}
study = {
    "name": "image_options",
    "phase": "discovery",
    "split": "discovery",
    "model_key": model_key,
    "head_freeze_path": str(head),
    "head_freeze_dataset_hash": "f5573e25f3aa2bf2",
    "tasks_path": str(task_root / "discovery/tasks.json"),
    "calibration_tasks_path": str(task_root / "calibration/tasks.json"),
    "task_assets_root": str(task_root),
    "protected_inputs": [str(prep), str(Path(base["data"]["directory"])), str(head.parent)],
    "protocol": protocol,
    "phase_settings": {
        "confirmation": {"tasks_path": str(task_root / "confirmation/tasks.json")}
    },
    "evidence_status": decision["evidence_status"],
}
if partial_rows_raw:
    partial_rows = Path(partial_rows_raw).resolve()
    if not partial_rows.is_file() or partial_rows.name != "raw.jsonl":
        raise SystemExit("E24_PARTIAL_ROWS_PATH must name an existing raw.jsonl")
    study["partial_rows_path"] = str(partial_rows)
    # Recovery applies only to the interrupted discovery phase. Confirmation
    # remains a fresh held-out run after the discovery layer is frozen.
    study["phase_settings"]["confirmation"]["partial_rows_path"] = None
config = {
    "schema": base["schema"],
    "kind": "stretch",
    "seed": 0,
    "output_dir": str(attempt),
    "source": {"root": str(source), "manifest": str(launch / "source_inventory.json")},
    "model": {
        **base["model"],
        "models_config": str(source / "configs/core7/next_three_models_pinned.json"),
    },
    "data": base["data"],
    "deployment": base["deployment"],
    "session_profile": {"calibration_batch_size": 2},
    "protocol": {
        "decision_path": str(decision_path),
        "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
    },
    "study": {"name": "image_options", "phases": ["discovery", "confirmation"], "config": study},
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
record = {
    "schema": "rachit.next-three-e24-launch.v1",
    "status": "complete",
    "primary_status": completion["status"],
    "scientific_verdict": completion.get("scientific_verdict"),
    "attempt_root": str(attempt),
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
test -s "${LAUNCH_ROOT}/gpu_monitor.csv"
test -s "${LAUNCH_ROOT}/completion.json"
