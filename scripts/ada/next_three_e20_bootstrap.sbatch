#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=next3_e20_boot
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/next3_e20_%x_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/next3_e20_%x_%j.err

set -euo pipefail
set -x

case "$(hostname -s)" in
  gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077)
    echo "[FATAL] ineligible Ada node: $(hostname -s)" >&2
    exit 90
    ;;
esac

: "${MODEL_KEY:?MODEL_KEY must be internvl35_8b or llava_ov_7b_si}"
: "${SOURCE_GIT_COMMIT:?exact full source commit required}"
: "${SOURCE_SNAPSHOT:?immutable source snapshot required}"
: "${E20_DISCOVERY_TARGET:=}"
: "${E20_PRIMARY_INSUFFICIENT_JOB:=}"
: "${E20_REUSE_TECHNICAL_RESULT_ROOT:=}"
: "${E20_RETRY_FROM:=}"
: "${E20_CONFIRMATION_SENSITIVITY_PAIRS:=}"
: "${E20_PREPARED_CACHE:=8}"
: "${E20_DEVICE_CUT:=}"
: "${E20_DEVICE_CUTS:=}"
: "${VLM_VISIBLE_GPU_SELECTION:=}"

# Some older Ada nodes mix Turing and unsupported Pascal cards under one
# untyped Slurm GRES.  A four-GPU allocation may explicitly retain only the
# preflight-verified Turing pair for PyTorch while still preventing another
# job from being assigned either incompatible card on that node.
if test -n "$VLM_VISIBLE_GPU_SELECTION"; then
  export CUDA_VISIBLE_DEVICES="$VLM_VISIBLE_GPU_SELECTION"
fi

BASE=${VLM_BASE:-/home2/rsm/anlp-vlm-mcqa}
DATASET_ROOT=${BASE}/inputs/clevr_mcq4_v3_full
ENV_ROOT=${VLM_ENV_ROOT:-${BASE}/envs/e20_qwen_torch28_tf4576_40839}
QWEN_PREP=${E20_PREP_ROOT:-${BASE}/prepared/e20_qwen_6505882a9dee55c4c4ad0eaf73bbab400a8910aa_cc594898137f460bfe9f0759e9844b3ce807cfb5}

case "$MODEL_KEY" in
  internvl35_8b)
    MODEL_ID=OpenGVLab/InternVL3_5-8B-HF
    REVISION=741a7d03020411e666c6109218ab71e08151ef86
    NUM_LAYERS=36
    DTYPE=float16
    ;;
  llava_ov_7b_si)
    MODEL_ID=llava-hf/llava-onevision-qwen2-7b-si-hf
    REVISION=cae7503dce88887557f73db1882328d21d9a7e21
    NUM_LAYERS=28
    DTYPE=float16
    ;;
  *)
    echo "[FATAL] unsupported MODEL_KEY=$MODEL_KEY" >&2
    exit 2
    ;;
esac

# Ada shared home currently cannot safely hold three additional 7--10B model
# copies.  Keep the reproducible public weights on this exact node's large
# scratch volume and pin all successor jobs to the same node.  Nodes may expose
# usable local capacity as either /scratch or /ssd_scratch, so the submission
# can select it explicitly with NODE_STORAGE_ROOT.  Scientific
# outputs, inventories and logs remain durable under shared home.
NODE_STORAGE_ROOT=${NODE_STORAGE_ROOT:-/scratch/rsm}
CHECKPOINT_ROOT=${NODE_STORAGE_ROOT}/anlp-vlm-mcqa/checkpoints/${MODEL_KEY}_${REVISION}
LAUNCH_ROOT=${BASE}/launch/next3_e20_${MODEL_KEY}_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}
RESULT_ROOT=${BASE}/results/next3_e20_${MODEL_KEY}_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}
mkdir -p "${BASE}/logs" "${BASE}/launch" "${BASE}/results" "$(dirname "$CHECKPOINT_ROOT")"
test ! -e "$LAUNCH_ROOT"
test ! -e "$RESULT_ROOT"
mkdir "$LAUNCH_ROOT" "$RESULT_ROOT"

echo "[runtime] job=${SLURM_JOB_ID} model=${MODEL_KEY} node=$(hostname -s) source=${SOURCE_GIT_COMMIT}"
hostname -s
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
df -h "$BASE" "$NODE_STORAGE_ROOT" || true

module load u22/cuda/12.1
if test -n "${VLM_ENV_ROOT:-}"; then
  test -x "${ENV_ROOT}/bin/python"
else
  source /home2/rsm/miniconda3/etc/profile.d/conda.sh
  conda activate motion-probe
fi
test -s "${ENV_ROOT}/completion.json"
export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${SOURCE_SNAPSHOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=180
export HF_HUB_ETAG_TIMEOUT=30

cd "$SOURCE_SNAPSHOT"
test "$(git rev-parse HEAD 2>/dev/null || printf '%s' "$SOURCE_GIT_COMMIT")" = "$SOURCE_GIT_COMMIT" || true
python - <<'PY'
import os
import torch, transformers
assert torch.cuda.is_available(), "CUDA unavailable"
cuts = [value for value in os.environ.get("E20_DEVICE_CUTS", "").replace(",", ":").split(":") if value]
expected = len(cuts) + 1 if cuts else 2
assert torch.cuda.device_count() == expected, (torch.cuda.device_count(), expected)
print({"torch": torch.__version__, "transformers": transformers.__version__,
       "cuda_devices": [torch.cuda.get_device_name(i) for i in range(expected)]})
PY

# Materialize each pinned public checkpoint once on the selected node.  A retry
# resumes the newest preserved partial directory from an interrupted download.
if test ! -s "${CHECKPOINT_ROOT}/stage_completion.json"; then
  STAGING_ROOT=$(find "${NODE_STORAGE_ROOT}/anlp-vlm-mcqa" -maxdepth 1 -type d \
    -name "${MODEL_KEY}_${REVISION}_*" -print 2>/dev/null | sort | tail -n 1)
  if test -z "$STAGING_ROOT"; then
    STAGING_ROOT=${NODE_STORAGE_ROOT}/anlp-vlm-mcqa/${MODEL_KEY}_${REVISION}_${SLURM_JOB_ID}
    test ! -e "$STAGING_ROOT"
  else
    echo "[resume] preserved partial checkpoint download: ${STAGING_ROOT}"
  fi
  mkdir -p "$(dirname "$STAGING_ROOT")"
  python - "$MODEL_ID" "$REVISION" "$STAGING_ROOT" <<'PY'
from huggingface_hub import snapshot_download
import sys
model_id, revision, target = sys.argv[1:]
print(snapshot_download(repo_id=model_id, revision=revision, local_dir=target,
                        ignore_patterns=["*.md", ".gitattributes", "original/*", "onnx/*"]))
PY
  python - "$STAGING_ROOT" "$MODEL_ID" "$REVISION" "$MODEL_KEY" <<'PY'
import json
from pathlib import Path
import sys
root, model_id, revision, model_key = Path(sys.argv[1]), *sys.argv[2:]
required = [root / "config.json", root / "model.safetensors.index.json"]
if any(not path.is_file() for path in required) or not list(root.glob("*.safetensors")):
    raise SystemExit("incomplete native safetensors checkpoint")
record = {"schema": "rachit.next-three-checkpoint-stage.v1", "status": "complete",
          "model_key": model_key, "hf_id": model_id, "revision": revision,
          "file_count": sum(path.is_file() for path in root.rglob("*")),
          "total_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file())}
(root / "stage_completion.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
print(record)
PY
  if test -e "$CHECKPOINT_ROOT"; then
    echo "[FATAL] another job created checkpoint without a valid completion marker" >&2
    exit 5
  fi
  mv "$STAGING_ROOT" "$CHECKPOINT_ROOT"
fi
test -s "${CHECKPOINT_ROOT}/stage_completion.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m vlm_mcqa.campaigns inventory --root "$SOURCE_SNAPSHOT" \
  --output "${LAUNCH_ROOT}/source_inventory.json" --kind source --commit "$SOURCE_GIT_COMMIT"
python -m vlm_mcqa.campaigns inventory --root "$CHECKPOINT_ROOT" \
  --output "${LAUNCH_ROOT}/checkpoint_inventory.json" --kind checkpoint \
  --hf-id "$MODEL_ID" --revision "$REVISION"

python - "$MODEL_KEY" "$NUM_LAYERS" "$DTYPE" "$SOURCE_SNAPSHOT" "$CHECKPOINT_ROOT" \
  "$DATASET_ROOT" "$QWEN_PREP" "$LAUNCH_ROOT" "$RESULT_ROOT" "$E20_DISCOVERY_TARGET" \
  "$E20_PRIMARY_INSUFFICIENT_JOB" "$E20_RETRY_FROM" "$E20_PREPARED_CACHE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

model_key, num_layers, dtype = sys.argv[1], int(sys.argv[2]), sys.argv[3]
source, checkpoint, data, qwen_prep, launch, results = map(Path, sys.argv[4:10])
target_override = sys.argv[10]
primary_insufficient_job = sys.argv[11]
retry_from = sys.argv[12]
prepared_cache = int(sys.argv[13])
if prepared_cache < 1:
    raise SystemExit("E20_PREPARED_CACHE must be positive")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def device_map():
    # Vision and the first half of the decoder share GPU 0.  The second half,
    # final norm, and unembedding use GPU 1.  This is explicit and contains no
    # CPU/disk/automatic placement.
    # InternVL needs the decoder weights almost evenly split: the earlier
    # 15/21 split left GPU 1 short by about 380 MiB during the real two-copy
    # calibration replay.  A 17/19 split moves two layers back to GPU 0 while
    # retaining more GPU-0 headroom than the original 18/18 placement.
    os = __import__("os")
    raw_cuts = os.environ.get("E20_DEVICE_CUTS", "")
    if raw_cuts:
        cuts = [int(value) for value in raw_cuts.replace(",", ":").split(":") if value]
    else:
        default_cut = 16 if model_key == "internvl35_8b" else 12
        cuts = [int(os.environ.get("E20_DEVICE_CUT") or default_cut)]
    if (not cuts or cuts != sorted(set(cuts)) or cuts[0] < 0
            or cuts[-1] >= num_layers - 1 or len(cuts) > 3):
        raise SystemExit("E20 device cuts must be unique increasing layer ends leaving work on every GPU")
    final_device = f"cuda:{len(cuts)}"
    mapping = {
        "model.vision_tower": "cuda:0",
        "model.multi_modal_projector": "cuda:0",
        "model.language_model.embed_tokens": "cuda:0",
        "model.language_model.rotary_emb": "cuda:0",
        "model.language_model.norm": final_device,
        "lm_head": final_device,
    }
    if model_key == "llava_ov_7b_si":
        # LLaVA-OneVision owns this parameter directly under `model`; it is
        # outside the vision tower/projector/language-model submodules.
        mapping["model.image_newline"] = "cuda:0"
    for layer in range(num_layers):
        device = next((index for index, cut in enumerate(cuts) if layer <= cut), len(cuts))
        mapping[f"model.language_model.layers.{layer}"] = f"cuda:{device}"
    return mapping

contrasts = qwen_prep / "fixed_text_contrasts.json"
rows = json.loads(contrasts.read_text())["contrasts"]
calibration = next(row for row in rows if row["split"] == "discovery")
protocol = source / f"configs/stretch/next_three/{model_key}.e20_protocol.json"
recipe = source / f"configs/stretch/next_three/{model_key}.e20_recipe.json"
decision = source / "configs/stretch/next_three_stretch_decision_2026-10-27.json"
head_source = source / f"consolidation/core7_release_results/{model_key}_core7/freeze.json"

# A model with fewer control-complete, clean-correct contrasts may run a
# prespecified reduced sensitivity after the 32-pair primary is recorded as
# insufficient.  The environment override changes only the discovery target,
# never the minimum, controls, layouts, or held-out confirmation protocol.
effective_protocol = json.loads(protocol.read_text())
evidence_status = "full_replication"
if target_override:
    if not primary_insufficient_job.isdigit():
        raise SystemExit("reduced E20 sensitivity requires E20_PRIMARY_INSUFFICIENT_JOB")
    target_override = int(target_override)
    available_floor = effective_protocol["sampling"]["discovery"]["min_pairs"]
    primary_target = effective_protocol["sampling"]["discovery"]["target_pairs"]
    if not available_floor <= target_override < primary_target:
        raise SystemExit("E20_DISCOVERY_TARGET must be >= min_pairs and below the frozen primary target")
    effective_protocol["sampling"]["discovery"]["target_pairs"] = target_override
    effective_protocol["evidence_status"] = "reduced_correct_only_sensitivity"
    effective_protocol["primary_insufficient_job"] = primary_insufficient_job
    effective_protocol["reduction_reason"] = (f"only {target_override} strict control-complete clean-correct "
                                               f"{model_key} discovery pairs exist")
    protocol = launch / "effective_protocol.json"
    protocol.write_text(json.dumps(effective_protocol, sort_keys=True, indent=2) + "\n")
    evidence_status = "reduced_correct_only_sensitivity"

base = {
    "schema": "vlm-mcqa-campaign/v1", "kind": "stretch", "seed": 0,
    "output_dir": str(results / "UNMATERIALIZED"),
    "source": {"root": str(source), "manifest": str(launch / "source_inventory.json")},
    "model": {"key": model_key,
              "models_config": str(source / "configs/core7/next_three_models_pinned.json"),
              "checkpoint_dir": str(checkpoint),
              "checkpoint_manifest": str(launch / "checkpoint_inventory.json")},
    "data": {"directory": str(data), "dataset_hash": "f5573e25f3aa2bf2",
             "image_aware_release_hash": "eb7b79e56977bbaeed5fc981c5c8e09a4e26eaf8b1702283abe2d4bbfc718cbd"},
    "deployment": {"dtype": dtype, "device_map": device_map(), "attn_implementation": "sdpa",
                   "offload_policy": "forbid", "batch_size": 1,
                   "prepared_cache": prepared_cache, "clean_cache": 96},
    # This remains a real batched-vs-single equivalence test.  Two copies are
    # sufficient to exercise the batched kernel and fit on 11 GiB GPUs.
    "session_profile": {"calibration_batch_size": 2},
    "protocol": {"decision_path": str(decision), "decision_sha256": sha(decision)},
    "study": {"name": "semantic_heads", "phases": ["discovery", "confirmation"],
              "config": {"protocol": str(protocol), "recipe": str(recipe),
                         "evidence_status": evidence_status,
                         "contrasts": str(contrasts),
                         "calibration_variant_ids": [calibration["donor_variant_id"], calibration["recipient_variant_id"]],
                         "calibration_contrast_id": calibration["contrast_id"], "tolerance": 0.06,
                         "protected_inputs": [str(data), str(head_source)]}}
}
if retry_from:
    base["retry_from"] = retry_from
confirmation_sensitivity = __import__("os").environ.get("E20_CONFIRMATION_SENSITIVITY_PAIRS", "")
if confirmation_sensitivity:
    if not retry_from:
        raise SystemExit("14-pair confirmation sensitivity requires E20_RETRY_FROM")
    confirmation_sensitivity = int(confirmation_sensitivity)
    base["study"]["config"]["confirmation_sample_sensitivity"] = {
        "target_pairs": confirmation_sensitivity,
        "min_pairs": confirmation_sensitivity,
        "primary_attempt": retry_from,
        "label": "reduced_heldout_sample_sensitivity",
        "reason": "run all deterministic control-complete held-out pairs retained by the frozen primary preparation",
    }

def write(name, config, output):
    value = json.loads(json.dumps(config))
    value["output_dir"] = str(output)
    (launch / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")

write("calibration.json", base, results / "calibration")
pilot_protocol = json.loads(protocol.read_text())
pilot_protocol["sampling"]["discovery"]["target_pairs"] = 8
pilot_protocol["sampling"]["discovery"]["min_pairs"] = 8
(launch / "pilot_protocol.json").write_text(json.dumps(pilot_protocol, sort_keys=True, indent=2) + "\n")
pilot = json.loads(json.dumps(base))
pilot["study"]["config"]["protocol"] = str(launch / "pilot_protocol.json")
write("pilot.json", pilot, results / "pilot")
write("science.json", base, results / "science")
(launch / "bootstrap.json").write_text(json.dumps({
    "schema": "rachit.next-three-e20-bootstrap.v1", "model_key": model_key,
    "num_layers": num_layers, "dtype": dtype, "device_map": device_map(),
    "calibration_contrast_id": calibration["contrast_id"], "evidence_status": evidence_status,
}, sort_keys=True, indent=2) + "\n")
PY

for config in calibration pilot science; do
  python -m vlm_mcqa.campaigns verify --config "${LAUNCH_ROOT}/${config}.json"
done

nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader -l 20 > "${LAUNCH_ROOT}/gpu_monitor.csv" &
MONITOR_PID=$!
cleanup_monitor() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup_monitor EXIT

if test -n "$E20_REUSE_TECHNICAL_RESULT_ROOT"; then
  python - "$MODEL_KEY" "$E20_REUSE_TECHNICAL_RESULT_ROOT" "${LAUNCH_ROOT}/reused_technical_evidence.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

model_key, root, output = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
calibration = json.loads((root / "calibration/calibration_result.json").read_text())
extension = json.loads((root / "pilot/extension_calibration.json").read_text())
measured = json.loads((root / "pilot/measured_study_result.json").read_text())
pilot = json.loads((root / "pilot/pilot_result.json").read_text())
attempt = json.loads((root / "pilot/attempt.json").read_text())
coverage = measured.get("coverage", {})
if attempt["config"]["model"]["key"] != model_key:
    raise SystemExit("reused pilot model mismatch")
if not (calibration.get("status") == "calibrated" and calibration.get("ordinary_all_passed") is True
        and calibration.get("study_all_passed") is True and extension.get("all_passed") is True):
    raise SystemExit("reused technical calibration did not pass")
if not (measured.get("execution_complete") is True
        and coverage.get("semantic_pairs") == 8 and coverage.get("semantic_evaluated") == 32
        and coverage.get("semantic_requested") == 32 and coverage.get("controls_evaluated") == 64
        and coverage.get("controls_requested") == 64 and coverage.get("unmatched_control_groups") == 0):
    raise SystemExit("reused pilot lacks complete eight-pair intervention coverage")
if not (pilot.get("measured_status") == "insufficient" and measured.get("scientific_verdict") == "insufficient_coverage"):
    raise SystemExit("reused pilot is not the known phase-minimum bookkeeping case")
files = [root / "calibration/calibration_result.json", root / "calibration/extension_calibration.json",
         root / "pilot/extension_calibration.json", root / "pilot/measured_study_result.json",
         root / "pilot/pilot/raw.jsonl", root / "pilot/pilot/measurements.csv"]
record = {"schema": "rachit.e20-reused-technical-evidence.v1", "status": "validated",
          "model_key": model_key, "source_result_root": str(root),
          "reason": "complete_eight_pair_pilot_was_false_insufficient_only_because_full_science_minimum_was_applied",
          "coverage": coverage,
          "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
print(record)
PY
else
  python -m vlm_mcqa.campaigns calibrate --config "${LAUNCH_ROOT}/calibration.json" --execute
  CAL_STATUS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "${RESULT_ROOT}/calibration/calibration_result.json")
  if test "$CAL_STATUS" != calibrated; then
    echo "[complete] calibration produced ${CAL_STATUS}; no scientific E20 launched"
    exit 0
  fi

  python -m vlm_mcqa.campaigns pilot --config "${LAUNCH_ROOT}/pilot.json" --execute
  PILOT_STATUS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "${RESULT_ROOT}/pilot/pilot_result.json")
  if test "$PILOT_STATUS" != technical_complete; then
    echo "[complete] pilot produced ${PILOT_STATUS}; no scientific E20 launched"
    exit 0
  fi
fi

python -m vlm_mcqa.campaigns run --config "${LAUNCH_ROOT}/science.json" --execute
cleanup_monitor
trap - EXIT

python - "$MODEL_KEY" "$SOURCE_GIT_COMMIT" "$LAUNCH_ROOT" "$RESULT_ROOT" <<'PY'
import json
from pathlib import Path
import sys
model_key, commit, launch, results = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
science_root = results / "science"
summary_path = science_root / "summary.json"
result_path = science_root / "result.json"
campaign_completion_path = science_root / "completion.json"
if summary_path.is_file():
    science_record = json.loads(summary_path.read_text())
    evidence_path = summary_path
elif result_path.is_file() and campaign_completion_path.is_file():
    # A valid insufficient campaign deliberately has no summary.json because
    # no held-out estimate exists.  Preserve that scientific outcome instead
    # of turning successful campaign bookkeeping into a Slurm failure.
    science_record = json.loads(result_path.read_text())
    campaign_completion = json.loads(campaign_completion_path.read_text())
    if campaign_completion.get("status") != "insufficient":
        raise SystemExit("missing science summary for a non-insufficient campaign")
    evidence_path = result_path
else:
    raise SystemExit("science run produced neither a summary nor a valid insufficient result")
record = {"schema": "rachit.next-three-e20-launch.v1", "status": "complete",
          "model_key": model_key, "source_git_commit": commit,
          "launch_root": str(launch), "result_root": str(results),
          "science_status": science_record.get("status"),
          "scientific_verdict": science_record.get("scientific_verdict"),
          "science_evidence": str(evidence_path),
          "scientific_run_performed": True}
(launch / "completion.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
print(record)
PY
test -s "${LAUNCH_ROOT}/gpu_monitor.csv"
test -s "${LAUNCH_ROOT}/completion.json"
echo "[complete] ${MODEL_KEY} E20 bootstrap/calibration/pilot/science"
