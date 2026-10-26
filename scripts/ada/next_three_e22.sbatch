#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=next3_e22
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:3
#SBATCH --time=1-00:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/next3_e22_%x_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/next3_e22_%x_%j.err

set -euo pipefail
set -x
case "$(hostname -s)" in
  gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077) exit 90 ;;
esac

: "${MODEL_KEY:?MODEL_KEY required}"
: "${SOURCE_GIT_COMMIT:?exact full source commit required}"
: "${SOURCE_SNAPSHOT:?immutable source snapshot required}"
: "${E20_SCIENCE_CONFIG:?completed model-specific E20 config required}"
: "${E22_PREP_ROOT:?outcome-blind model-specific E22 preparation required}"
: "${E22_OPTION_LAYERS:?translated candidate layers required}"
: "${E22_NUM_LAYERS:?decoder layer count required}"
: "${IMAGE_REPAIR_ENABLED:=false}"
: "${E20_RESTORATION_EVIDENCE_SHA256:=}"

BASE=${VLM_BASE:-/home2/rsm/anlp-vlm-mcqa}
ENV_ROOT=${VLM_ENV_ROOT:-${BASE}/envs/e20_qwen_torch28_tf4576_40839}
LAUNCH_ROOT=${BASE}/launch/next3_e22_${MODEL_KEY}_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}
ATTEMPT_ROOT=${BASE}/results/next3_e22_${MODEL_KEY}_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}
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
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2
print({"visible_gpus": torch.cuda.device_count(),
       "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
PY
python -m vlm_mcqa.campaigns inventory --root "$SOURCE_SNAPSHOT" \
  --output "${LAUNCH_ROOT}/source_inventory.json" --kind source --commit "$SOURCE_GIT_COMMIT"

python - "$MODEL_KEY" "$SOURCE_GIT_COMMIT" "$SOURCE_SNAPSHOT" "$E20_SCIENCE_CONFIG" \
  "$E22_PREP_ROOT" "$LAUNCH_ROOT" "$ATTEMPT_ROOT" "$E22_OPTION_LAYERS" \
  "$E22_NUM_LAYERS" "$IMAGE_REPAIR_ENABLED" "$E20_RESTORATION_EVIDENCE_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(model_key, commit, source_raw, e20_raw, prep_raw, launch_raw, attempt_raw,
 layers_raw, num_layers_raw, image_raw, evidence_raw) = sys.argv[1:]
source, e20, prep, launch, attempt = map(Path, (source_raw, e20_raw, prep_raw,
                                                launch_raw, attempt_raw))
layers = [int(value) for value in layers_raw.replace(":", ",").split(",") if value]
num_layers = int(num_layers_raw)
image_enabled = image_raw.lower() == "true"
evidence = evidence_raw or None
if (not layers or layers != sorted(set(layers))
        or any(not 0 <= layer < num_layers for layer in layers)):
    raise SystemExit("E22 candidate layers must be unique, increasing and model-valid")
if image_enabled != (evidence is not None):
    raise SystemExit("image-repair decision and E20 evidence identity must agree")
if evidence is not None and (len(evidence) != 64 or any(c not in "0123456789abcdef" for c in evidence)):
    raise SystemExit("E20 evidence must be a lowercase SHA256")
base = json.loads(e20.read_text())
if base["model"]["key"] != model_key:
    raise SystemExit("E20 config/model mismatch")
calibration = json.loads((prep / "calibration.json").read_text())
rules = json.loads((prep / "rules.json").read_text())
if rules["stage"] != num_layers:
    raise SystemExit("E22 rules must read content at the model's final hidden stage")
head = source / f"consolidation/core7_release_results/{model_key}_core7/freeze.json"
if not head.is_file():
    raise SystemExit(f"missing native frozen heads: {head}")
decision = {
    "schema": "rachit.next-three-e22-decision.v1",
    "approved": True,
    "study": "failures",
    "model_key": model_key,
    "source_git_commit": commit,
    "dataset_hash": "f5573e25f3aa2bf2",
    "design": "model-native truth-aligned-content/wrong-symbol E22 replication",
    "rationale": ("Replicate the frozen Qwen E22 design with outcome-blind model-specific "
                  "candidate cohorts, native heads, translated option layers, and disjoint confirmation."),
    "option_candidate_layers": layers,
    "image_repair_enabled": image_enabled,
    "e20_restoration_evidence_sha256": evidence,
}
decision_path = launch / "decision.json"
decision_path.write_text(json.dumps(decision, sort_keys=True, indent=2) + "\n")
protocol = {
    "allowed_strata": ["truth_aligned_content_with_wrong_final_symbol"],
    "discovery_allowed_splits": ["discovery", "behavior"],
    "strata_rules_sha256": rules["rules_sha256"],
    "strata_rules_path": str(prep / "rules.json"),
    "tie_tolerance": 0.0,
    "layer_tie_tolerance": 0.01,
    "head_source": str(head),
    "head_source_sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
    "head_k": 4,
    "random_sets": 20,
    "seed": 0,
    "bootstrap_seed": 0,
    "bootstrap_samples": 2000,
    "min_stratum_scenes": 16,
    "max_donor_reuse": 1,
    "option_scope": "truth",
    "option_candidate_layers": layers,
    "image_repair_enabled": image_enabled,
    "e20_restoration_evidence_sha256": evidence,
    "sites": {
        "image": {"site": "resid_post", "layer": 0},
        "options": {"site": "resid_post", "layer": layers[0]},
        "final": {"site": "resid_post", "layer": num_layers - 1},
    },
    "donor_classes": ["same_image_remap", "semantic_match", "unrelated_same_label", "wrong_donor"],
    "primary_conditions_by_donor_class": {
        "same_image_remap": ["options_repair", "head_repair"],
        "semantic_match": ["options_repair", "head_repair"],
        "unrelated_same_label": ["head_repair"],
        "wrong_donor": ["head_repair"],
    },
}
study = {
    "name": "failures",
    "phase": "discovery",
    "split": "behavior",
    "batch_size": 1,
    "candidate_scope_path": str(prep / "discovery_scope.json"),
    "calibration_variant_ids": calibration["calibration_variant_ids"],
    "calibration_case": calibration["calibration_case"],
    "tolerance": 0.06,
    "protected_inputs": [str(prep), base["data"]["directory"], str(head.parent)],
    "protocol": protocol,
    "phase_settings": {"confirmation": {
        "candidate_scope_path": str(prep / "confirmation_scope.json"),
        "split": "confirmation",
    }},
}
config = {
    "schema": base["schema"],
    "kind": "stretch",
    "seed": 0,
    "output_dir": str(attempt),
    "source": {"root": str(source), "manifest": str(launch / "source_inventory.json")},
    "model": {**base["model"],
              "models_config": str(source / "configs/core7/next_three_models_pinned.json")},
    "data": base["data"],
    "deployment": base["deployment"],
    "session_profile": base.get("session_profile", {"calibration_batch_size": 2}),
    "protocol": {"decision_path": str(decision_path),
                 "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest()},
    "study": {"name": "failures", "phases": ["discovery", "confirmation"], "config": study},
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
output.write_text(json.dumps({"status": "complete", "primary_status": completion["status"],
                              "scientific_verdict": completion.get("scientific_verdict"),
                              "attempt_root": str(attempt)}, sort_keys=True, indent=2) + "\n")
PY
