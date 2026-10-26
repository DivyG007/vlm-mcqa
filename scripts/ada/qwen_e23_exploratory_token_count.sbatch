#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=rachit_e23_explore
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e23_explore_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e23_explore_%j.err
set -euo pipefail
set -x
case "$(hostname -s)" in gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077) exit 90;; esac
: "${SOURCE_GIT_COMMIT:?}"
: "${SOURCE_ROOT:?}"
: "${SOURCE_MANIFEST:?}"
: "${E20_ENV_ROOT:?}"
: "${TASK_PREP_ROOT:?}"
: "${E23_IMAGE_LAYERS:=0,8,16,23}"
: "${E23_HEAD_COUNT:=4}"
: "${E23_QUERY_GROUPS:=final:options}"

BASE=/home2/rsm/anlp-vlm-mcqa
LAUNCH_ROOT="${BASE}/launch/spatial_exploratory_token_count_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}"
ATTEMPT_ROOT="${BASE}/results/spatial_exploratory_token_count_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}"
mkdir -p "${BASE}/logs" "${BASE}/launch"
test ! -e "${LAUNCH_ROOT}"
test ! -e "${ATTEMPT_ROOT}"
mkdir "${LAUNCH_ROOT}"

module load u22/cuda/12.1
source /home2/rsm/miniconda3/etc/profile.d/conda.sh
conda activate motion-probe
export PATH="${E20_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${SOURCE_ROOT}"
python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
PY

python - "${SOURCE_ROOT}" "${SOURCE_MANIFEST}" "${TASK_PREP_ROOT}" "${LAUNCH_ROOT}" "${ATTEMPT_ROOT}" "${E23_IMAGE_LAYERS}" "${E23_HEAD_COUNT}" "${E23_QUERY_GROUPS}" <<'PY'
import hashlib, json, sys
from pathlib import Path

source_root, source_manifest, prep, launch, attempt, image_layers_raw, head_count_raw, query_groups_raw = sys.argv[1:]
source_root, prep, launch, attempt = map(Path, (source_root, prep, launch, attempt))
image_layers = [int(x) for x in image_layers_raw.replace(':', ',').split(',') if x]
head_count = int(head_count_raw)
query_groups = [x for x in query_groups_raw.replace(',', ':').split(':') if x]
if not image_layers or len(image_layers) != len(set(image_layers)) or any(not 0 <= x < 28 for x in image_layers):
    raise SystemExit('E23_IMAGE_LAYERS must be unique comma-separated layers in [0,27]')
if head_count not in {4, 16}:
    raise SystemExit('E23_HEAD_COUNT must be 4 or 16')
if (len(query_groups) != len(set(query_groups)) or 'final' not in query_groups or 'options' not in query_groups
        or any(x not in {'final', 'options', 'text_question', 'visual_options'} for x in query_groups)):
    raise SystemExit('E23_QUERY_GROUPS must uniquely include final and options')
base = Path('/home2/rsm/anlp-vlm-mcqa')
old = json.loads((base / 'prepared/e20_qwen_6505882a9dee55c4c4ad0eaf73bbab400a8910aa_cc594898137f460bfe9f0759e9844b3ce807cfb5/campaign_base.json').read_text())
decision = {
  'schema': 'rachit.qwen-e23-exploratory-decision.v1', 'approved': True,
  'study': 'spatial', 'evidence_status': 'exploratory_sensitivity_not_frozen_primary',
  'model_key': 'qwen25vl_7b', 'revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
  'dataset_hash': 'f5573e25f3aa2bf2', 'source_git_commit': source_root.name,
  'primary_failure_job': '41211',
  'rationale': 'The frozen translated-shape control was geometrically infeasible for the disconnected aggregate option region before scientific inference. This sensitivity run preserves all other rules and uses equal-token background controls.',
}
if image_layers != [0, 8, 16, 23]:
    decision.update({
      'schema': 'rachit.qwen-e23-layer-closure-decision.v1',
      'evidence_status': 'posthoc_closure_sensitivity',
      'rationale': 'This closure sensitivity tests whether the original coarse residual grid missed an intermediate spatial layer. It preserves the original result and uses a fixed expanded grid.',
      'image_layers': image_layers,
    })
if head_count != 4 or query_groups != ['final', 'options']:
    decision.update({
      'schema': 'rachit.qwen-e23-head-query-closure-decision.v1',
      'evidence_status': 'posthoc_closure_sensitivity',
      'head_count': head_count,
      'attention_query_groups': query_groups,
    })
decision_path = launch / 'decision.json'
decision_path.write_text(json.dumps(decision, sort_keys=True, indent=2) + '\n')
head = source_root / 'consolidation/core7_release_results/qwen25vl_7b_core7/m3_heads.json'
protocol = {
  'target_units': 24, 'min_units': 16, 'seed': 0, 'max_tokens': 1024,
  'max_region_tokens': 256, 'calibration_tasks': 2, 'bootstrap_samples': 2000,
  'random_head_sets': 20, 'query_budget_per_group': 8, 'tolerance': 0.06,
  'minimum_accuracy': 0.75, 'minimum_image_gap': 0.20, 'minimum_effect': 0.0,
  'layer_tie_tolerance': 0.01, 'mapping_rule': 'center', 'min_fraction': None,
  'control_mode': 'token_count', 'attention_query_groups': query_groups,
  'image_layers': image_layers, 'head_count': head_count,
}
study = {
  'name': 'spatial', 'phase': 'discovery', 'split': 'discovery', 'model_key': 'qwen25vl_7b',
  'head_freeze_path': str(head), 'head_freeze_dataset_hash': 'f5573e25f3aa2bf2',
  'tasks_path': str(prep / 'discovery/tasks.json'),
  'calibration_tasks_path': str(prep / 'calibration/tasks.json'),
  'task_assets_root': str(base / 'inputs/clevr_mcq4_v3_full'),
  'protected_inputs': [str(prep), str(base / 'inputs/clevr_mcq4_v3_full'), str(head.parent)],
  'protocol': protocol,
  'phase_settings': {'confirmation': {'tasks_path': str(prep / 'confirmation/tasks.json')}},
  'evidence_status': 'exploratory_sensitivity_not_frozen_primary',
}
config = {
  'schema': old['schema'], 'kind': 'stretch', 'seed': 0, 'output_dir': str(attempt),
  'source': {'root': str(source_root), 'manifest': source_manifest},
  'model': {**old['model'], 'models_config': str(source_root / 'configs/core7/qwen25vl_7b_pinned.json')},
  'data': old['data'], 'deployment': old['deployment'],
  'protocol': {'decision_path': str(decision_path),
               'decision_sha256': hashlib.sha256(decision_path.read_bytes()).hexdigest()},
  'study': {'name': 'spatial', 'phases': ['discovery', 'confirmation'], 'config': study},
}
(launch / 'campaign.json').write_text(json.dumps(config, sort_keys=True, indent=2) + '\n')
PY

python -m unittest tests.test_campaigns_study_integration tests.test_stretch_prospective_tasks -v
python -m vlm_mcqa.campaigns verify --config "${LAUNCH_ROOT}/campaign.json"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 15 > "${LAUNCH_ROOT}/gpu_monitor.csv" &
MONITOR_PID=$!
cleanup(){ kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT
python -m vlm_mcqa.campaigns run --config "${LAUNCH_ROOT}/campaign.json" --execute
cleanup
trap - EXIT
python - "${ATTEMPT_ROOT}" "${LAUNCH_ROOT}/completion.json" <<'PY'
import json,sys
from pathlib import Path
attempt,out=map(Path,sys.argv[1:])
c=json.loads((attempt/'completion.json').read_text())
if c.get('status') not in {'complete','insufficient'}: raise SystemExit(c)
out.write_text(json.dumps({'status':'complete','primary_status':c['status'],
  'evidence_status':'exploratory_sensitivity_not_frozen_primary',
  'scientific_verdict':c.get('scientific_verdict'),'attempt_root':str(attempt)},indent=2,sort_keys=True)+'\n')
PY
