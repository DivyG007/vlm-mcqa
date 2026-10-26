#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=rachit_e24_explore
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e24_explore_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/rachit_e24_explore_%j.err
set -euo pipefail
set -x
case "$(hostname -s)" in gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077) exit 90;; esac
: "${SOURCE_GIT_COMMIT:?}"
: "${SOURCE_ROOT:?}"
: "${SOURCE_MANIFEST:?}"
: "${E20_ENV_ROOT:?}"
: "${TASK_PREP_ROOT:?}"

BASE=/home2/rsm/anlp-vlm-mcqa
LAUNCH_ROOT="${BASE}/launch/image_options_exploratory_no_unrelated_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}"
ATTEMPT_ROOT="${BASE}/results/image_options_exploratory_no_unrelated_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}"
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

python - "${SOURCE_ROOT}" "${SOURCE_MANIFEST}" "${TASK_PREP_ROOT}" "${LAUNCH_ROOT}" "${ATTEMPT_ROOT}" <<'PY'
import hashlib, json, sys
from pathlib import Path
source_root, source_manifest, prep, launch, attempt = sys.argv[1:]
source_root, prep, launch, attempt = map(Path, (source_root, prep, launch, attempt))
base = Path('/home2/rsm/anlp-vlm-mcqa')
old = json.loads((base / 'prepared/e20_qwen_6505882a9dee55c4c4ad0eaf73bbab400a8910aa_cc594898137f460bfe9f0759e9844b3ce807cfb5/campaign_base.json').read_text())
decision = {
  'schema': 'rachit.qwen-e24-exploratory-decision.v1', 'approved': True,
  'study': 'image_options', 'evidence_status': 'exploratory_sensitivity_not_frozen_primary',
  'model_key': 'qwen25vl_7b', 'revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
  'dataset_hash': 'f5573e25f3aa2bf2', 'source_git_commit': source_root.name,
  'primary_result_job': '41210',
  'rationale': 'Frozen primary job 41210 was valid but insufficient solely because every cross-parent semantic-matched unrelated donor was unavailable. This sensitivity preserves the prospective cohort and all estimand controls except that structurally impossible auxiliary donor control.',
}
decision_path = launch / 'decision.json'
decision_path.write_text(json.dumps(decision, sort_keys=True, indent=2) + '\n')
head = source_root / 'consolidation/core7_release_results/qwen25vl_7b_core7/m3_heads.json'
root = prep / 'e24'
protocol = {
  'target_units': 32, 'min_units': 24, 'seed': 0, 'max_tokens': 2048,
  'max_region_tokens': 512, 'calibration_tasks': 2, 'bootstrap_samples': 2000,
  'random_head_sets': 20, 'query_budget_per_group': 8, 'tolerance': 0.06,
  'minimum_accuracy': 0.75, 'minimum_image_gap': 0.20, 'minimum_effect': 0.0,
  'layer_tie_tolerance': 0.01, 'mapping_rule': 'center', 'min_fraction': None,
  'control_mode': 'translated_shape', 'attention_query_groups': ['final'],
  'image_layers': [0, 8, 16, 23], 'head_count': 4,
  'require_unrelated_donor_control': False,
}
study = {
  'name': 'image_options', 'phase': 'discovery', 'split': 'discovery', 'model_key': 'qwen25vl_7b',
  'head_freeze_path': str(head), 'head_freeze_dataset_hash': 'f5573e25f3aa2bf2',
  'tasks_path': str(root / 'discovery/tasks.json'),
  'calibration_tasks_path': str(root / 'calibration/tasks.json'),
  'task_assets_root': str(root),
  'protected_inputs': [str(prep), str(base / 'inputs/clevr_mcq4_v3_full'), str(head.parent)],
  'protocol': protocol,
  'phase_settings': {'confirmation': {'tasks_path': str(root / 'confirmation/tasks.json')}},
  'evidence_status': 'exploratory_sensitivity_not_frozen_primary',
}
config = {
  'schema': old['schema'], 'kind': 'stretch', 'seed': 0, 'output_dir': str(attempt),
  'source': {'root': str(source_root), 'manifest': source_manifest},
  'model': {**old['model'], 'models_config': str(source_root / 'configs/core7/qwen25vl_7b_pinned.json')},
  'data': old['data'], 'deployment': old['deployment'],
  'protocol': {'decision_path': str(decision_path),
               'decision_sha256': hashlib.sha256(decision_path.read_bytes()).hexdigest()},
  'study': {'name': 'image_options', 'phases': ['discovery', 'confirmation'], 'config': study},
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
