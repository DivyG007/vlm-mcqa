#!/bin/bash
#SBATCH -A rsm
#SBATCH --partition=u22
#SBATCH --job-name=rachit_qwen_e22
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --output=/home2/rsm/anlp-vlm-mcqa/logs/rachit_qwen_e22_%j.log
#SBATCH --error=/home2/rsm/anlp-vlm-mcqa/logs/rachit_qwen_e22_%j.err
set -euo pipefail
set -x
case "$(hostname -s)" in gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077) exit 90;; esac
: "${SOURCE_GIT_COMMIT:?}"
: "${SOURCE_ROOT:?}"
: "${SOURCE_MANIFEST:?}"
: "${E20_ENV_ROOT:?}"
: "${E22_PREP_ROOT:?}"
: "${IMAGE_REPAIR_ENABLED:=false}"
: "${E20_RESTORATION_EVIDENCE_SHA256:=}"
: "${E22_DONOR_SCOPE:=primary}"
: "${E22_OPTION_LAYERS:=20,23,25}"
: "${E22_OPTION_SCOPE:=truth}"
: "${E22_OPTION_SITE:=resid_post}"

BASE=/home2/rsm/anlp-vlm-mcqa
REVISION=cc594898137f460bfe9f0759e9844b3ce807cfb5
LAUNCH_ROOT="${BASE}/launch/failures_qwen_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}"
ATTEMPT_ROOT="${BASE}/results/failures_qwen_${SOURCE_GIT_COMMIT}_${SLURM_JOB_ID}"
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

python - "${SOURCE_ROOT}" "${SOURCE_MANIFEST}" "${E22_PREP_ROOT}" "${LAUNCH_ROOT}" "${ATTEMPT_ROOT}" "${IMAGE_REPAIR_ENABLED}" "${E20_RESTORATION_EVIDENCE_SHA256}" "${E22_DONOR_SCOPE}" "${E22_OPTION_LAYERS}" "${E22_OPTION_SCOPE}" "${E22_OPTION_SITE}" <<'PY'
import hashlib, json, sys
from pathlib import Path

source_root, source_manifest, prep, launch, attempt = sys.argv[1:6]
image_enabled = sys.argv[6].lower() == 'true'
e20_evidence = sys.argv[7] or None
donor_scope = sys.argv[8]
option_layers = [int(x) for x in sys.argv[9].replace(':', ',').split(',') if x]
option_scope = sys.argv[10]
option_site = sys.argv[11]
source_root, prep, launch, attempt = map(Path, (source_root, prep, launch, attempt))
if donor_scope not in {'primary', 'covered', 'same_image_remap'}:
    raise SystemExit('E22_DONOR_SCOPE must be primary, covered, or same_image_remap')
if not option_layers or len(option_layers) != len(set(option_layers)) or any(not 0 <= x < 27 for x in option_layers):
    raise SystemExit('E22_OPTION_LAYERS must be unique comma-separated layers in [0,26]')
if option_scope not in {'truth', 'all'}:
    raise SystemExit('E22_OPTION_SCOPE must be truth or all')
if option_site not in {'resid_pre', 'resid_post'}:
    raise SystemExit('E22_OPTION_SITE must be resid_pre or resid_post')
if image_enabled and (e20_evidence is None or len(e20_evidence) != 64):
    raise SystemExit('enabled E22 image repair requires a 64-hex E20 evidence SHA')
if not image_enabled and e20_evidence is not None:
    raise SystemExit('disabled E22 image repair cannot claim E20 evidence')
base = Path('/home2/rsm/anlp-vlm-mcqa')
old = json.loads((base / 'prepared/e20_qwen_6505882a9dee55c4c4ad0eaf73bbab400a8910aa_cc594898137f460bfe9f0759e9844b3ce807cfb5/campaign_base.json').read_text())
calibration = json.loads((prep / 'calibration.json').read_text())
rules = json.loads((prep / 'rules.json').read_text())
decision = {
  'schema': 'rachit.qwen-e22-decision.v1', 'approved': True, 'study': 'failures',
  'model_key': 'qwen25vl_7b', 'revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
  'dataset_hash': 'f5573e25f3aa2bf2', 'source_git_commit': source_root.name,
  'design': '2026-10-27 frozen Qwen E22 truth-aligned-content/wrong-symbol stratum',
  'rationale': 'User authorized Qwen-first E22 with 32 control-complete discovery scenes, blind held-out confirmation preparation, layers 20/23/25, and layer-27 diagnostic reference.',
}
if donor_scope in {'covered', 'same_image_remap'}:
    decision.update({
      'schema': 'rachit.qwen-e22-exploratory-decision.v1',
      'evidence_status': 'exploratory_sensitivity_not_frozen_primary',
      'primary_result_job': '41280',
      'design': ('E22 sensitivity restricted to the control-complete same-image remap donor class'
                 if donor_scope == 'same_image_remap'
                 else 'E22 sensitivity restricted to the two donor classes with measured primary-run coverage'),
      'rationale': ('The all-option audit had 27 failure and 26 matched-control scenes for same_image_remap, but semantic_match had only 15 matched controls and blocked layer selection. This bounded successor isolates the prespecified control-complete donor class.'
                    if donor_scope == 'same_image_remap'
                    else 'Frozen primary job 41280 was valid but insufficient because unrelated_same_label and wrong_donor had zero evaluable scenes. This labeled sensitivity preserves all other rules and tests only same_image_remap and semantic_match.'),
    })
if option_layers != [20, 23, 25]:
    decision.update({
      'schema': 'rachit.qwen-e22-layer-closure-decision.v1',
      'evidence_status': 'posthoc_closure_sensitivity',
      'design': 'E22 prespecified expanded late-layer audit on the covered donor cohort',
      'rationale': 'This closure sensitivity tests whether the original negative arose from sparse layer candidates. It preserves the original result and evaluates a fixed expanded late-layer grid.',
      'option_candidate_layers': option_layers,
    })
if option_scope != 'truth' or option_site != 'resid_post':
    decision.update({
      'schema': 'rachit.qwen-e22-site-scope-closure-decision.v1',
      'evidence_status': 'posthoc_closure_sensitivity',
      'design': 'E22 option-token scope and residual-boundary audit',
      'option_scope': option_scope,
      'option_site': option_site,
    })
decision_path = launch / 'decision.json'
decision_path.write_text(json.dumps(decision, sort_keys=True, indent=2) + '\n')
head = source_root / 'consolidation/core7_release_results/qwen25vl_7b_core7/m3_heads.json'
protocol = {
  'allowed_strata': ['truth_aligned_content_with_wrong_final_symbol'],
  'discovery_allowed_splits': ['discovery', 'behavior'],
  'strata_rules_sha256': rules['rules_sha256'], 'strata_rules_path': str(prep / 'rules.json'),
  'tie_tolerance': 0.0, 'layer_tie_tolerance': 0.01,
  'head_source': str(head), 'head_source_sha256': hashlib.sha256(head.read_bytes()).hexdigest(),
  'head_k': 4, 'random_sets': 20, 'seed': 0, 'bootstrap_seed': 0, 'bootstrap_samples': 2000,
  'min_stratum_scenes': 16, 'max_donor_reuse': 1, 'option_scope': option_scope,
  'option_candidate_layers': option_layers,
  'image_repair_enabled': image_enabled, 'e20_restoration_evidence_sha256': e20_evidence,
  'sites': {'image': {'site': 'resid_post', 'layer': 0},
            'options': {'site': option_site, 'layer': 20},
            'final': {'site': 'resid_post', 'layer': 27}},
  'donor_classes': ['same_image_remap', 'semantic_match', 'unrelated_same_label', 'wrong_donor'],
  'primary_conditions_by_donor_class': {
    'same_image_remap': ['options_repair', 'head_repair'],
    'semantic_match': ['options_repair', 'head_repair'],
    'unrelated_same_label': ['head_repair'],
    'wrong_donor': ['head_repair'],
  },
}
if donor_scope == 'covered':
  protocol['donor_classes'] = ['same_image_remap', 'semantic_match']
  protocol['primary_conditions_by_donor_class'] = {
    'same_image_remap': ['options_repair', 'head_repair'],
    'semantic_match': ['options_repair', 'head_repair'],
  }
elif donor_scope == 'same_image_remap':
  protocol['donor_classes'] = ['same_image_remap']
  protocol['primary_conditions_by_donor_class'] = {
    'same_image_remap': ['options_repair', 'head_repair'],
  }
study = {
  'name': 'failures', 'phase': 'discovery', 'split': 'behavior', 'batch_size': 1,
  'strata_manifest': str(prep / 'discovery_strata.json'), 'strata_phase': 'discovery',
  'calibration_variant_ids': calibration['calibration_variant_ids'],
  'calibration_case': calibration['calibration_case'], 'tolerance': 0.06,
  'protected_inputs': [str(prep), str(base / 'inputs/clevr_mcq4_v3_full'), str(head.parent)],
  'protocol': protocol,
  'phase_settings': {'confirmation': {'candidate_scope_path': str(prep / 'confirmation_scope.json'),
                                      'split': 'confirmation'}},
}
if donor_scope in {'covered', 'same_image_remap'}:
  study['evidence_status'] = 'exploratory_sensitivity_not_frozen_primary'
config = {
  'schema': old['schema'], 'kind': 'stretch', 'seed': 0, 'output_dir': str(attempt),
  'source': {'root': str(source_root), 'manifest': source_manifest},
  'model': {**old['model'], 'models_config': str(source_root / 'configs/core7/qwen25vl_7b_pinned.json')},
  'data': old['data'], 'deployment': old['deployment'],
  'protocol': {'decision_path': str(decision_path),
               'decision_sha256': hashlib.sha256(decision_path.read_bytes()).hexdigest()},
  'study': {'name': 'failures', 'phases': ['discovery', 'confirmation'], 'config': study},
}
(launch / 'campaign.json').write_text(json.dumps(config, sort_keys=True, indent=2) + '\n')
PY

python -m unittest tests.test_campaigns_study_integration tests.test_stretch_failure_preparation tests.test_stretch_failure_protocol -v
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
  'scientific_verdict':c.get('scientific_verdict'),'attempt_root':str(attempt)},indent=2,sort_keys=True)+'\n')
PY
