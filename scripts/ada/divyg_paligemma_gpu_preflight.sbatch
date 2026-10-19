#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --job-name=divyg_pali_gpucheck
#SBATCH --nodelist=gnode0[51-99]
#SBATCH --exclude=gnode065,gnode066,gnode077
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:2
#SBATCH --time=0-00:10:00
#SBATCH --output=/home2/divyg/anlp-vlm-mcqa/logs/divyg_pali_gpucheck_%j.log
#SBATCH --error=/home2/divyg/anlp-vlm-mcqa/logs/divyg_pali_gpucheck_%j.err

set -euo pipefail
set -x
case "$(hostname -s)" in
  gnode0[0-4][0-9]|gnode050|gnode065|gnode066|gnode077) exit 90 ;;
esac
: "${SOURCE_GIT_COMMIT:?exact source commit required}"
: "${VLM_ENV_ROOT:?verified cu126 runtime required}"

test -s "${VLM_ENV_ROOT}/completion.json"
export PATH="${VLM_ENV_ROOT}/bin:${PATH}"
python - "$SOURCE_GIT_COMMIT" <<'PY'
import json
from pathlib import Path
import sys
import torch

commit = sys.argv[1]
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
rows = []
for index in range(2):
    device = torch.device(f"cuda:{index}")
    props = torch.cuda.get_device_properties(device)
    row = {
        "logical_index": index,
        "name": props.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory": props.total_memory,
        "kernels": {},
    }
    for dtype in (torch.float16, torch.bfloat16):
        try:
            left = torch.randn(512, 512, device=device, dtype=dtype)
            right = torch.randn(512, 512, device=device, dtype=dtype)
            value = (left @ right).float()
            torch.cuda.synchronize(device)
            row["kernels"][str(dtype)] = bool(torch.isfinite(value).all().item())
        except Exception as error:
            row["kernels"][str(dtype)] = type(error).__name__
    rows.append(row)
record = {
    "schema": "divyg.paligemma-gpu-preflight.v1",
    "status": "complete",
    "source_git_commit": commit,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpus": rows,
}
output = Path(f"/home2/divyg/anlp-vlm-mcqa/preflight/paligemma_gpu_{__import__('os').environ['SLURM_JOB_ID']}.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
print(json.dumps(record, sort_keys=True, indent=2))
PY
