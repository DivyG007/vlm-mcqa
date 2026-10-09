# Tuned Lens on Jarvislabs — runbook

Instance: A100 80GB **spot** (₹84.24/h), PyTorch container, 150 GB storage (expand on
resume for the full four-model run). Every phase is resumable, so a pre-empted spot instance
is resumed and `run_model.sh` re-run with the same arguments. W&B: `vlm-mcqa/tuned-lens`.

## 0. Before creating the instance (laptop, branch worktree)

The branch must be committed so every run records an exact commit (owner approval).

```bash
cd ../ANLP-project-tuned-lens
C=$(git rev-parse --short HEAD)
git bundle create /tmp/tuned_lens_$C.bundle codex/divyanshu-tuned-lens
```

## 1. Create and log in (owner)

```bash
jl create --gpu A100-80GB --spot --storage 150 --name tuned-lens --template pytorch
jl list                       # instance ID
jl ssh <ID>
pip install -U wandb "huggingface_hub[cli]"
wandb login                   # key from https://wandb.ai/authorize
hf auth login                 # read token; needed for gated PaliGemma
exit
```

## 2. Upload and set up

```bash
SSH=$(jl ssh <ID> --print-command)
$SSH 'mkdir -p /home/tuned_lens/bundles'
jl upload <ID> /tmp/tuned_lens_$C.bundle /home/tuned_lens/bundles/
jl upload <ID> ../ANLP-project/clevr_mcq4_v3_official3k_cuda45pairs_handoff_20261001.tar.gz /home/tuned_lens/bundles/
jl upload <ID> scripts/tuned_lens/setup_instance.sh /home/tuned_lens/
$SSH 'bash /home/tuned_lens/setup_instance.sh 2>&1 | tee /home/tuned_lens/logs/setup.log'
```

Setup verifies the dataset (image-aware hash `eb7b79e5…`), runs the tuned-lens CPU tests and
reports whether W&B is logged in.

## 3. Trial run (Qwen pilot) — only after owner approval

```bash
$SSH 'tmux new -d -s tl_pilot "bash /home/tuned_lens/repo/scripts/tuned_lens/run_model.sh qwen25vl_7b pilot_01 --pilot"'
$SSH 'tail -5 /home/tuned_lens/runs/qwen25vl_7b/pilot_01/run.log'
```

The pilot (12 train + 4 val scenes, 50 steps per stage, 16 behavior items) is exploratory and
measures seconds per prompt, seconds per training step, peak GPU memory and checkpoint size;
`estimate.json` projects the full run with a 25–40% margin. The full-run corpus and
hyperparameters stay frozen as in `configs/tuned_lens/` regardless of pilot lens outcomes.

## 4. Full runs (one model at a time, owner approval per model)

```bash
$SSH 'tmux new -d -s tl_qwen "bash /home/tuned_lens/repo/scripts/tuned_lens/run_model.sh qwen25vl_7b full_01"'
```

Order: `qwen25vl_7b` → `internvl35_8b` → `llava_ov_7b_si` → `paligemma2_10b_mix` (after
Yashav's Pali quirk report). After pre-emption: `jl resume <ID>` and re-run the same command.

## 5. Bring back compact evidence, then pause/destroy

```bash
for f in cache/cache_manifest.json translators/train_summary.json eval/comparison_behavior.json run.log; do
  jl download <ID> /home/tuned_lens/runs/<model>/<run>/$f runs/<model>/<run>/$f
done
jl download <ID> -r /home/tuned_lens/runs/<model>/<run>/eval/tables runs/<model>/<run>/eval/tables
jl download <ID> -r /home/tuned_lens/runs/<model>/<run>/eval/figures runs/<model>/<run>/eval/figures
jl pause <ID>       # keeps storage (billed); jl destroy <ID> ends all billing
```

Hidden-state caches, translator checkpoints and weights stay off Git.
