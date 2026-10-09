"""``python -m vlm_mcqa.tuned_lens <command> ...``

Commands (each writes into one run directory per model, ``--out``):

cache     load the model, run Core-7 calibration, cache fit-position stages
train     fit translators (one stage at a time; resumable)
evaluate  Logit vs Tuned Lens on the Core-7 behavior split (M4/M5/M7 rows)
estimate  project full-run time/cost from a measured pilot run directory
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "configs" / "tuned_lens"


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _configs(args) -> dict[str, Any]:
    root = Path(args.config_dir)
    return {"corpus": _read(root / "corpus.json"), "train": _read(root / "train.json"),
            "models": _read(root / "models.json"), "wandb": _read(root / "wandb.json")}


def _tracker(args, configs, *, out_dir: Path, job: str, extra: dict[str, Any]):
    from ..core7.tracking import Tracker

    os.environ.setdefault("CORE7_WANDB_ENTITY", configs["wandb"]["entity"])
    os.environ.setdefault("CORE7_WANDB_PROJECT", configs["wandb"]["project"])
    tag = "pilot" if args.pilot else "full"
    return Tracker.start(enabled=not args.no_wandb, out_dir=out_dir, job_type=job, group=args.model,
                         name=f"{args.model}_{job}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}",
                         tags=["tuned-lens", tag, args.model], config={"model_key": args.model, "pilot": args.pilot,
                                                                       "command": job, **extra})


def _adapter(args, configs):
    from ..core7.adapters import VLMAdapter, load_model_specs

    entry = configs["models"][args.model]
    spec = load_model_specs(REPO / entry["models_config"])[args.model]
    spec = dataclasses.replace(spec, device_map=entry.get("device_map", "cuda"),
                               attn_implementation=entry.get("attn_implementation", spec.attn_implementation))
    started = time.time()
    adapter = VLMAdapter.load(spec, cache_dir=args.hf_cache)
    print(f"[load] {spec.key} {spec.hf_id}@{spec.revision} layers={adapter.num_layers} "
          f"in {time.time() - started:.1f}s", flush=True)
    return adapter, spec


def _session(adapter, args, out_dir: Path):
    from ..core7.session import Session, load_profiles

    profile = dict(load_profiles()["profiles"]["full_core7"])
    return Session(adapter, args.data, out_dir, profile, seed=0)


def _calibrate(adapter, args, out_dir: Path) -> dict[str, Any]:
    from ..core7.calibration import run_calibration

    session = _session(adapter, args, out_dir)
    report = run_calibration(session)
    session.write_json("calibration.json", report)
    failed = [k for k, g in report["gates"].items() if not g["passed"]]
    print(f"[calibration] all_passed={report['all_passed']} failed={failed} softcap={report['softcap']}", flush=True)
    if not report["all_passed"]:
        raise RuntimeError(f"Core-7 calibration failed: {failed}")
    return report


def _manifest(args, configs, spec, adapter) -> dict[str, Any]:
    from ..core7.session import provenance

    return provenance({"schema": "vlm-mcqa-tuned-lens-run/v1", "model_key": spec.key, "hf_id": spec.hf_id,
                       "revision": spec.revision, "dtype": spec.dtype, "attn": spec.attn_implementation,
                       "num_layers": adapter.num_layers, "translator_stages": adapter.num_layers,
                       "hidden_size": int(adapter.handle.info["hidden_size"]),
                       "vocab_size": int(adapter.handle.lm_head.weight.shape[0]),
                       "translator_params": adapter.num_layers * (adapter.handle.info["hidden_size"] ** 2
                                                                  + adapter.handle.info["hidden_size"]),
                       "pilot": args.pilot, "configs": configs, "data_dir": str(args.data)})


def cmd_cache(args) -> None:
    import torch

    from .cache import build_cache
    from .corpus import load_corpus

    configs = _configs(args)
    out = Path(args.out)
    corpus = load_corpus(args.data, configs["corpus"], pilot=args.pilot)
    adapter, spec = _adapter(args, configs)
    manifest = _manifest(args, configs, spec, adapter)
    tracker = _tracker(args, configs, out_dir=out, job="cache",
                       extra={"corpus": corpus["manifest"], "manifest": {k: v for k, v in manifest.items() if k != "configs"}})
    try:
        calibration = _calibrate(adapter, args, out / "calibration")
        tracker.summary({"all_passed": calibration["all_passed"],
                         "gates": {k: {"value": g["value"], "passed": g["passed"]} for k, g in calibration["gates"].items()}},
                        "calibration")
        from PIL import Image
        images: dict[str, Any] = {}

        def load_image(relative: str):
            if relative not in images:
                if len(images) >= 64:
                    images.clear()
                with Image.open(Path(args.data) / relative) as loaded:
                    images[relative] = loaded.convert("RGB")
            return images[relative]

        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        started = time.time()
        results = {}
        for split in ("train", "val"):
            results[split] = build_cache(adapter, load_image, corpus[split], out / "cache" / split,
                                         shard_size=int(configs["corpus"]["shard_size"]), tracker=tracker)
        seconds = time.time() - started
        gate_limit = float(configs["corpus"]["final_gate_limit"])
        worst = max(r["max_final_gate"] for r in results.values())
        cache_manifest = {**manifest, "corpus": corpus["manifest"], "splits": results, "softcap": calibration["softcap"],
                          "calibration_all_passed": calibration["all_passed"], "cache_seconds": seconds,
                          "seconds_per_variant": seconds / max(sum(r["variants"] for r in results.values()), 1),
                          "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
                          "final_gate_max": worst, "final_gate_limit": gate_limit,
                          "final_gate_passed": worst <= gate_limit, "wandb_url": tracker.url}
        _write(out / "cache" / "cache_manifest.json", cache_manifest)
        tracker.summary({k: cache_manifest[k] for k in ("cache_seconds", "seconds_per_variant", "peak_gpu_gib",
                                                       "final_gate_max", "final_gate_passed")}, "cache")
        tracker.summary({f"{s}_tokens": r["tokens"] for s, r in results.items()}, "cache")
        if not cache_manifest["final_gate_passed"]:
            raise RuntimeError(f"final-stage lens differs from model logits by {worst} > {gate_limit}")
        tracker.finish("complete")
    except BaseException:
        tracker.finish("failed")
        raise


def cmd_train(args) -> None:
    import torch

    from .cache import load_stage
    from .train import train_stage
    from .translator import TunedLens

    configs = _configs(args)
    out = Path(args.out)
    cache_manifest = _read(out / "cache" / "cache_manifest.json")
    if not cache_manifest.get("final_gate_passed"):
        raise RuntimeError("cache gates did not pass; refusing to train")
    train_config = dict(configs["train"])
    if args.pilot:
        train_config["steps"] = int(train_config["pilot_steps"])
    adapter, spec = _adapter(args, configs)
    adapter.handle.softcap = cache_manifest["softcap"]  # selected by Core-7 calibration during caching
    num_layers, hidden = adapter.num_layers, int(adapter.handle.info["hidden_size"])
    if (num_layers, hidden) != (cache_manifest["num_layers"], cache_manifest["hidden_size"]):
        raise RuntimeError("model shape differs from the cache manifest")
    device = adapter.handle.lm_head.weight.device
    lens = TunedLens(num_layers, hidden).to(device)
    stages = [int(s) for s in args.stages.split(",")] if args.stages else list(range(num_layers))
    tracker = _tracker(args, configs, out_dir=out, job="train",
                       extra={"train": train_config, "stages": stages, "cache_manifest_softcap": cache_manifest["softcap"]})
    try:
        final_train = load_stage(out / "cache" / "train", num_layers)
        final_val = load_stage(out / "cache" / "val", num_layers)
        summaries = []
        for stage in stages:
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            summary = train_stage(adapter.handle, lens, stage,
                                  (load_stage(out / "cache" / "train", stage), final_train),
                                  (load_stage(out / "cache" / "val", stage), final_val),
                                  train_config, out / "translators", tracker=tracker)
            summaries.append({k: v for k, v in summary.items() if k != "history"})
            _write(out / "translators" / "train_summary.json",
                   {"model_key": spec.key, "train_config": train_config, "pilot": args.pilot, "stages": summaries,
                    "wandb_url": tracker.url})
        tracker.finish("complete")
    except BaseException:
        tracker.finish("failed")
        raise


def cmd_evaluate(args) -> None:
    from .evaluate import compare_reference, run_evaluation, write_figures
    from .translator import load_lens

    configs = _configs(args)
    out = Path(args.out)
    cache_manifest = _read(out / "cache" / "cache_manifest.json")
    adapter, spec = _adapter(args, configs)
    lens, hashes = load_lens(out / "translators", adapter.num_layers, int(adapter.handle.info["hidden_size"]))
    lens = lens.to(adapter.handle.lm_head.weight.device)
    tracker = _tracker(args, configs, out_dir=out, job="evaluate",
                       extra={"translator_sha256": hashes, "n_items": args.n_items})
    try:
        calibration = _calibrate(adapter, args, out / "eval")
        if calibration["softcap"] != cache_manifest["softcap"]:
            raise RuntimeError("softcap selected at evaluation differs from the cache")
        session = _session(adapter, args, out / "eval")
        summary = run_evaluation(session, lens, n_items=args.n_items, tracker=tracker)
        if args.reference:
            summary["reference_check"] = compare_reference(out / "eval" / "lens_behavior_logit.jsonl", args.reference)
            tracker.summary(summary["reference_check"], "reference")
        summary["figures"] = write_figures(out / "eval", "behavior")
        summary["translator_sha256"] = hashes
        _write(out / "eval" / "comparison_behavior.json", summary)
        tracker.figures(out / "eval")
        tracker.artifact(out / "eval", f"{spec.key}_tuned_lens_eval")
        tracker.finish("complete")
        print(json.dumps(summary["milestones"], indent=2), flush=True)
    except BaseException:
        tracker.finish("failed")
        raise


def cmd_estimate(args) -> None:
    """Project full-corpus time from a pilot run directory (measured, plus margin)."""
    configs = _configs(args)
    out = Path(args.out)
    cache = _read(out / "cache" / "cache_manifest.json")
    train = _read(out / "translators" / "train_summary.json")
    pilot_steps = int(train["train_config"]["steps"])
    full_steps = int(configs["train"]["steps"])
    per_step = sum(s["seconds"] for s in train["stages"]) / max(len(train["stages"]), 1) / pilot_steps
    variants_full = int(args.full_variants)
    stages = cache["num_layers"]
    cache_h = cache["seconds_per_variant"] * variants_full / 3600
    train_h = per_step * full_steps * stages / 3600
    eval_h = cache["seconds_per_variant"] * 2 * int(args.eval_variants) / 3600
    base = cache_h + train_h + eval_h
    result = {"model_key": cache["model_key"], "measured_seconds_per_variant": cache["seconds_per_variant"],
              "measured_seconds_per_train_step": per_step, "peak_gpu_gib_cache": cache["peak_gpu_gib"],
              "peak_gpu_gib_train": max(s["peak_gpu_gib"] for s in train["stages"]),
              "hours": {"cache": cache_h, "train": train_h, "evaluate": eval_h, "total": base,
                        "with_25pct_margin": base * 1.25, "with_40pct_margin": base * 1.40},
              "note": "evaluate assumes ~2x a cache forward per behavior variant (decoder replay + full forward)"}
    _write(out / "estimate.json", result)
    print(json.dumps(result, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("cache", "train", "evaluate", "estimate"))
    parser.add_argument("--model", required=True, help="key in configs/tuned_lens/models.json")
    parser.add_argument("--data", type=Path, required=True, help="frozen CLEVR-MCQ-4 v3 dataset directory")
    parser.add_argument("--out", type=Path, required=True, help="run directory for this model")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--hf-cache", help="Hugging Face cache directory")
    parser.add_argument("--pilot", action="store_true", help="exploratory pilot corpus and step count")
    parser.add_argument("--stages", help="comma-separated stages to train (default: all)")
    parser.add_argument("--n-items", type=int, help="limit behavior items in evaluate (pilot only)")
    parser.add_argument("--reference", type=Path, help="released Core-7 lens_behavior.jsonl for the parity check")
    parser.add_argument("--full-variants", type=int, default=5040)
    parser.add_argument("--eval-variants", type=int, default=3072)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(argv)
    if args.n_items and not args.pilot:
        parser.error("--n-items is only allowed for exploratory pilot runs")
    {"cache": cmd_cache, "train": cmd_train, "evaluate": cmd_evaluate, "estimate": cmd_estimate}[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
