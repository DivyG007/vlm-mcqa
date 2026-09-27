"""Command-line entry point: ``python -m vlm_mcqa.core7 <command> ...``.

Commands
--------
calibrate  hook/replay calibration gates for one model
screen     calibration + behavioural screening (Step 4; also dataset calibration)
select     choose family winners and the two Core-7 models from screen runs
run        full Core-7 on one split (discovery, or confirmation with --freeze)
freeze     freeze discovery decisions (layers, components, heads) to JSON
confirm    test a confirmation run against a freeze file
plot       re-render figures for a run directory
compare    cross-model normalized-depth comparison and completion matrix
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

RUN_METHODS = ("behavior", "lens", "patching", "components", "heads")


def _enforce_device_placement(adapter) -> None:
    """Fail closed when a run explicitly requires real multi-GPU sharding."""
    if os.environ.get("CORE7_REQUIRE_MULTI_GPU") != "1":
        return
    placement = adapter.describe()
    device_map = placement.get("hf_device_map")
    if not device_map:
        raise RuntimeError("CORE7_REQUIRE_MULTI_GPU=1 but the model has no hf_device_map")
    devices = set(device_map.values())
    forbidden = {d for d in devices if d in {"cpu", "disk", "meta"}}
    if forbidden:
        raise RuntimeError(f"unplanned model offload in hf_device_map: {sorted(forbidden)}")
    cuda_devices = {d.removeprefix("cuda:") for d in devices
                    if d.isdigit() or d.startswith("cuda:")}
    if len(cuda_devices) < 2:
        raise RuntimeError(f"multi-GPU required but hf_device_map uses {sorted(devices)}")
    print(f"[placement] verified sharding across logical CUDA devices {sorted(cuda_devices)}", flush=True)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="key in configs/core7/models.json")
    parser.add_argument("--data", type=Path, required=True, help="finalized CLEVR-MCQ-4 directory")
    parser.add_argument("--out", type=Path, required=True, help="result directory (created)")
    parser.add_argument("--profile", default="smoke", help="smoke | full_core7")
    parser.add_argument("--models-config", type=Path)
    parser.add_argument("--profiles-config", type=Path)
    parser.add_argument("--cache-dir", help="Hugging Face cache directory (default: HF_HOME)")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", help="e.g. cuda, cuda:1, auto")
    parser.add_argument("--attn", help="attention implementation override (sdpa, eager)")
    parser.add_argument("--batch-size", type=int, help="override profile batch size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-uncalibrated", action="store_true",
                        help="continue even if calibration gates fail (development only)")
    _wandb_flag(parser)


def _wandb_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-wandb", action="store_true",
                        help="do not log to W&B (destination is configurable with CORE7_WANDB_ENTITY/PROJECT)")


def _tracker(args, *, out_dir: Path, name: str, job_type: str, config: dict, tags=(), group=None):
    from .tracking import Tracker

    enabled = not args.no_wandb and os.environ.get("CORE7_NO_WANDB") != "1"
    return Tracker.start(enabled=enabled, out_dir=out_dir, name=name, job_type=job_type,
                         config=config, tags=list(tags), group=group)


def _session(args):
    from .adapters import VLMAdapter, load_model_specs
    from .session import Session, load_profiles, provenance

    specs = load_model_specs(args.models_config)
    spec = specs[args.model]
    overrides = {k: v for k, v in (("dtype", args.dtype), ("device_map", args.device_map),
                                    ("attn_implementation", args.attn)) if v}
    spec = dataclasses.replace(spec, **overrides)
    profiles = load_profiles(args.profiles_config)
    profile = dict(profiles["profiles"][args.profile])
    if args.batch_size:
        profile["batch_size"] = args.batch_size
    print(f"[load] {spec.key} <- {spec.hf_id} dtype={spec.dtype} device_map={spec.device_map}", flush=True)
    started = time.time()
    adapter = VLMAdapter.load(spec, cache_dir=args.cache_dir)
    _enforce_device_placement(adapter)
    print(f"[load] done in {time.time() - started:.1f}s layers={adapter.num_layers}", flush=True)
    session = Session(adapter, args.data, args.out, profile, seed=args.seed)
    run = provenance({
        "command": sys.argv, "model_key": spec.key, "family": spec.family, "hf_id": spec.hf_id,
        "params_b": spec.params_b, "lm_backbone": spec.lm_backbone, "model_spec": dataclasses.asdict(spec),
        "profile_name": args.profile, "profile": profile, "data_dir": str(args.data),
        "dataset_hash": session.manifest.get("dataset_hash"), "num_layers": adapter.num_layers,
        "image_aware_release_hash": session.release_manifest.get("image_aware_release_hash"),
        "adapter": adapter.describe(), "load_seconds": time.time() - started, "status": "running",
    })
    job = args.command + (f"-{args.split}" if getattr(args, "split", None) else "")
    session.tracker = _tracker(
        args, out_dir=args.out, name=f"{spec.key}_{job}_{args.profile}_{time.strftime('%Y%m%d_%H%M%S')}",
        job_type=job, group=spec.key, tags=[spec.family, args.profile, getattr(args, "split", None)],
        config={"model_key": spec.key, "family": spec.family, "hf_id": spec.hf_id, "params_b": spec.params_b,
                "lm_backbone": spec.lm_backbone, "dtype": spec.dtype, "attn": spec.attn_implementation,
                "profile_name": args.profile, "profile": profile, "dataset_hash": run["dataset_hash"],
                "image_aware_release_hash": run["image_aware_release_hash"],
                "data_dir": str(args.data), "num_layers": adapter.num_layers, "seed": args.seed,
                "source_git_commit": run["source_git_commit"], "command": job})
    run["wandb_url"] = session.tracker.url
    session.write_json("run.json", run)
    return session, run, profiles


def _calibrate(session, args) -> dict:
    from .calibration import run_calibration

    report = run_calibration(session)
    failed = [k for k, g in report["gates"].items() if not g["passed"]]
    print(f"[calibration] all_passed={report['all_passed']} failed={failed} softcap={report['softcap']}", flush=True)
    session.tracker.summary({"all_passed": report["all_passed"],
                             "gates": {k: {"value": g["value"], "passed": g["passed"]} for k, g in report["gates"].items()},
                             "informational": report["informational"]}, "calibration")
    if failed and not args.allow_uncalibrated:
        raise SystemExit(f"calibration failed: {failed} (see calibration.json)")
    return report


def _finish(session, run, status: str, **extra) -> None:
    run.update(status=status, finished_unix=time.time(), **extra)
    session.write_json("run.json", run)
    tracker = getattr(session, "tracker", None)
    if tracker is not None:
        tracker.summary({"method_seconds": run.get("method_seconds", {})})
        tracker.artifact(session.out_dir, f"{run['model_key']}-{run.get('split') or 'screen'}-{run['profile_name']}")
        tracker.finish(status)


def cmd_calibrate(args) -> None:
    session, run, _ = _session(args)
    args.allow_uncalibrated = True
    report = _calibrate(session, args)
    _finish(session, run, "complete", calibration_passed=report["all_passed"])


def cmd_screen(args) -> None:
    from .behavior import run_behavior

    session, run, _ = _session(args)
    try:
        cal = _calibrate(session, args)
        p = session.profile
        summary = run_behavior(session, split=args.split, n_items=p["screen_items"], robust_items=p["robust_items"],
                               compliance_items=p["compliance_items"], controls=True,
                               pair_splits=("discovery", "confirmation") if args.split == "screening" else ())
        print(json.dumps({k: summary[k] for k in ("letters_accuracy", "worst_position_accuracy",
                                                  "clean_minus_shuffled", "pair_yield")}, indent=2, default=str))
        session.tracker.summary(summary, f"behavior_{args.split}")
        _finish(session, run, "complete", calibration_passed=cal["all_passed"])
    except BaseException as error:
        _finish(session, run, "failed", error=repr(error), traceback=traceback.format_exc())
        raise


def cmd_run(args) -> None:
    from .aggregate import aggregate_run
    from .behavior import run_behavior
    from .components import run_components
    from .heads import load_freeze, run_heads
    from .lens import run_lens
    from .patching import run_patching
    from .plots import plot_run

    methods = [m for m in RUN_METHODS if m in args.methods.split(",")]
    if args.split == "confirmation" and not args.freeze:
        raise SystemExit("confirmation runs must pass --freeze (decisions frozen on discovery)")
    freeze = load_freeze(args.freeze)
    session, run, _ = _session(args)
    run.update(split=args.split, methods=methods, freeze=str(args.freeze) if args.freeze else None,
               freeze_hash=(freeze or {}).get("freeze_hash"),
               components_from=str(args.components_from) if args.components_from else None)
    if args.components_from:
        if args.split != "discovery" or methods != ["heads"] or freeze:
            raise SystemExit("--components-from is only valid for a heads-only discovery correction")
        source_dir = Path(args.components_from)
        source_run_path = source_dir / "run.json"
        source_rows = source_dir / "components_discovery.jsonl"
        if not source_run_path.is_file() or not source_rows.is_file():
            raise SystemExit("--components-from must contain run.json and components_discovery.jsonl")
        source_run = json.loads(source_run_path.read_text())
        for field in ("model_key", "dataset_hash", "image_aware_release_hash"):
            if source_run.get(field) != run.get(field):
                raise SystemExit(f"component source {field} differs from the correction run")
        if source_run.get("status") != "complete" or source_rows.stat().st_size == 0:
            raise SystemExit("component source is incomplete or empty")
        run["components_source_git_commit"] = source_run.get("source_git_commit")
        run["components_source_sha256"] = hashlib.sha256(source_rows.read_bytes()).hexdigest()
        session.write_json("run.json", run)
    if freeze and freeze.get("model_key") != args.model:
        raise SystemExit(f"freeze file belongs to {freeze.get('model_key')}, not {args.model}")
    if freeze and (freeze.get("dataset_hash") != run["dataset_hash"]
                   or freeze.get("source_git_commit") != run["source_git_commit"]):
        raise SystemExit("freeze dataset hash or source commit differs from the confirmation run")
    if freeze and freeze.get("image_aware_release_hash") not in (None, run.get("image_aware_release_hash")):
        raise SystemExit("freeze image-aware release hash differs from the confirmation run")
    try:
        _calibrate(session, args)
        p = session.profile
        timings = {}
        for method in methods:
            started = time.time()
            print(f"[run] {method} on {args.split}", flush=True)
            if method == "behavior":
                run_behavior(session, split="behavior", n_items=p["behavior_items"], robust_items=p["behavior_items"],
                             compliance_items=min(p["compliance_items"], p["behavior_items"]), controls=True,
                             pair_splits=(args.split,))
            elif method == "lens":
                run_lens(session, split="behavior", n_items=p["behavior_items"])
            elif method == "patching":
                result = run_patching(session, split=args.split)
            elif method == "components":
                result = run_components(session, split=args.split)
            elif method == "heads":
                result = run_heads(session, split=args.split, freeze=freeze,
                                   components_dir=args.components_from)
            if method in ("behavior", "lens"):
                result = json.loads((args.out / f"{method}_behavior_summary.json").read_text())
            timings[method] = time.time() - started
            run["method_seconds"] = timings
            session.write_json("run.json", run)
            session.tracker.log({method: result, "seconds": {method: timings[method]}})
        agg = aggregate_run(args.out, args.split)
        figures = plot_run(args.out)
        session.tracker.aggregate(agg, args.split)
        session.tracker.figures(args.out, figures)
        _finish(session, run, "complete", figures=figures)
        print(f"[run] complete -> {args.out}", flush=True)
    except BaseException as error:
        _finish(session, run, "failed", error=repr(error), traceback=traceback.format_exc())
        raise


def cmd_freeze(args) -> None:
    from .aggregate import freeze

    frozen = freeze(args.run, args.out)
    print(json.dumps(frozen, indent=2))
    tracker = _tracker(args, out_dir=args.out.parent, name=f"{frozen['model_key']}_freeze_{frozen['freeze_hash']}",
                       job_type="freeze", group=frozen["model_key"], config=frozen)
    tracker.summary(frozen, "freeze")
    tracker.finish()


def cmd_freeze_heads(args) -> None:
    from .aggregate import freeze_heads

    frozen = freeze_heads(args.run, args.out)
    print(json.dumps(frozen, indent=2))


def cmd_confirm_heads(args) -> None:
    from .aggregate import confirm_heads

    report = confirm_heads(args.run, args.freeze)
    print(json.dumps(report["checks"], indent=2))
    if not report["checks"]["execution_complete"]:
        raise SystemExit("heads-only confirmation evidence is incomplete or inconsistent")


def cmd_confirm(args) -> None:
    from .aggregate import confirm

    report = confirm(args.run, args.freeze)
    print(json.dumps(report["checks"], indent=2, default=str))
    model = json.loads((Path(args.run) / "run.json").read_text()).get("model_key", "unknown")
    tracker = _tracker(args, out_dir=Path(args.run), name=f"{model}_confirm_{report['freeze_hash']}",
                       job_type="confirm", group=model, config={"model_key": model, "freeze_hash": report["freeze_hash"]})
    tracker.summary(report["checks"], "confirm")
    tracker.finish()
    if not report["checks"]["execution_complete"]:
        raise SystemExit("confirmation evidence is incomplete or inconsistent")


def cmd_plot(args) -> None:
    from .aggregate import aggregate_run
    from .plots import plot_run

    for split in ("discovery", "confirmation"):
        if (Path(args.run) / f"patching_{split}.jsonl").exists():
            aggregate_run(args.run, split)
    print("\n".join(plot_run(args.run)))


def cmd_select(args) -> None:
    from .plots import plot_screening
    from .report import select
    from .session import load_profiles

    dirs = sorted(p.parent for p in Path(args.screen_root).glob("*/behavior_screening_summary.json"))
    thresholds = load_profiles(args.profiles_config)["selection"]
    decision = select(dirs, thresholds, args.out)
    figures = plot_screening(decision, args.out)
    print((args.out / "screening_table.md").read_text())
    tracker = _tracker(args, out_dir=args.out, name=f"selection_{time.strftime('%Y%m%d_%H%M%S')}",
                       job_type="select", config={"thresholds": thresholds, "screen_root": str(args.screen_root)})
    if tracker.run:
        import wandb

        rows = decision["table"]
        columns = sorted({k for r in rows for k in r if not isinstance(r[k], (list, dict))})
        tracker.run.log({"screening_table": wandb.Table(columns=columns, data=[[r.get(c) for c in columns] for r in rows]),
                         **{f"figures/{Path(f).stem}": wandb.Image(str(args.out / f)) for f in figures}})
        tracker.run.summary.update({"core7_models": decision["core7_models"],
                                    "family_winners": {f: w["model_key"] for f, w in decision["family_winners"].items()}})
    tracker.artifact(args.out, "selection")
    tracker.finish()


def cmd_compare(args) -> None:
    from .plots import plot_compare
    from .report import compare

    result = compare(args.runs, args.out)
    figures = plot_compare(result, args.out)
    print("\n".join(figures))
    tracker = _tracker(args, out_dir=args.out, name=f"compare_{time.strftime('%Y%m%d_%H%M%S')}",
                       job_type="compare", config={"runs": [str(r) for r in args.runs]})
    if tracker.run:
        import wandb

        tracker.run.log({f"figures/{Path(f).stem}": wandb.Image(str(args.out / f)) for f in figures})
    for entry in result["models"]:
        tracker.summary({k: v for k, v in entry.items() if k.endswith(":depth")}, entry["model_key"])
    tracker.artifact(args.out, "comparison")
    tracker.finish()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m vlm_mcqa.core7", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("calibrate")
    _common(p)
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("screen")
    _common(p)
    p.add_argument("--split", default="screening", help="screening (model selection) or calibration (dataset difficulty)")
    p.set_defaults(func=cmd_screen)
    p = sub.add_parser("run")
    _common(p)
    p.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    p.add_argument("--methods", default=",".join(RUN_METHODS))
    p.add_argument("--components-from", type=Path,
                   help="existing compatible run directory supplying M2 rows for a heads-only discovery rerun")
    p.add_argument("--freeze", type=Path)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("freeze")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    _wandb_flag(p)
    p.set_defaults(func=cmd_freeze)
    p = sub.add_parser("freeze-heads")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_freeze_heads)
    p = sub.add_parser("confirm-heads")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.set_defaults(func=cmd_confirm_heads)
    p = sub.add_parser("confirm")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    _wandb_flag(p)
    p.set_defaults(func=cmd_confirm)
    p = sub.add_parser("plot")
    p.add_argument("--run", type=Path, required=True)
    p.set_defaults(func=cmd_plot)
    p = sub.add_parser("select")
    p.add_argument("--screen-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profiles-config", type=Path)
    _wandb_flag(p)
    p.set_defaults(func=cmd_select)
    p = sub.add_parser("compare")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    _wandb_flag(p)
    p.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
