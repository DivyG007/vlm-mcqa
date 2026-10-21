"""Explicit future CPU-analysis CLI. Importing this module performs no work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import validate_output_path
from .evidence import exposure_inventory, read_rows, source_record, synthesize_core7
from .failure_analysis import analyze_failures


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_new(output: Path, report: dict, inputs: list[Path]) -> None:
    """Only a new directory, never inside an immutable input tree."""
    protected = [source if source.is_dir() else source.parent for source in inputs]
    output = validate_output_path(output, [*inputs, *protected])
    # Transport/source bookkeeping is added by the CLI before finalization.
    # Bind those final bytes' semantic payload too, rather than retaining a hash
    # computed before provenance fields were attached.
    if "manifest_sha256" in report:
        body = {key: value for key, value in report.items() if key != "manifest_sha256"}
        report["manifest_sha256"] = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    # Strict serialization: do not persist NaN as valid scientific measurements.
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")
    from .reporting import write_analysis_artifacts
    write_analysis_artifacts(output, report)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    synthesis = commands.add_parser("synthesis", help="Read curated exports; never infer new outcomes")
    synthesis.add_argument("--curated-root", type=Path, required=True)
    synthesis.add_argument("--output", type=Path, required=True)
    failures = commands.add_parser("failures", help="Descriptive saved behavior/lens join")
    failures.add_argument("--dataset", type=Path, required=True)
    failures.add_argument("--behavior", type=Path, required=True)
    failures.add_argument("--lens", type=Path, required=True)
    failures.add_argument("--scope", type=Path, required=True,
                          help="JSON object with prospective stages and expected_variant_ids")
    failures.add_argument("--tie-tolerance", type=float, default=0.0)
    failures.add_argument("--output", type=Path, required=True)
    design = commands.add_parser("failure-design", help="Freeze descriptive strata and matched correct controls")
    design.add_argument("--dataset", type=Path, required=True)
    design.add_argument("--report", type=Path, required=True)
    design.add_argument("--rules", type=Path, required=True,
                        help="Prospective stage, flags, reliability, scene counts and matching rules")
    design.add_argument("--output", type=Path, required=True)
    exposure = commands.add_parser("exposure", help="Known-used union, never an untouched verdict")
    exposure.add_argument("--dataset", type=Path, required=True)
    exposure.add_argument("--sources", type=Path, required=True,
                          help="JSON object: sources{name:path}, required_source_names[list]")
    exposure.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be a new directory")
    if args.command == "synthesis":
        report = synthesize_core7(args.curated_root)
        inputs = [args.curated_root]
    else:
        from .data import load_dataset

        dataset = load_dataset(args.dataset)
        if args.command == "failure-design":
            from .failure_protocol import evaluate_failure_rules, freeze_failure_rules
            rules = freeze_failure_rules(_json(args.rules))
            report = evaluate_failure_rules(_json(args.report), dataset, rules)
            paths = [args.report, args.rules]
            inputs = [args.dataset, *paths]
            report["sources"] = [source_record(p) for p in paths]
        elif args.command == "failures":
            paths = [args.behavior, args.lens, args.scope]
            missing = [str(p) for p in paths if not p.is_file()]
            if missing:
                report = {"schema_version": "stretch.failures.v1", "status": "missing_input",
                          "missing_inputs": missing,
                          "warning": "Unavailable raw data is not evidence of zero failures."}
            else:
                scope = _json(args.scope)
                report = analyze_failures(dataset, read_rows(args.behavior), read_rows(args.lens),
                                          stages=scope["stages"], expected_variant_ids=scope["expected_variant_ids"],
                                          tie_tolerance=args.tie_tolerance)
            inputs = [args.dataset, *paths]
            report["sources"] = [source_record(p) for p in paths]
        else:
            specification = _json(args.sources)
            # Relative paths are relative to the source inventory, not the shell cwd.
            paths = {name: (args.sources.parent / Path(path)).resolve()
                     for name, path in specification["sources"].items()}
            report = exposure_inventory(dataset, paths, specification["required_source_names"])
            report["source_specification"] = source_record(args.sources)
            inputs = [args.dataset, args.sources, *paths.values()]
        report["dataset_provenance"] = dataset.get("provenance")
    report["execution_boundary"] = "CPU saved-evidence analysis only; no inference, services or compute-host selection"
    report["provenance_boundary"] = ("Input byte hashes recorded, but source commit/model revision/calibration "
                                      "bindings require separate audit against historical run manifests.")
    _write_new(args.output, report, inputs)


if __name__ == "__main__":
    main()
