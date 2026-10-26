#!/usr/bin/env python3
"""Aggregate next-three attempt results without turning insufficiency into zero."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vlm_mcqa.stretch.replication_prep import classify_result  # noqa: E402


def summarize(attempt: Path) -> dict:
    result_path = attempt / "result.json"
    failure_path = attempt / "failure.json"
    if failure_path.is_file() or not result_path.is_file():
        return {"attempt": str(attempt), "status": "failed_or_incomplete",
                "result_label": classify_result(technical_complete=False, behavior_sufficient=False,
                                                controls_valid=False, confirmation_complete=False,
                                                primary_positive=None)}
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    status = result.get("status")
    verdict = str(result.get("scientific_verdict", ""))
    behavior = "behavior" not in verdict or "insufficient" not in verdict
    controls = not any(word in verdict for word in ("control", "geometry", "incomplete"))
    complete = status == "complete" and "confirmation" in result.get("phases", result)
    positive_words = ("replicated", "selective", "rescue_and", "positive")
    negative_words = ("no_confirmed", "non_replication", "negative")
    primary = True if any(word in verdict for word in positive_words) else (
        False if any(word in verdict for word in negative_words) else None)
    reduced = "reduced_correct_only" in verdict
    label = classify_result(technical_complete=status in {"complete", "insufficient"},
                            behavior_sufficient=behavior, controls_valid=controls,
                            confirmation_complete=complete, primary_positive=primary,
                            reduced_route=reduced, mixed=primary is None and status == "complete")
    config = json.loads((attempt / "attempt.json").read_text(encoding="utf-8-sig"))["config"]
    return {"attempt": str(attempt), "model": config["model"]["key"],
            "study": config.get("study", {}).get("name", "core7"), "status": status,
            "scientific_verdict": verdict, "result_label": label}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempts", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [summarize(path.resolve()) for path in args.attempts]
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"schema": "vlm-mcqa.next-three-summary.v1", "rows": rows},
                                    indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
