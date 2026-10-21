"""Campaign plan/verify/run/audit entry point; model creation is opt-in only."""
from __future__ import annotations

import argparse
import json

from .config import load_config, plan_campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "verify", "calibrate", "pilot", "run", "audit", "inventory", "digest", "retry-plan"))
    parser.add_argument("--config")
    parser.add_argument("--attempt")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root")
    parser.add_argument("--output")
    parser.add_argument("--kind", choices=("source", "checkpoint"))
    parser.add_argument("--commit")
    parser.add_argument("--hf-id")
    parser.add_argument("--revision")
    args = parser.parse_args(argv)
    if args.command == "inventory":
        if not args.root or not args.output or not args.kind:
            parser.error("inventory requires --root --output --kind")
        from .preparation import prepare_inventory
        result = prepare_inventory(args.root, args.output, kind=args.kind, commit=args.commit,
                                   hf_id=args.hf_id, revision=args.revision)
    elif args.command == "digest":
        if not args.attempt:
            parser.error("digest requires --attempt <file>")
        from .verification import file_hash
        result = {"path": args.attempt, "sha256": file_hash(args.attempt)}
    elif args.command == "audit":
        if not args.attempt:
            parser.error("audit requires --attempt")
        from .audit import audit_attempt
        result = audit_attempt(args.attempt)
    else:
        if not args.config:
            parser.error("--config required")
        config = load_config(args.config)
        if args.command == "plan":
            result = plan_campaign(config)
        elif args.command == "retry-plan":
            if not args.attempt:
                parser.error("retry-plan requires --attempt <previous attempt>")
            from .retry import plan_retry
            result = plan_retry(args.attempt, config)
        elif args.command == "verify":
            from .verification import verify_inputs
            result = verify_inputs(config)
        elif args.command == "calibrate":
            if not args.execute:
                parser.error("calibrate requires explicit --execute")
            from .runner import calibrate_campaign
            result = calibrate_campaign(config, execute=True)
        elif args.command == "pilot":
            if not args.execute:
                parser.error("pilot requires explicit --execute")
            from .runner import technical_pilot_campaign
            result = technical_pilot_campaign(config, execute=True)
        else:
            if not args.execute:
                parser.error("run requires explicit --execute; authorization must come from the user")
            from .runner import run_campaign
            result = run_campaign(config, execute=True)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
