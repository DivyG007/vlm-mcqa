"""Preparation-only CLI. There is deliberately no model/run subcommand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build_fixed_text_contrasts, load_dataset, write_new_json
from .protocol import compile_recipe, load_frozen_heads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare stretch artifacts; never load a model")
    commands = parser.add_subparsers(dest="command", required=True)
    contrasts = commands.add_parser("contrasts", help="derive structural fixed-text image candidates")
    contrasts.add_argument("--dataset", type=Path, required=True)
    contrasts.add_argument("--output", type=Path, required=True)
    contrasts.add_argument("--schemes", nargs="+", default=["letters"])
    contrasts.add_argument("--splits", nargs="+", default=["discovery", "confirmation"])
    recipe = commands.add_parser("recipe", help="compile frozen-head intervention conditions")
    recipe.add_argument("--heads", type=Path, required=True)
    recipe.add_argument("--model-key", required=True)
    recipe.add_argument("--dataset-hash", required=True)
    recipe.add_argument("--num-heads", type=int, required=True)
    recipe.add_argument("--num-layers", type=int, required=True)
    recipe.add_argument("--image-layer", type=int, required=True)
    recipe.add_argument("--image-site", choices=("resid_pre", "resid_post"), required=True)
    recipe.add_argument("--expanded-controls", action="store_true")
    recipe.add_argument("--seed", type=int, required=True)
    recipe.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "contrasts":
        dataset = load_dataset(args.dataset)
        result = {"schema": "stretch-fixed-text-contrasts-v1", "provenance": dataset["provenance"],
                  "source_manifest": dataset["manifest"], "release_manifest": dataset["release_manifest"],
                  "contrasts": build_fixed_text_contrasts(dataset, schemes=args.schemes, splits=args.splits),
                  "runtime_validation": "pending", "historical_exposure": "not_audited"}
        inputs = [args.dataset] if args.dataset.is_dir() else [args.dataset, args.dataset.parent]
    else:
        frozen = load_frozen_heads(args.heads, model_key=args.model_key, dataset_hash=args.dataset_hash)
        result = compile_recipe(frozen, num_heads=args.num_heads, num_layers=args.num_layers,
                                image_layer=args.image_layer, image_site=args.image_site,
                                seed=args.seed, expanded_controls=args.expanded_controls)
        result["head_source"] = frozen
        # The recipe hash covers the recipe itself, not this external envelope.
        result = {"recipe": {k: v for k, v in result.items() if k != "head_source"}, "head_source": frozen}
        inputs = [args.heads, args.heads.parent]
    write_new_json(args.output, result, inputs=inputs)
    return 0
