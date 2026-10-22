"""Four-text-choice IconQA validation subset in the CLEVR-MCQ-4 schema.

Expects the official IconQA release layout::

    iconqa_data/iconqa/<split>/choose_txt/<question_id>/{image.png,data.json}

where ``data.json`` holds ``question``, ``choices`` and ``answer`` (index).
Only questions with exactly four distinct text choices are kept. There are no
counterfactual pairs, so the subset supports behaviour, logit-lens and
answer-symbol analyses only.

    python -m vlm_mcqa.clevr_mcq4.iconqa --iconqa-root /data/iconqa_data/iconqa \
        --split test --num-items 300 --out-dir /data/iconqa_mcq4_v1
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from .build import expand


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iconqa-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-items", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.iconqa_root / args.split / "choose_txt"
    candidates = []
    for qdir in sorted(p for p in root.iterdir() if p.is_dir()):
        data = json.loads((qdir / "data.json").read_text())
        choices = [str(c).strip() for c in data["choices"]]
        if len(choices) != 4 or len(set(choices)) != 4:
            continue
        candidates.append((qdir, data, choices))
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: args.num_items]
    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
    items = []
    for qdir, data, choices in selected:
        qid = qdir.name
        image = f"images/iconqa_{qid}.png"
        shutil.copy2(qdir / "image.png", args.out_dir / image)
        items.append({
            "item_id": f"iconqa_{qid}",
            "source": "iconqa",
            "dataset_version": "iconqa_mcq4_v1",
            "split": "iconqa",
            "scene_id": f"iconqa_{qid}",
            "counterfactual_pair_id": None,
            "pair_member": None,
            "image": image,
            "question": data["question"].strip(),
            "option_contents": choices,
            "semantic_answer": choices[int(data["answer"])],
            "answer_type": "text",
            "question_family": data.get("ques_type", "choose_txt"),
            "program": [],
            "program_depth": 0,
        })
    manifest = expand(args.out_dir, items, seed=args.seed)
    print(json.dumps({"eligible": len(candidates), "selected": len(items), **manifest}, indent=2))


if __name__ == "__main__":
    main()
