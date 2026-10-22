"""Reproducible CPU-only IconQA inventory, leakage audit and pairing EDA.

Does not alter the dataset or run models. Candidate pairs are observational,
not counterfactual edits or model-eligible experimental samples.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def norm(s):
    return " ".join(str(s).strip().split()).casefold()


def numeric(s):
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", s.strip()))


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dist(values):
    a = np.array(list(values))
    return dict(zip(["min", "p25", "median", "p75", "p95", "p99", "max"],
                    map(float, np.percentile(a, [0, 25, 50, 75, 95, 99, 100])))) if len(a) else {}


def pair_key(row, mode="strict"):
    d = row["data"]
    q = d["question"] if mode == "strict" else norm(d["question"])
    opts = tuple(sorted(d["choices"])) if mode == "strict" else tuple(sorted(map(norm, d["choices"])))
    return d["grade"], d["label"], q, opts


def audit_one(args):
    root, pid, d, skills = args
    folder = root / "iconqa" / d["split"] / d["ques_type"] / pid
    row = {"pid": pid, "data": d, "skills": skills, "errors": []}
    try:
        local = json.loads((folder / "data.json").read_text(encoding="utf-8"))
        if local != {k: v for k, v in d.items() if k != "split"}:
            row["errors"].append("local_metadata_mismatch")
    except Exception as e:
        row["errors"].append("metadata:" + str(e))
    assets = ["image.png"] + (d.get("choices", []) if d["ques_type"] == "choose_img" else [])
    row["asset_bytes"] = 0
    row["assets_checked"] = 0
    for name in assets:
        try:
            raw = (folder / name).read_bytes()
            with Image.open(io.BytesIO(raw)) as im:
                im.load()  # full decode, not only PNG header validation
                row["asset_bytes"] += len(raw)
                row["assets_checked"] += 1
                if name == "image.png":
                    rgb = im.convert("RGB")
                    row.update(width=im.width, height=im.height, mode=im.mode,
                               byte_sha256=hashlib.sha256(raw).hexdigest(),
                               pixel_sha256=hashlib.sha256(f"{im.size}:".encode() + rgb.tobytes()).hexdigest())
                    if d["ques_type"] == "choose_txt":
                        small = np.asarray(rgb.convert("L").resize((9, 8)))
                        row["dhash"] = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")
        except Exception as e:
            row["errors"].append(name + ":" + str(e))
    c = d.get("choices", [])
    if c and (not isinstance(d["answer"], int) or not 0 <= d["answer"] < len(c)):
        row["errors"].append("answer_out_of_range")
    row["answer_text"] = str(c[d["answer"]]) if c and isinstance(d["answer"], int) and 0 <= d["answer"] < len(c) else str(d["answer"])
    row["mcq4"] = d["ques_type"] == "choose_txt" and len(c) == 4 and len(set(map(norm, c))) == 4 and not row["errors"]
    return row


def make_pairs(rows, keyfunc, blocked=None):
    """Deterministic greedy matching, no source PID or exact pixels reused.

    Most frequent answer is paired with the most frequent different answer.
    This is a feasible matching count, not a maximum graph matching claim.
    """
    groups = defaultdict(list)
    for r in rows:
        if r["pixel_sha256"] not in (blocked or set()):
            groups[keyfunc(r)].append(r)
    pairs = []
    used_pixels = set()
    qualified = 0
    for key, group in sorted(groups.items(), key=lambda x: str(x[0])):
        by_answer = defaultdict(list)
        seen = set()
        for r in sorted(group, key=lambda r: int(r["pid"])):
            px = r["pixel_sha256"]
            if px not in seen and px not in used_pixels:
                by_answer[r["answer_text"]].append(r)
                seen.add(px)
        if len(by_answer) < 2:
            continue
        qualified += 1
        while len([a for a in by_answer if by_answer[a]]) >= 2:
            answers = sorted([a for a in by_answer if by_answer[a]], key=lambda a: (-len(by_answer[a]), a))
            x = by_answer[answers[0]].pop(0)
            # Similarity only prioritizes visual review. It cannot prove minimality.
            candidates = by_answer[answers[1]]
            y = min(candidates, key=lambda y: ((x.get("dhash", 0) ^ y.get("dhash", 0)).bit_count(), int(y["pid"])))
            candidates.remove(y)
            used_pixels.update([x["pixel_sha256"], y["pixel_sha256"]])
            pairs.append({"x": x["pid"], "y": y["pid"], "source_split": x["data"]["split"],
                          "question": x["data"]["question"], "options_x": x["data"]["choices"],
                          "options_y": y["data"]["choices"], "answer_x": x["answer_text"],
                          "answer_y": y["answer_text"], "grade": x["data"]["grade"],
                          "label": x["data"]["label"], "skills": x["skills"],
                          "dhash_distance": (x.get("dhash", 0) ^ y.get("dhash", 0)).bit_count(),
                          "same_size": (x["width"], x["height"]) == (y["width"], y["height"]),
                          "status": "unreviewed_natural_pair"})
    return pairs, qualified


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    problems = json.loads((root / "problems.json").read_text())
    skills = json.loads((root / "pid2skills.json").read_text())
    splits = json.loads((root / "pid_splits.json").read_text())
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(audit_one, ((root, pid, d, skills[pid]) for pid, d in problems.items()))):
            rows.append(row)
            if (i + 1) % 10000 == 0:
                print(f"audited {i+1}/{len(problems)} problems in {time.time()-t0:.1f}s", flush=True)
    jsonl(out / "inventory.jsonl", rows)
    print("image/metadata scan saved; auditing split indices and directory coverage", flush=True)
    byid = {r["pid"]: r for r in rows}
    summary = {"identity": "ICONQA-EDA-20261026", "root": str(root), "problems": len(rows),
               "assets_decoded": sum(r["assets_checked"] for r in rows),
               "asset_bytes": sum(r["asset_bytes"] for r in rows),
               "errors": [{"pid": r["pid"], "errors": r["errors"]} for r in rows if r["errors"]],
               "metadata_sha256": {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
                                     ["metadata.json", "problems.json", "pid_splits.json", "pid2skills.json", "skills.json", "LICENSE.md"]}}
    split_sets = {key: set(ids) for key, ids in splits.items()}
    summary["split_index_mismatches"] = sum(pid not in split_sets[f'{d["ques_type"]}_{d["split"]}'] for pid, d in problems.items())
    # Unexpected/missing directories and assets, independent of metadata inventory.
    actual_dirs = set()
    extra_files = []
    for split in ("train", "val", "test"):
        for task in ("choose_txt", "choose_img", "fill_in_blank"):
            for folder in (root/"iconqa"/split/task).iterdir():
                if folder.is_dir():
                    actual_dirs.add((split, task, folder.name))
                    d = problems.get(folder.name, {})
                    expected = {"data.json", "image.png"} | (set(d.get("choices", [])) if task == "choose_img" else set())
                    extra_files.extend(str(f.relative_to(root)) for f in folder.iterdir() if f.name not in expected)
    expected_dirs = {(d["split"], d["ques_type"], pid) for pid,d in problems.items()}
    summary["directory_audit"] = {"actual": len(actual_dirs), "missing": sorted(expected_dirs-actual_dirs),
                                  "extra": sorted(actual_dirs-expected_dirs), "unexpected_files": extra_files}
    print("directory coverage saved in memory; computing duplicates and natural pairs", flush=True)
    summary["task_split"] = {s: dict(Counter(r["data"]["ques_type"] for r in rows if r["data"]["split"] == s)) for s in ["train", "val", "test"]}
    summary["choice_counts"] = {task: dict(sorted(Counter(len(r["data"].get("choices", [])) for r in rows if r["data"]["ques_type"] == task).items())) for task in ["choose_txt", "choose_img"]}
    summary["grades"] = dict(Counter(r["data"]["grade"] for r in rows))
    summary["skills_multilabel"] = dict(Counter(s for r in rows for s in r["skills"]))
    summary["fields"] = dict(Counter(k for r in rows for k in r["data"]))
    summary["image_dimensions"] = {"width": dist(r["width"] for r in rows if "width" in r), "height": dist(r["height"] for r in rows if "height" in r),
                                    "modes": dict(Counter(r.get("mode") for r in rows)), "tiny_min_side_lt28": sum(min(r.get("width",0),r.get("height",0)) < 28 for r in rows)}
    summary["question_words"] = dist(len(r["data"]["question"].split()) for r in rows)
    summary["duplicate_choices"] = {t: sum(len(set(map(norm,r["data"].get("choices",[])))) != len(r["data"].get("choices",[])) for r in rows if r["data"]["ques_type"] == t) for t in ["choose_txt", "choose_img"]}
    # Exact pixel components across all tasks, including identical images re-encoded.
    pixels = defaultdict(list)
    for r in rows:
        if "pixel_sha256" in r:
            pixels[r["pixel_sha256"]].append(r)
    duplicates = [v for v in pixels.values() if len(v)>1]
    cross = [v for v in duplicates if len({r["data"]["split"] for r in v})>1]
    summary["exact_pixel_duplicates"] = {"unique_images": len(pixels), "duplicate_groups": len(duplicates), "rows_in_duplicate_groups": sum(map(len, duplicates)),
                                          "cross_split_groups": len(cross), "cross_split_rows": sum(map(len,cross))}
    jsonl(out / "duplicate_image_groups.jsonl", [[r["pid"] for r in g] for g in duplicates])
    conflicts = defaultdict(list)
    for r in rows:
        if r["data"]["ques_type"] == "choose_txt" and "pixel_sha256" in r:
            conflicts[(r["pixel_sha256"], norm(r["data"]["question"]), tuple(sorted(map(norm, r["data"]["choices"]))))].append(r)
    conflict_groups = [g for g in conflicts.values() if len({norm(r["answer_text"]) for r in g})>1]
    summary["conflicting_text_labels"] = [[r["pid"] for r in g] for g in conflict_groups]
    bad_pixels = {r["pixel_sha256"] for g in conflict_groups for r in g}
    mcq = [r for r in rows if r["mcq4"]]
    summary["mcq4"] = {}
    all_pairs = []
    for split in ["train", "val", "test"]:
        rs = [r for r in mcq if r["data"]["split"] == split]
        other_pixels = {r["pixel_sha256"] for r in rows if r["data"]["split"] != split and "pixel_sha256" in r}
        clean = [r for r in rs if r["pixel_sha256"] not in other_pixels | bad_pixels]
        pairs, groups = make_pairs(clean, pair_key)
        # With exactly four original choices, a shared option set is mandatory.
        relaxed, relaxed_groups = make_pairs(clean, lambda r: pair_key(r, "normalized"))
        all_pairs.extend(pairs)
        top = Counter(r["answer_text"] for r in rs)
        # Leave-one-out question-only majority diagnostic; descriptive, not a trained baseline.
        qgroups = defaultdict(list)
        for r in rs:
            qgroups[norm(r["data"]["question"])].append(r)
        covered = wins = 0
        for g in qgroups.values():
            for r in g:
                c = Counter(x["answer_text"] for x in g if x["pid"] != r["pid"] and x["answer_text"] in r["data"]["choices"])
                if c:
                    covered += 1
                    wins += sorted(c, key=lambda a: (-c[a], a))[0] == r["answer_text"]
        summary["mcq4"][split] = {"items": len(rs), "exclusive_nonconflicting_items": len(clean),
            "grades": dict(Counter(r["data"]["grade"] for r in rs)),
            "skills": dict(Counter(s for r in rs for s in r["skills"])),
            "skill_combinations": dict(Counter("+".join(sorted(r["skills"])) for r in rs)),
            "original_answer_positions": dict(Counter(r["data"]["answer"] for r in rs)),
            "top_answers": top.most_common(20), "unique_answers": len(top),
            "numeric_only_options": sum(all(numeric(c) for c in r["data"]["choices"]) for r in rs),
            "all_options_single_word": sum(all(len(c.split())==1 for c in r["data"]["choices"]) for r in rs),
            "option_prefix_collision_word": sum(len({norm(c).split()[0] for c in r["data"]["choices"]})<4 for r in rs),
            "with_hint": sum("hint" in r["data"] for r in rs),
            "question_unique": len(qgroups), "top_questions": Counter(r["data"]["question"] for r in rs).most_common(15),
            "width": dist(r["width"] for r in rs), "height": dist(r["height"] for r in rs),
            "strict_pairs_greedy": len(pairs), "strict_pair_groups": groups,
            "normalized_pairs_greedy": len(relaxed), "normalized_pair_groups": relaxed_groups,
            "pair_skills": dict(Counter(s for pair in pairs for s in pair["skills"])),
            "pair_grade": dict(Counter(pair["grade"] for pair in pairs)),
            "pair_numeric_options": sum(all(numeric(c) for c in pair["options_x"]) for pair in pairs),
            "pair_same_size": sum(pair["same_size"] for pair in pairs),
            "pair_dhash_distance": dist(pair["dhash_distance"] for pair in pairs),
            "question_prior_LOO": {"correct": wins, "covered": covered, "accuracy": wins/covered if covered else None}}
    jsonl(out / "natural_pair_candidates.jsonl", all_pairs)
    # Additional candidate sources are counted, not silently included in MCQ4.
    summary["extensions"] = {}
    for split in ["train", "val", "test"]:
        txt5 = [r for r in rows if r["data"]["split"]==split and r["data"]["ques_type"]=="choose_txt" and len(r["data"]["choices"])==5 and len(set(map(norm,r["data"]["choices"])))==5 and not r["errors"]]
        other = {r["pixel_sha256"] for r in rows if r["data"]["split"]!=split and "pixel_sha256" in r}
        pp, gg = make_pairs(txt5, pair_key, other | bad_pixels)
        fib = [r for r in rows if r["data"]["split"]==split and r["data"]["ques_type"]=="fill_in_blank"]
        summary["extensions"][split] = {"five_text_items":len(txt5), "same_five_option_natural_pairs":len(pp), "five_pair_groups":gg,
            "fill_in_blank":len(fib), "numeric_fill_answers":sum(numeric(r["answer_text"]) for r in fib)}
    summary["elapsed_seconds"] = time.time()-t0
    dump(out/"summary.json", summary)
    # Small static scientific report figure.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,2,figsize=(13,9),layout="constrained")
    tasks=["choose_img","choose_txt","fill_in_blank"]
    bottom=np.zeros(3)
    for split,color in zip(["train","val","test"],["#26547c","#2a9d8f","#e9c46a"]):
        vals=[summary["task_split"][split][t] for t in tasks]
        ax[0,0].bar(tasks,vals,bottom=bottom,label=split,color=color); bottom+=vals
    ax[0,0].set_title("All 107,439 source questions"); ax[0,0].legend()
    for i,t in enumerate(["choose_txt","choose_img"]):
        cs=summary["choice_counts"][t]; ax[0,1].bar(np.array(list(cs),dtype=int)+i*.32-.16,list(cs.values()),width=.32,label=t)
    ax[0,1].set_title("Original multiple-choice cardinality"); ax[0,1].set_xticks([2,3,4,5]); ax[0,1].legend()
    sk=Counter(s for r in mcq for s in r["skills"])
    ax[1,0].barh(list(sk),list(sk.values()),color="#2a9d8f"); ax[1,0].set_title("Native four-text-choice skills (multilabel)")
    ss=["train","val","test"]; vals=[summary["mcq4"][s]["strict_pairs_greedy"] for s in ss]
    ax[1,1].bar(ss,vals,color=["#26547c","#2a9d8f","#e9c46a"])
    for i,n in enumerate(vals): ax[1,1].text(i,n,str(n),ha="center",va="bottom")
    ax[1,1].set_title("Unreviewed, disjoint natural pairs after exact-leakage filter")
    fig.savefig(out/"eda_overview.png",dpi=180); plt.close(fig)
    # Original assets, ordinary contact-sheet assembly, no image editing model.
    selected=[]
    for split in ["val","test"]:
        pp=[p for p in all_pairs if p["source_split"]==split]
        seen=set()
        for pair in sorted(pp,key=lambda p:(p["dhash_distance"],int(p["x"]))):
            key=tuple(pair["skills"])
            if key not in seen:
                selected.append(pair); seen.add(key)
            if len([p for p in selected if p["source_split"]==split])>=6: break
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",16)
    for page,start in enumerate(range(0,len(selected),6),1):
        sub=selected[start:start+6]; canvas=Image.new("RGB",(1200, len(sub)*290),"#f6f5f1"); draw=ImageDraw.Draw(canvas)
        for k,pair in enumerate(sub):
            y=k*290
            for side,xpos in [("x",10),("y",410)]:
                r=byid[pair[side]]; d=r["data"]
                img=Image.open(root/"iconqa"/d["split"]/d["ques_type"]/r["pid"]/"image.png").convert("RGB")
                thumb=ImageOps.contain(img,(380,220)); canvas.paste(thumb,(xpos,y+35))
                draw.text((xpos,y+8),f'{side}: {r["pid"]} | answer: {r["answer_text"]}',fill="black",font=font)
            import textwrap
            lines=[pair["source_split"]+" | "+",".join(pair["skills"]),*textwrap.wrap(pair["question"],43),
                   *textwrap.wrap(str(pair["options_x"]),43),f'dHash distance: {pair["dhash_distance"]}',"UNREVIEWED NATURAL PAIR"]
            draw.multiline_text((815,y+12),"\n".join(lines),fill="black",font=font,spacing=5)
        canvas.save(out/f"pair_contact_sheet_{page}.png")
    dump(out/"visual_review_sample.json",selected)
    print(json.dumps({"problems":len(rows),"errors":len(summary["errors"]),"mcq4":{s:{k:v for k,v in summary["mcq4"][s].items() if k in ["items","exclusive_nonconflicting_items","strict_pairs_greedy","strict_pair_groups"]} for s in ss},"out":str(out),"seconds":time.time()-t0},indent=2))


if __name__ == "__main__":
    main()
