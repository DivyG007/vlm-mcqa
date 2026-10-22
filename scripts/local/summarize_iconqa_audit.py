"""Cached-inventory extension: template, rational-choice and perceptual audit.

Perceptual edges are review candidates, not automatically equivalent images.
No models, tokenizer downloads, source edits, or final experimental selection.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import argparse
import csv
import importlib.util
import itertools
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("iconqa_eda", Path(__file__).with_name("analyze_iconqa.py"))
    eda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eda)
    rows = [json.loads(line) for line in args.inventory.read_text(encoding="utf-8").splitlines()]
    native = [r for r in rows if r["mcq4"]]
    groups = defaultdict(list)
    families = defaultdict(list)
    exact = defaultdict(list)
    for r in native:
        d = r["data"]
        families[(d["grade"],d["label"])].append(r)
        groups[(d["question"],tuple(sorted(d["choices"])),r["width"],r["height"])].append(r)
        exact[eda.pair_key(r)].append(r)
    table = []
    for (grade,label), rs in sorted(families.items()):
        table.append({"grade":grade,"label":label,"total":len(rs),
                      **{s:sum(r["data"]["split"]==s for r in rs) for s in ["train","val","test"]},
                      "answer_counts":dict(Counter(r["answer_text"] for r in rs)),
                      "skills":sorted(set(s for r in rs for s in r["skills"]))})
    eda.dump(args.out/"native_families.json",table)
    eda.jsonl(args.out/"strict_pairable_groups.jsonl",[
        {"grade":key[0],"label":key[1],"question":key[2],"options":key[3],"n":len(rs),
         "answer_counts":dict(Counter(r["answer_text"] for r in rs))}
        for key,rs in sorted(exact.items(),key=lambda x:str(x[0])) if len({r["answer_text"] for r in rs})>1])
    equivalent = []
    for r in native:
        try:
            values = [Fraction(c) for c in r["data"]["choices"]]
        except (ValueError,ZeroDivisionError):
            continue
        if len(set(values))<4:
            equivalent.append(r["pid"])
    near=[]
    for rs in groups.values():
        for x,y in itertools.combinations(rs,2):
            if x["data"]["split"]==y["data"]["split"]:
                continue
            distance=(x["dhash"]^y["dhash"]).bit_count()
            if distance<=2:
                near.append({"x":x["pid"],"y":y["pid"],"distance":distance,
                             "same_answer":x["answer_text"]==y["answer_text"],
                             "family":x["data"]["grade"]+"_"+x["data"]["label"],
                             "status":"unreviewed_perceptual_neighbor"})
    ids={r[k] for r in near for k in ["x","y"]}
    eda.jsonl(args.out/"perceptual_cross_split_neighbors.jsonl",near)
    result={"equivalent_rational_choice_ids":equivalent,
            "perceptual_cross_split_edges":len(near), "same_answer_edges":sum(r["same_answer"] for r in near),
            "different_answer_edges":sum(not r["same_answer"] for r in near),"flagged_source_ids":len(ids),
            "flagged_families":dict(Counter(r["family"] for r in near)),"conservative_review_quarantine":{},
            "same_size_matching":{}}
    # Keep the same exact conflict policy as the primary EDA, then conservatively
    # quarantine every flagged native PID; this is an alternative planning bound.
    conflicts=defaultdict(list)
    for r in rows:
        if r["data"]["ques_type"]=="choose_txt" and "pixel_sha256" in r:
            conflicts[(r["pixel_sha256"],eda.norm(r["data"]["question"]),tuple(sorted(map(eda.norm,r["data"]["choices"]))))].append(r)
    bad={r["pixel_sha256"] for g in conflicts.values() if len({eda.norm(r["answer_text"]) for r in g})>1 for r in g}
    for split in ["train","val","test"]:
        other={r["pixel_sha256"] for r in rows if r["data"]["split"]!=split and "pixel_sha256" in r}
        clean=[r for r in native if r["data"]["split"]==split and r["pid"] not in ids and r["pixel_sha256"] not in other|bad]
        pairs,_=eda.make_pairs(clean,eda.pair_key)
        result["conservative_review_quarantine"][split]={"items":len(clean),"pairs":len(pairs),
            "pair_skills":dict(Counter(s for r in pairs for s in r["skills"])),
            "families":dict(Counter(r["data"]["grade"]+"_"+r["data"]["label"] for r in clean))}
        eda.jsonl(args.out/f"quarantined_review_pool_{split}.jsonl",pairs)
        same_size, size_groups = eda.make_pairs(clean, lambda r: (eda.pair_key(r),r["width"],r["height"]))
        result["same_size_matching"][split] = {"same_raw_dimensions_pairs":len(same_size),"groups":size_groups,
            "skills":dict(Counter(s for pair in same_size for s in pair["skills"]))}
        eda.jsonl(args.out/f"same_size_review_pool_{split}.jsonl",same_size)
    eda.dump(args.out/"supplement_summary.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
