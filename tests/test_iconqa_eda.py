"""Small CPU fixtures for EDA pair counts and source/pixel independence."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("iconqa_eda", Path(__file__).parents[1] / "scripts/local/analyze_iconqa.py")
eda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda)


def row(pid, answer, pixel=None, question="question"):
    return {"pid": str(pid), "answer_text": answer, "pixel_sha256": pixel or str(pid),
            "dhash": pid, "width": 100, "height": 100, "skills": ["probability"],
            "data": {"grade": "grade1", "label": "S2", "question": question,
                     "choices": ["a", "b", "c", "d"], "split": "val"}}


def test_pairing_requires_changed_answer_and_shared_question():
    pairs, _ = eda.make_pairs([row(1,"a"), row(2,"a"), row(3,"b",question="other")], eda.pair_key)
    assert not pairs


def test_duplicate_pixels_and_blocked_components_do_not_inflate_pairs():
    rows = [row(1,"a","same"), row(2,"b","same"), row(3,"b","blocked"), row(4,"c")]
    pairs, _ = eda.make_pairs(rows, eda.pair_key, {"blocked"})
    assert len(pairs) == 1
    assert {pairs[0]["x"], pairs[0]["y"]} == {"1", "4"}


def test_pair_members_and_pixels_are_disjoint_across_groups():
    rows = [row(1,"a","shared"),row(2,"b"),row(3,"a","shared", "other"),row(4,"b",question="other")]
    pairs, _ = eda.make_pairs(rows, eda.pair_key)
    assert len(pairs) == 1


def test_greedy_pairing_handles_answer_imbalance_deterministically():
    rows = [row(i+1,a) for i,a in enumerate(["a"]*6+["b"]*2+["c"]*2)]
    pairs, _ = eda.make_pairs(rows, eda.pair_key)
    assert len(pairs) == 4
    assert pairs == eda.make_pairs(rows, eda.pair_key)[0]
    ids = [p[k] for p in pairs for k in ("x","y")]
    assert len(ids) == len(set(ids))
