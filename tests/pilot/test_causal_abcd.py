"""Pilot A/B/C/D causal runner helpers (vlm_mcqa.eval.qwen_causal_abcd)."""

from __future__ import annotations

import unittest

from vlm_mcqa.eval.qwen_causal_abcd import (
    _text_token_groups,
    _window_specs,
    build_contrasts,
    select_pair_ids,
)


def _record(
    pair: str,
    *,
    member: str,
    family: str,
    index: int,
    key: str,
) -> dict[str, object]:
    return {
        "pair_id": pair,
        "pair_member": member,
        "variant_family": family,
        "variant_key": key,
        "label_scheme": "letters",
        "correct_index": index,
        "correct_label": "ABCD"[index],
        "sample_id": f"{pair}_{member}_{family}_{key}",
    }


def _content_record(
    pair: str,
    *,
    member: str,
    correct_content: str,
    options: list[str],
) -> dict[str, object]:
    index = options.index(correct_content)
    record = _record(
        pair,
        member=member,
        family="cross_image",
        index=index,
        key="shared",
    )
    record.update(
        {
            "correct_content": correct_content,
            "option_contents": options,
            "labels": list("ABCD"),
            "question": "What color is the circle?",
            "prompt": "placeholder",
        }
    )
    return record


class _WhitespaceTokenizer:
    """Maps fixed strings to ids; leading-space variants get distinct ids."""

    all_special_ids = [999]

    mapping = {
        "What color?": [91, 92],
        " What color?": [93, 94],
        "\nWhat color?": [95, 96],
        "A. red": [21, 22, 31],
        "B. blue": [24, 22, 32],
        "C. green": [25, 22, 33],
        "D. yellow": [26, 22, 34],
        "A": [21],
        "B": [24],
        "C": [25],
        "D": [26],
        "red": [131],
        "blue": [132],
        "green": [133],
        "yellow": [134],
        " red": [31],
        " blue": [32],
        " green": [33],
        " yellow": [34],
    }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("test tokenizer expects no special tokens")
        return self.mapping.get(text, [777])


class CausalAbcdRunnerTests(unittest.TestCase):
    def test_text_groups_fall_back_to_leading_whitespace_tokens(self) -> None:
        input_ids = [999, 100, 999, 11, 12, 21, 22, 31, 24, 22, 32, 25, 22, 33, 26, 22, 34]
        record = {
            "question": "What color?",
            "labels": list("ABCD"),
            "option_contents": ["red", "blue", "green", "yellow"],
        }
        groups = _text_token_groups(
            tokenizer=_WhitespaceTokenizer(), input_ids=input_ids, record=record, image_positions=[1]
        )
        self.assertEqual(groups["question"], [3, 4])
        self.assertEqual(groups["option_labels"], [5, 8, 11, 14])
        self.assertEqual(groups["option_content"], [7, 10, 13, 16])

    def test_window_specs_are_inclusive_and_bounded(self) -> None:
        self.assertEqual(
            _window_specs(num_layers=6, widths=[2], start_layer=2, end_layer=5),
            [(2, 3), (3, 4), (4, 5)],
        )

    def test_contrasts_include_visual_and_two_position_pairs(self) -> None:
        records = [
            _record("pair_00000", member="a", family="cross_image", index=0, key="shared"),
            _record("pair_00000", member="b", family="cross_image", index=1, key="shared"),
        ]
        records.extend(
            _record(
                "pair_00000",
                member="a",
                family="answer_position",
                index=index,
                key=f"correct_at_{index}",
            )
            for index in range(4)
        )
        contrasts = build_contrasts(
            records, selected_pairs=["pair_00000"], contrast_types=["visual", "position"]
        )
        self.assertEqual([contrast.contrast_type for contrast in contrasts], ["visual", "position", "position"])

    def test_content_contrast_changes_content_and_keeps_symbol(self) -> None:
        options = ["red", "blue", "green", "yellow"]
        records = [
            _content_record(
                "pair_00000", member="a", correct_content="red", options=options
            ),
            _content_record(
                "pair_00000", member="b", correct_content="blue", options=options
            ),
        ]
        contrasts = build_contrasts(
            records, selected_pairs=["pair_00000"], contrast_types=["content"]
        )
        self.assertEqual(len(contrasts), 1)
        contrast = contrasts[0]
        self.assertEqual(contrast.source["correct_content"], "red")
        self.assertEqual(contrast.target["correct_content"], "blue")
        self.assertEqual(contrast.source["correct_label"], contrast.target["correct_label"])
        self.assertEqual(contrast.target["correct_index"], 0)
        self.assertEqual(contrast.target["option_contents"], ["blue", "red", "green", "yellow"])

    def test_discovery_and_confirmation_pairs_are_disjoint(self) -> None:
        records = [
            {"pair_id": f"pair_{index:05d}", "label_scheme": "letters"}
            for index in range(64)
        ]
        discovery = select_pair_ids(records, split="discovery", num_pairs=32)
        confirmation = select_pair_ids(records, split="confirmation", num_pairs=32)
        self.assertEqual(len(discovery), 32)
        self.assertEqual(len(confirmation), 32)
        self.assertFalse(set(discovery) & set(confirmation))


if __name__ == "__main__":
    unittest.main()
