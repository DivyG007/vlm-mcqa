"""Family prompt formatting and token-span location (core7.adapters)."""

from __future__ import annotations

import unittest

from support import HAS_TORCH, requires_torch

if HAS_TORCH:
    from vlm_mcqa.core7.adapters import locate, paligemma_mcq_prompt


class MergingTokenizer:
    """Word-level stub that, like Qwen's BPE, emits "?\\n" as a single token."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _pieces(self, text: str) -> list[str]:
        pieces, word = [], ""
        i = 0
        while i < len(text):
            if text.startswith("?\n", i):
                pieces += [word] if word else []
                pieces.append("?\n")
                word, i = "", i + 2
                continue
            ch = text[i]
            if ch in " \n?.":
                pieces += [word] if word else []
                pieces.append(ch)
                word = ""
            else:
                word += ch
            i += 1
        return pieces + ([word] if word else [])

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab.setdefault(p, len(self.vocab)) for p in self._pieces(text)]


@requires_torch
class PaliGemmaPromptTests(unittest.TestCase):
    def test_choices_precede_first_newline(self) -> None:
        common = "What color is the cube?\nA. red\nB. blue\nC. green\nD. yellow\nRespond with only the option label.\nAnswer:"
        rendered = paligemma_mcq_prompt(common, "answer en ")
        self.assertEqual(rendered.count("\n"), 0)
        self.assertTrue(rendered.startswith("answer en What color is the cube? Options: A. red"))
        self.assertIn("D. yellow", rendered)
        self.assertNotIn("Answer:", rendered)


@requires_torch
class TokenSpanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = MergingTokenizer()
        self.prompt = "<img> What color is the cube?\nA. red\nB. blue\nAnswer:"
        self.ids = self.tok.encode(self.prompt)

    def test_span_whose_last_token_merges_with_newline(self) -> None:
        span = locate(self.tok, self.ids, "What color is the cube?", 0)
        self.assertTrue(span)
        self.assertEqual(self.ids[span[0]], self.tok.encode("What")[0])
        self.assertEqual(self.ids[span[-1]], self.tok.encode("?\n")[0])

    def test_exact_match_is_unchanged(self) -> None:
        span = locate(self.tok, self.ids, "A. red", 0)
        self.assertEqual([self.ids[i] for i in span], self.tok.encode("A. red"))

    def test_absent_text_returns_empty_span(self) -> None:
        self.assertEqual(locate(self.tok, self.ids, "Which shape is it?", 0), [])


if __name__ == "__main__":
    unittest.main()
