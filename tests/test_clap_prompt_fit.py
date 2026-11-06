"""CLAP prompts longer than the text encoder are shortened, others are untouched.

A question of about 2,000 characters crashed CLAP's text tower, and the caller
silently answered with every audio candidate unscored (results/edge-cases/).
Uses a stand-in tokenizer, so no model weights are needed.
"""

from __future__ import annotations

from gist.audio.clap import HuggingFaceClapAudioScorer


class _WordTokenizer:
    """One token per word, plus two special tokens, with a 512-token limit."""

    model_max_length = 512

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        ids = list(range(len(text.split())))
        return {"input_ids": ([0, *ids, 0] if add_special_tokens else ids)}

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"w{i}" for i in ids)


class _Processor:
    tokenizer = _WordTokenizer()


def _scorer() -> HuggingFaceClapAudioScorer:
    scorer = HuggingFaceClapAudioScorer()
    scorer._processor = _Processor()
    return scorer


def test_prompt_within_limit_is_returned_unchanged() -> None:
    prompt = "the sound of: where do they go snorkelling?"
    assert _scorer()._fit_prompt(prompt) is prompt


def test_prompt_at_exact_limit_is_unchanged() -> None:
    prompt = " ".join(["word"] * 510)  # 510 words + 2 special tokens = 512
    assert _scorer()._fit_prompt(prompt) is prompt


def test_overlong_prompt_is_cut_to_fit_with_special_tokens() -> None:
    prompt = " ".join(["word"] * 2000)
    fitted = _scorer()._fit_prompt(prompt)
    assert len(fitted.split()) == 510
