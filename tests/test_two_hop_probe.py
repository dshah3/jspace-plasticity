from __future__ import annotations

from pathlib import Path

import pytest

from jspace_plasticity.evals.two_hop_probe import (
    ANTHROPIC_DATA_REVISION,
    answer_continuation,
    capability_verdict,
    expected_token_id,
    load_probe_swap,
)


def test_vendored_anthropic_probe_swap_is_hash_bound() -> None:
    root = Path(__file__).resolve().parents[1]
    rows, manifest = load_probe_swap(
        root / "data/evals/anthropic-probe-swap.json"
    )
    assert len(rows) == 90
    assert manifest["source_revision"] == ANTHROPIC_DATA_REVISION
    spider = next(row for row in rows if row["name"] == "spider-legs")
    assert spider == {
        "name": "spider-legs",
        "category": "multihop",
        "prompt": "Fact: The number of legs on the animal that spins webs is ",
        "intermediate": "spider",
        "answer": "8",
        "swap_to": "ant",
        "swap_answer": "6",
    }


def test_two_hop_answer_continuation_respects_existing_whitespace() -> None:
    assert answer_continuation("The answer is ", "eight") == "eight"
    assert answer_continuation("The answer is", "eight") == " eight"
    with pytest.raises(ValueError, match="nonempty"):
        answer_continuation("", "eight")


def test_expected_token_requires_one_roundtripping_qwen_token() -> None:
    class Tokenizer:
        def encode(self, text, **kwargs):  # noqa: ANN001
            assert kwargs == {"add_special_tokens": False}
            mapping = {
                "The answer is": [10, 11, 12],
                "The answer is eight": [10, 11, 12, 8],
                "The answer is two words": [10, 11, 12, 1, 2],
            }
            return mapping[text]

        def decode(self, token_ids):  # noqa: ANN001
            return " eight" if token_ids == [8] else "bad"

    tokenizer = Tokenizer()
    token_id, continuation, reason = expected_token_id(
        tokenizer, "The answer is", "eight"
    )
    assert (token_id, continuation, reason) == (8, " eight", None)
    token_id, _, reason = expected_token_id(
        tokenizer, "The answer is", "two words"
    )
    assert token_id is None
    assert reason == "answer_token_count=2"


def test_expected_token_rejects_a_retokenized_prompt_boundary() -> None:
    class Tokenizer:
        def encode(self, text, **kwargs):  # noqa: ANN001
            assert kwargs == {"add_special_tokens": False}
            return [1, 2] if text == "Prompt" else [1, 99, 3]

        def decode(self, token_ids):  # noqa: ANN001
            return " answer"

    token_id, continuation, reason = expected_token_id(
        Tokenizer(), "Prompt", "answer"
    )
    assert token_id is None
    assert continuation == " answer"
    assert reason == "prompt_boundary_retokenized"


def test_two_hop_capability_gate_requires_accuracy_and_token_coverage() -> None:
    passed_rows = [
        {"token_compatible": True, "exact": float(index < 8)}
        for index in range(10)
    ]
    verdict = capability_verdict(passed_rows)
    assert verdict["passed"]
    assert verdict["accuracy"] == pytest.approx(0.8)
    assert verdict["correct_rows"] == 8

    low_coverage = passed_rows[:7] + [
        {"token_compatible": False, "exact": None} for _ in range(3)
    ]
    verdict = capability_verdict(low_coverage)
    assert not verdict["passed"]
    assert not verdict["clauses"]["tokenization_coverage"]

    count_gate = capability_verdict(
        passed_rows, min_accuracy=0.0, min_correct=9
    )
    assert not count_gate["passed"]
    assert not count_gate["clauses"]["clean_correct_count"]
