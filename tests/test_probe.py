from jspace_plasticity.probe import (
    candidate_from_text,
    explicit_final_candidate,
    parse_generated_candidate,
)


def test_candidate_from_text_returns_last_complete_node() -> None:
    nodes = ("amber", "river", "stone")
    assert candidate_from_text("First amber, finally river.", nodes) == "river"


def test_candidate_from_text_ignores_partial_word() -> None:
    nodes = ("amber", "river")
    assert candidate_from_text("ambergris arrives", nodes) is None


def test_explicit_final_candidate_handles_markdown() -> None:
    nodes = ("amber", "river", "stone")
    text = "**Final Node:** `river`\nChecking again: amber -> stone"
    assert explicit_final_candidate(text, nodes) == "river"


def test_explicit_final_candidate_uses_last_explicit_revision() -> None:
    nodes = ("amber", "river", "stone")
    text = "Final: amber. Correction. The result is **stone**."
    assert explicit_final_candidate(text, nodes) == "stone"


def test_explicit_final_candidate_ignores_intermediate_result_labels() -> None:
    nodes = ("amber", "river", "stone")
    text = "Final result: river. Recheck step 1. Result: stone."
    assert explicit_final_candidate(text, nodes) == "river"


def test_generated_parser_prefers_explicit_final_over_trailing_work() -> None:
    nodes = ("amber", "river", "stone")
    parsed = parse_generated_candidate(
        "Final node: river. Rechecking: river -> stone", nodes
    )
    assert parsed.value == "river"
    assert parsed.source == "explicit_final"


def test_generated_parser_uses_answer_after_think() -> None:
    nodes = ("amber", "river", "stone")
    parsed = parse_generated_candidate("trace amber</think>\nriver", nodes)
    assert parsed.value == "river"
    assert parsed.source == "after_think"


def test_generated_parser_falls_back_to_last_candidate() -> None:
    nodes = ("amber", "river", "stone")
    parsed = parse_generated_candidate("amber then stone", nodes)
    assert parsed.value == "stone"
    assert parsed.source == "last_candidate"
