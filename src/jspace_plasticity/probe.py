"""Shared helpers for unconstrained graph-task generation probes."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedCandidate:
    value: str | None
    source: str


def candidate_from_text(text: str, nodes: tuple[str, ...]) -> str | None:
    """Return the last complete node name mentioned in generated text."""
    matches: list[tuple[int, str]] = []
    for node in nodes:
        pattern = rf"(?<![A-Za-z]){re.escape(node)}(?![A-Za-z])"
        matches.extend(
            (match.start(), node.lower())
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        )
    return max(matches)[1] if matches else None


def explicit_final_candidate(text: str, nodes: tuple[str, ...]) -> str | None:
    """Return the last node attached to an explicit final-result marker."""
    if not nodes:
        return None
    candidates = "|".join(re.escape(node) for node in nodes)
    label = r"(?:final\s+node|final\s+result|final|the\s+result)"
    separator = r"(?:\s+is\b|\s*[:=])"
    formatting = r"[\s`*_]*"
    pattern = re.compile(
        rf"\b{label}\b{separator}{formatting}({candidates})(?![A-Za-z])",
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    return matches[-1].group(1).lower() if matches else None


def strict_direct_candidate(text: str, nodes: tuple[str, ...]) -> str | None:
    """Parse a node only when the completion contains no reasoning or extra answer.

    The prompt itself ends in ``Final node:``, so the intended completion is one
    node token. An echoed final marker and terminal punctuation are tolerated to
    avoid turning harmless surface variation into reward noise.
    """

    if not nodes:
        return None
    candidates = "|".join(re.escape(node) for node in nodes)
    pattern = re.compile(
        rf"^\s*(?:final\s+node\s*:\s*)?({candidates})[.!]?\s*$",
        flags=re.IGNORECASE,
    )
    match = pattern.fullmatch(text)
    return match.group(1).lower() if match else None


def parse_generated_candidate(text: str, nodes: tuple[str, ...]) -> ParsedCandidate:
    """Parse a generated answer with a preregistered order of precedence."""
    explicit = explicit_final_candidate(text, nodes)
    if explicit is not None:
        return ParsedCandidate(explicit, "explicit_final")

    if "</think>" in text:
        after_thinking = candidate_from_text(text.rsplit("</think>", 1)[1], nodes)
        if after_thinking is not None:
            return ParsedCandidate(after_thinking, "after_think")

    fallback = candidate_from_text(text, nodes)
    return ParsedCandidate(fallback, "last_candidate" if fallback else "none")
