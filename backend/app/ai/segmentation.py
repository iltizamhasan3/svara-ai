"""Deterministic sentence and simple clause segmentation for Week 4."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.ai.preprocessing import normalize_text


SEGMENTATION_VERSION = "segmentation-v1"
_MISSING_TEXT_VALUES = {"", "na", "n/a", "nan", "none", "null"}
_CLOSING_CHARS = frozenset('"\'”’»)]}')
_ABBREVIATIONS = {
    "a.n",
    "an",
    "dr",
    "etc",
    "no",
    "ny",
    "prof",
    "pt",
    "sdr",
    "sd",
    "smp",
    "sma",
    "yth",
}
_CLAUSE_BOUNDARY_RE = re.compile(
    r"\s+(?:akan\s+)?(?:tetapi|tapi|namun|sedangkan|meskipun|walaupun)\s+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnalysisUnit:
    """A traceable sentence or clause passed to downstream AI components."""

    source_row_number: int
    unit_index: int
    text: str


def _token_before(text: str, position: int) -> str:
    match = re.search(r"([\w./-]+)$", text[:position], flags=re.UNICODE)
    return match.group(1).casefold() if match else ""


def _is_sentence_boundary(text: str, position: int) -> bool:
    character = text[position]
    if character not in ".!?。！？":
        return False

    if character == ".":
        previous_character = text[position - 1] if position else ""
        next_character = text[position + 1] if position + 1 < len(text) else ""
        if previous_character.isdigit() and next_character.isdigit():
            return False
        if _token_before(text, position).rstrip(".") in _ABBREVIATIONS:
            return False

    end = position + 1
    while end < len(text) and text[end] in ".!?。！？":
        end += 1
    while end < len(text) and text[end] in _CLOSING_CHARS:
        end += 1
    return end == len(text) or text[end].isspace()


def split_sentences(text: object) -> list[str]:
    """Split text on sentence punctuation while preserving meaningful text."""

    normalized = normalize_text(text)
    if normalized.casefold() in _MISSING_TEXT_VALUES:
        return []

    sentences: list[str] = []
    start = 0
    position = 0
    while position < len(normalized):
        if _is_sentence_boundary(normalized, position):
            end = position + 1
            while end < len(normalized) and normalized[end] in ".!?。！？":
                end += 1
            while end < len(normalized) and normalized[end] in _CLOSING_CHARS:
                end += 1
            fragment = normalized[start:end].strip()
            if fragment:
                sentences.append(fragment)
            start = end
            while start < len(normalized) and normalized[start].isspace():
                start += 1
            position = start
            continue
        position += 1

    tail = normalized[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_simple_clauses(text: object) -> list[str]:
    """Split a sentence on contrastive Indonesian conjunctions.

    This deliberately handles only high-precision conjunctions. Splitting on
    every ``dan`` or comma would create fragments that are too short and could
    damage topic and sentiment context.
    """

    normalized = normalize_text(text)
    if normalized.casefold() in _MISSING_TEXT_VALUES:
        return []
    clauses = [part.strip(" \t\r\n,;:") for part in _CLAUSE_BOUNDARY_RE.split(normalized)]
    return [clause for clause in clauses if clause]


def segment_text(text: object, *, split_clauses: bool = False) -> list[str]:
    """Return deterministic sentence units, optionally refined into clauses."""

    units: list[str] = []
    for sentence in split_sentences(text):
        units.extend(split_simple_clauses(sentence) if split_clauses else [sentence])
    return units


def segment_rows(
    rows: Iterable[Mapping[str, object]],
    feedback_column: str,
    *,
    split_clauses: bool = False,
) -> list[AnalysisUnit]:
    """Segment feedback rows while preserving source row and unit ordering."""

    if not feedback_column.strip():
        raise ValueError("feedback_column is required")

    units: list[AnalysisUnit] = []
    for source_row_number, row in enumerate(rows, start=1):
        segments = segment_text(row.get(feedback_column), split_clauses=split_clauses)
        for unit_index, text in enumerate(segments, start=1):
            units.append(
                AnalysisUnit(
                    source_row_number=source_row_number,
                    unit_index=unit_index,
                    text=text,
                )
            )
    return units
