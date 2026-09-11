from collections.abc import Iterable
from uuid import uuid4

from app.schemas.ai import AIResult, ProbabilityMap, TopicKeyword


class MockAnalysisPipeline:
    """Deterministic stand-in for the future IndoBERT + BERTopic pipeline."""

    _negative_markers = {"tidak", "lambat", "buruk", "gagal", "error", "susah", "crash"}
    _positive_markers = {"bagus", "baik", "mudah", "lengkap", "cepat", "puas", "mantap"}
    _topic_markers = {
        1: ("OTP / Login / Verifikasi", {"otp", "login", "verifikasi"}),
        2: ("Performa / Loading / Crash", {"loading", "lambat", "crash", "error"}),
    }

    def run(self, rows: list[dict[str, object]], feedback_column: str) -> list[AIResult]:
        results: list[AIResult] = []
        for source_row_number, row in enumerate(rows, start=1):
            raw_text = str(row.get(feedback_column) or "").strip()
            if not raw_text:
                continue
            results.append(self._predict(raw_text, source_row_number))
        return results

    def _predict(self, text: str, source_row_number: int) -> AIResult:
        tokens = set(text.lower().replace(".", "").split())
        has_negative = bool(tokens & self._negative_markers)
        has_positive = bool(tokens & self._positive_markers)

        if has_negative:
            sentiment = "negative"
            probabilities = ProbabilityMap(positive=0.02, neutral=0.04, negative=0.94)
        elif has_positive:
            sentiment = "positive"
            probabilities = ProbabilityMap(positive=0.94, neutral=0.04, negative=0.02)
        else:
            sentiment = "neutral"
            probabilities = ProbabilityMap(positive=0.08, neutral=0.84, negative=0.08)

        topic_id = -1
        keywords: list[TopicKeyword] = []
        for candidate_id, (_, markers) in self._topic_markers.items():
            matched = [marker for marker in markers if marker in tokens]
            if matched:
                topic_id = candidate_id
                keywords = [
                    TopicKeyword(keyword=keyword, weight=0.18 - index * 0.02, rank=index + 1)
                    for index, keyword in enumerate(sorted(matched))
                ]
                break

        return AIResult(
            unit_id=uuid4(),
            source_row_number=source_row_number,
            text=text,
            sentiment=sentiment,
            confidence=max(probabilities.positive, probabilities.neutral, probabilities.negative),
            probabilities=probabilities,
            topic_id=topic_id,
            keywords=keywords,
        )


def keyword_counts(results: Iterable[AIResult]) -> dict[str, tuple[int, int | None]]:
    counts: dict[str, tuple[int, int | None]] = {}
    for result in results:
        for keyword in result.keywords:
            count, existing_topic = counts.get(keyword.keyword, (0, result.topic_id if result.topic_id >= 0 else None))
            counts[keyword.keyword] = (count + 1, existing_topic)
    return counts
