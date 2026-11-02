"""M2-002: deterministic document-quality rules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityConfig:
    min_words: int = 30
    max_words: int = 100_000
    min_alpha_ratio: float = 0.50
    max_duplicate_line_fraction: float = 0.30

    def __post_init__(self):
        if self.min_words < 0 or self.max_words < self.min_words:
            raise ValueError("Invalid word-count limits")
        if not 0 <= self.min_alpha_ratio <= 1:
            raise ValueError("min_alpha_ratio must be between 0 and 1")
        if not 0 <= self.max_duplicate_line_fraction <= 1:
            raise ValueError(
                "max_duplicate_line_fraction must be between 0 and 1"
            )


def alphabetic_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(ch.isalpha() for ch in chars) / len(chars)


def duplicate_line_fraction(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0

    return (len(lines) - len(set(lines))) / len(lines)


def rejection_reason(
    text: str,
    config: QualityConfig = QualityConfig(),
) -> str | None:
    """Return the first rejection reason, or None if accepted."""

    if not isinstance(text, str):
        raise TypeError("Document text must be a string")

    words = len(text.split())

    if words < config.min_words:
        return "too_short"

    if words > config.max_words:
        return "too_long"

    if alphabetic_ratio(text) < config.min_alpha_ratio:
        return "low_alpha_ratio"

    if duplicate_line_fraction(text) > config.max_duplicate_line_fraction:
        return "repeated_lines"

    return None
