"""M2-002 correctness tests for quality filtering."""

import pytest

from data_pipeline.quality_filter import (
    QualityConfig,
    alphabetic_ratio,
    duplicate_line_fraction,
    rejection_reason,
)


def test_good_document_passes():
    cfg = QualityConfig(min_words=3)

    assert rejection_reason(
        "Language models learn from many documents.", cfg
    ) is None


def test_short_document_rejected():
    cfg = QualityConfig(min_words=10)

    assert rejection_reason("Too short.", cfg) == "too_short"


def test_long_document_rejected():
    cfg = QualityConfig(min_words=1, max_words=5)

    assert rejection_reason(
        "one two three four five six", cfg
    ) == "too_long"


def test_low_alpha_ratio_rejected():
    cfg = QualityConfig(min_words=1)

    assert rejection_reason(
        "12345 !!!!! 67890", cfg
    ) == "low_alpha_ratio"


def test_repeated_lines_rejected():
    cfg = QualityConfig(min_words=1)

    text = "Repeated content\n" * 8 + "Unique ending"

    assert rejection_reason(text, cfg) == "repeated_lines"


def test_unicode_letters_count_as_alphabetic():
    assert alphabetic_ratio("Café") == 1.0
    assert alphabetic_ratio("模型") == 1.0


def test_duplicate_fraction():
    text = "first\nfirst\nsecond\nthird"

    assert duplicate_line_fraction(text) == 0.25


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        QualityConfig(min_alpha_ratio=1.5)

    with pytest.raises(ValueError):
        QualityConfig(min_words=10, max_words=5)


def test_rule_order_is_deterministic():
    cfg = QualityConfig(min_words=10)

    # Both too short and low-alpha; length takes precedence.
    assert rejection_reason("123 !!!", cfg) == "too_short"
