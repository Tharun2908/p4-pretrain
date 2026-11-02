"""Integration tests for M2 streaming quality filtering."""

import json

import pytest

from data_pipeline.filter_stream import filter_jsonl
from data_pipeline.quality_filter import QualityConfig


def write_docs(path, docs):
    with path.open("w", encoding="utf-8") as f:
        for doc_id, text in enumerate(docs):
            f.write(json.dumps({
                "id": str(doc_id),
                "source": "test",
                "text": text,
            }) + "\n")


def test_filter_counts_and_preserves_accepted(tmp_path):
    src = tmp_path / "input.jsonl"
    dst = tmp_path / "output.jsonl"

    write_docs(src, [
        "Language models learn from diverse documents.",
        "Too short.",
        "word " * 26,
        "12345 !!!!! 67890",
        "Repeated content\n" * 8 + "Unique ending",
    ])

    config = QualityConfig(min_words=3, max_words=25)
    stats = filter_jsonl(src, dst, config)

    assert stats["documents_seen"] == 5
    assert stats["accepted"] == 1
    assert stats["rejected"] == 4

    assert stats["reasons"] == {
        "too_short": 1,
        "too_long": 1,
        "low_alpha_ratio": 1,
        "repeated_lines": 1,
    }

    docs = [
        json.loads(line)
        for line in dst.read_text(encoding="utf-8").splitlines()
    ]

    assert len(docs) == 1
    assert docs[0]["id"] == "0"
    assert docs[0]["text"] == (
        "Language models learn from diverse documents."
    )


def test_malformed_input_preserves_output(tmp_path):
    src = tmp_path / "input.jsonl"
    dst = tmp_path / "output.jsonl"

    dst.write_text("previous output\n", encoding="utf-8")

    src.write_text(
        '{"text":"Valid document with enough words."}\n'
        '{invalid json}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        filter_jsonl(src, dst, QualityConfig(min_words=1))

    assert dst.read_text(encoding="utf-8") == "previous output\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_blank_lines_not_counted_as_documents(tmp_path):
    src = tmp_path / "input.jsonl"
    dst = tmp_path / "output.jsonl"

    src.write_text(
        '\n{"text":"A valid document."}\n\n',
        encoding="utf-8",
    )

    stats = filter_jsonl(src, dst, QualityConfig(min_words=1))

    assert stats["documents_seen"] == 1
    assert stats["accepted"] == 1
    assert stats["blank_lines_skipped"] == 2
