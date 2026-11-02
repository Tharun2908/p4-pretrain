"""Correctness tests for M2 document ingestion."""

import json

import pytest

from data_pipeline.ingest import ingest


def write_docs(path, docs):
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def read_docs(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_unicode_normalization_and_fingerprint(tmp_path):
    src = tmp_path / "input.jsonl"
    dst = tmp_path / "output.jsonl"

    write_docs(src, [
        {"id": "a", "source": "test", "text": "Cafe\u0301\r\nHello"},
        {"id": "b", "source": "test", "text": "Café\nHello"},
    ])

    ingest(src, dst)
    docs = read_docs(dst)

    assert len(docs) == 2
    assert docs[0]["text"] == "Café\nHello"
    assert docs[0]["sha256"] == docs[1]["sha256"]
    assert docs[0]["id"] != docs[1]["id"]


def test_same_id_different_sources_allowed(tmp_path):
    src = tmp_path / "input.jsonl"
    dst = tmp_path / "output.jsonl"

    write_docs(src, [
        {"id": "001", "source": "wiki", "text": "Document A"},
        {"id": "001", "source": "books", "text": "Document B"},
    ])

    ingest(src, dst)
    assert len(read_docs(dst)) == 2


def test_duplicate_id_same_source_rejected(tmp_path):
    src = tmp_path / "input.jsonl"
    dst = tmp_path / "output.jsonl"

    write_docs(src, [
        {"id": "001", "source": "wiki", "text": "Document A"},
        {"id": "001", "source": "wiki", "text": "Document B"},
    ])

    with pytest.raises(ValueError, match="Duplicate document ID"):
        ingest(src, dst)


def test_malformed_json_reports_line(tmp_path):
    src = tmp_path / "input.jsonl"
    dst = tmp_path / "output.jsonl"

    src.write_text(
        '{"id":"1","text":"Valid"}\n'
        '{invalid json}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        ingest(src, dst)


def test_blank_documents_are_skipped(tmp_path):
    src = tmp_path / "input.jsonl"
    dst = tmp_path / "output.jsonl"

    write_docs(src, [
        {"id": "001", "source": "test", "text": "   "},
        {"id": "002", "source": "test", "text": "Valid text"},
    ])

    ingest(src, dst)

    docs = read_docs(dst)
    assert len(docs) == 1
    assert docs[0]["id"] == "002"


def test_failed_ingestion_preserves_previous_output(tmp_path):
    src = tmp_path / "input.jsonl"
    dst = tmp_path / "output.jsonl"

    dst.write_text("previous valid output\n", encoding="utf-8")

    src.write_text(
        '{"id":"1","text":"Valid"}\n'
        '{invalid json}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        ingest(src, dst)

    assert dst.read_text(encoding="utf-8") == "previous valid output\n"
