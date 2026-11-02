"""M2-003 tests for disk-backed exact deduplication."""

import hashlib
import json

import pytest

from data_pipeline.exact_dedup import exact_deduplicate


def doc(doc_id, source, text):
    return {
        "id": doc_id,
        "source": source,
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def write_docs(path, docs):
    path.write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs),
        encoding="utf-8",
    )


def read_docs(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_first_occurrence_kept_across_sources(tmp_path):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_docs(src, [
        doc("a", "web", "The cat sat on the mat."),
        doc("b", "books", "Different content."),
        doc("c", "wiki", "The cat sat on the mat."),
    ])
    stats = exact_deduplicate(src, dst)
    assert stats["documents_seen"] == 3
    assert stats["unique"] == 2
    assert stats["duplicates_removed"] == 1
    assert [d["id"] for d in read_docs(dst)] == ["a", "b"]


def test_different_texts_are_retained(tmp_path):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_docs(src, [doc("a", "web", "Alpha"), doc("b", "web", "Beta")])
    assert exact_deduplicate(src, dst)["unique"] == 2


def test_deterministic_repeated_execution(tmp_path):
    src = tmp_path / "in.jsonl"
    write_docs(src, [doc("x", "s", "Repeat"), doc("y", "s", "Repeat")])
    out1, out2 = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    assert exact_deduplicate(src, out1) == exact_deduplicate(src, out2)
    assert out1.read_bytes() == out2.read_bytes()


def test_corrupted_fingerprint_preserves_existing_output(tmp_path):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    good = doc("a", "s", "Good")
    bad = doc("b", "s", "Other")
    bad["sha256"] = "0" * 64
    write_docs(src, [good, bad])
    dst.write_text("old complete output\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        exact_deduplicate(src, dst)
    assert dst.read_text(encoding="utf-8") == "old complete output\n"
    assert list(tmp_path.glob(".exact-dedup-index-*")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_bad_json_preserves_existing_output(tmp_path):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text(
        json.dumps(doc("a", "s", "Good")) + "\n{bad json}\n",
        encoding="utf-8",
    )
    dst.write_text("old complete output\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        exact_deduplicate(src, dst)
    assert dst.read_text(encoding="utf-8") == "old complete output\n"


def test_same_input_output_rejected(tmp_path):
    src = tmp_path / "same.jsonl"
    write_docs(src, [doc("a", "s", "Text")])
    with pytest.raises(ValueError, match="must differ"):
        exact_deduplicate(src, src)
    assert len(read_docs(src)) == 1
