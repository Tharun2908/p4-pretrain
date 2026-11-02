"""M2-003 near-deduplication integration tests."""

import hashlib
import json

import pytest

from data_pipeline.minhash_lsh import jaccard, word_shingles
from data_pipeline.near_dedup import near_deduplicate


def make_doc(doc_id, text):
    return {
        "id": doc_id,
        "source": "fixture",
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def write_docs(path, docs):
    path.write_text(
        "".join(json.dumps(d) + "\n" for d in docs),
        encoding="utf-8",
    )


def read_docs(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_identical_shingles_form_one_group(tmp_path):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"

    write_docs(src, [
        make_doc("a", "The transformer learns token relationships."),
        make_doc("b", "The transformer learns token relationships!"),
        make_doc("c", "Completely different words appear here."),
    ])

    stats = near_deduplicate(src, dst)

    assert stats["retained"] == 2
    assert stats["removed"] == 1

    output = read_docs(dst)
    assert [d["id"] for d in output] == ["a", "c"]
    assert output[0]["dedup_group"]["size"] == 2


def test_transitive_similarity_group(tmp_path):
    """A~B and B~C, but A and C are below threshold."""

    a = " ".join(f"w{i}" for i in range(0, 100))
    b = " ".join(f"w{i}" for i in range(8, 108))
    c = " ".join(f"w{i}" for i in range(16, 116))

    assert jaccard(word_shingles(a, 1), word_shingles(b, 1)) >= 0.8
    assert jaccard(word_shingles(b, 1), word_shingles(c, 1)) >= 0.8
    assert jaccard(word_shingles(a, 1), word_shingles(c, 1)) < 0.8

    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_docs(src, [
        make_doc("A", a),
        make_doc("B", b),
        make_doc("C", c),
    ])

    stats = near_deduplicate(
        src, dst, threshold=0.8, shingle_size=1
    )

    assert stats["retained"] == 1
    assert stats["removed"] == 2
    assert stats["verified_similarity_edges"] >= 2

    record = read_docs(dst)[0]
    assert record["id"] == "A"
    assert [x["id"] for x in record["dedup_group"]["members"]] == [
        "A", "B", "C"
    ]


def test_deterministic_output(tmp_path):
    src = tmp_path / "in.jsonl"
    write_docs(src, [
        make_doc("a", "Language model training data"),
        make_doc("b", "Language model training data"),
    ])

    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    assert near_deduplicate(src, one) == near_deduplicate(src, two)
    assert one.read_bytes() == two.read_bytes()


def test_failure_preserves_output(tmp_path):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"

    valid = make_doc("a", "Some valid training text")
    invalid = make_doc("b", "Different training text")
    invalid["sha256"] = "0" * 64

    write_docs(src, [valid, invalid])
    dst.write_text("previous output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        near_deduplicate(src, dst)

    assert dst.read_text(encoding="utf-8") == "previous output\n"
