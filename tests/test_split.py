"""M2-004 tests for reproducible group-aware splitting."""

import hashlib
import json

import pytest

from data_pipeline.split import split_groups


def doc(i, members=None):
    text = f"unique text {i}"
    members = members or [("fixture", i)]

    return {
        "id": i,
        "source": "fixture",
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "dedup_group": {
            "group_id": f"group-{i}",
            "size": len(members),
            "members": [
                {"source": source, "id": doc_id}
                for source, doc_id in members
            ],
        },
    }


def put(path, docs):
    path.write_text(
        "".join(json.dumps(d) + "\n" for d in docs),
        encoding="utf-8",
    )


def read(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
    ]


def test_groups_indivisible_and_balanced_by_group_count(tmp_path):
    a = doc(
        "A",
        [("fixture", "A"), ("fixture", "B"), ("fixture", "C")],
    )
    docs = [a] + [doc(str(i)) for i in range(7)]

    path = tmp_path / "in.jsonl"
    put(path, docs)

    result = split_groups(path, tmp_path / "split", 0.25, 42)

    assert result["train_groups"] == 6
    assert result["validation_groups"] == 2

    assert len(read(tmp_path / "split" / "train.jsonl")) == 6
    assert len(read(tmp_path / "split" / "val.jsonl")) == 2

    memberships = {
        (source, doc_id)
        for group in result["groups"]
        for source, doc_id in group["members"]
    }

    assert len(memberships) == 10


def test_input_order_does_not_change_assignment(tmp_path):
    docs = [doc(str(i)) for i in range(8)]

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    put(first, docs)
    put(second, list(reversed(docs)))

    result_a = split_groups(first, tmp_path / "out1", 0.25, 17)
    result_b = split_groups(second, tmp_path / "out2", 0.25, 17)

    assert result_a == result_b

    for filename in ("train.jsonl", "val.jsonl", "manifest.json"):
        assert (
            (tmp_path / "out1" / filename).read_bytes()
            == (tmp_path / "out2" / filename).read_bytes()
        )


def test_group_id_does_not_control_assignment(tmp_path):
    a, b = doc("a"), doc("b")

    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"

    put(first, [a, b])

    a["dedup_group"]["group_id"] = "different-name"
    put(second, [a, b])

    result_a = split_groups(first, tmp_path / "s1")
    result_b = split_groups(second, tmp_path / "s2")

    assert result_a["groups"] == result_b["groups"]


def test_overlapping_members_rejected(tmp_path):
    docs = [
        doc("a", [("fixture", "a"), ("fixture", "shared")]),
        doc("b", [("fixture", "b"), ("fixture", "shared")]),
    ]

    src = tmp_path / "input.jsonl"
    dest = tmp_path / "output"
    put(src, docs)

    with pytest.raises(ValueError, match="multiple groups"):
        split_groups(src, dest)

    assert not dest.exists()


def test_corrupted_fingerprint_rejected(tmp_path):
    docs = [doc("a"), doc("b")]
    docs[1]["sha256"] = "0" * 64

    src = tmp_path / "input.jsonl"
    dest = tmp_path / "output"
    put(src, docs)

    with pytest.raises(ValueError, match="sha256"):
        split_groups(src, dest)

    assert not dest.exists()


def test_destination_not_overwritten(tmp_path):
    src = tmp_path / "input.jsonl"
    dest = tmp_path / "output"

    put(src, [doc("a"), doc("b")])

    dest.mkdir()
    (dest / "keep").write_text("unchanged")

    with pytest.raises(FileExistsError):
        split_groups(src, dest)

    assert (dest / "keep").read_text() == "unchanged"


def test_small_input_and_invalid_fraction(tmp_path):
    src = tmp_path / "input.jsonl"
    put(src, [doc("only")])

    with pytest.raises(ValueError, match="at least two"):
        split_groups(src, tmp_path / "one")

    with pytest.raises(ValueError, match="val_fraction"):
        split_groups(src, tmp_path / "two", val_fraction=1)
