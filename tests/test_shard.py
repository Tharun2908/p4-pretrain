"""M2-006 sharding tests: lossless roundtrip and publication safety."""

import hashlib
import json

import numpy as np
import pytest

from data_pipeline.shard import shard_splits


def setup_tokens(
    root,
    train=(1, 2, 3, 50256, 4, 5, 6, 7),
    val=(8, 50256, 9),
):
    root.mkdir()
    parts = {}

    for name, ids in (("train", train), ("val", val)):
        payload = np.asarray(ids, dtype="<u2").tobytes()
        (root / f"{name}.bin").write_bytes(payload)
        parts[name] = {
            "total_tokens": len(ids),
            "boundary_tokens": ids.count(50256),
            "output_sha256": hashlib.sha256(payload).hexdigest(),
        }

    (root / "tokenization_manifest.json").write_text(
        json.dumps({
            "format": "flat_uint16_le_v1",
            "tokenizer": "gpt2",
            "vocab_size": 50257,
            "eot_token_id": 50256,
            "splits": parts,
        }),
        encoding="utf-8",
    )
    return root


def test_lossless_reassembly_and_ranges(tmp_path):
    src = setup_tokens(tmp_path / "tokens")
    dst = tmp_path / "shards"
    result = shard_splits(src, dst, 3)

    assert result["splits"]["train"]["shard_count"] == 3
    assert [
        s["tokens"] for s in result["splits"]["train"]["shards"]
    ] == [3, 3, 2]

    for split in ("train", "val"):
        pieces = result["splits"][split]["shards"]

        assert b"".join(
            (dst / p["file"]).read_bytes() for p in pieces
        ) == (src / f"{split}.bin").read_bytes()

        assert pieces[0]["start_token"] == 0
        assert pieces[-1]["end_token_exclusive"] == (
            result["splits"][split]["tokens"]
        )

        assert all(
            p["sha256"] == hashlib.sha256(
                (dst / p["file"]).read_bytes()
            ).hexdigest()
            for p in pieces
        )


def test_deterministic_output(tmp_path):
    src = setup_tokens(tmp_path / "tokens")
    shard_splits(src, tmp_path / "one", 5)
    shard_splits(src, tmp_path / "two", 5)

    assert {
        p.name: p.read_bytes()
        for p in (tmp_path / "one").iterdir()
    } == {
        p.name: p.read_bytes()
        for p in (tmp_path / "two").iterdir()
    }


def test_bad_source_checksum_not_published(tmp_path):
    src = setup_tokens(tmp_path / "tokens")

    (src / "val.bin").write_bytes(
        np.asarray([10, 11], dtype="<u2").tobytes()
    )

    with pytest.raises(ValueError, match="content differs"):
        shard_splits(src, tmp_path / "out", 3)

    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.*"))


def test_odd_byte_count_rejected(tmp_path):
    src = setup_tokens(tmp_path / "tokens")

    with (src / "train.bin").open("ab") as f:
        f.write(b"x")

    with pytest.raises(ValueError, match="odd byte"):
        shard_splits(src, tmp_path / "out", 3)

    assert not (tmp_path / "out").exists()


def test_out_of_vocab_rejected(tmp_path):
    src = setup_tokens(
        tmp_path / "tokens", train=(1, 60000, 3)
    )

    with pytest.raises(ValueError, match="outside GPT-2"):
        shard_splits(src, tmp_path / "out", 3)


def test_existing_output_preserved(tmp_path):
    src = setup_tokens(tmp_path / "tokens")
    dst = tmp_path / "out"
    dst.mkdir()
    (dst / "keep").write_text("untouched")

    with pytest.raises(FileExistsError):
        shard_splits(src, dst, 3)

    assert (dst / "keep").read_text() == "untouched"


def test_empty_and_bad_shard_size(tmp_path):
    src = setup_tokens(tmp_path / "tokens", train=())

    with pytest.raises(ValueError, match="empty"):
        shard_splits(src, tmp_path / "out", 3)

    for value in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            shard_splits(src, tmp_path / "out", value)
