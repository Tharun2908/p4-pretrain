"""M2-005 tests for GPT-2 tokenization and binary output."""

import hashlib
import json

import numpy as np
import pytest

tiktoken = pytest.importorskip("tiktoken")

from data_pipeline.tokenize import tokenize_splits


def doc(doc_id, text):
    return {
        "id": doc_id,
        "source": "fixture",
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def make_splits(root, train_docs, val_docs):
    root.mkdir()

    for name, documents in (
        ("train", train_docs),
        ("val", val_docs),
    ):
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(d) + "\n" for d in documents),
            encoding="utf-8",
        )

    (root / "manifest.json").write_text(
        '{"seed":42}\n', encoding="utf-8"
    )


def read_tokens(path):
    return np.fromfile(path, dtype="<u2").tolist()


def test_exact_token_stream_and_boundaries(tmp_path):
    split = tmp_path / "split"
    output = tmp_path / "tokens"

    make_splits(
        split,
        [doc("a", "Hello world"), doc("b", "Another document")],
        [doc("c", "Validation text")],
    )

    result = tokenize_splits(split, output)
    enc = tiktoken.get_encoding("gpt2")

    expected = (
        enc.encode_ordinary("Hello world")
        + [enc.eot_token]
        + enc.encode_ordinary("Another document")
        + [enc.eot_token]
    )

    assert read_tokens(output / "train.bin") == expected
    assert result["splits"]["train"]["documents"] == 2
    assert result["splits"]["train"]["boundary_tokens"] == 2


def test_special_token_string_is_literal_text(tmp_path):
    split = tmp_path / "split"
    output = tmp_path / "tokens"
    text = "The string <|endoftext|> appears in this document."

    make_splits(split, [doc("a", text)], [doc("b", "Validation")])
    tokenize_splits(split, output)

    enc = tiktoken.get_encoding("gpt2")
    expected = enc.encode_ordinary(text) + [enc.eot_token]

    assert read_tokens(output / "train.bin") == expected
    assert expected.count(enc.eot_token) == 1


def test_deterministic_binary_output(tmp_path):
    split = tmp_path / "split"
    make_splits(
        split,
        [doc("a", "One"), doc("b", "Two")],
        [doc("c", "Three")],
    )

    tokenize_splits(split, tmp_path / "out1")
    tokenize_splits(split, tmp_path / "out2")

    for filename in (
        "train.bin",
        "val.bin",
        "tokenization_manifest.json",
    ):
        assert (
            (tmp_path / "out1" / filename).read_bytes()
            == (tmp_path / "out2" / filename).read_bytes()
        )


def test_invalid_validation_does_not_publish_partial_data(tmp_path):
    split = tmp_path / "split"
    output = tmp_path / "tokens"

    bad = doc("b", "Corrupted")
    bad["sha256"] = "0" * 64

    make_splits(split, [doc("a", "Good training text")], [bad])

    with pytest.raises(ValueError, match="sha256"):
        tokenize_splits(split, output)

    assert not output.exists()
    assert list(tmp_path.glob(".tokens.*")) == []


def test_existing_output_preserved(tmp_path):
    split = tmp_path / "split"
    output = tmp_path / "tokens"

    make_splits(split, [doc("a", "Train")], [doc("b", "Val")])

    output.mkdir()
    (output / "keep.txt").write_text("existing")

    with pytest.raises(FileExistsError):
        tokenize_splits(split, output)

    assert (output / "keep.txt").read_text() == "existing"


def test_empty_split_is_rejected(tmp_path):
    split = tmp_path / "split"
    output = tmp_path / "tokens"

    make_splits(split, [doc("a", "Train")], [])

    with pytest.raises(ValueError, match="empty split"):
        tokenize_splits(split, output)

    assert not output.exists()
