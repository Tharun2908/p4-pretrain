"""M2-007 offline source extraction tests."""

import hashlib
import json

import pytest

from data_pipeline.fetch_fineweb import export_rows


def example(i, text="Document content"):
    return {
        "id": f"id-{i}",
        "text": text,
        "url": f"https://example.org/{i}",
        "dump": "CC-MAIN-2024-10",
        "date": "2024-01-01",
    }


def test_export_with_provenance(tmp_path):
    output = tmp_path / "sample"
    result = export_rows(
        [example(1), example(2, " "), example(3)],
        output,
        limit=2,
    )

    docs = [
        json.loads(line)
        for line in (output / "raw.jsonl").read_text().splitlines()
    ]
    provenance = [
        json.loads(line)
        for line in (output / "provenance.jsonl").read_text().splitlines()
    ]

    assert [d["id"] for d in docs] == ["id-1", "id-3"]
    assert [p["id"] for p in provenance] == ["id-1", "id-3"]
    assert result["rows_seen"] == 3
    assert result["documents_exported"] == 2
    assert result["empty_rows_skipped"] == 1
    assert result["license"] == "ODC-By 1.0"
    assert result["raw_sha256"] == hashlib.sha256(
        (output / "raw.jsonl").read_bytes()
    ).hexdigest()


def test_deterministic_export(tmp_path):
    rows = [example(i) for i in range(5)]

    export_rows(rows, tmp_path / "one", 3)
    export_rows(rows, tmp_path / "two", 3)

    for filename in (
        "raw.jsonl", "provenance.jsonl", "source_manifest.json"
    ):
        assert (
            (tmp_path / "one" / filename).read_bytes()
            == (tmp_path / "two" / filename).read_bytes()
        )


def test_bad_row_not_published(tmp_path):
    output = tmp_path / "sample"
    bad = example(2)
    bad.pop("id")

    with pytest.raises(ValueError, match="document ID"):
        export_rows([example(1), bad], output)

    assert not output.exists()
    assert not list(tmp_path.glob(".sample.*"))


def test_existing_output_preserved(tmp_path):
    output = tmp_path / "sample"
    output.mkdir()
    (output / "keep").write_text("unchanged")

    with pytest.raises(FileExistsError):
        export_rows([example(1)], output)

    assert (output / "keep").read_text() == "unchanged"


def test_invalid_limit(tmp_path):
    for limit in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            export_rows([example(1)], tmp_path / "sample", limit)
