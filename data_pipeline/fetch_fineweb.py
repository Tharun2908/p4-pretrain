"""M2-007: bounded, reproducible FineWeb-Edu source extraction."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_rows(rows, output_dir: Path, limit: int = 500):
    """Export a bounded iterable into a new, auditable directory."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))

    try:
        seen_ids = set()
        rows_seen = 0
        written = 0
        empty_skipped = 0

        with (
            (staging / "raw.jsonl").open("w", encoding="utf-8") as raw,
            (staging / "provenance.jsonl").open("w", encoding="utf-8") as provenance,
        ):
            for row in rows:
                rows_seen += 1

                if not isinstance(row, dict):
                    raise ValueError(f"Invalid row {rows_seen}")

                doc_id = row.get("id")
                content = row.get("text")

                if not isinstance(doc_id, str) or not doc_id.strip():
                    raise ValueError(f"Missing document ID at row {rows_seen}")
                if not isinstance(content, str):
                    raise ValueError(f"Invalid text at row {rows_seen}")
                if doc_id in seen_ids:
                    raise ValueError(f"Repeated document ID: {doc_id}")

                seen_ids.add(doc_id)

                if not content.strip():
                    empty_skipped += 1
                    continue

                raw.write(json.dumps({
                    "id": doc_id,
                    "source": "fineweb-edu",
                    "text": content,
                }, ensure_ascii=False) + "\n")

                provenance.write(json.dumps({
                    "id": doc_id,
                    "url": row.get("url"),
                    "dump": row.get("dump"),
                    "date": row.get("date"),
                    "file_path": row.get("file_path"),
                }, ensure_ascii=False) + "\n")

                written += 1
                if written == limit:
                    break

            for stream in (raw, provenance):
                stream.flush()
                os.fsync(stream.fileno())

        if written == 0:
            raise ValueError("No nonempty documents found")

        manifest = {
            "dataset": DATASET,
            "config": CONFIG,
            "revision": REVISION,
            "license": "ODC-By 1.0",
            "terms": "https://commoncrawl.org/terms-of-use",
            "selection": "first_n_nonempty_stream_rows",
            "requested_documents": limit,
            "rows_seen": rows_seen,
            "documents_exported": written,
            "empty_rows_skipped": empty_skipped,
            "raw_sha256": file_sha256(staging / "raw.jsonl"),
            "provenance_sha256": file_sha256(
                staging / "provenance.jsonl"
            ),
        }

        with (staging / "source_manifest.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        if output_dir.exists():
            raise FileExistsError(output_dir)

        os.rename(staging, output_dir)
        return manifest

    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    # Delayed import lets the exporter be tested without network access.
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET,
        name=CONFIG,
        split="train",
        streaming=True,
        revision=REVISION,
    )

    manifest = export_rows(dataset, args.output_dir, args.limit)
    print(json.dumps(manifest, indent=2))
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
