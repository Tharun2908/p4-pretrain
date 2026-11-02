"""M2-001: deterministic, failure-safe JSONL document ingestion."""

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    """Normalize Unicode and line endings without lowercasing."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text).strip()


def ingest(input_path: Path, output_path: Path) -> None:
    """Validate, normalize, fingerprint and atomically write JSONL."""

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    written = 0
    empty = 0
    tmp_path = None

    try:
        with input_path.open("r", encoding="utf-8") as src:
            # Temporary file is on the same filesystem as the output.
            # This allows os.replace() to rename it atomically.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as dst:

                tmp_path = Path(dst.name)

                for line_number, line in enumerate(src, start=1):
                    if not line.strip():
                        continue

                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON at line {line_number}"
                        ) from exc

                    if not isinstance(doc, dict):
                        raise ValueError(
                            f"Expected object at line {line_number}"
                        )

                    doc_id = doc.get("id")
                    source = doc.get("source", "unknown")
                    text = doc.get("text")

                    # Validate identifiers independently of text content.
                    if not isinstance(doc_id, str) or not doc_id.strip():
                        raise ValueError(
                            f"Invalid id at line {line_number}"
                        )

                    if not isinstance(source, str) or not source.strip():
                        raise ValueError(
                            f"Invalid source at line {line_number}"
                        )

                    if not isinstance(text, str):
                        raise ValueError(
                            f"Invalid text type at line {line_number}"
                        )

                    key = (source, doc_id)

                    if key in seen_ids:
                        raise ValueError(
                            f"Duplicate document ID: {key}"
                        )

                    seen_ids.add(key)

                    # Blank text is skipped, not treated as malformed input.
                    text = normalize_text(text)

                    if not text:
                        empty += 1
                        continue

                    fingerprint = hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest()

                    record = {
                        "id": doc_id,
                        "source": source,
                        "text": text,
                        "sha256": fingerprint,
                    }

                    dst.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    written += 1

                # Flush data before publishing the completed output.
                dst.flush()
                os.fsync(dst.fileno())

        # Atomic publication: only after successful processing.
        os.replace(tmp_path, output_path)

    finally:
        # If validation or writing fails, remove incomplete temp output.
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    print(f"Documents written: {written}")
    print(f"Empty documents skipped: {empty}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    ingest(args.input, args.output)
