"""M2-003: exact document deduplication using a disk-backed SQLite index."""

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path


def exact_deduplicate(input_path: Path, output_path: Path) -> dict[str, int]:
    """Keep the first document per content hash, in input order.

    The SQLite index resides in temporary disk storage, not a growing Python set.
    Existing output remains unchanged if processing raises an exception.
    """
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "documents_seen": 0,
        "unique": 0,
        "duplicates_removed": 0,
        "blank_lines_skipped": 0,
    }
    tmp_output = None

    try:
        with tempfile.TemporaryDirectory(
            dir=output_path.parent, prefix=".exact-dedup-index-"
        ) as index_dir:
            conn = sqlite3.connect(str(Path(index_dir) / "seen.sqlite"))
            try:
                conn.execute("CREATE TABLE seen (sha256 TEXT PRIMARY KEY)")
                with input_path.open("r", encoding="utf-8") as src:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=output_path.parent,
                        prefix=f".{output_path.name}.",
                        suffix=".tmp",
                        delete=False,
                    ) as dst:
                        tmp_output = Path(dst.name)
                        for line_number, line in enumerate(src, start=1):
                            if not line.strip():
                                stats["blank_lines_skipped"] += 1
                                continue

                            try:
                                doc = json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise ValueError(
                                    f"Invalid JSON at line {line_number}"
                                ) from exc

                            if not isinstance(doc, dict):
                                raise ValueError(
                                    f"Expected document object at line {line_number}"
                                )
                            if not all(
                                isinstance(doc.get(field), str) and doc[field].strip()
                                for field in ("id", "source", "text")
                            ):
                                raise ValueError(
                                    f"Invalid id, source, or text at line {line_number}"
                                )

                            digest = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
                            if doc.get("sha256") != digest:
                                raise ValueError(
                                    f"Missing or incorrect sha256 at line {line_number}"
                                )

                            stats["documents_seen"] += 1
                            inserted = conn.execute(
                                "INSERT OR IGNORE INTO seen (sha256) VALUES (?)", (digest,)
                            ).rowcount

                            if inserted:
                                dst.write(line if line.endswith("\n") else line + "\n")
                                stats["unique"] += 1
                            else:
                                stats["duplicates_removed"] += 1

                        dst.flush()
                        os.fsync(dst.fileno())
                conn.commit()
            finally:
                conn.close()

            # Publish only after the full input has been validated and processed.
            os.replace(tmp_output, output_path)

    finally:
        if tmp_output is not None:
            tmp_output.unlink(missing_ok=True)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stats = exact_deduplicate(args.input, args.output)
    print(json.dumps(stats, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
