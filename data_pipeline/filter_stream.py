"""M2-002: streaming quality filtering with rejection statistics."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from data_pipeline.quality_filter import QualityConfig, rejection_reason


REASONS = (
    "too_short",
    "too_long",
    "low_alpha_ratio",
    "repeated_lines",
)


def filter_jsonl(
    input_path: Path,
    output_path: Path,
    config: QualityConfig,
) -> dict:
    """Filter ingested JSONL without loading the corpus into memory."""

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "documents_seen": 0,
        "accepted": 0,
        "rejected": 0,
        "blank_lines_skipped": 0,
        "reasons": {reason: 0 for reason in REASONS},
    }

    tmp_path = None

    try:
        with input_path.open("r", encoding="utf-8") as src:
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

                    text = doc.get("text")
                    if not isinstance(text, str):
                        raise ValueError(
                            f"Missing or invalid text at line {line_number}"
                        )

                    stats["documents_seen"] += 1
                    reason = rejection_reason(text, config)

                    if reason is not None:
                        stats["rejected"] += 1
                        stats["reasons"][reason] += 1
                        continue

                    # Preserve the original ingested record.
                    dst.write(line if line.endswith("\n") else line + "\n")
                    stats["accepted"] += 1

                dst.flush()
                os.fsync(dst.fileno())

        # Publish only a fully processed output.
        os.replace(tmp_path, output_path)

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    assert (
        stats["documents_seen"]
        == stats["accepted"] + stats["rejected"]
    )

    return stats


def main():
    defaults = QualityConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument("--min-words", type=int, default=defaults.min_words)
    parser.add_argument("--max-words", type=int, default=defaults.max_words)
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=defaults.min_alpha_ratio,
    )
    parser.add_argument(
        "--max-duplicate-line-fraction",
        type=float,
        default=defaults.max_duplicate_line_fraction,
    )

    args = parser.parse_args()

    config = QualityConfig(
        min_words=args.min_words,
        max_words=args.max_words,
        min_alpha_ratio=args.min_alpha_ratio,
        max_duplicate_line_fraction=args.max_duplicate_line_fraction,
    )

    stats = filter_jsonl(args.input, args.output, config)

    print(json.dumps(stats, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
