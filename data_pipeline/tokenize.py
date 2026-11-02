"""M2-005: deterministic GPT-2 tokenization into flat uint16 files.

Output is compatible with train.py's MemmapBatches on little-endian
systems. Files are published together via staging-directory rename.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import tiktoken


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize_file(input_path: Path, output_path: Path, enc) -> dict:
    stats = {
        "documents": 0,
        "content_tokens": 0,
        "boundary_tokens": 0,
        "total_tokens": 0,
    }

    digest = hashlib.sha256()

    with (
        input_path.open("r", encoding="utf-8") as src,
        output_path.open("wb") as dst,
    ):
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{input_path}: invalid JSON at line {line_number}"
                ) from exc

            if not isinstance(doc, dict) or not all(
                isinstance(doc.get(k), str) and doc[k].strip()
                for k in ("id", "source", "text", "sha256")
            ):
                raise ValueError(
                    f"{input_path}: invalid document at line {line_number}"
                )

            actual_digest = hashlib.sha256(
                doc["text"].encode("utf-8")
            ).hexdigest()

            if doc["sha256"] != actual_digest:
                raise ValueError(
                    f"{input_path}: incorrect sha256 at line {line_number}"
                )

            # Literal special-token strings in source text are treated
            # as ordinary text, not as tokenizer control tokens.
            tokens = enc.encode_ordinary(doc["text"])

            # Explicit separator between documents.
            tokens.append(enc.eot_token)

            if max(tokens) > 65535:
                raise ValueError("Token ID exceeds uint16 capacity")

            payload = np.asarray(tokens, dtype="<u2").tobytes()
            dst.write(payload)
            digest.update(payload)

            stats["documents"] += 1
            stats["content_tokens"] += len(tokens) - 1
            stats["boundary_tokens"] += 1
            stats["total_tokens"] += len(tokens)

        dst.flush()
        os.fsync(dst.fileno())

    if stats["documents"] == 0:
        raise ValueError(f"{input_path}: empty split")

    stats["input_sha256"] = file_sha256(input_path)
    stats["output_sha256"] = digest.hexdigest()

    return stats


def tokenize_splits(split_dir: Path, output_dir: Path) -> dict:
    """Tokenize train and val, publishing only a complete new directory."""

    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")

    required = [
        split_dir / "train.jsonl",
        split_dir / "val.jsonl",
        split_dir / "manifest.json",
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    enc = tiktoken.get_encoding("gpt2")

    if enc.n_vocab != 50257 or enc.eot_token != 50256:
        raise ValueError("Unexpected GPT-2 tokenizer configuration")

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))

    try:
        stats = {}

        for split in ("train", "val"):
            stats[split] = tokenize_file(
                split_dir / f"{split}.jsonl",
                staging / f"{split}.bin",
                enc,
            )

        metadata = {
            "format": "flat_uint16_le_v1",
            "tokenizer": "gpt2",
            "vocab_size": enc.n_vocab,
            "eot_token_id": enc.eot_token,
            "split_manifest_sha256": file_sha256(
                split_dir / "manifest.json"
            ),
            "splits": stats,
        }

        with (staging / "tokenization_manifest.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.rename(staging, output_dir)

    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stats = tokenize_splits(args.split_dir, args.output_dir)
    print(json.dumps(stats, indent=2))
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
