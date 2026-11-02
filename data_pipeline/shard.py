"""M2-006: losslessly shard flat uint16 GPT-2 token streams."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_splits(
    token_dir: Path, output_dir: Path, max_tokens_per_shard: int = 1_000_000
) -> dict:
    """Produce ordered, checksummed shards without changing any token IDs."""
    if (
        isinstance(max_tokens_per_shard, bool)
        or not isinstance(max_tokens_per_shard, int)
        or max_tokens_per_shard <= 0
    ):
        raise ValueError("max_tokens_per_shard must be a positive integer")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")

    source_manifest_path = token_dir / "tokenization_manifest.json"
    with source_manifest_path.open("r", encoding="utf-8") as f:
        source_meta = json.load(f)

    if (
        source_meta.get("format") != "flat_uint16_le_v1"
        or source_meta.get("tokenizer") != "gpt2"
        or source_meta.get("vocab_size") != 50257
        or source_meta.get("eot_token_id") != 50256
    ):
        raise ValueError("Unsupported tokenization manifest configuration")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ))

    try:
        result = {
            "format": "flat_uint16_le_sharded_v1",
            "source_tokenization_manifest_sha256": sha256_file(
                source_manifest_path
            ),
            "tokenizer": "gpt2",
            "vocab_size": 50257,
            "eot_token_id": 50256,
            "max_tokens_per_shard": max_tokens_per_shard,
            "splits": {},
        }

        for split in ("train", "val"):
            src_path = token_dir / f"{split}.bin"
            expected = source_meta["splits"][split]

            if src_path.stat().st_size % 2:
                raise ValueError(
                    f"{src_path}: odd byte length for uint16 stream"
                )

            stream_hash = hashlib.sha256()
            shards = []
            token_count = 0
            boundary_count = 0

            with src_path.open("rb") as src:
                chunks = iter(
                    lambda: src.read(max_tokens_per_shard * 2), b""
                )

                for index, payload in enumerate(chunks):
                    values = np.frombuffer(payload, dtype="<u2")

                    if bool(np.any(values >= 50257)):
                        raise ValueError(
                            f"{src_path}: token ID outside GPT-2 vocabulary"
                        )

                    n = len(values)
                    count_eot = int(
                        np.count_nonzero(values == 50256)
                    )
                    name = f"{split}-{index:06d}.bin"

                    with (staging / name).open("wb") as dst:
                        dst.write(payload)
                        dst.flush()
                        os.fsync(dst.fileno())

                    shards.append({
                        "file": name,
                        "start_token": token_count,
                        "end_token_exclusive": token_count + n,
                        "tokens": n,
                        "eot_tokens": count_eot,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    })

                    stream_hash.update(payload)
                    token_count += n
                    boundary_count += count_eot

            if (
                token_count != expected["total_tokens"]
                or boundary_count != expected["boundary_tokens"]
                or stream_hash.hexdigest() != expected["output_sha256"]
            ):
                raise ValueError(
                    f"{split}: content differs from tokenization manifest"
                )

            if not shards:
                raise ValueError(f"{split}: empty token stream")

            result["splits"][split] = {
                "tokens": token_count,
                "eot_tokens": boundary_count,
                "sha256": stream_hash.hexdigest(),
                "shard_count": len(shards),
                "shards": shards,
            }

        with (staging / "shard_manifest.json").open(
            "w", encoding="utf-8"
        ) as dst:
            json.dump(result, dst, indent=2, sort_keys=True)
            dst.write("\n")
            dst.flush()
            os.fsync(dst.fileno())

        if output_dir.exists():
            raise FileExistsError(
                f"Output already exists: {output_dir}"
            )

        os.rename(staging, output_dir)

    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-tokens-per-shard", type=int, default=1_000_000
    )
    args = parser.parse_args()

    result = shard_splits(
        args.token_dir, args.output_dir, args.max_tokens_per_shard
    )

    print(json.dumps({
        split: {
            key: value
            for key, value in info.items()
            if key != "shards"
        }
        for split, info in result["splits"].items()
    }, indent=2))
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
