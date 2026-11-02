"""M2-003: MinHash/LSH near-deduplication with connected groups.

Small-corpus reference implementation. Holds document shingles and
signatures in RAM; large-corpus scaling is a separate milestone.
"""

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from data_pipeline.minhash_lsh import (
    BandLSH,
    MinHasher,
    jaccard,
    word_shingles,
)


def near_deduplicate(
    input_path: Path,
    output_path: Path,
    threshold: float = 0.8,
    shingle_size: int = 3,
    bands: int = 14,
    rows: int = 8,
    seed: int = 42,
) -> dict:
    """Keep the first document in each detected similarity component."""

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if shingle_size <= 0 or bands <= 0 or rows <= 0:
        raise ValueError("shingle_size, bands and rows must be positive")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")

    hasher = MinHasher(num_perm=bands * rows, seed=seed)
    index = BandLSH(bands=bands, rows=rows)

    documents = []
    shingle_sets = []
    parent = []
    document_ids = set()

    stats = {
        "documents_seen": 0,
        "candidate_pairs_checked": 0,
        "verified_similarity_edges": 0,
        "groups": 0,
        "retained": 0,
        "removed": 0,
    }

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Smaller input index becomes the canonical root.
            parent[max(ra, rb)] = min(ra, rb)

    with input_path.open("r", encoding="utf-8") as src:
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
                raise ValueError(f"Invalid document at line {line_number}")

            if not all(
                isinstance(doc.get(k), str) and doc[k].strip()
                for k in ("id", "source", "text", "sha256")
            ):
                raise ValueError(
                    f"Invalid document fields at line {line_number}"
                )

            digest = hashlib.sha256(
                doc["text"].encode("utf-8")
            ).hexdigest()

            if doc["sha256"] != digest:
                raise ValueError(
                    f"Incorrect fingerprint at line {line_number}"
                )

            identity = (doc["source"], doc["id"])
            if identity in document_ids:
                raise ValueError(
                    f"Duplicate document identity at line {line_number}"
                )
            document_ids.add(identity)

            shingles = word_shingles(doc["text"], k=shingle_size)
            if not shingles:
                raise ValueError(
                    f"No word shingles at line {line_number}"
                )

            signature = hasher.signature(shingles)
            current = len(documents)

            documents.append(doc)
            shingle_sets.append(shingles)
            parent.append(current)

            # Search all previously indexed documents, including those
            # that may already belong to a duplicate group.
            candidates = sorted(
                int(i) for i in index.candidates(signature)
            )

            for previous in candidates:
                stats["candidate_pairs_checked"] += 1

                if jaccard(shingles, shingle_sets[previous]) >= threshold:
                    union(current, previous)
                    stats["verified_similarity_edges"] += 1

            # Index EVERY document, not just group representatives.
            # Otherwise A~B~C bridging relationships may be missed.
            index.add(str(current), signature)

    groups = defaultdict(list)
    for i in range(len(documents)):
        groups[find(i)].append(i)

    stats["documents_seen"] = len(documents)
    stats["groups"] = len(groups)
    stats["retained"] = len(groups)
    stats["removed"] = len(documents) - len(groups)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as dst:
            tmp_path = Path(dst.name)

            for root in sorted(groups):
                members = groups[root]
                representative = min(members)

                record = dict(documents[representative])
                record["dedup_group"] = {
                    "group_id": f"group-{representative:08d}",
                    "size": len(members),
                    "members": [
                        {
                            "id": documents[i]["id"],
                            "source": documents[i]["source"],
                        }
                        for i in members
                    ],
                }

                dst.write(json.dumps(record, ensure_ascii=False) + "\n")

            dst.flush()
            os.fsync(dst.fileno())

        os.replace(tmp_path, output_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--shingle-size", type=int, default=3)
    parser.add_argument("--bands", type=int, default=14)
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()

    stats = near_deduplicate(
        args.input,
        args.output,
        threshold=args.threshold,
        shingle_size=args.shingle_size,
        bands=args.bands,
        rows=args.rows,
    )

    print(json.dumps(stats, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
