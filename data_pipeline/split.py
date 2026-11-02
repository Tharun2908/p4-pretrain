"""M2-004: deterministic, group-aware train/validation splitting.

Input: one representative per detected near-duplicate component.
Output: train.jsonl, val.jsonl and manifest.json in a new directory.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def canonical_members(doc, line_number):
    """Validate a group and return sorted, stable member identities."""
    if not isinstance(doc, dict):
        raise ValueError(f"Expected document object at line {line_number}")

    fields = ("id", "source", "text", "sha256")
    if not all(
        isinstance(doc.get(k), str) and doc[k].strip()
        for k in fields
    ):
        raise ValueError(f"Invalid document fields at line {line_number}")

    digest = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
    if doc["sha256"] != digest:
        raise ValueError(f"Incorrect sha256 at line {line_number}")

    group = doc.get("dedup_group")
    if not isinstance(group, dict) or not isinstance(
        group.get("members"), list
    ):
        raise ValueError(
            f"Missing dedup_group.members at line {line_number}"
        )

    size = group.get("size")
    if not isinstance(size, int) or isinstance(size, bool):
        raise ValueError(f"Invalid dedup_group.size at line {line_number}")

    identities = []

    for member in group["members"]:
        if not isinstance(member, dict) or not all(
            isinstance(member.get(k), str) and member[k].strip()
            for k in ("source", "id")
        ):
            raise ValueError(f"Invalid group member at line {line_number}")

        identities.append((member["source"], member["id"]))

    if (
        not identities
        or len(identities) != size
        or len(set(identities)) != size
    ):
        raise ValueError(f"Invalid group membership at line {line_number}")

    if (doc["source"], doc["id"]) not in identities:
        raise ValueError(
            f"Representative absent from group at line {line_number}"
        )

    return [list(identity) for identity in sorted(identities)]


def split_groups(
    input_path: Path,
    output_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Split detected duplicate groups as indivisible units."""

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1 (exclusive)")

    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    groups = {}
    seen_members = set()
    seen_fingerprints = set()

    with input_path.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

            members = canonical_members(doc, line_number)
            identities = set(map(tuple, members))

            if seen_members & identities:
                raise ValueError(
                    f"A member occurs in multiple groups at line {line_number}"
                )

            seen_members.update(identities)

            if doc["sha256"] in seen_fingerprints:
                raise ValueError(
                    f"Exact duplicate representatives across groups "
                    f"at line {line_number}"
                )

            seen_fingerprints.add(doc["sha256"])

            # Stable across input-row and group-member ordering.
            serialized = json.dumps(
                members, ensure_ascii=False, separators=(",", ":")
            )

            key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

            if key in groups:
                raise ValueError(f"Duplicate group at line {line_number}")

            groups[key] = {
                "record": doc,
                "members": members,
                "size": len(members),
            }

    if len(groups) < 2:
        raise ValueError(
            "Need at least two detected groups for a nonempty "
            "train and validation split"
        )

    # Target a group count, keeping both splits nonempty.
    n_val = max(
        1,
        min(
            len(groups) - 1,
            int(len(groups) * val_fraction + 0.5),
        ),
    )

    # Rank groups deterministically using seed + stable group key.
    ranked = sorted(
        groups,
        key=lambda key: (
            hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(),
            key,
        ),
    )

    val_keys = set(ranked[:n_val])

    manifest = {
        "seed": seed,
        "requested_validation_fraction": val_fraction,
        "strategy": "group_count_ranked_sha256_v1",
        "train_groups": len(groups) - n_val,
        "validation_groups": n_val,
        "train_member_documents": sum(
            item["size"]
            for key, item in groups.items()
            if key not in val_keys
        ),
        "validation_member_documents": sum(
            item["size"]
            for key, item in groups.items()
            if key in val_keys
        ),
        "groups": [
            {
                "key": key,
                "split": "val" if key in val_keys else "train",
                "members": groups[key]["members"],
            }
            for key in sorted(groups)
        ],
    }

    # Build all output files in a staging directory.
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )

    try:
        with (
            (staging / "train.jsonl").open("w", encoding="utf-8") as train,
            (staging / "val.jsonl").open("w", encoding="utf-8") as val,
        ):
            for key in sorted(groups):
                stream = val if key in val_keys else train
                stream.write(
                    json.dumps(
                        groups[key]["record"], ensure_ascii=False
                    ) + "\n"
                )

            train.flush()
            os.fsync(train.fileno())

            val.flush()
            os.fsync(val.fileno())

        with (staging / "manifest.json").open(
            "w", encoding="utf-8"
        ) as out:
            json.dump(manifest, out, ensure_ascii=False, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())

        # Publish the completed directory.
        os.rename(staging, output_dir)

    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    manifest = split_groups(
        args.input,
        args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print(json.dumps(
        {k: v for k, v in manifest.items() if k != "groups"},
        indent=2,
    ))
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
