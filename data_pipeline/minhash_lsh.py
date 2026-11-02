"""M2-003: deterministic word-shingle MinHash and in-memory LSH prototype."""

import hashlib
import random
import re
from collections import defaultdict

PRIME = (1 << 61) - 1


def word_shingles(text: str, k: int = 3) -> set[str]:
    """Case-folded word k-shingles; punctuation is ignored for matching only."""
    if k <= 0:
        raise ValueError("k must be positive")

    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)

    if not words:
        return set()

    if len(words) < k:
        return {"\x1f".join(words)}

    return {
        "\x1f".join(words[i:i + k])
        for i in range(len(words) - k + 1)
    }


def jaccard(a: set[str], b: set[str]) -> float:
    """Exact set Jaccard. Empty sets are not treated as duplicates."""
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


class MinHasher:
    """Repeatable family of hash functions approximating MinHash permutations."""

    def __init__(self, num_perm: int = 112, seed: int = 42):
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")

        rng = random.Random(seed)

        self.coefficients = [
            (rng.randrange(1, PRIME), rng.randrange(PRIME))
            for _ in range(num_perm)
        ]

    def signature(self, shingles: set[str]) -> tuple[int, ...]:
        if not shingles:
            raise ValueError("Cannot compute MinHash for empty shingle set")

        hashed = [
            int.from_bytes(
                hashlib.blake2b(
                    s.encode("utf-8"), digest_size=8
                ).digest(),
                "big",
            ) % PRIME
            for s in sorted(shingles)
        ]

        return tuple(
            min((a * value + b) % PRIME for value in hashed)
            for a, b in self.coefficients
        )


class BandLSH:
    """LSH index. Band matches produce candidates, not duplicate decisions."""

    def __init__(self, bands: int = 14, rows: int = 8):
        if bands <= 0 or rows <= 0:
            raise ValueError("bands and rows must be positive")

        self.bands = bands
        self.rows = rows

        self.buckets: dict[
            tuple[int, tuple[int, ...]], set[str]
        ] = defaultdict(set)

        self.ids: set[str] = set()

    def _band_keys(self, signature: tuple[int, ...]):
        if len(signature) != self.bands * self.rows:
            raise ValueError(
                "Signature length does not match bands * rows"
            )

        for band in range(self.bands):
            lo = band * self.rows
            yield (band, signature[lo:lo + self.rows])

    def candidates(self, signature: tuple[int, ...]) -> set[str]:
        result: set[str] = set()

        for key in self._band_keys(signature):
            result.update(self.buckets.get(key, ()))

        return result

    def add(self, doc_id: str, signature: tuple[int, ...]) -> None:
        if doc_id in self.ids:
            raise ValueError(
                f"Document ID already indexed: {doc_id}"
            )

        keys = list(self._band_keys(signature))

        for key in keys:
            self.buckets[key].add(doc_id)

        self.ids.add(doc_id)
