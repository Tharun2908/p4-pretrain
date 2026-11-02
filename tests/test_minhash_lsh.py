"""M2-003 tests for MinHash and LSH."""

import pytest

from data_pipeline.minhash_lsh import (
    BandLSH,
    MinHasher,
    jaccard,
    word_shingles,
)


def test_shingles_normalize_case_and_punctuation():
    assert word_shingles(
        "The CAT, sat on the mat.", 2
    ) == word_shingles(
        "the cat sat on the mat", 2
    )


def test_jaccard_is_exact():
    a = word_shingles("a b c d", 2)
    b = word_shingles("a b c e", 2)

    assert jaccard(a, b) == 0.5


def test_short_and_empty_documents():
    assert word_shingles("one two", 3) == {"one\x1ftwo"}
    assert word_shingles("!!!") == set()
    assert jaccard(set(), set()) == 0.0


def test_minhash_repeatable_across_orderings():
    mh = MinHasher()
    shingles = word_shingles(
        "Transformer layers use attention on token sequences", 3
    )

    assert mh.signature(shingles) == MinHasher().signature(
        set(reversed(sorted(shingles)))
    )
    assert len(mh.signature(shingles)) == 112


def test_minhash_approximates_jaccard():
    a = {str(i) for i in range(100)}
    b = {str(i) for i in range(50, 150)}

    true_similarity = jaccard(a, b)

    mh = MinHasher(num_perm=256)
    x, y = mh.signature(a), mh.signature(b)

    estimate = sum(u == v for u, v in zip(x, y)) / 256

    assert abs(estimate - true_similarity) < 0.12


def test_lsh_identical_signatures_are_candidates():
    mh = MinHasher()
    shingles = word_shingles(
        "This language model uses multiple heads for attention"
    )

    signature = mh.signature(shingles)

    lsh = BandLSH()
    lsh.add("doc-a", signature)

    assert lsh.candidates(signature) == {"doc-a"}


def test_lsh_candidates_require_exact_verification():
    lsh = BandLSH(bands=2, rows=2)
    lsh.add("doc-a", (1, 2, 3, 4))

    # Matching one band is enough to become a candidate.
    assert lsh.candidates((1, 2, 8, 9)) == {"doc-a"}

    # Candidate status alone does not prove Jaccard >= 0.8.
    assert jaccard(
        word_shingles("a b c d", 2),
        word_shingles("a b c e", 2),
    ) < 0.8

    assert lsh.candidates((7, 8, 9, 10)) == set()


def test_invalid_signature_and_duplicate_id():
    lsh = BandLSH(bands=2, rows=2)

    with pytest.raises(ValueError, match="length"):
        lsh.candidates((1, 2))

    lsh.add("same", (1, 2, 3, 4))

    with pytest.raises(ValueError, match="already indexed"):
        lsh.add("same", (5, 6, 7, 8))

    with pytest.raises(ValueError, match="empty"):
        MinHasher().signature(set())
