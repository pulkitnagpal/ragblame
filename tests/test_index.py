import numpy as np

from ragblame.index import VectorIndex, recall_at_k


def unit_vectors(n, d, seed=0):
    """n random directions in d dimensions, each scaled to length 1."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_exact_returns_self_first():
    vecs = unit_vectors(100, 32)
    index = VectorIndex(vecs, kind="exact")
    hits = index.search(vecs[7], 5)
    assert hits[0][0] == 7
    assert abs(hits[0][1] - 1.0) < 1e-5
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_hnsw_matches_exact_on_tiny_set():
    vecs = unit_vectors(50, 32, seed=1)
    queries = unit_vectors(10, 32, seed=2)
    ann = VectorIndex(vecs, kind="hnsw", ef_search=64)
    exact = VectorIndex(vecs, kind="exact")
    for q in queries:
        assert [i for i, _ in ann.search(q, 5)] == [i for i, _ in exact.search(q, 5)]
    assert recall_at_k(ann, exact, queries, 5) == 1.0


def test_recall_drops_with_tiny_ef():
    vecs = unit_vectors(5000, 64, seed=3)
    queries = unit_vectors(200, 64, seed=4)
    ann = VectorIndex(vecs, kind="hnsw", m=4, ef_construction=8, ef_search=1)
    exact = VectorIndex(vecs, kind="exact")
    assert recall_at_k(ann, exact, queries, 5) < 0.9


def test_search_skips_missing_ids():
    vecs = unit_vectors(3, 8)
    hits = VectorIndex(vecs, kind="exact").search(vecs[0], 10)
    assert len(hits) == 3
    assert all(i >= 0 for i, _ in hits)
