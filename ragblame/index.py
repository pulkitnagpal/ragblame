"""Vector search: find the stored vectors closest to a query vector.

Every chunk becomes a vector (see embed.py). Because the vectors are L2-normalised, the inner
product of two of them is their cosine similarity: 1.0 means "same direction", 0 means unrelated.
"Search" means: give me the k stored vectors with the highest inner product with the query.

There are two ways to do that here:

- exact ("flat"): compare the query with every stored vector. Always right, but the cost grows
  with the number of vectors.
- HNSW (Hierarchical Navigable Small World): approximate nearest-neighbour search. At build time
  each vector is linked to about `m` near neighbours, in a few layers of graphs (sparse at the top,
  dense at the bottom). A search starts at the top, greedily walks towards the query, and drops a
  layer each time it can't get closer. It keeps a shortlist of `ef_search` best candidates while
  walking; a small shortlist is fast but can wander past the true neighbours. That is a real,
  silent failure: the right chunk exists, but the index never returns it.

The defaults (m=16, ef_construction=40, ef_search=16) are FAISS's own defaults, not sabotage.
"""
from __future__ import annotations

import faiss
import numpy as np


def _as_matrix(vectors: np.ndarray) -> np.ndarray:
    """FAISS wants a contiguous 2-D float32 array, one row per vector."""
    return np.ascontiguousarray(np.atleast_2d(vectors), dtype=np.float32)


class VectorIndex:
    """A FAISS index over `vectors`, searched by inner product. Row i of `vectors` gets id i."""

    def __init__(self, vectors: np.ndarray, kind: str = "hnsw", m: int = 16,
                 ef_construction: int = 40, ef_search: int = 16):
        vectors = _as_matrix(vectors)
        self.kind = kind
        self.index = self._build(vectors.shape[1], kind, m, ef_construction, ef_search)
        # Insertion order matters for HNSW (the graph is built greedily), so keep the caller's order.
        self.index.add(vectors)

    @staticmethod
    def _build(d: int, kind: str, m: int, ef_construction: int, ef_search: int):
        """Make an empty exact (IndexFlatIP) or approximate (IndexHNSWFlat) index of dimension d."""
        if kind == "exact":
            return faiss.IndexFlatIP(d)
        if kind == "hnsw":
            index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction   # shortlist size while linking the graph
            index.hnsw.efSearch = ef_search               # shortlist size while searching
            return index
        raise ValueError(f"unknown index kind {kind!r}; use 'hnsw' or 'exact'")

    def __len__(self) -> int:
        return self.index.ntotal

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """The k best (id, inner-product score) pairs for one query vector, best first."""
        scores, ids = self.index.search(_as_matrix(qvec), k)
        # FAISS pads with id -1 when it found fewer than k results.
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def recall_at_k(ann: VectorIndex, exact: VectorIndex, queries: np.ndarray, k: int) -> float:
    """How much of the true top-k the approximate index finds, averaged over queries (1.0 = all of it)."""
    queries = _as_matrix(queries)
    overlaps = []
    for q in queries:
        found = {i for i, _ in ann.search(q, k)}
        truth = {i for i, _ in exact.search(q, k)}
        overlaps.append(len(found & truth) / k)
    return float(np.mean(overlaps)) if overlaps else 0.0
