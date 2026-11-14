"""Reranking: rescore search candidates with a slower, smarter model.

The embedder encodes the question and each passage *separately*, then compares two vectors. That
is fast (passages are encoded once, ahead of time) but coarse: the model never sees the question
and the passage together.

A cross-encoder reads the question and one passage *together*, as a single input, and outputs one
relevance score. Every word of the question can attend to every word of the passage, so it catches
things like "this passage mentions pulls but for the wrong account type". The price: it has to run
once per (question, passage) pair at question time, so it is only used on the top ~20 candidates
that vector search returned.

Scores are raw model outputs (logits for ms-marco-MiniLM). Only their order matters.
"""
from __future__ import annotations

from ragblame import models
from ragblame.embed import best_device

BATCH_SIZE = 32


class Reranker:
    """A sentence-transformers CrossEncoder that scores (query, passage) pairs."""

    def __init__(self, name: str):
        from sentence_transformers import CrossEncoder
        self.name = name
        self.model = CrossEncoder(name, device=best_device())

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text, in input order. Higher means more relevant."""
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False,
                                    convert_to_numpy=True)
        scores = scores.reshape(len(texts), -1)
        if scores.shape[1] != 1:
            raise ValueError(f"{self.name} returns {scores.shape[1]} scores per pair; expected one")
        return [float(s) for s in scores[:, 0]]


models.register("reranker", Reranker)
