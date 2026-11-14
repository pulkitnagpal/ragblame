"""The RAG pipeline: the Pipeline protocol any bot can implement, and the demo bot's settings.

A pipeline is a chain of stages. ragblame needs to call each stage on its own so it can swap one
for a perfect version and re-run the rest:

    chunks()  -> every piece the docs were cut into
    search()  -> nearest pieces to the question (approximate index, or exact when exact=True)
    rerank()  -> the same candidates, reordered by a smarter model
    pack()    -> which reranked pieces go into the prompt, in what order, within the budget
    answer()  -> the LLM's answer given the question and the packed pieces
    rechunk() -> optional: a copy of the pipeline whose chunks keep one gold span whole
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ragblame.types import Chunk, Gold, Hit, Trace

DEFAULT_PROMPT = ("You are Docker's support assistant. Answer the question using only the context below. "
                  "Be brief. If the context does not contain the answer, say you don't know.")


@dataclass(frozen=True)
class Config:
    # chunking
    chunker: str = "fixed"            # "fixed" or "heading"
    chunk_words: int = 220
    overlap_words: int = 20
    # embedding and search
    embedder: str = "BAAI/bge-small-en-v1.5"
    index: str = "hnsw"               # "hnsw" or "exact"
    hnsw_m: int = 16
    ef_construction: int = 40
    ef_search: int = 16
    k_candidates: int = 20
    # ranking and packing
    reranker: str | None = "cross-encoder/ms-marco-MiniLM-L6-v2"   # None: keep search order
    reranker_mode: str = "normal"     # "normal" or "reversed" (a benchmark fault)
    top_n: int = 5
    budget_words: int = 900
    # generation
    generator: str = "mlx-community/Qwen3.5-4B-MLX-4bit"
    system_prompt: str = DEFAULT_PROMPT
    max_tokens: int = 256
    # benchmark faults
    corrupt_docs: frozenset[str] = field(default_factory=frozenset)   # docs whose chunk vectors are scrambled
    exclude_docs: frozenset[str] = field(default_factory=frozenset)   # docs left out of the bot entirely


@runtime_checkable
class Pipeline(Protocol):
    def chunks(self) -> list[Chunk]: ...
    def search(self, query: str, k: int, *, exact: bool = False) -> list[Hit]: ...
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...
    def pack(self, hits: list[Hit]) -> list[Chunk]: ...
    def answer(self, query: str, chunks: list[Chunk]) -> str: ...
    def run(self, query: str, question_id: str = "") -> Trace: ...
    # Optional: def rechunk(self, gold: Gold) -> "Pipeline": ...


# ---------------------------------------------------------------------------------------------
# The demo bot: a small, honest RAG pipeline where every stage can be swapped or broken on purpose.
# ---------------------------------------------------------------------------------------------

import dataclasses
import hashlib
import threading

import numpy as np

from ragblame import models


def _load_model_modules() -> None:
    """Importing these registers their model loaders with ragblame.models."""
    import ragblame.embed  # noqa: F401
    import ragblame.generate  # noqa: F401
    import ragblame.rerank  # noqa: F401


def _scrambled(chunk_id: str, dim: int) -> np.ndarray:
    """A fixed random unit vector: what a broken embedding looks like (it points nowhere useful)."""
    rng = np.random.default_rng(int(hashlib.sha256(chunk_id.encode()).hexdigest()[:8], 16))
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# The local models (MPS and MLX) aren't safe to call from several threads at once, but diagnosis
# runs questions in parallel so that slow Claude judge calls overlap. Every model call takes this lock.
MODEL_LOCK = threading.RLock()


def _words(text: str) -> int:
    return len(text.split())


class DemoBot:
    """The reference RAG bot ragblame ships with. Implements the Pipeline protocol, plus rechunk()."""

    def __init__(self, docs, config: Config = Config(), *, chunk_override=None):
        from ragblame.chunking import fixed_chunks, heading_chunks
        from ragblame.index import VectorIndex

        _load_model_modules()
        self.all_docs = list(docs)
        self.config = config
        # A fault can remove docs entirely: then the bot simply never saw them.
        self.docs = [d for d in self.all_docs if d.id not in config.exclude_docs]
        self.doc_by_id = {d.id: d for d in self.docs}
        # Stage 1, chunking: cut every doc into pieces small enough to search and to fit in a prompt.
        if chunk_override is not None:
            self._chunks = [c for c in chunk_override if c.doc_id in self.doc_by_id]
        elif config.chunker == "heading":
            self._chunks = heading_chunks(self.docs)
        else:
            self._chunks = fixed_chunks(self.docs, words=config.chunk_words, overlap=config.overlap_words)
        # Stage 2, embedding: turn every chunk into a vector (its "meaning coordinates").
        self.embedder = models.get("embedder", config.embedder)
        with MODEL_LOCK:
            vecs = self.embedder.encode([c.text for c in self._chunks]) if self._chunks else np.zeros((0, 384), np.float32)
        if config.corrupt_docs:
            vecs = vecs.copy()
            for i, c in enumerate(self._chunks):
                if c.doc_id in config.corrupt_docs:
                    vecs[i] = _scrambled(c.id, vecs.shape[1])
        self._vectors = vecs
        # Stage 3, the index: a fast, approximate way to find the nearest vectors (plus an exact one,
        # used only for instrumentation and for the "perfect index" swap).
        self._ann = VectorIndex(vecs, kind=config.index, m=config.hnsw_m,
                                ef_construction=config.ef_construction, ef_search=config.ef_search)
        self._exact = VectorIndex(vecs, kind="exact")

    # -- Pipeline protocol --------------------------------------------------------------------

    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def search(self, query: str, k: int, *, exact: bool = False) -> list[Hit]:
        """Nearest chunks to the question. exact=True checks every chunk; otherwise the index takes shortcuts."""
        if not self._chunks:
            return []
        with MODEL_LOCK:
            q = self.embedder.encode([query], query=True)[0]
        found = (self._exact if exact else self._ann).search(q, min(k, len(self._chunks)))
        return [Hit(self._chunks[i], float(s)) for i, s in found]

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Stage 4: a slower, smarter model re-reads each candidate next to the question and reorders them."""
        if not hits or self.config.reranker is None:
            return list(hits)
        with MODEL_LOCK:
            scores = models.get("reranker", self.config.reranker).score(query, [h.chunk.text for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=self.config.reranker_mode != "reversed")
        return [Hit(h.chunk, float(s)) for h, s in ranked]

    def pack(self, hits: list[Hit]) -> list[Chunk]:
        """Stage 5: choose what goes into the prompt, in order, without going over the word budget."""
        packed, used = [], 0
        for h in hits[: self.config.top_n]:
            n = _words(h.chunk.text)
            if used + n <= self.config.budget_words:
                packed.append(h.chunk)
                used += n
            elif not packed:
                # The first chunk alone is too big: keep its first budget_words words rather than nothing.
                words = h.chunk.text.split()[: self.config.budget_words]
                packed.append(dataclasses.replace(h.chunk, text=" ".join(words)))
                break
            else:
                break
        return packed

    def answer(self, query: str, chunks: list[Chunk]) -> str:
        """Stage 6: the LLM writes the answer from the question and the packed chunks."""
        from ragblame.generate import build_messages

        messages = build_messages(self.config.system_prompt, query, chunks)
        with MODEL_LOCK:
            return models.get("generator", self.config.generator).generate(messages, max_tokens=self.config.max_tokens)

    def run(self, query: str, question_id: str = "") -> Trace:
        from ragblame.generate import build_messages

        exact = self.search(query, 50, exact=True)          # instrumentation: where would perfect search put things?
        candidates = self.search(query, self.config.k_candidates)
        reranked = self.rerank(query, candidates)
        packed = self.pack(reranked)
        messages = build_messages(self.config.system_prompt, query, packed)
        prompt = "\n\n".join(m["content"] for m in messages)
        answer = self.answer(query, packed)
        return Trace(question_id, exact, candidates, reranked, packed, prompt, answer)

    # -- Hooks that unlock more experiments -----------------------------------------------------

    def rechunk(self, gold: Gold) -> "DemoBot":
        """A copy of this bot whose chunks keep the gold span whole (the "perfect chunking" swap)."""
        from ragblame.chunking import rechunk_around

        doc = self.doc_by_id.get(gold.doc_id)
        if doc is None:
            return self
        return DemoBot(self.all_docs, self.config, chunk_override=rechunk_around(self._chunks, doc, gold))

    def with_config(self, **changes) -> "DemoBot":
        """A copy with some settings changed. Models and cached vectors are shared, so this is cheap."""
        return DemoBot(self.all_docs, dataclasses.replace(self.config, **changes))
