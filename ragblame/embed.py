"""Embeddings: turn text into vectors so that similar meanings land close together.

An embedding model reads a piece of text and outputs a fixed-length list of numbers (384 for
bge-small, 1024 for Qwen3-Embedding-0.6B). It was trained so that a question and the passage that
answers it get vectors pointing in nearly the same direction. We L2-normalise every vector (scale
it to length 1), so the inner product of two vectors is their cosine similarity, which is what
index.py searches by.

Two details matter for retrieval:

- Query prompts. Some models were trained with an instruction in front of the *question* but not
  the passage ("Represent this sentence for searching relevant passages: ..."). Leaving it off
  quietly costs a few points of retrieval quality, so queries and passages are encoded differently
  and cached separately.
- Truncation. bge-small reads at most 512 tokens and silently ignores the rest of a long chunk.

Encoding is the slow part of building the bot, so vectors are cached on disk per model and only
texts that miss the cache reach the model.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from ragblame import models

BATCH_SIZE = 64

# How each model wants its queries marked. Passages never get a prompt.
QUERY_PROMPTS: dict[str, dict[str, str]] = {
    "BAAI/bge-small-en-v1.5": {"prompt": "Represent this sentence for searching relevant passages: "},
    # Qwen3-Embedding ships a named "query" prompt (an "Instruct: ... Query:" template) in its config.
    "Qwen/Qwen3-Embedding-0.6B": {"prompt_name": "query"},
}


def ragblame_home() -> Path:
    """Where ragblame keeps caches: $RAGBLAME_HOME if set, else `.ragblame/` in the repo root."""
    env = os.environ.get("RAGBLAME_HOME")
    return Path(env) if env else Path(__file__).resolve().parent.parent / ".ragblame"


def cache_key(text: str, query: bool) -> str:
    """The cache key for one text. Queries and passages differ, because the query prompt changes the vector."""
    return hashlib.sha256((("q:" if query else "p:") + text).encode("utf-8")).hexdigest()


def best_device() -> str:
    """Apple's GPU (MPS) when available, else the CPU."""
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def normalise(vecs: np.ndarray) -> np.ndarray:
    """Scale each row to length 1 so inner product equals cosine similarity."""
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


class Embedder:
    """A sentence-transformers embedding model with an on-disk vector cache."""

    def __init__(self, name: str):
        self.name = name
        self.cache_path = ragblame_home() / "cache" / "emb" / f"{name.replace('/', '__')}.npz"
        self._model = None
        self._cache: dict[str, np.ndarray] | None = None   # loaded on first use

    @property
    def model(self):
        """The underlying SentenceTransformer, loaded only when a text actually needs encoding."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name, device=best_device())
        return self._model

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        """Vectors for `texts`, one float32 unit-length row each, in input order."""
        cache = self._load_cache()
        keys = [cache_key(t, query) for t in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in cache))   # unique, in order
        if missing:
            text_of = dict(zip(keys, texts))
            fresh = self._encode_with_model([text_of[k] for k in missing], query)
            cache.update(zip(missing, fresh))
            self._save_cache()
        if not keys:
            return np.zeros((0, self._dimension()), dtype=np.float32)
        return np.stack([cache[k] for k in keys]).astype(np.float32)

    def _encode_with_model(self, texts: list[str], query: bool) -> np.ndarray:
        """Run the model on texts that missed the cache, adding the query prompt if asked."""
        prompt = QUERY_PROMPTS.get(self.name, {}) if query else {}
        vecs = self.model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False, **prompt)
        return normalise(vecs)

    def _dimension(self) -> int:
        """Vector length, from a cached vector if there is one, else from the model."""
        if self._cache:
            return len(next(iter(self._cache.values())))
        # sentence-transformers 6 renamed get_sentence_embedding_dimension to get_embedding_dimension.
        dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(dim())

    def _load_cache(self) -> dict[str, np.ndarray]:
        """Read the model's cache file once, the first time it's needed."""
        if self._cache is None:
            self._cache = {}
            if self.cache_path.exists():
                with np.load(self.cache_path) as data:
                    self._cache = dict(zip(data["keys"].tolist(), data["vecs"]))
        return self._cache

    def _save_cache(self) -> None:
        """Write the whole cache, via a temp file so a crash never leaves a half-written one."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp.npz")
        keys = list(self._cache)
        np.savez(tmp, keys=np.array(keys), vecs=np.stack([self._cache[k] for k in keys]))
        os.replace(tmp, self.cache_path)


models.register("embedder", Embedder)
