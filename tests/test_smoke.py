"""Task 1 smoke tests: every local model loads and does something sensible. Slow, real models."""
import numpy as np
import pytest

from ragblame.index import VectorIndex

pytestmark = pytest.mark.slow


def test_mlx_generates_without_thinking():
    from ragblame import models
    from ragblame.generate import Generator  # noqa: F401  (registers the loader)
    from ragblame.pipeline import Config

    reply = models.get("generator", Config().generator).generate(
        [{"role": "user", "content": "Reply with the word ok"}], max_tokens=16)
    assert "ok" in reply.lower()
    assert "<think>" not in reply


def test_bge_small_embeds_on_mps(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGBLAME_HOME", str(tmp_path))
    from ragblame.embed import Embedder

    emb = Embedder("BAAI/bge-small-en-v1.5")
    vecs = emb.encode(["docker run", "docker build"])
    assert vecs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert emb.model.device.type == "mps"


def test_faiss_hnsw_and_flat():
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((100, 64)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    flat = VectorIndex(vecs, kind="exact")
    hnsw = VectorIndex(vecs, kind="hnsw")
    assert flat.search(vecs[42], 1)[0][0] == 42
    assert len(hnsw.search(vecs[42], 5)) == 5
