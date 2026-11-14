import numpy as np
import pytest

BGE = "BAAI/bge-small-en-v1.5"
pytestmark = pytest.mark.slow


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point the vector cache at a throwaway folder so tests never touch the real one."""
    monkeypatch.setenv("RAGBLAME_HOME", str(tmp_path))
    return tmp_path


def count_encode_calls(embedder, monkeypatch):
    """Wrap the model's encode so the test can see which texts actually reached the model."""
    calls = []
    real = embedder.model.encode

    def counting(texts, *args, **kwargs):
        calls.append(list(texts))
        return real(texts, *args, **kwargs)

    monkeypatch.setattr(embedder.model, "encode", counting)
    return calls


def test_bge_shape_and_norm(home):
    from ragblame.embed import Embedder

    vecs = Embedder(BGE).encode(["How do I log in to Docker Hub?", "docker login"])
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)


def test_cache_avoids_reencode(home, monkeypatch):
    from ragblame.embed import Embedder

    emb = Embedder(BGE)
    calls = count_encode_calls(emb, monkeypatch)
    first = emb.encode(["alpha", "beta"])
    assert calls == [["alpha", "beta"]]

    again = emb.encode(["beta", "alpha"])
    assert len(calls) == 1                      # all hits, no model call
    assert np.array_equal(again, first[::-1])   # rows follow the input order

    emb.encode(["alpha", "gamma"])
    assert calls[-1] == ["gamma"]               # only the miss is encoded

    # A fresh Embedder reads the vectors back from disk.
    fresh = Embedder(BGE)
    fresh_calls = count_encode_calls(fresh, monkeypatch)
    assert np.allclose(fresh.encode(["alpha", "beta", "gamma"])[:2], first)
    assert fresh_calls == []
    assert list(home.glob("cache/emb/*.npz"))


def test_query_and_passage_cached_separately(home, monkeypatch):
    from ragblame.embed import Embedder

    emb = Embedder(BGE)
    calls = count_encode_calls(emb, monkeypatch)
    text = "How many pulls do I get on Docker Hub?"
    p = emb.encode([text])
    q = emb.encode([text], query=True)
    assert len(calls) == 2                      # the query version is a cache miss
    assert not np.allclose(p, q)                # the query prompt changes the vector
    emb.encode([text], query=True)
    emb.encode([text])
    assert len(calls) == 2
