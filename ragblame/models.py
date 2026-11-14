"""One place that loads the local models, once per process, and lets tests swap in fakes.

    embedder = get("embedder", "BAAI/bge-small-en-v1.5")
    override("embedder", "BAAI/bge-small-en-v1.5", FakeEmbedder())   # in tests
"""
from __future__ import annotations

from typing import Any, Callable

_loaded: dict[tuple[str, str], Any] = {}
_overrides: dict[tuple[str, str], Any] = {}
_loaders: dict[str, Callable[[str], Any]] = {}


def register(kind: str, loader: Callable[[str], Any]) -> None:
    """A module that knows how to load a kind of model ("embedder", "reranker", "generator") registers here."""
    _loaders[kind] = loader


def get(kind: str, name: str) -> Any:
    key = (kind, name)
    if key in _overrides:
        return _overrides[key]
    if key not in _loaded:
        if kind not in _loaders:
            raise KeyError(f"no loader registered for {kind!r}")
        _loaded[key] = _loaders[kind](name)
    return _loaded[key]


def override(kind: str, name: str, obj: Any) -> None:
    _overrides[(kind, name)] = obj


def clear_overrides() -> None:
    _overrides.clear()
