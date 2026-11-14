import pytest

pytestmark = pytest.mark.slow


def test_relevant_passage_scores_higher():
    from ragblame import models
    from ragblame.pipeline import Config
    from ragblame.rerank import Reranker  # noqa: F401  (registers the loader)

    reranker = models.get("reranker", Config().reranker)
    scores = reranker.score(
        "How many image pulls does a Docker Hub personal account get?",
        ["Docker Desktop runs on macOS, Windows and Linux.",
         "Personal accounts are limited to 200 pulls per 6 hour period on Docker Hub.",
         "Compose files describe multi-container applications."],
    )
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    assert scores[1] == max(scores)


def test_empty_texts_returns_empty():
    from ragblame import models
    from ragblame.pipeline import Config
    from ragblame.rerank import Reranker  # noqa: F401

    assert models.get("reranker", Config().reranker).score("anything", []) == []
