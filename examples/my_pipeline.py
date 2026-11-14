"""A minimal pipeline of your own, wired to ragblame: keyword search (BM25) and any LLM you like.

ragblame needs to call each stage on its own, so a pipeline exposes them as methods. This one has no
rechunk() hook, so ragblame skips the chunking experiment for it and says so in the verdict.

    from ragblame.diagnose import diagnose
    from ragblame.judge import judge
    verdict = diagnose(bot, question, bot.run(question.text, question.id), judge_fn=judge)
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from ragblame.chunking import fixed_chunks
from ragblame.types import Hit, Trace


class KeywordBot:
    def __init__(self, docs, generate, top_n: int = 3):
        self._chunks = fixed_chunks(docs, words=150, overlap=15)
        self._bm25 = BM25Okapi([c.text.lower().split() for c in self._chunks])
        self.generate = generate          # any function (question, list_of_passages) -> answer
        self.top_n = top_n

    def chunks(self):
        return list(self._chunks)

    def search(self, query, k, *, exact=False):
        # BM25 already scores every chunk, so "exact" search is the same search.
        scores = self._bm25.get_scores(query.lower().split())
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [Hit(self._chunks[i], float(scores[i])) for i in best]

    def rerank(self, query, hits):
        return list(hits)                 # no reranker: keep the search order

    def pack(self, hits):
        return [h.chunk for h in hits[: self.top_n]]

    def answer(self, query, chunks):
        return self.generate(query, [c.text for c in chunks])

    def run(self, query, question_id=""):
        exact = self.search(query, 50, exact=True)
        candidates = self.search(query, 20)
        reranked = self.rerank(query, candidates)
        packed = self.pack(reranked)
        return Trace(question_id, exact, candidates, reranked, packed, query, self.answer(query, packed))
