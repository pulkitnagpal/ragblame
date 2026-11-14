"""Tests for the diagnosis engine, using a scripted fake pipeline instead of real models.

The fake bot answers "RIGHT" only when an answer-bearing chunk is packed first (or when it is
handed the gold passage directly, the oracle), and "WRONG" otherwise. That makes every swap's
effect easy to predict by hand.
"""
import pytest

from ragblame.diagnose import diagnose, explain, loss_point
from ragblame.pipeline import Config
from ragblame.types import Chunk, Gold, Hit, Judgement, Question, Trace, Verdict

DOC = "d"
CHUNKS = [Chunk(f"{DOC}#{i}", DOC, i * 100, (i + 1) * 100, f"text {i}") for i in range(10)]
INTACT = Gold(DOC, 310, 360, "the answer")    # sits inside d#3
SPLIT = Gold(DOC, 350, 450, "the answer")     # straddles d#3 and d#4
# After re-chunking, one chunk holds the whole split span.
WHOLE = Chunk(f"{DOC}#3b", DOC, 300, 500, "text 3 and 4")
RECHUNKED = CHUNKS[:3] + [WHOLE] + CHUNKS[5:]


def cid(i):
    return i if isinstance(i, str) else f"{DOC}#{i}"


def fake_judge(question, answer):
    label = {"RIGHT": "correct", "PART": "partial"}.get(answer, "wrong")
    return Judgement(label, f"fake judge saw {answer!r}")


class FakePipeline:
    """A pipeline whose every stage is scripted.

    ann / exact: the chunk order the approximate index / exact search return.
    right_ids:   chunks that answer the question when packed first.
    reranker:    "good" puts answer-bearing chunks first; "bad" pushes them to the end.
    bury:        the packer moves answer-bearing chunks to the end of the prompt.
    right / oracle: what the bot says with a good chunk first / with only the gold passage.
    """

    config = Config(k_candidates=4, top_n=2)

    def __init__(self, *, ann, exact, chunks=CHUNKS, right_ids=("d#3",), reranker="good",
                 bury=False, right="RIGHT", oracle="RIGHT"):
        self._chunks = list(chunks)
        self._by_id = {c.id: c for c in chunks}
        self.ann, self.exact = [cid(i) for i in ann], [cid(i) for i in exact]
        self.right_ids, self.reranker, self.bury = set(right_ids), reranker, bury
        self.right, self.oracle = right, oracle
        self.log = []

    def _hits(self, ids):
        return [Hit(self._by_id[i], 1.0 - n / 100) for n, i in enumerate(ids)]

    def chunks(self):
        return list(self._chunks)

    def search(self, query, k, *, exact=False):
        self.log.append(("search", exact))
        return self._hits(self.exact if exact else self.ann)[:k]

    def rerank(self, query, hits):
        self.log.append(("rerank",))
        good = [h for h in hits if h.chunk.id in self.right_ids]
        rest = [h for h in hits if h.chunk.id not in self.right_ids]
        return rest + good if self.reranker == "bad" else good + rest

    def pack(self, hits):
        self.log.append(("pack",))
        packed = [h.chunk for h in hits[: self.config.top_n]]
        if self.bury:
            packed = ([c for c in packed if c.id not in self.right_ids]
                      + [c for c in packed if c.id in self.right_ids])
        return packed

    def answer(self, query, chunks):
        self.log.append(("answer", [c.id for c in chunks]))
        if chunks and chunks[0].id == "gold":
            return self.oracle
        if chunks and chunks[0].id in self.right_ids:
            return self.right
        return "WRONG"

    def run(self, query, question_id=""):
        self.log.append(("run",))
        exact = self.search(query, 50, exact=True)
        candidates = self.search(query, self.config.k_candidates)
        reranked = self.rerank(query, candidates)
        packed = self.pack(reranked)
        return Trace(question_id, exact, candidates, reranked, packed, "prompt", self.answer(query, packed))


class RechunkPipeline(FakePipeline):
    """A fake pipeline that also has the optional rechunk hook."""

    def __init__(self, *, fixed=None, **kw):
        super().__init__(**kw)
        self.fixed = fixed
        self.rechunk_calls = []

    def rechunk(self, gold):
        self.rechunk_calls.append(gold)
        return self.fixed


def question(gold=INTACT):
    return Question("q1", "How many pulls?", "200", gold)


def run_diagnosis(pipeline, gold=INTACT):
    q = question(gold)
    trace = pipeline.run(q.text, q.id)
    pipeline.log.clear()
    return diagnose(pipeline, q, trace, judge_fn=fake_judge)


def swaps(verdict):
    return [e.swap for e in verdict.experiments]


# Scenario: the approximate index misses d#3 but exact search finds it.
INDEX_MISS = dict(ann=[0, 1, 2, 4, 5], exact=[3, 0, 1, 2, 4])


# ---- loss_point: reading the trace, no re-runs -------------------------------------------------

def make_trace(exact, candidates, reranked, packed):
    hits = lambda ids: [Hit(CHUNKS[i], 0.5) for i in ids]
    return Trace("q1", hits(exact), hits(candidates), hits(reranked), [CHUNKS[i] for i in packed], "", "")


@pytest.mark.parametrize("trace, intact, expected", [
    (make_trace([3], [3], [3], [3]), False, "chunking"),
    (make_trace([0, 1, 2, 4, 3], [3], [3], [3]), True, "embedding"),   # gold is 5th, k is 4
    (make_trace([3, 0], [0, 1], [0, 1], [0]), True, "index"),
    (make_trace([3, 0], [0, 3], [0, 1, 3], [0]), True, "ranking"),     # gold is 3rd, top_n is 2
    (make_trace([3, 0], [3, 0], [3, 0], [0]), True, "packing"),        # gold never packed
    (make_trace([3, 0], [3, 0], [3, 0], [0, 3]), True, "packing"),     # gold packed, not first
    (make_trace([3, 0], [3, 0], [3, 0], [3, 0]), True, None),
])
def test_loss_point_each_stage(trace, intact, expected):
    assert loss_point(trace, {"d#3"}, intact, k_candidates=4, top_n=2) == expected


# ---- diagnose: the verdicts ---------------------------------------------------------------------

def test_docs_gap_when_no_gold():
    bot = FakePipeline(**INDEX_MISS)
    v = run_diagnosis(bot, gold=None)
    assert v.cause == "docs_gap"
    assert v.experiments == [] and v.loss_point is None
    assert bot.log == []   # nothing was re-run


def test_generation_when_oracle_fails():
    bot = FakePipeline(**INDEX_MISS, oracle="WRONG")
    v = run_diagnosis(bot)
    assert v.cause == "generation"
    assert swaps(v) == ["oracle"]
    assert bot.log == [("answer", ["gold"])]   # the generator saw only the gold passage


def test_single_swap_verdict_index():
    v = run_diagnosis(FakePipeline(**INDEX_MISS))
    assert (v.cause, v.stages, v.loss_point) == ("index", ["index"], "index")
    assert swaps(v) == ["oracle", "index"]
    assert v.experiments[1].judgement.label == "correct"


@pytest.mark.parametrize("bot, stage", [
    (FakePipeline(ann=[0, 1, 2, 4, 5, 3], exact=[0, 1, 2, 4, 5, 3]), "embedding"),
    (FakePipeline(ann=[3, 0, 1, 2], exact=[3, 0, 1, 2], reranker="bad"), "ranking"),
    (FakePipeline(ann=[3, 0, 1, 2], exact=[3, 0, 1, 2], bury=True), "packing"),
])
def test_single_swap_verdict_other_stages(bot, stage):
    v = run_diagnosis(bot)
    assert (v.cause, v.stages, v.loss_point) == (stage, [stage], stage)
    assert swaps(v) == ["oracle", stage]


def test_single_swap_verdict_chunking_uses_rechunk():
    fixed = FakePipeline(ann=["d#3b", 0, 1], exact=["d#3b", 0, 1], chunks=RECHUNKED, right_ids=("d#3b",))
    bot = RechunkPipeline(fixed=fixed, ann=[3, 4, 0], exact=[3, 4, 0], right_ids=())
    v = run_diagnosis(bot, gold=SPLIT)
    assert (v.cause, v.stages, v.loss_point) == ("chunking", ["chunking"], "chunking")
    assert bot.rechunk_calls == [SPLIT]
    assert ("run",) in fixed.log   # the re-chunked copy ran the whole pipeline


def test_multiple_when_only_combination_fixes():
    # Exact search finds d#3, but the reranker then buries it: neither swap alone is enough.
    v = run_diagnosis(FakePipeline(**INDEX_MISS, reranker="bad"))
    assert (v.cause, v.stages, v.loss_point) == ("multiple", ["index", "ranking"], "index")
    assert [e.judgement.label for e in v.experiments] == ["correct", "wrong", "correct"]


def test_distracted_when_gold_packed_first_but_wrong():
    v = run_diagnosis(FakePipeline(ann=[3, 0, 1], exact=[3, 0, 1], right="WRONG"))
    assert (v.cause, v.loss_point) == ("distracted", None)
    assert swaps(v) == ["oracle", "packing"]
    assert v.experiments[0].judgement.label == "correct"   # the evidence: gold alone works


def test_partial_does_not_count_as_fix():
    v = run_diagnosis(FakePipeline(**INDEX_MISS, right="PART"))
    assert v.cause == "unresolved"
    assert swaps(v) == ["oracle", "index", "index+ranking", "index+ranking+packing"]
    assert {e.judgement.label for e in v.experiments[1:]} == {"partial"}


def test_missing_rechunk_hook_recorded():
    v = run_diagnosis(FakePipeline(ann=[3, 4, 0], exact=[3, 4, 0], right_ids=()), gold=SPLIT)
    skipped = v.experiments[1]
    assert skipped.swap == "chunking"
    assert skipped.description == "skipped: this pipeline has no rechunk hook"
    assert skipped.judgement == Judgement("wrong", "not run")
    # The downstream swaps still run, on the split pieces, and here none of them helps.
    assert v.cause == "unresolved" and v.loss_point == "chunking"
    assert swaps(v)[2:] == ["embedding", "embedding+index", "embedding+index+ranking",
                            "embedding+index+ranking+packing"]


def test_index_swap_never_calls_rechunk():
    bot = RechunkPipeline(fixed=None, **INDEX_MISS)
    v = run_diagnosis(bot)
    assert v.cause == "index"
    assert bot.rechunk_calls == []
    assert ("search", True) in bot.log and ("search", False) not in bot.log


def test_experiments_recorded_in_order():
    v = run_diagnosis(FakePipeline(**INDEX_MISS, reranker="bad"))
    assert swaps(v) == ["oracle", "index", "index+ranking"]
    assert v.experiments[1].description == "Exact search instead of the approximate index"
    assert all(e.description and e.answer for e in v.experiments)


# ---- explain --------------------------------------------------------------------------------------

def test_explain_names_the_fix():
    calls = []

    def ask(prompt, *, model=None):
        calls.append((prompt, model))
        return {"explanation": "The index lost it.", "fix": "Try ef-64."}

    q = question()
    trace = FakePipeline(**INDEX_MISS).run(q.text, q.id)
    got = explain(q, trace, Verdict("multiple", ["index", "ranking"], "index"), ask_json=ask)
    assert got == ("The index lost it.", "Try ef-64.")
    prompt, model = calls[0]
    assert model == "sonnet" and "ef-64" in prompt and "multiple" in prompt


def test_packing_swap_keeps_whole_gold_chunk_when_budget_truncates():
    """A perfect packing stage gets the whole gold passage into the prompt, even when the
    configured budget would cut it (found by the benchmark's tiny-budget fault)."""
    from ragblame.diagnose import _pack_gold_first
    from ragblame.types import Chunk, Hit

    gold = Chunk("d#1", "d", 0, 40, "the answer is sixty days for annual plans")
    other = Chunk("d#2", "d", 41, 60, "unrelated text here")

    class TinyPacker:
        def pack(self, hits):   # a 3-word budget: truncates whatever comes first
            first = hits[0].chunk
            return [Chunk(first.id, first.doc_id, first.start, first.end, " ".join(first.text.split()[:3]))]

    packed = _pack_gold_first(TinyPacker(), [Hit(other, 0.9), Hit(gold, 0.8)], {"d#1"}, top_n=5)
    assert packed[0].text == gold.text
