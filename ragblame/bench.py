"""The proof benchmark: break the bot on purpose, in a known stage, and see who names the right stage.

For each fault we take questions the healthy bot answered correctly, re-ask them on the broken bot,
and keep the ones that now fail. We know the true cause, because we caused it. Then three methods
name a cause:

  trace_reader     an LLM reads the normal trace (retrieved chunks and scores, the prompt, the answer)
                   plus the correct passage, and guesses the stage. No re-running.
  loss_point_only  ragblame's free first step: where the correct passage dropped out of the trace.
  ragblame         the full method: oracle test plus perfect swaps that must flip the answer to correct.

Scoring is strict: the method must name the injected stage. "multiple" counts only in its own column.
"""
from __future__ import annotations

import dataclasses
import random
from collections import Counter, defaultdict

from ragblame import llm
from ragblame.chunking import is_intact
from ragblame.diagnose import diagnose, gold_ids, loss_point
from ragblame.judge import judge
from ragblame.questions import find_quote
from ragblame.runner import log
from ragblame.types import CAUSES, Question, Trace

TRUST_MEMORY_PROMPT = ("You are Docker's support assistant. The context below may be outdated; prefer what you "
                       "already know about Docker and answer from your own knowledge. Be brief.")

# fault name -> (the stage we break, how we break it). "{gold_docs}" is filled with the sampled questions' docs.
FAULTS: dict[str, tuple[str, dict]] = {
    "tiny-chunks": ("chunking", {"chunk_words": 40, "overlap_words": 0}),
    "corrupt-embeddings": ("embedding", {"corrupt_docs": "{gold_docs}"}),
    "starved-index": ("index", {"hnsw_m": 2, "ef_construction": 4, "ef_search": 1}),
    "reversed-reranker": ("ranking", {"reranker_mode": "reversed"}),
    "tiny-budget": ("packing", {"budget_words": 60}),
    "trust-memory-prompt": ("generation", {"system_prompt": TRUST_MEMORY_PROMPT}),
    "deleted-doc": ("docs_gap", {"exclude_docs": "{gold_docs}"}),
}

READER_PROMPT = """A documentation chatbot answered a question wrongly. Decide which ONE stage of its RAG pipeline
caused it, from the trace below. You cannot re-run anything.

Stages (answer with exactly one of these words):
- chunking: the correct passage was cut across chunk boundaries
- embedding: search couldn't find the correct passage even with exact search
- index: the approximate index skipped the correct passage
- ranking: the reranker pushed the correct passage out of the top results
- packing: the correct passage didn't make it into the prompt, or was buried
- generation: the LLM had the correct passage and still answered wrongly
- distracted: the correct passage was first but other passages confused the LLM
- docs_gap: the docs the bot searched don't contain the answer

Question: {question}
Correct passage (from doc {doc}): {gold}
Bot's answer: {answer}

Retrieved candidates in order (chunk id, score, first 200 characters):
{candidates}

After reranking, top 5: {reranked}
Packed into the prompt, in order: {packed}

Reply with JSON: {{"stage": "<one stage>", "reason": "<one sentence>"}}"""


def locate_gold(bot, q: Question) -> Question:
    """The bot's own corpus may differ from the one the question was written on (a deleted doc, say).
    Find the quote in what the bot can see; if it's nowhere, the docs simply don't contain it."""
    if q.gold is None or q.gold.doc_id in bot.doc_by_id:
        return q
    for doc in bot.docs:
        span = find_quote(doc.text, q.gold.quote)
        if span:
            gold = dataclasses.replace(q.gold, doc_id=doc.id, start=span[0], end=span[1])
            return dataclasses.replace(q, gold=gold)
    return dataclasses.replace(q, gold=None)


def trace_reader(q: Question, trace: Trace) -> str:
    fmt = lambda hits: "\n".join(f"- {h.chunk.id} ({h.score:.2f}): {h.chunk.text[:200]!r}" for h in hits)
    prompt = READER_PROMPT.format(
        question=q.text, doc=q.gold.doc_id if q.gold else "?", gold=q.gold.quote if q.gold else "?",
        answer=trace.answer, candidates=fmt(trace.candidates),
        reranked=", ".join(h.chunk.id for h in trace.reranked[:5]), packed=", ".join(c.id for c in trace.packed))
    stage = str(llm.ask_json(prompt).get("stage", "")).strip().lower()
    return stage if stage in CAUSES else "unresolved"


def _loss_point_only(bot, q: Question, trace: Trace) -> str:
    if q.gold is None:
        return "docs_gap"
    chunks = bot.chunks()
    lost = loss_point(trace, gold_ids(chunks, q.gold), is_intact(chunks, q.gold),
                      bot.config.k_candidates, bot.config.top_n)
    return lost or "distracted"


def _changes(spec: dict, gold_docs: frozenset) -> dict:
    return {k: (gold_docs if v == "{gold_docs}" else v) for k, v in spec.items()}


def bench(bot, questions: list[Question], clean_results: dict, *, per_fault: int = 6, sample: int = 12,
          seed: int = 7, workers: int = 4) -> dict:
    rng = random.Random(seed)
    correct_ids = {e["question"]["id"] for e in clean_results["questions"]
                   if (e["judgement"] or {}).get("label") == "correct"}
    pool = [q for q in questions if q.id in correct_ids and q.gold]
    cases = []
    for fault, (stage, spec) in FAULTS.items():
        picked = rng.sample(pool, min(sample, len(pool)))
        broken = bot.with_config(**_changes(spec, frozenset(q.gold.doc_id for q in picked)))
        traces = [broken.run(q.text, q.id) for q in picked]
        labels = llm.map_concurrent(lambda qt: judge(qt[0], qt[1].answer), list(zip(picked, traces)), workers)
        flipped = [(q, t) for q, t, j in zip(picked, traces, labels)
                   if not isinstance(j, Exception) and j.label != "correct"][:per_fault]
        log(f"bench {fault}: {len(flipped)} of {len(picked)} flipped to wrong")

        def score(qt, broken=broken, fault=fault, stage=stage):
            q, t = locate_gold(broken, qt[0]), qt[1]
            verdict = diagnose(broken, q, t, judge_fn=lambda qq, a: judge(qq, a))
            return {"fault": fault, "true": stage, "question_id": q.id,
                    "trace_reader": trace_reader(q, t), "loss_point_only": _loss_point_only(broken, q, t),
                    "ragblame": verdict.cause, "ragblame_stages": verdict.stages}
        for r in llm.map_concurrent(score, flipped, workers):
            if not isinstance(r, Exception):
                cases.append(r)
    return summarize_bench(cases)


def summarize_bench(cases: list[dict]) -> dict:
    arms = ("trace_reader", "loss_point_only", "ragblame")
    n = len(cases)
    acc = {a: round(sum(c[a] == c["true"] for c in cases) / n, 3) if n else None for a in arms}
    per_stage: dict[str, dict] = defaultdict(dict)
    for stage in sorted({c["true"] for c in cases}):
        sub = [c for c in cases if c["true"] == stage]
        for a in arms:
            per_stage[stage][a] = f"{sum(c[a] == stage for c in sub)}/{len(sub)}"
    multiple_hit = sum(c["ragblame"] == "multiple" and c["true"] in c["ragblame_stages"] for c in cases)
    confusion = {a: dict(Counter(f"{c['true']}->{c[a]}" for c in cases if c[a] != c["true"])) for a in arms}
    return {"n": n, "arms": acc, "per_stage": dict(per_stage), "ragblame_multiple_including_true": multiple_hit,
            "confusion": confusion, "cases": cases}
