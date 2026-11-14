"""Diagnosis: find which stage of the pipeline caused a wrong answer.

The idea is a controlled experiment. We know where the true answer lives in the docs (the gold
passage), so we can re-run the question with one stage made *perfect* (a "swap") and see whether
the answer becomes correct. If fixing that one stage fixes the answer, that stage is to blame.

The steps (spec section 5):
  1. No gold passage at all?                    -> docs_gap   (the docs don't contain the answer)
  2. Hand the generator only the gold passage.  -> generation, if it still gets it wrong
  3. Read the trace: where was the gold passage lost? (the "loss point", no re-runs needed)
  4. Swap that stage for a perfect one. Fixed?  -> blame that stage
  5. Else add the later stages' swaps too.      -> multiple, if a combination fixes it
  6. Gold was packed first and still wrong?     -> distracted (the other passages confused it)
  7. Nothing helped.                            -> unresolved

Each swap is stage-local: it keeps everything upstream exactly as the trace recorded it, and
recomputes only its own stage and the stages after it, using the pipeline's own functions.
"""
from __future__ import annotations

from typing import Callable

from ragblame.chunking import gold_chunks, is_intact
from ragblame.fixes import FIX_FOR_CAUSE, FIXES
from ragblame.types import STAGES, Chunk, Experiment, Gold, Hit, Judgement, Question, Trace, Verdict

# Used when the pipeline has no `config` attribute telling us its own settings.
DEFAULT_K = 20       # how many candidates search returns
DEFAULT_TOP_N = 5    # how many reranked hits the packer may use

# Plain-English descriptions of each experiment, shown in the report.
DESCRIPTIONS = {
    "oracle": "Gave the generator only the gold passage, with the same prompt",
    "chunking": "Re-chunked the docs so the gold passage sits whole in one chunk, then re-ran everything",
    "embedding": "Added the gold chunk to the search results, as a perfect embedder would",
    "index": "Exact search instead of the approximate index",
    "ranking": "Moved the gold chunk to the top of the reranked list",
    "packing": "Packed the gold chunk first in the prompt, then the other chunks, within the same budget",
}
SKIPPED_RECHUNK = "skipped: this pipeline has no rechunk hook"


# ---- Reading the trace -----------------------------------------------------------------------

def gold_ids(chunks: list[Chunk], gold: Gold) -> set[str]:
    """Ids of the chunks that count as "the gold chunk".

    If some chunk holds the whole gold span, only chunks that hold all of it count: a neighbour
    that overlaps a few words of the span can't answer the question. If the span was split, no
    chunk holds it whole, so the pieces count (their loss is a chunking problem anyway).
    """
    if is_intact(chunks, gold):
        return {c.id for c in chunks if c.doc_id == gold.doc_id and c.start <= gold.start and c.end >= gold.end}
    return {c.id for c in gold_chunks(chunks, gold)}


def _has_gold(hits: list[Hit], ids: set[str]) -> bool:
    """True if any of these hits is a gold chunk."""
    return any(h.chunk.id in ids for h in hits)


def loss_point(trace: Trace, gold_ids: set[str], intact: bool, k_candidates: int, top_n: int) -> str | None:
    """The earliest stage where the gold passage went missing, read from the trace alone.

    Walk the stages in the order data flows through them and stop at the first one where the
    gold chunk drops out. None means it survived all the way to the top of the prompt.
    """
    if not intact:
        return "chunking"      # the answer was cut in two, so no single chunk holds it
    if not _has_gold(trace.exact[:k_candidates], gold_ids):
        return "embedding"     # even a perfect search with these vectors ranks it too low
    if not _has_gold(trace.candidates, gold_ids):
        return "index"         # exact search found it, the approximate index didn't
    if not _has_gold(trace.reranked[:top_n], gold_ids):
        return "ranking"       # it was a candidate, but the reranker pushed it out of the top n
    if not trace.packed or trace.packed[0].id not in gold_ids:
        return "packing"       # it passed the reranker but wasn't placed first in the prompt
    return None


# ---- Perfect swaps ---------------------------------------------------------------------------

def _settings(pipeline) -> tuple[int, int]:
    """The pipeline's own k (candidates) and top_n, or the defaults if it doesn't say."""
    config = getattr(pipeline, "config", None)
    return getattr(config, "k_candidates", DEFAULT_K), getattr(config, "top_n", DEFAULT_TOP_N)


def _oracle_chunk(gold: Gold) -> Chunk:
    """The gold passage wrapped as a chunk, so the generator can be handed it directly."""
    return Chunk("gold", gold.doc_id, gold.start, gold.end, gold.quote)


def _with_gold_first(chunks: list[Chunk], ids: set[str], hits: list[Hit], k: int) -> list[Hit]:
    """The gold chunks, then the given hits, without repeats, cut to k (the embedding swap)."""
    gold = [Hit(c, 1.0) for c in chunks if c.id in ids]   # 1.0: a perfect embedder's top score
    seen, out = set(), []
    for hit in gold + hits:
        if hit.chunk.id not in seen:
            seen.add(hit.chunk.id)
            out.append(hit)
    return out[:k]


def _promote_gold(reranked: list[Hit], candidates: list[Hit], ids: set[str]) -> list[Hit]:
    """The ranking swap: gold candidates first, then the reranker's own order for the rest.

    It only reorders what search returned; it can't add a chunk search never found.
    """
    in_reranked = [h for h in reranked if h.chunk.id in ids]
    dropped = [h for h in candidates if h.chunk.id in ids and h.chunk.id not in {g.chunk.id for g in in_reranked}]
    return in_reranked + dropped + [h for h in reranked if h.chunk.id not in ids]


def _pack_gold_first(pipeline, reranked: list[Hit], ids: set[str], top_n: int) -> list[Chunk]:
    """The packing swap: the whole gold chunk first in the prompt, then the others the packer chose.

    It only uses chunks the reranker passed (its top n). We hand the packer the hits with gold
    moved to the front, and if the packer still doesn't put gold first, we move it there.
    """
    passed = [h for h in reranked[:top_n] if h.chunk.id in ids]
    if not passed:
        return pipeline.pack(reranked)    # nothing gold got this far, so nothing to move
    packed = pipeline.pack(passed + [h for h in reranked if h not in passed])
    # A perfect packing stage gets the whole gold chunk in, first: the packer may have left it out,
    # moved it, or truncated it to fit the budget, and none of that should survive the swap.
    first = passed[0].chunk
    return [first] + [c for c in packed if c.id != first.id]


def _rerun(pipeline, question: Question, trace: Trace, stages: set[str], k: int, top_n: int) -> str:
    """Answer the question again with the given stages made perfect, in pipeline order.

    `pipeline` is already the re-chunked copy when "chunking" is among the stages. Stages before
    the first swapped one reuse the trace; from there on, every stage is recomputed.
    """
    q = question.text
    if stages == {"chunking"}:
        return pipeline.run(q, question.id).answer   # new chunks: run the whole pipeline again
    fresh = "chunking" in stages    # new chunks mean nothing in the old trace still applies
    chunks = pipeline.chunks()
    ids = gold_ids(chunks, question.gold)

    # Search: which candidates come back?
    if "index" in stages:
        candidates = pipeline.search(q, k, exact=True)    # exact search: no approximate-index misses
    elif fresh:
        candidates = pipeline.search(q, k)
    else:
        candidates = list(trace.candidates)
    if "embedding" in stages:
        candidates = _with_gold_first(chunks, ids, candidates, k)
    changed = fresh or bool(stages & {"embedding", "index"})

    # Ranking: which candidates come first?
    reranked = pipeline.rerank(q, candidates) if changed else list(trace.reranked)
    if "ranking" in stages:
        reranked = _promote_gold(reranked, candidates, ids)
        changed = True

    # Packing: what goes into the prompt, in what order?
    packed = pipeline.pack(reranked) if changed else list(trace.packed)
    if "packing" in stages:
        packed = _pack_gold_first(pipeline, reranked, ids, top_n)
    return pipeline.answer(q, packed)


# ---- The verdict -----------------------------------------------------------------------------

def _record(verdict: Verdict, swap: str, description: str, answer: str, question: Question,
            judge_fn: Callable[[Question, str], Judgement]) -> bool:
    """Judge one experiment's answer, add it to the verdict, and say whether it was a fix."""
    judgement = judge_fn(question, answer)
    verdict.experiments.append(Experiment(swap, description, answer, judgement))
    return judgement.label == "correct"   # partial is still a failure, so it's not a fix


def diagnose(pipeline, question: Question, trace: Trace, *,
             judge_fn: Callable[[Question, str], Judgement]) -> Verdict:
    """Find the cause of one wrong answer by re-running it with stages made perfect."""
    gold = question.gold
    if gold is None:
        return Verdict("docs_gap")

    # Reading the trace costs nothing, so we always record where the gold was lost.
    k, top_n = _settings(pipeline)
    chunks = pipeline.chunks()
    lost = loss_point(trace, gold_ids(chunks, gold), is_intact(chunks, gold), k, top_n)
    verdict = Verdict("unresolved", loss_point=lost)

    # The oracle: if the generator fails even with the perfect passage, retrieval isn't the issue.
    oracle = pipeline.answer(question.text, [_oracle_chunk(gold)])
    if not _record(verdict, "oracle", DESCRIPTIONS["oracle"], oracle, question, judge_fn):
        verdict.cause = "generation"
        return verdict

    if lost is None:
        # Gold reached the top of the prompt and the answer was still wrong. Putting gold first
        # changes nothing here; the oracle passing is the evidence that the other passages
        # distracted the generator. We still try the packing swap, so the report shows it.
        answer = _rerun(pipeline, question, trace, {"packing"}, k, top_n)
        if _record(verdict, "packing", DESCRIPTIONS["packing"], answer, question, judge_fn):
            verdict.cause, verdict.stages = "packing", ["packing"]
        else:
            verdict.cause = "distracted"
        return verdict

    # The swap ladder: the loss point's swap alone, then add each later stage's swap in turn.
    rechunked = None
    applied: list[str] = []
    for stage in STAGES[STAGES.index(lost):]:
        if stage == "chunking":
            if not hasattr(pipeline, "rechunk"):
                verdict.experiments.append(Experiment("chunking", SKIPPED_RECHUNK, "", Judgement("wrong", "not run")))
                continue
            rechunked = pipeline.rechunk(gold)
        applied.append(stage)
        answer = _rerun(rechunked or pipeline, question, trace, set(applied), k, top_n)
        description = "; plus ".join(DESCRIPTIONS[s] for s in applied)
        if _record(verdict, "+".join(applied), description, answer, question, judge_fn):
            verdict.stages = list(applied)
            verdict.cause = applied[0] if len(applied) == 1 else "multiple"
            return verdict
    return verdict   # nothing fixed it: unresolved


# ---- Explaining it ---------------------------------------------------------------------------

EXPLAIN_PROMPT = """You are explaining why a documentation chatbot answered a question wrongly.
The reader is not an expert in RAG (retrieval-augmented generation).

Question: {question}
Reference answer: {reference}
Gold passage from the docs: {gold}
Bot's answer: {answer}

Where the gold chunk went through the pipeline:
{journey}

Diagnosis: {cause}{stages}
Experiments (each re-ran the question with some stages made perfect):
{experiments}

{fix_hint}

Reply with JSON only: {{"explanation": "<two or three plain-English sentences on what happened>",
"fix": "<one line: the change to try>"}}"""


def suggested_fix(verdict: Verdict) -> str | None:
    """The name of the fix (from fixes.FIXES) to suggest for this verdict, if there is one."""
    if verdict.cause == "multiple":
        return FIX_FOR_CAUSE.get(verdict.stages[0]) if verdict.stages else None
    return FIX_FOR_CAUSE.get(verdict.cause)


def _position(hits: list[Hit], ids: set[str], name: str) -> str:
    """One line saying where (1-based) the first gold chunk sat in a list of hits."""
    for n, hit in enumerate(hits, 1):
        if hit.chunk.id in ids:
            return f"- {name}: rank {n} of {len(hits)}"
    return f"- {name}: not present ({len(hits)} results)"


def _journey(question: Question, trace: Trace) -> str:
    """Where the gold chunk ranked at each stage of the trace, in plain lines."""
    if question.gold is None:
        return "(the docs have no passage that answers this)"
    seen = trace.exact + trace.candidates + trace.reranked + [Hit(c, 0.0) for c in trace.packed]
    ids = gold_ids([h.chunk for h in seen], question.gold)
    return "\n".join([
        _position(trace.exact, ids, "exact search (instrumentation)"),
        _position(trace.candidates, ids, "approximate index"),
        _position(trace.reranked, ids, "after the reranker"),
        _position([Hit(c, 0.0) for c in trace.packed], ids, "packed into the prompt"),
    ])


def _fix_hint(verdict: Verdict) -> str:
    """Tell Claude which fix to name, so the suggestion matches what `ragblame fix` can run."""
    name = suggested_fix(verdict)
    if name:
        return f'For the fix, recommend "{name}": {FIXES[name]["description"]}'
    if verdict.cause == "docs_gap":
        return "For the fix, recommend adding the missing information to the docs."
    return "No tested change fixed this; for the fix, suggest reading the trace by hand."


def explain(question: Question, trace: Trace, verdict: Verdict, *, ask_json=None) -> tuple[str, str]:
    """Ask Claude for a short plain-English explanation of the verdict, and a one-line fix."""
    if ask_json is None:
        from ragblame.llm import ask_json   # loaded lazily, so tests can run without Claude
    experiments = "\n".join(f"- {e.description}: {e.judgement.label} ({e.judgement.reason})"
                            for e in verdict.experiments) or "- none"
    prompt = EXPLAIN_PROMPT.format(
        question=question.text, reference=question.answer,
        gold=question.gold.quote if question.gold else "(no supporting passage)",
        answer=trace.answer, journey=_journey(question, trace), cause=verdict.cause,
        stages=f" ({', '.join(verdict.stages)})" if verdict.stages else "",
        experiments=experiments, fix_hint=_fix_hint(verdict))
    reply = ask_json(prompt, model="sonnet")
    if "explanation" not in reply or "fix" not in reply:
        raise ValueError(f"explain reply is missing a field: {sorted(reply)}")
    return str(reply["explanation"]).strip(), str(reply["fix"]).strip()
