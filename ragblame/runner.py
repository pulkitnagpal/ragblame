"""Runs the whole loop: the bot answers every question, Claude judges, ragblame diagnoses the failures.

Local models run one at a time (they share MODEL_LOCK); Claude calls overlap across questions, which
is where nearly all the waiting is.
"""
from __future__ import annotations

import dataclasses
import random
import time
from datetime import datetime
from pathlib import Path

from ragblame import llm
from ragblame.diagnose import diagnose, explain, suggested_fix
from ragblame.fixes import FIXES
from ragblame.judge import agreement, judge, reference_label
from ragblame.results import save, summarize
from ragblame.types import Judgement, Question, Trace

KEEP_EXACT = 20   # results keep the top of the exact-search instrumentation, not all 50 hits


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _trim(trace: Trace) -> dict:
    d = dataclasses.asdict(trace)
    d["exact"] = d["exact"][:KEEP_EXACT]
    return d


def answer_all(bot, questions: list[Question]) -> list[Trace]:
    """The bot answers every question (local models, one at a time)."""
    traces = []
    for i, q in enumerate(questions, 1):
        traces.append(bot.run(q.text, q.id))
        if i % 10 == 0 or i == len(questions):
            log(f"answered {i}/{len(questions)}")
    return traces


def judge_all(questions: list[Question], traces: list[Trace], workers: int = 4) -> list:
    """Claude marks each answer correct, partial or wrong (calls overlap)."""
    return llm.map_concurrent(lambda qt: judge(qt[0], qt[1].answer), list(zip(questions, traces)), workers)


def _diagnose_one(bot, q: Question, trace: Trace):
    def judge_fn(question, answer):
        return judge(question, answer)
    verdict = diagnose(bot, q, trace, judge_fn=judge_fn)
    try:
        verdict.explanation, verdict.fix = explain(q, trace, verdict)
    except llm.LLMError as err:   # the verdict stands even if the explanation call fails
        verdict.explanation, verdict.fix = f"(explanation unavailable: {err})", suggested_fix(verdict) or ""
    return verdict


def run(bot, questions: list[Question], *, run_id: str | None = None, diagnose_failures: bool = True,
        workers: int = 4, out_root: Path = Path("runs")) -> dict:
    t0, calls0 = time.time(), llm.calls_made()
    run_id = run_id or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log(f"run {run_id}: {len(questions)} questions")
    traces = answer_all(bot, questions)
    judgements = judge_all(questions, traces, workers)
    entries = []
    for q, tr, j in zip(questions, traces, judgements):
        err = str(j) if isinstance(j, Exception) else None
        entries.append({"question": dataclasses.asdict(q), "trace": _trim(tr),
                        "judgement": None if err else dataclasses.asdict(j), "verdict": None, "error": err})
    log("judged: " + str(summarize(entries)))

    if diagnose_failures:
        failed = [i for i, e in enumerate(entries) if e["judgement"] and e["judgement"]["label"] != "correct"]
        log(f"diagnosing {len(failed)} failures")
        verdicts = llm.map_concurrent(lambda i: _diagnose_one(bot, questions[i], traces[i]), failed, workers)
        for i, v in zip(failed, verdicts):
            if isinstance(v, Exception):
                entries[i]["error"] = f"diagnosis failed: {v}"
            else:
                entries[i]["verdict"] = dataclasses.asdict(v)

    results = {
        "id": run_id, "created": datetime.now().isoformat(timespec="seconds"),
        "config": {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in dataclasses.asdict(bot.config).items()},
        "docs": {d.id: {"title": d.title, "url": d.url} for d in bot.all_docs},
        "questions": entries, "summary": summarize(entries),
        "stats": {"seconds": round(time.time() - t0), "claude_calls": llm.calls_made() - calls0},
    }
    save(results, out_root / run_id / "results.json")
    log(f"done in {results['stats']['seconds']}s with {results['stats']['claude_calls']} Claude calls: {results['summary']}")
    return results


def calibrate(results: dict, questions: list[Question], n: int = 20, seed: int = 7, workers: int = 4) -> dict:
    """Check the judge: a stricter model (Opus) re-marks n answers, stratified by the judge's label."""
    by_id = {q.id: q for q in questions}
    rng = random.Random(seed)
    pool = [e for e in results["questions"] if e["judgement"]]
    by_label: dict[str, list] = {}
    for e in pool:
        by_label.setdefault(e["judgement"]["label"], []).append(e)
    picked = []
    while len(picked) < min(n, len(pool)):
        for label in sorted(by_label):
            if by_label[label] and len(picked) < n:
                picked.append(by_label[label].pop(rng.randrange(len(by_label[label]))))
    refs = llm.map_concurrent(lambda e: reference_label(by_id[e["question"]["id"]], e["trace"]["answer"]), picked, workers)
    pairs = [(e, r) for e, r in zip(picked, refs) if not isinstance(r, Exception)]
    a = [Judgement(**e["judgement"]) for e, _ in pairs]
    b = [r for _, r in pairs]
    results["calibration"] = {
        "n": len(pairs), "agreement": round(agreement(a, b), 3),
        "pairs": [{"question_id": e["question"]["id"], "judge": e["judgement"]["label"], "reference": r.label,
                   "answer": e["trace"]["answer"], "reference_reason": r.reason} for e, r in pairs],
    }
    return results


def simulate_fix(bot, questions: list[Question], results: dict, fix_name: str, workers: int = 4) -> dict:
    """Apply one realistic fix to the whole bot, re-answer and re-judge everything, compare."""
    fix = FIXES[fix_name]
    fixed_bot = bot.with_config(**fix["changes"])
    log(f"fix {fix_name}: re-answering {len(questions)} questions")
    traces = answer_all(fixed_bot, questions)
    judged = judge_all(questions, traces, workers)
    before = {e["question"]["id"]: (e["judgement"] or {}).get("label") for e in results["questions"]}
    after = {q.id: (None if isinstance(j, Exception) else j.label) for q, j in zip(questions, judged)}
    count = lambda labels: {k: sum(1 for v in labels.values() if v == k) for k in ("correct", "partial", "wrong")}
    results["fix_simulation"] = {
        "fix": fix_name, "description": fix["description"],
        "before": count(before), "after": count(after),
        "fixed_ids": sorted(i for i in after if after[i] == "correct" and before.get(i) != "correct"),
        "broken_ids": sorted(i for i in after if before.get(i) == "correct" and after[i] != "correct"),
    }
    log(f"fix {fix_name}: {results['fix_simulation']['before']} -> {results['fix_simulation']['after']}")
    return results
