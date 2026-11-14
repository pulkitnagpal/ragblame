import copy
import json
import re
from pathlib import Path

import pytest

from ragblame.report import render

FIXTURE = Path(__file__).parent / "fixtures" / "results_sample.json"


@pytest.fixture
def sample():
    return json.loads(FIXTURE.read_text())


def _chunk(doc_id, n, start, end, text):
    return {"id": f"{doc_id}#{n}", "doc_id": doc_id, "start": start, "end": end, "text": text}


def _minimal(entries, **extra):
    from ragblame.results import summarize
    return {"id": "t1", "created": "2026-11-14T10:00:00", "config": {}, "questions": entries,
            "summary": summarize(entries), **extra}


def _entry(qid, cause, *, quote="the answer", text="q?", answer="bot", trace=None, gold=True):
    g = {"doc_id": "d", "start": 0, "end": len(quote), "quote": quote} if gold else None
    return {"question": {"id": qid, "text": text, "answer": "ref", "gold": g},
            "trace": trace, "judgement": {"label": "wrong", "reason": "r"},
            "verdict": {"cause": cause, "stages": [cause], "loss_point": None, "experiments": [],
                        "explanation": "e", "fix": "f"},
            "error": None}


def _metric(html, name):
    m = re.search(rf'data-metric="{name}"[^>]*>\s*([-+]?\d+)', html)
    assert m, f"metric {name} missing"
    return int(m.group(1))


def test_render_counts_match_summary(sample):
    html = render(sample)
    s = sample["summary"]
    assert _metric(html, "total") == s["total"]
    for key in ("correct", "partial", "wrong"):
        assert _metric(html, key) == s[key]
    fs = sample["fix_simulation"]
    assert _metric(html, "after_fix") == fs["after"]["correct"]
    assert _metric(html, "after_fix_delta") == fs["after"]["correct"] - fs["before"]["correct"]
    n_failures = sum(s["causes"].values())
    assert f"Why the {n_failures} failures happened" in html
    assert html.count('class="failure-card"') == n_failures


def test_cause_bar_lists_every_cause_present(sample):
    html = render(sample)
    bar = re.search(r'<div class="cause-bar".*?</div>', html, re.S).group(0)   # segments are spans
    legend = re.search(r'<ul class="cause-legend".*?</ul>', html, re.S).group(0)
    present = sample["summary"]["causes"]
    for cause, n in present.items():
        assert f'data-cause="{cause}"' in bar
        assert re.search(rf'data-cause="{cause}".*?{n}', legend, re.S)
    assert "the answer was cut across two chunks" in legend
    assert "the docs don&#x27;t contain the answer" in legend
    for absent in {"distracted", "unresolved"} - set(present):
        assert f'data-cause="{absent}"' not in bar + legend


def test_split_chunk_view_only_for_chunking(sample):
    html = render(sample)
    n_chunking = sum(1 for e in sample["questions"] if (e["verdict"] or {}).get("cause") == "chunking")
    assert html.count('class="split-view"') == n_chunking == 2

    # q02's two real chunks are shown, and their highlighted parts rebuild the gold quote.
    q02 = next(e for e in sample["questions"] if e["question"]["id"] == "q02")
    card = re.search(r'<article class="failure-card" id="q02".*?</article>', html, re.S).group(0)
    marks = re.findall(r"<mark[^>]*>(.*?)</mark>", card, re.S)
    assert len(marks) == 2
    joined = " ".join(m.strip() for m in marks)
    quote = q02["question"]["gold"]["quote"].replace("&", "&amp;")
    assert set(joined.split()) == set(quote.split())

    # q12's chunks aren't in the trace, so it falls back to splitting the quote.
    card12 = re.search(r'<article class="failure-card" id="q12".*?</article>', html, re.S).group(0)
    assert 'class="split-view"' in card12 and "not in the trace" in card12

    only_ranking = _minimal([_entry("r1", "ranking")])
    assert 'class="split-view"' not in render(only_ranking)


def test_html_escapes_doc_text():
    evil = "<script>alert(1)</script>"
    text = "Intro words " + evil + " and more"
    gs, ge = text.index(evil), text.index(evil) + len(evil)
    c0 = _chunk("d", 0, 0, gs + 10, "Title\n\n" + text[:gs + 10])
    c1 = _chunk("d", 1, gs + 10, len(text), text[gs + 10:])
    trace = {"question_id": "x1", "exact": [{"chunk": c0, "score": 0.9}, {"chunk": c1, "score": 0.8}],
             "candidates": [], "reranked": [], "packed": [c0], "prompt": "p", "answer": "<img src=x onerror=alert(2)>"}
    e = _entry("x1", "chunking", quote=evil, text="Is <b>this</b> safe?", answer="a", trace=trace)
    e["question"]["gold"].update(start=gs, end=ge)
    e["verdict"]["explanation"] = "<i>explained</i>"
    html = render(_minimal([e], docs={"d": {"title": "<t>", "url": "javascript:alert(3)"}}))
    assert evil not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<b>this</b>" not in html and "&lt;b&gt;this&lt;/b&gt;" in html
    assert "<img src=x" not in html
    assert "<i>explained</i>" not in html
    assert "javascript:" not in html
    assert "<script" not in html.lower()


def test_fix_card_only_when_fix_simulation_present(sample):
    html = render(sample)
    assert html.count('data-section="fix"') == 1
    fs = sample["fix_simulation"]
    fix = re.search(r'<section[^>]*data-section="fix".*?</section>', html, re.S).group(0)
    assert fs["fix"] in fix
    assert f"Fixed {len(fs['fixed_ids'])}" in fix and f"broke {len(fs['broken_ids'])}" in fix

    without = copy.deepcopy(sample)
    del without["fix_simulation"]
    html2 = render(without)
    assert 'data-section="fix"' not in html2
    assert 'data-metric="after_fix"' not in html2


def test_calibration_disagreements_listed(sample):
    html = render(sample)
    cal = sample["calibration"]
    section = re.search(r'<section[^>]*data-section="calibration".*?</section>', html, re.S).group(0)
    assert "How much to trust the judge" in section
    assert "Reference labels were made by a separate, stricter model (Opus), not by a person." in section
    assert f"{round(cal['agreement'] * 100)}%" in section and str(cal["n"]) in section
    disagree = [p for p in cal["pairs"] if p["judge"] != p["reference"]]
    assert disagree
    rows = re.findall(r'<tr data-qid="([^"]+)"', section)
    assert rows == [p["question_id"] for p in disagree]

    without = copy.deepcopy(sample)
    del without["calibration"]
    assert 'data-section="calibration"' not in render(without)


def test_page_is_self_contained(sample):
    html = render(sample)
    doc_urls = {d["url"] for d in sample["docs"].values()}
    for url in re.findall(r'(?:src|href)\s*=\s*["\']([^"\']*)', html):
        if url.startswith(("http:", "https:", "//")):
            assert url in doc_urls, url
    assert any(u in html for u in doc_urls)
    assert "<script src" not in html.lower()
    assert "<link" not in html.lower()
    assert "@import" not in html and "url(" not in html
    assert "@media (prefers-color-scheme: dark)" in html


def test_bench_arms_table_when_present(sample):
    assert 'data-section="bench"' not in render(sample)
    with_bench = copy.deepcopy(sample)
    with_bench["bench"] = {"arms": [{"name": "ragblame", "accuracy": 0.91}, {"name": "trace-reader", "accuracy": 0.64}]}
    html = render(with_bench)
    section = re.search(r'<section[^>]*data-section="bench".*?</section>', html, re.S).group(0)
    assert "Proof benchmark" in section and "trace-reader" in section and "91%" in section
    # dict form is accepted too
    with_bench["bench"] = {"arms": {"ragblame": 0.9}}
    assert "90%" in render(with_bench)


def test_failures_ordered_most_informative_first(sample):
    html = render(sample)
    order = re.findall(r'<article class="failure-card" id="[^"]+" data-cause="([^"]+)"', html)
    rank = ["chunking", "index", "ranking", "packing", "embedding", "multiple", "generation", "distracted",
            "docs_gap", "unresolved"]
    assert order == sorted(order, key=rank.index)
