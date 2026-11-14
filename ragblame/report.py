"""Render a results file (see results.py) as one self-contained HTML page.

No external scripts, fonts, styles or images: everything is inline, and the only links are to the
source docs. Every piece of text that came from the docs, the questions or an LLM goes through
html.escape before it reaches the page.

Optional extra key read here: results["docs"] = {doc_id: {"title": ..., "url": ...}} (or {doc_id: url}),
used to link each gold quote to its page.
"""
from __future__ import annotations

from datetime import datetime
from html import escape

from .results import summarize

# Most informative first: these name a fixable retrieval stage.
CAUSE_ORDER = ("chunking", "index", "ranking", "packing", "embedding", "multiple", "generation", "distracted",
               "docs_gap", "unresolved")

CAUSE_NAMES = {
    "chunking": "Chunking", "embedding": "Embedding", "index": "Index", "ranking": "Ranking",
    "packing": "Packing", "generation": "LLM", "distracted": "Distracted", "docs_gap": "Docs gap",
    "multiple": "Multiple", "unresolved": "Unresolved",
}

CAUSE_LINES = {
    "chunking": "the answer was cut across two chunks",
    "embedding": "search couldn't find the right passage even with exact search",
    "index": "the approximate index skipped the right passage",
    "ranking": "the reranker pushed the right passage out of the top results",
    "packing": "the right passage didn't make it into the prompt, or was buried",
    "generation": "the LLM had the right passage and still answered wrong",
    "distracted": "the right passage was first, but the other passages confused the LLM",
    "docs_gap": "the docs don't contain the answer",
    "multiple": "more than one stage lost the right passage, and only fixing them together worked",
    "unresolved": "no single fix or combination of fixes made the answer right",
}

LABEL_WORDS = {"correct": "right", "partial": "partial", "wrong": "still wrong"}

CONTEXT_CHARS = 140   # chunk text shown either side of the highlighted part in the split view


def e(value) -> str:
    return escape("" if value is None else str(value), quote=True)


def _safe_url(url) -> str | None:
    if isinstance(url, str) and url.startswith(("https://", "http://")):
        return url
    return None


# ---------------------------------------------------------------- header and metrics

def _short_model(name) -> str:
    if not name:
        return "none"
    return str(name).rsplit("/", 1)[-1]


def _config_line(cfg: dict) -> str:
    if not cfg:
        return ""
    parts = []
    chunker = cfg.get("chunker")
    if chunker:
        size = f" {cfg['chunk_words']} words" if cfg.get("chunk_words") else ""
        ov = f", {cfg['overlap_words']} overlap" if cfg.get("overlap_words") else ""
        parts.append(f"{chunker} chunks{size}{ov}" if chunker == "fixed" else f"{chunker} chunks")
    if cfg.get("embedder"):
        parts.append(f"embedder {_short_model(cfg['embedder'])}")
    if cfg.get("index"):
        idx = cfg["index"]
        if idx == "hnsw" and cfg.get("ef_search"):
            idx += f" (M {cfg.get('hnsw_m', '?')}, efSearch {cfg['ef_search']})"
        parts.append(f"index {idx}")
    if "reranker" in cfg:
        parts.append(f"reranker {_short_model(cfg['reranker'])}")
    if cfg.get("generator"):
        parts.append(f"generator {_short_model(cfg['generator'])}")
    return " · ".join(e(p) for p in parts)


def _date(created) -> str:
    try:
        dt = datetime.fromisoformat(created)
        return f"{dt.day} {dt:%b %Y, %H:%M}"
    except (TypeError, ValueError):
        return str(created or "")


def _metric(label, key, value, cls="", extra="") -> str:
    return (f'<div class="metric"><div class="metric-label">{e(label)}</div>'
            f'<div class="metric-value {cls}" data-metric="{key}">{int(value)}{extra}</div></div>')


def _metrics(summary: dict, fix: dict | None) -> str:
    cards = [_metric("Questions", "total", summary.get("total", 0)),
             _metric("Correct", "correct", summary.get("correct", 0), "ok"),
             _metric("Partial", "partial", summary.get("partial", 0), "warn"),
             _metric("Wrong", "wrong", summary.get("wrong", 0), "bad")]
    if summary.get("error"):
        cards.append(_metric("Errors", "error", summary["error"], "muted"))
    if fix:
        after = fix.get("after", {}).get("correct", 0)
        delta = after - fix.get("before", {}).get("correct", summary.get("correct", 0))
        cls = "ok" if delta > 0 else "bad" if delta < 0 else "muted"
        cards.append(_metric("Correct after fix", "after_fix", after, "",
                             f' <span class="delta {cls}"><span class="sr">change </span>'
                             f'<span data-metric="after_fix_delta">{delta:+d}</span></span>'))
    return f'<div class="metrics">{"".join(cards)}</div>'


# ---------------------------------------------------------------- cause bar

def _ordered_causes(causes: dict) -> list[tuple[str, int]]:
    known = [(c, causes[c]) for c in CAUSE_ORDER if causes.get(c)]
    other = [(c, n) for c, n in causes.items() if c not in CAUSE_ORDER and n]
    return known + other


def _cause_section(causes: dict) -> str:
    items = _ordered_causes(causes)
    total = sum(n for _, n in items)
    if not total:
        return ""
    segs, legend = [], []
    for cause, n in items:
        pct = 100 * n / total
        name = CAUSE_NAMES.get(cause, cause)
        label = f"{e(name)} {n}" if pct >= 12 else (str(n) if pct >= 5 else "")
        segs.append(f'<span class="seg c-{e(cause)}" data-cause="{e(cause)}" style="width:{pct:.2f}%" '
                    f'title="{e(name)}: {n}">{label}</span>')
        legend.append(f'<li data-cause="{e(cause)}"><span class="swatch c-{e(cause)}" aria-hidden="true"></span>'
                      f'<span class="legend-name">{e(name)}</span> <span class="legend-n">{n}</span>'
                      f'<span class="legend-line">{e(CAUSE_LINES.get(cause, ""))}</span></li>')
    aria = ", ".join(f"{CAUSE_NAMES.get(c, c)} {n}" for c, n in items)
    return (f'<section class="block" data-section="causes"><h2>Why the {total} failures happened</h2>'
            f'<div class="cause-bar" role="img" aria-label="{e(aria)}">{"".join(segs)}</div>'
            f'<ul class="cause-legend">{"".join(legend)}</ul></section>')


# ---------------------------------------------------------------- fix simulation

def _counts(d: dict) -> str:
    return (f'<span class="ok">{int(d.get("correct", 0))} correct</span> · '
            f'<span class="warn">{int(d.get("partial", 0))} partial</span> · '
            f'<span class="bad">{int(d.get("wrong", 0))} wrong</span>')


def _ids(ids) -> str:
    return ", ".join(f'<a href="#{e(i)}">{e(i)}</a>' for i in ids) if ids else "none"


def _fix_section(fix: dict) -> str:
    fixed, broken = fix.get("fixed_ids") or [], fix.get("broken_ids") or []
    return (f'<section class="block card fix-card" data-section="fix">'
            f'<div class="card-head"><h2>Simulated fix: <code>{e(fix.get("fix"))}</code></h2></div>'
            f'<p class="muted">{e(fix.get("description"))}</p>'
            f'<div class="before-after"><div><div class="metric-label">Before</div>{_counts(fix.get("before", {}))}</div>'
            f'<div class="arrow" aria-hidden="true">→</div>'
            f'<div><div class="metric-label">After</div>{_counts(fix.get("after", {}))}</div></div>'
            f'<p>Fixed {len(fixed)} question{"" if len(fixed) == 1 else "s"} and broke {len(broken)}.</p>'
            f'<p class="small muted">Fixed: {_ids(fixed)}<br>Broke: {_ids(broken)}</p></section>')


# ---------------------------------------------------------------- failure cards

def _trace_chunks(trace: dict | None) -> list[dict]:
    """Every distinct chunk the trace mentions, packed first."""
    if not trace:
        return []
    seen, out = set(), []
    pools = [trace.get("packed") or []]
    pools += [[h.get("chunk") for h in trace.get(k) or []] for k in ("candidates", "exact", "reranked")]
    for pool in pools:
        for c in pool:
            if c and c.get("id") not in seen:
                seen.add(c.get("id"))
                out.append(c)
    return out


def _gold_chunks(trace: dict | None, gold: dict) -> list[dict]:
    gs, ge = gold["start"], gold["end"]
    hits = [c for c in _trace_chunks(trace)
            if c.get("doc_id") == gold["doc_id"] and c.get("start", 0) < ge and c.get("end", 0) > gs]
    return sorted(hits, key=lambda c: c["start"])


def _chunk_pane(chunk: dict, gold: dict) -> str:
    text = chunk.get("text") or ""
    span = chunk["end"] - chunk["start"]
    base = len(text) - span                    # the doc slice sits at the end of chunk.text
    if base < 0:
        base = 0
    hs = base + max(gold["start"], chunk["start"]) - chunk["start"]
    he = base + min(gold["end"], chunk["end"]) - chunk["start"]
    lo, hi = max(0, hs - CONTEXT_CHARS), min(len(text), he + CONTEXT_CHARS)
    if lo > 0:                                  # snap the context to whole words
        sp = text.find(" ", lo, hs)
        lo = sp + 1 if sp != -1 else lo
    if hi < len(text):
        sp = text.rfind(" ", he, hi)
        hi = sp if sp != -1 else hi
    before = ("…" if lo > 0 else "") + text[lo:hs]
    after = text[he:hi] + ("…" if hi < len(text) else "")
    num = str(chunk.get("id", "")).rsplit("#", 1)[-1]
    return (f'<div class="chunk"><div class="chunk-label">chunk {e(num)} '
            f'<span class="muted">· chars {int(chunk["start"])}–{int(chunk["end"])}</span></div>'
            f'<div class="chunk-text">{e(before)}<mark>{e(text[hs:he])}</mark>{e(after)}</div></div>')


def _split_view(trace: dict | None, gold: dict) -> str:
    chunks = _gold_chunks(trace, gold)
    if chunks:
        panes = "".join(_chunk_pane(c, gold) for c in chunks)
        note = "The highlighted text is the part of the answer inside each chunk."
    else:
        quote = gold.get("quote") or ""
        # No chunk in the trace: split at the nearest word boundary to the middle.
        mid = len(quote) // 2
        cut = quote.rfind(" ", 0, mid + 1)
        cut = cut if cut > 0 else mid
        halves = [quote[:cut].rstrip(), quote[cut:].lstrip()]
        panes = "".join(f'<div class="chunk"><div class="chunk-label">part {i + 1}</div>'
                        f'<div class="chunk-text"><mark>{e(h)}</mark></div></div>' for i, h in enumerate(halves))
        note = ("The original chunks were not in the trace (search never returned them), so this shows the quote "
                "split roughly where a boundary fell.")
    return (f'<div class="sub">What happened</div><div class="split-view">{panes}</div>'
            f'<p class="small muted">{note}</p>')


def _doc_ref(gold: dict, docs: dict) -> str:
    info = docs.get(gold.get("doc_id")) if isinstance(docs, dict) else None
    if isinstance(info, str):
        info = {"url": info}
    info = info or {}
    title = info.get("title") or gold.get("doc_id")
    url = _safe_url(info.get("url"))
    label = f'<code>{e(gold.get("doc_id"))}</code>'
    if url:
        return f'<a class="doc-link" href="{e(url)}" title="{e(title)}">{label}</a>'
    return label


def _experiments(exps: list[dict]) -> str:
    if not exps:
        return ""
    rows = []
    for x in exps:
        label = (x.get("judgement") or {}).get("label")
        if label is None:
            mark, cls, word, sr = "–", "muted", "skipped", "skipped"
        elif label == "correct":
            mark, cls, word, sr = "✓", "ok", LABEL_WORDS[label], "fixed"
        else:
            mark, cls, word, sr = "✗", "bad" if label == "wrong" else "warn", LABEL_WORDS.get(label, label), "not fixed"
        rows.append(f'<li class="exp"><span class="exp-mark {cls}" aria-hidden="true">{mark}</span>'
                    f'<span class="sr">{sr}: </span><span class="exp-desc">{e(x.get("description") or x.get("swap"))}'
                    f'</span><span class="exp-label {cls}">{e(word)}</span></li>')
    return f'<div class="sub">Experiments, one part swapped at a time</div><ul class="exps">{"".join(rows)}</ul>'


def _badge(verdict: dict) -> str:
    cause = verdict.get("cause", "unresolved")
    text = f"Cause: {CAUSE_NAMES.get(cause, cause).lower()}"
    if cause == "multiple" and verdict.get("stages"):
        text += " (" + " + ".join(verdict["stages"]) + ")"
    return f'<span class="badge c-{e(cause)}">{e(text)}</span>'


def _failure_card(entry: dict, i: int, n: int, docs: dict) -> str:
    q, v = entry.get("question") or {}, entry["verdict"]
    trace, gold = entry.get("trace"), q.get("gold")
    cause = v.get("cause", "unresolved")
    answer = (trace or {}).get("answer", "")
    label = (entry.get("judgement") or {}).get("label")
    judged = f' <span class="small muted">(judged {e(label)})</span>' if label else ""
    if gold:
        docs_say = f'<span class="ok">{e(gold.get("quote"))}</span> <span class="small muted">— {_doc_ref(gold, docs)}</span>'
    else:
        docs_say = '<span class="muted">Nothing. No passage in your docs answers this.</span>'
    parts = [
        f'<article class="failure-card" id="{e(q.get("id"))}" data-cause="{e(cause)}">',
        f'<div class="card-head"><span class="card-title">Failure {i} of {n} '
        f'<span class="muted small">· {e(q.get("id"))}</span></span>{_badge(v)}</div>',
        f'<dl class="qa"><dt>Question</dt><dd>{e(q.get("text"))}</dd>'
        f'<dt>Bot said</dt><dd><span class="bad">{e(answer)}</span>{judged}</dd>'
        f'<dt>Your docs say</dt><dd>{docs_say}</dd></dl>',
        _experiments(v.get("experiments") or []),
    ]
    if cause == "chunking" and gold:
        parts.append(_split_view(trace, gold))
    expl = e(v.get("explanation"))
    fix = f' <strong>Suggested fix:</strong> {e(v.get("fix"))}' if v.get("fix") else ""
    if expl or fix:
        parts.append(f'<p class="explain">{expl}{fix}</p>')
    parts.append("</article>")
    return "".join(parts)


def _failures(entries: list[dict], docs: dict) -> str:
    diagnosed = [x for x in entries if x.get("verdict")]
    if not diagnosed:
        return ""
    rank = {c: k for k, c in enumerate(CAUSE_ORDER)}
    diagnosed.sort(key=lambda x: rank.get(x["verdict"].get("cause"), len(rank)))   # stable: keeps question order
    cards = "".join(_failure_card(x, i + 1, len(diagnosed), docs) for i, x in enumerate(diagnosed))
    return f'<section class="block" data-section="failures"><h2>Each failure</h2>{cards}</section>'


# ---------------------------------------------------------------- appendices

def _excerpt(text, n=160) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _calibration(cal: dict) -> str:
    agreement = cal.get("agreement")
    pct = f"{round(agreement * 100)}%" if isinstance(agreement, (int, float)) else "n/a"
    pairs = cal.get("pairs") or []
    dis = [p for p in pairs if p.get("judge") != p.get("reference")]
    if dis:
        rows = "".join(f'<tr data-qid="{e(p.get("question_id"))}"><td>{e(p.get("question_id"))}</td>'
                       f'<td>{e(p.get("judge"))}</td><td>{e(p.get("reference"))}</td>'
                       f'<td>{e(_excerpt(p.get("answer")))}</td></tr>' for p in dis)
        table = (f'<table class="cal-table"><thead><tr><th>Question</th><th>Judge</th><th>Reference</th><th>Answer</th></tr></thead>'
                 f'<tbody>{rows}</tbody></table>')
    else:
        table = '<p class="muted">The judge agreed with every reference label.</p>'
    return (f'<section class="block card" data-section="calibration"><h2>How much to trust the judge</h2>'
            f'<p>The judge agreed with the reference label on <strong>{pct}</strong> of '
            f'<strong>{int(cal.get("n", len(pairs)))}</strong> answers. '
            f'{len(dis)} disagreement{"" if len(dis) == 1 else "s"}:</p>{table}'
            f'<p class="small muted">Reference labels were made by a separate, stricter model (Opus), '
            f'not by a person.</p></section>')


def _fmt_acc(v) -> str:
    if isinstance(v, dict):
        v = v.get("accuracy")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return e(v if v is not None else "n/a")
    return f"{round(v * 100)}%" if v <= 1 else e(v)


def _bench(bench: dict) -> str:
    arms = bench.get("arms") if isinstance(bench, dict) else None
    if isinstance(arms, dict):
        arms = [{"name": k, "accuracy": v} for k, v in arms.items()]
    if arms:
        rows = "".join(f'<tr><td>{e(a.get("name") or a.get("arm"))}</td><td>{_fmt_acc(a.get("accuracy", a))}</td></tr>'
                       for a in arms if isinstance(a, dict))
        body = f'<table><thead><tr><th>Arm</th><th>Accuracy</th></tr></thead><tbody>{rows}</tbody></table>'
    else:
        body = '<p class="muted">Benchmark results are attached to this run but not rendered yet.</p>'
    return f'<section class="block card" data-section="bench"><h2>Proof benchmark</h2>{body}</section>'


# ---------------------------------------------------------------- page

CSS = """
:root{
  --bg:#faf9f5;--surface-1:#f1efe8;--surface-2:#ffffff;--border:rgba(0,0,0,.14);
  --text:#1f1e1b;--text-2:#5f5e5a;--muted:#888780;--ok:#3b6d11;--bad:#a32d2d;--warn:#854f0b;
  --mark-bg:#faece7;--mark-fg:#712b13;--radius:8px;
  --font:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --chunking-bg:#f5c4b3;--chunking-fg:#712b13;
  --embedding-bg:#b5d4f4;--embedding-fg:#0c447c;
  --index-bg:#afa9ec;--index-fg:#26215c;
  --ranking-bg:#cecbf6;--ranking-fg:#3c3489;
  --packing-bg:#9fe1cb;--packing-fg:#085041;
  --multiple-bg:#f4c0d1;--multiple-fg:#72243e;
  --generation-bg:#d3d1c7;--generation-fg:#444441;
  --distracted-bg:#fac775;--distracted-fg:#633806;
  --docs_gap-bg:#ebe9e1;--docs_gap-fg:#444441;
  --unresolved-bg:#c0bfb8;--unresolved-fg:#2c2c2a;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#1a1a18;--surface-1:#252522;--surface-2:#20201e;--border:rgba(255,255,255,.16);
    --text:#ecebe6;--text-2:#b4b2a9;--muted:#8c8a82;--ok:#97c459;--bad:#f09595;--warn:#ef9f27;
    --mark-bg:#712b13;--mark-fg:#fae5dc;
    --chunking-bg:#8a3a1e;--chunking-fg:#fae5dc;
    --embedding-bg:#185fa5;--embedding-fg:#e6f1fb;
    --index-bg:#3c3489;--index-fg:#eeedfe;
    --ranking-bg:#534ab7;--ranking-fg:#eeedfe;
    --packing-bg:#0f6e56;--packing-fg:#e1f5ee;
    --multiple-bg:#993556;--multiple-fg:#fbeaf0;
    --generation-bg:#5f5e5a;--generation-fg:#f1efe8;
    --distracted-bg:#854f0b;--distracted-fg:#faeeda;
    --docs_gap-bg:#3d3d3a;--docs_gap-fg:#d3d1c7;
    --unresolved-bg:#2c2c2a;--unresolved-fg:#b4b2a9;
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 var(--font)}
main{max-width:860px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:24px;font-weight:600;margin:0 0 4px}
h2{font-size:16px;font-weight:600;margin:0 0 12px}
a{color:inherit;text-underline-offset:2px}
code{font-family:var(--mono);font-size:.9em}
.header{margin-bottom:28px}
.meta{color:var(--text-2);font-size:13px}
.config{color:var(--muted);font-size:13px;margin-top:6px;overflow-wrap:anywhere}
.block{margin:0 0 32px}
.card,.failure-card{background:var(--surface-2);border:.5px solid var(--border);border-radius:12px;padding:18px 20px}
.failure-card{margin-bottom:16px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:32px}
.metric{background:var(--surface-1);border-radius:var(--radius);padding:12px 14px}
.metric-label{color:var(--text-2);font-size:13px}
.metric-value{font-size:26px;font-weight:500;line-height:1.3}
.delta{font-size:14px;font-weight:500}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}.muted{color:var(--muted)}
.small{font-size:13px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.cause-bar{display:flex;height:30px;border-radius:var(--radius);overflow:hidden;margin-bottom:14px;font-size:13px}
.seg{display:flex;align-items:center;padding:0 8px;white-space:nowrap;overflow:hidden;min-width:3px}
.cause-legend{list-style:none;margin:0;padding:0;display:grid;gap:6px;font-size:14px}
.cause-legend li{display:grid;grid-template-columns:14px auto auto minmax(0,1fr);column-gap:8px;align-items:baseline}
.swatch{width:12px;height:12px;border-radius:3px;display:inline-block;align-self:center}
.legend-name{font-weight:500}.legend-n{color:var(--text-2)}
.legend-line{color:var(--text-2)}
.legend-line::before{content:"— "}
""" + "".join(f".c-{c}{{background:var(--{c}-bg);color:var(--{c}-fg)}}\n" for c in CAUSE_ORDER) + """
.c-unresolved{background:repeating-linear-gradient(135deg,var(--unresolved-bg) 0 6px,var(--docs_gap-bg) 6px 12px)}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.card-head h2{margin:0}
.card-title{font-weight:600}
.badge{padding:2px 10px;border-radius:var(--radius);font-size:13px;white-space:nowrap}
.qa{display:grid;grid-template-columns:max-content minmax(0,1fr);column-gap:14px;row-gap:6px;margin:0 0 14px}
.qa dt{color:var(--text-2)}
.qa dd{margin:0;overflow-wrap:anywhere}
.sub{color:var(--text-2);font-size:13px;margin:14px 0 6px}
.exps{list-style:none;margin:0;padding:0;display:grid;gap:6px}
.exp{display:grid;grid-template-columns:18px minmax(0,1fr) auto;column-gap:8px;align-items:baseline}
.exp-mark{font-weight:600}
.exp-label{font-size:13px;white-space:nowrap}
.split-view{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px;margin-bottom:4px}
.chunk{border:.5px solid var(--border);border-radius:var(--radius);padding:10px;background:var(--surface-1);min-width:0}
.chunk-label{color:var(--muted);font-size:12px;margin-bottom:4px}
.chunk-text{font-family:var(--mono);font-size:12.5px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}
mark{background:var(--mark-bg);color:var(--mark-fg);border-radius:2px;padding:0 1px}
.explain{margin:14px 0 0;line-height:1.6}
.before-after{display:flex;align-items:center;gap:16px;flex-wrap:wrap;background:var(--surface-1);border-radius:var(--radius);padding:12px 14px;margin:12px 0}
.arrow{font-size:20px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:.5px solid var(--border);vertical-align:top;overflow-wrap:anywhere}
th{color:var(--text-2);font-weight:500}
.cal-table th:nth-child(-n+3),.cal-table td:nth-child(-n+3){width:1%;white-space:nowrap}
footer{color:var(--muted);font-size:12px;margin-top:40px}
@media (max-width:520px){
  main{padding-top:20px}
  .card,.failure-card{padding:14px}
  .qa{grid-template-columns:minmax(0,1fr)}
  .qa dt{margin-top:4px}
  .cause-legend li{grid-template-columns:14px auto minmax(0,1fr)}
  .legend-line{grid-column:2/-1}
  .legend-line::before{content:""}
  th,td{padding:6px 4px}
}
"""


def render(results: dict) -> str:
    entries = results.get("questions") or []
    summary = results.get("summary") or summarize(entries)
    fix = results.get("fix_simulation")
    docs = results.get("docs") or {}
    run_id = results.get("id", "")
    cfg = _config_line(results.get("config") or {})
    body = [
        f'<header class="header"><h1>ragblame report</h1>'
        f'<div class="meta">Run <code>{e(run_id)}</code> · {e(_date(results.get("created")))}</div>'
        + (f'<div class="config">{cfg}</div>' if cfg else "") + "</header>",
        _metrics(summary, fix),
        _cause_section(summary.get("causes") or {}),
    ]
    if fix:
        body.append(_fix_section(fix))
    body.append(_failures(entries, docs))
    if results.get("calibration"):
        body.append(_calibration(results["calibration"]))
    if results.get("bench"):
        body.append(_bench(results["bench"]))
    body.append('<footer>Generated by ragblame. Each verdict comes from re-running the question with one stage '
                'made perfect.</footer>')
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>ragblame report {e(run_id)}</title><style>{CSS}</style></head>"
            f"<body><main>{''.join(body)}</main></body></html>\n")
