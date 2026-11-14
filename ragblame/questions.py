"""Write test questions from the docs with Claude, and pin each one to its gold quote.

Claude is asked for realistic support questions, a short answer and a supporting quote copied
from the page. A quote is kept only if `find_quote` locates it in the page, which tolerates
whitespace, smart quotes, dashes, markdown table pipes and bold/code/link markup, and then the
question's Gold holds the exact character span of the original text.
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path

from ragblame.types import Doc, Gold, Question

MIN_WORDS = 80        # pages shorter than this are skipped
MAX_PROMPT_WORDS = 3000
PER_PAGE = 2          # at most this many questions are kept from one page

QUESTION_PROMPT = """You are writing a test set for a support chatbot that answers questions about Docker \
from the Docker documentation. Below is one page of those docs.

Write 1 or 2 questions that a real Docker user might type into a support chat, each answerable from \
this page alone.
- Word each question the way a user would, in their own words, about their own situation. Do not copy \
the page's phrasing or headings.
- Mix the kinds of facts you ask about, as real users do: facts stated in prose, items in a list, and \
values in a table.
- Do not ask about the page itself, its navigation, links or "what does this page cover".
- For each question give a short reference answer (one sentence at most).
- For each question give a supporting quote copied exactly, character for character, from the page: \
one sentence, one list item, or one table row together with that table's header row. Do not \
paraphrase, shorten, fix or reformat the quote.

Reply with only a JSON object in this form:
{{"questions": [{{"question": "...", "answer": "...", "quote": "..."}}]}}

Page title: {title}
URL: {url}

----- PAGE START -----
{text}
----- PAGE END -----
"""

# --- quote matching -------------------------------------------------------------------------

_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", "…": "...", " ": " ", " ": " ", " ": " ", "|": " ",
}
_SEPARATOR_LINE = re.compile(r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$", re.M)
_LINK = re.compile(r"(\[)([^\]\n]*)(\]\([^)\s]*(?:\s+\"[^\"]*\")?\))")
_BOLD = re.compile(r"\*\*|__")


def _normalise(text: str) -> tuple[str, list[int]]:
    """A matching view of `text` and, for each of its characters, the index it came from.

    Lowercased; smart quotes and dashes made plain; pipes, table separator lines and runs of
    whitespace become one space; **bold**, `code` and [link](url) markup is dropped."""
    out: list[str | None] = list(text)
    for m in _SEPARATOR_LINE.finditer(text):
        for i in range(m.start(), m.end()):
            out[i] = " "
    for m in _LINK.finditer(text):
        out[m.start(1)] = ""
        for i in range(m.start(3), m.end(3)):
            out[i] = ""
    for m in _BOLD.finditer(text):
        for i in range(m.start(), m.end()):
            out[i] = ""
    chars: list[str] = []
    index: list[int] = []
    for i, c in enumerate(out):
        c = _CHAR_MAP.get(c, c)
        if not c:
            continue
        if c.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                index.append(i)
            continue
        for ch in c:
            low = ch.lower()
            chars.append(low if len(low) == 1 else ch)
            index.append(i)
    while chars and chars[-1] == " ":
        chars.pop()
        index.pop()
    return "".join(chars), index


def _quote_lines(quote: str) -> list[str]:
    return [n for n in (_normalise(line)[0] for line in quote.splitlines()) if n]


def find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """(start, end) offsets into the original `text` where `quote` appears, or None.

    A table quote of a header row plus a data row that are not next to each other in the
    table matches the data row alone, provided the header row appears above it."""
    nq = _normalise(quote)[0]
    if not nq:
        return None
    nt, index = _normalise(text)
    pos = nt.find(nq)
    if pos != -1:
        return index[pos], index[pos + len(nq) - 1] + 1
    lines = _quote_lines(quote)
    if len(lines) < 2:
        return None
    header, row = lines[:-1], lines[-1]
    pos = nt.find(row)
    while pos != -1:
        start = index[pos]
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", start)
        on_table_row = "|" in text[line_start:len(text) if line_end == -1 else line_end]
        if on_table_row and _in_order(nt[:pos], header):
            return start, index[pos + len(row) - 1] + 1
        pos = nt.find(row, pos + 1)
    return None


def _in_order(haystack: str, parts: list[str]) -> bool:
    at = 0
    for p in parts:
        at = haystack.find(p, at)
        if at == -1:
            return False
        at += len(p)
    return True


# --- writing questions ----------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def build_prompt(doc: Doc) -> str:
    words = list(re.finditer(r"\S+", doc.text))
    text = doc.text
    if len(words) > MAX_PROMPT_WORDS:
        text = doc.text[:words[MAX_PROMPT_WORDS - 1].end()] + "\n[... rest of page truncated ...]"
    return QUESTION_PROMPT.format(title=doc.title, url=doc.url, text=text)


def _sample_order(docs: list[Doc], seed: int) -> list[Doc]:
    """Every eligible page in a random order weighted by word count (sampling without
    replacement: Efraimidis-Spirakis keys u ** (1 / w))."""
    rng = random.Random(seed)
    keyed = []
    for d in sorted(docs, key=lambda d: d.id):
        w = _word_count(d.text)
        if w >= MIN_WORDS:
            keyed.append((rng.random() ** (1.0 / w), d.id, d))
    keyed.sort(key=lambda k: (-k[0], k[1]))
    return [d for _, _, d in keyed]


def write_questions(docs: list[Doc], n: int = 100, *, seed: int = 7, ask_json=None,
                    workers: int = 4) -> tuple[list[Question], int]:
    """Up to `n` questions with verified gold quotes, and how many quotes were rejected."""
    from ragblame import llm

    ask_json = ask_json or llm.ask_json
    pages = _sample_order(docs, seed)
    kept: list[Question] = []
    rejected = 0
    while pages and len(kept) < n:
        # Enough pages to reach n if every page yields PER_PAGE questions, and at least one per worker.
        size = max(workers, math.ceil((n - len(kept)) / PER_PAGE))
        batch, pages = pages[:size], pages[size:]
        replies = llm.map_concurrent(lambda d: ask_json(build_prompt(d)), batch, workers=workers)
        if all(isinstance(r, Exception) for r in replies):
            raise llm.LLMError(f"every question-writing call in a batch failed: {replies[0]}")
        for d, reply in zip(batch, replies):
            if isinstance(reply, Exception):
                continue
            items = reply.get("questions") if isinstance(reply, dict) else None
            for item in (items if isinstance(items, list) else [])[:PER_PAGE]:
                if len(kept) >= n:
                    break
                fields = [item.get(k) if isinstance(item, dict) else None for k in ("question", "answer", "quote")]
                if not all(isinstance(f, str) and f.strip() for f in fields):
                    rejected += 1
                    continue
                question, answer, quote = (f.strip() for f in fields)
                found = find_quote(d.text, quote)
                if found is None:
                    rejected += 1
                    continue
                start, end = found
                kept.append(Question(f"q{len(kept) + 1:03d}", question, answer,
                                     Gold(d.id, start, end, d.text[start:end])))
            if len(kept) >= n:
                break
    return kept, rejected


# --- JSONL ----------------------------------------------------------------------------------

def save_questions(qs: list[Question], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


def load_questions(path) -> list[Question]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        g = d.get("gold")
        out.append(Question(d["id"], d["text"], d["answer"], Gold(**g) if g else None))
    return out
