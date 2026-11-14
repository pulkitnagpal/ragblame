"""Find the true answer for a question nobody wrote an answer key for (say, a customer complaint).

Diagnosis needs to know where the right answer lives in the docs: the "gold" passage. For the test
sets, Claude wrote it alongside each question. For a new question, we search the docs two ways
(exact vector search, which matches meaning, and BM25 keyword search, which matches exact terms like
"RHEL" or "$10 million"), show Claude the candidates, and ask it to copy the sentence that answers the
question. The quote must exist verbatim in the docs, so Claude can't invent an answer key.
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from ragblame.questions import find_quote
from ragblame.types import Gold

PER_METHOD = 6   # candidates from each search method

FIND_PROMPT = """A user asked a documentation chatbot this question:

{question}

Below are passages from the documentation, numbered. Find the passage that answers the question and
copy, exactly and character for character, the sentence, list item or table row that contains the answer.
If none of the passages answers it, set "passage" to null.

{passages}

Reply with only JSON: {{"passage": <number or null>, "quote": "<exact text copied from that passage>",
"answer": "<the correct answer in one short sentence>"}}"""


def _candidates(question: str, bot) -> list:
    """Chunks most likely to hold the answer: vector search plus keyword search, best first, no repeats."""
    chunks = bot.chunks()
    by_vector = [h.chunk for h in bot.search(question, PER_METHOD, exact=True)]
    bm25 = BM25Okapi([c.text.lower().split() for c in chunks])
    scores = bm25.get_scores(question.lower().split())
    by_keyword = [chunks[i] for i in sorted(range(len(chunks)), key=lambda i: -scores[i])[:PER_METHOD]]
    seen, out = set(), []
    for c in by_vector + by_keyword:
        if c.id not in seen:
            seen.add(c.id)
            out.append(c)
    return out


def find_gold(question: str, bot, *, ask_json=None) -> tuple[Gold | None, str]:
    """Return (where the answer lives in the docs, the correct answer), or (None, "") when the docs
    don't answer the question, which ragblame reports as a docs gap."""
    if ask_json is None:
        from ragblame.llm import ask_json
    cands = _candidates(question, bot)
    passages = "\n\n".join(f"[{i}] ({c.doc_id})\n{c.text}" for i, c in enumerate(cands, 1))
    reply = ask_json(FIND_PROMPT.format(question=question, passages=passages))
    quote, answer = str(reply.get("quote") or "").strip(), str(reply.get("answer") or "").strip()
    if not reply.get("passage") or not quote:
        return None, ""
    docs = {d.id: d for d in bot.docs}
    n = reply["passage"]
    first = cands[n - 1].doc_id if isinstance(n, int) and 1 <= n <= len(cands) else None
    # Look in the doc Claude pointed at first, then anywhere: it sometimes gets the number wrong.
    order = ([docs[first]] if first in docs else []) + [d for d in bot.docs if d.id != first]
    for doc in order:
        span = find_quote(doc.text, quote)
        if span:
            return Gold(doc.id, span[0], span[1], doc.text[span[0]:span[1]]), answer
    return None, ""   # the quote isn't really in the docs: don't trust it
