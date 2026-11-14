"""The judge: an LLM that grades the bot's answer as correct, partial or wrong.

Grading free-text answers by string match fails ("200 pulls" vs "two hundred pulls"), so we ask
Claude to compare the bot's answer with the reference answer and the gold passage from the docs.

Because the judge is itself a model, we check it: `reference_label` grades the same answers with a
different, stricter setup (Opus, fact by fact), and `agreement` measures how often the two agree.
"""
from __future__ import annotations

from ragblame.types import LABELS, Judgement, Question

NO_GOLD = "(no supporting passage)"

JUDGE_PROMPT = """You are grading a support bot's answer against the documentation.

Question: {question}
Reference answer: {reference}
Supporting passage from the docs: {gold}
Bot's answer: {answer}

Rules:
- The same meaning in different words is "correct".
- Extra detail is fine, as long as it is correct.
- An answer that is right but misses a qualifier the question needs (a limit, a condition, a plan name) is "partial".
- An answer that contradicts the docs is "wrong".
- "I don't know", or refusing to answer, is "wrong".
- Judge only against the supporting passage and the reference answer, not your own knowledge.

Reply with JSON only: {{"label": "correct" | "partial" | "wrong", "reason": "<one sentence>"}}"""

REFERENCE_PROMPT = """You are carefully grading a support bot's answer against the documentation.

Question: {question}
Reference answer: {reference}
Supporting passage from the docs: {gold}
Bot's answer: {answer}

Work in two steps.
1. List every fact from the supporting passage that a complete answer to this question must state,
   including any limit, condition or plan name the question depends on.
2. Check each fact against the bot's answer: stated (in any wording), missing, or contradicted.

Then label the answer:
- "correct": every required fact is stated and nothing contradicts the passage (extra correct detail is fine);
- "partial": nothing is contradicted, but at least one required fact is missing;
- "wrong": any fact is contradicted, or the answer says "I don't know" or refuses.
Use only the passage and the reference answer, not your own knowledge.

Reply with JSON only:
{{"facts": [{{"fact": "<fact>", "status": "stated" | "missing" | "contradicted"}}],
  "label": "correct" | "partial" | "wrong", "reason": "<one sentence>"}}"""


def _default_ask_json():
    """Load the real Claude wrapper only when it's needed, so tests can run without it."""
    from ragblame.llm import ask_json
    return ask_json


def _fill(template: str, question: Question, answer: str) -> str:
    """Put the question, reference, gold quote and bot answer into a prompt template."""
    gold = question.gold.quote if question.gold else NO_GOLD
    return template.format(question=question.text, reference=question.answer, gold=gold, answer=answer)


def _parse(reply: dict) -> Judgement:
    """Turn the model's JSON reply into a Judgement, refusing any label we don't know."""
    label = str(reply.get("label", "")).strip().lower()
    if label not in LABELS:
        raise ValueError(f"judge returned an invalid label: {reply.get('label')!r}")
    return Judgement(label, str(reply.get("reason", "")).strip())


def judge(question: Question, answer: str, *, model: str = "sonnet", ask_json=None) -> Judgement:
    """Grade one bot answer as correct, partial or wrong, with a one-sentence reason."""
    ask_json = ask_json or _default_ask_json()
    return _parse(ask_json(_fill(JUDGE_PROMPT, question, answer), model=model))


def reference_label(question: Question, answer: str, *, ask_json=None) -> Judgement:
    """Grade an answer the slow, strict way (Opus, fact by fact) to check the everyday judge."""
    ask_json = ask_json or _default_ask_json()
    return _parse(ask_json(_fill(REFERENCE_PROMPT, question, answer), model="opus"))


def agreement(a: list[Judgement], b: list[Judgement]) -> float:
    """Share of answers where two graders gave exactly the same label (0.0 to 1.0)."""
    if len(a) != len(b):
        raise ValueError(f"can't compare {len(a)} judgements with {len(b)}")
    if not a:
        raise ValueError("no judgements to compare")
    return sum(x.label == y.label for x, y in zip(a, b)) / len(a)
