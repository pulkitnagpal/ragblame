import dataclasses

import pytest

from ragblame.fixes import FIX_FOR_CAUSE, FIXES, QUOTE_FIRST_PROMPT
from ragblame.judge import agreement, judge, reference_label
from ragblame.pipeline import DEFAULT_PROMPT, Config
from ragblame.types import CAUSES, Gold, Judgement, Question

QUOTE = "Personal accounts get 200 pulls per 6 hours."
QUESTION = Question("q1", "How many pulls does a personal account get?", "200 pulls per 6 hours",
                    Gold("hub/usage", 0, len(QUOTE), QUOTE))


class FakeAsk:
    """Stands in for ragblame.llm.ask_json: remembers each call and returns a scripted reply."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, prompt, *, model=None):
        self.calls.append((prompt, model))
        return self.reply


def test_judge_parses_label():
    ask = FakeAsk({"label": "partial", "reason": "Leaves out the 6-hour window."})
    got = judge(QUESTION, "200 pulls", ask_json=ask)
    assert got == Judgement("partial", "Leaves out the 6-hour window.")
    assert ask.calls[0][1] == "sonnet"


def test_judge_invalid_label_raises():
    with pytest.raises(ValueError):
        judge(QUESTION, "200 pulls", ask_json=FakeAsk({"label": "mostly right", "reason": "x"}))
    with pytest.raises(ValueError):
        judge(QUESTION, "200 pulls", ask_json=FakeAsk({"reason": "no label at all"}))


def test_judge_prompt_contains_gold_quote():
    ask = FakeAsk({"label": "correct", "reason": "Matches."})
    judge(QUESTION, "You get 200 pulls every 6 hours.", ask_json=ask)
    prompt = ask.calls[0][0]
    for part in (QUOTE, QUESTION.text, QUESTION.answer, "You get 200 pulls every 6 hours."):
        assert part in prompt

    no_gold = dataclasses.replace(QUESTION, gold=None)
    judge(no_gold, "I don't know.", ask_json=ask)
    assert "(no supporting passage)" in ask.calls[1][0]


def test_reference_label_uses_opus():
    ask = FakeAsk({"facts": ["200 pulls", "per 6 hours"], "label": "correct", "reason": "Both facts present."})
    got = reference_label(QUESTION, "200 pulls per 6 hours", ask_json=ask)
    assert got.label == "correct"
    prompt, model = ask.calls[0]
    assert model == "opus"
    assert QUOTE in prompt and "fact" in prompt.lower()


def test_agreement_counts_exact_matches():
    a = [Judgement(x, "") for x in ("correct", "partial", "wrong", "correct")]
    b = [Judgement(x, "") for x in ("correct", "wrong", "wrong", "partial")]
    assert agreement(a, b) == 0.5
    with pytest.raises(ValueError):
        agreement(a, b[:2])


def test_fix_for_every_cause_is_known_or_none():
    assert set(FIX_FOR_CAUSE) == set(CAUSES)
    for cause, fix in FIX_FOR_CAUSE.items():
        assert fix is None or fix in FIXES, cause
    for name in ("docs_gap", "unresolved", "multiple"):
        assert FIX_FOR_CAUSE[name] is None
    config_fields = {f.name for f in dataclasses.fields(Config)}
    for name, fix in FIXES.items():
        assert fix["description"]
        assert set(fix["changes"]) <= config_fields, name
        dataclasses.replace(Config(), **fix["changes"])   # every fix makes a valid Config
    assert QUOTE_FIRST_PROMPT.startswith(DEFAULT_PROMPT) and "quote" in QUOTE_FIRST_PROMPT.lower()
