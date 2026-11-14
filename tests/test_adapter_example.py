import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from my_pipeline import KeywordBot  # noqa: E402

from ragblame.diagnose import SKIPPED_RECHUNK, diagnose  # noqa: E402
from ragblame.pipeline import Pipeline  # noqa: E402
from ragblame.types import Doc, Gold, Judgement, Question  # noqa: E402

FACT = ("Personal accounts get 200 pulls per 6 hours, while unauthenticated users get 100 pulls per IPv4 "
        "address or IPv6 subnet in the same window, and Team accounts get unlimited pulls.")
TEXT = " ".join(["filler"] * 130) + " " + FACT + " " + " ".join(["more"] * 60)


def test_example_pipeline_runs_diagnose_and_records_skipped_rechunk():
    doc = Doc("hub/usage", "Usage", "https://example.test", TEXT)
    start = TEXT.index(FACT)
    q = Question("q1", "How many pulls do personal accounts get?", "200 per 6 hours",
                 Gold(doc.id, start, start + len(FACT), FACT))
    bot = KeywordBot([doc], generate=lambda query, passages: "RIGHT" if any(FACT in p for p in passages) else "WRONG")
    assert isinstance(bot, Pipeline)
    judge = lambda question, answer: Judgement("correct" if answer == "RIGHT" else "wrong", "")
    verdict = diagnose(bot, q, bot.run(q.text, q.id), judge_fn=judge)
    assert verdict.loss_point == "chunking"           # 150-word chunks with 15 words of overlap split the 28-word fact
    assert any(e.description == SKIPPED_RECHUNK for e in verdict.experiments)
