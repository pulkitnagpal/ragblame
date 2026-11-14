import pytest

from ragblame.generate import build_messages, strip_think
from ragblame.types import Chunk


def test_strip_think_removes_blocks():
    text = "<think>let me reason\nabout it</think>\n\nThe limit is 200 pulls.<think>again</think>  \n"
    assert strip_think(text) == "The limit is 200 pulls."


def test_strip_think_unterminated():
    # The model started thinking and ran out of tokens before closing the block.
    assert strip_think("<think>I should check the context and then") == ""
    # An unterminated block at the start swallows everything after it.
    assert strip_think("  <think>still going... 200 pulls?") == ""
    # A stray closing tag (the opening one was in the prompt) drops the reasoning before it.
    assert strip_think("reasoning here</think>\n\nThe answer.") == "The answer."
    # Plain answers are only trimmed.
    assert strip_think("  ok \n") == "ok"


def test_build_messages_numbers_chunks():
    chunks = [Chunk("a#0", "a", 0, 5, "First chunk."), Chunk("b#3", "b", 10, 20, "Second chunk.")]
    msgs = build_messages("Be brief.", "How many pulls?", chunks)
    assert msgs[0] == {"role": "system", "content": "Be brief."}
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == ("Context:\n[1] First chunk.\n\n[2] Second chunk.\n\n"
                                  "Question: How many pulls?")
    assert len(msgs) == 2


@pytest.mark.slow
def test_real_generation_says_ok():
    from ragblame import models
    from ragblame.generate import Generator  # noqa: F401  (registers the loader)
    from ragblame.pipeline import Config

    gen = models.get("generator", Config().generator)
    reply = gen.generate([{"role": "user", "content": "Reply with the word ok"}], max_tokens=16)
    assert "ok" in reply.lower()
    assert "<think>" not in reply
