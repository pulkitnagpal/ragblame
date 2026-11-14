"""Generation: the LLM reads the packed chunks and writes the answer.

The prompt is a chat: a system message (the bot's instructions) and a user message holding the
numbered context chunks and then the question. The model's chat template turns that list of
messages into the exact token sequence it was trained on.

Decoding is how the model picks each next token from its probability distribution. We decode
greedily (temperature 0: always take the single most likely token), so the same prompt always
gives the same answer. That matters here: ragblame re-runs a question with one stage swapped and
compares answers, and random sampling would make differences that no stage caused.

Qwen3.5 "thinks" by default (writes reasoning inside <think>...</think> before answering). We turn
that off in the chat template and still strip any think block that slips through.
"""
from __future__ import annotations

import re

from ragblame import models
from ragblame.types import Chunk

_CLOSED_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_messages(system: str, question: str, chunks: list[Chunk]) -> list[dict]:
    """The chat sent to the generator: instructions, then numbered context, then the question."""
    context = "\n\n".join(f"[{n}] {c.text}" for n, c in enumerate(chunks, start=1))
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning and trim whitespace, leaving only the answer.

    Handles three shapes: closed blocks anywhere; an opening <think> that never closes (the model
    ran out of tokens mid-thought), which drops everything after it; and a stray </think> with no
    opening tag (the template put <think> in the prompt), which drops everything before it.
    """
    text = _CLOSED_THINK.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


class Generator:
    """A local MLX chat model that answers deterministically."""

    def __init__(self, name: str):
        from mlx_lm import load
        self.name = name
        self.model, self.tokenizer = load(name)

    def prompt_text(self, messages: list[dict]) -> str:
        """The exact text the model sees, from its chat template, with thinking switched off."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def generate(self, messages: list[dict], max_tokens: int = 256) -> str:
        """Greedy (temperature 0) answer to `messages`, at most `max_tokens` tokens, think blocks removed."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        output = generate(self.model, self.tokenizer, self.prompt_text(messages),
                          max_tokens=max_tokens, sampler=make_sampler(temp=0.0), verbose=False)
        return strip_think(output)


models.register("generator", Generator)
