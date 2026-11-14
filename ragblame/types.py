"""Data shared by every part of ragblame. All of it serialises with dataclasses.asdict."""
from __future__ import annotations

from dataclasses import dataclass, field

# Every verdict names one of these causes.
CAUSES = ("docs_gap", "generation", "chunking", "embedding", "index", "ranking", "packing",
          "multiple", "distracted", "unresolved")
# Pipeline stages in the order data flows through them; diagnosis looks for the earliest loss.
STAGES = ("chunking", "embedding", "index", "ranking", "packing")
# What the judge may say about an answer. Only "correct" counts as right, or as a fix.
LABELS = ("correct", "partial", "wrong")


@dataclass(frozen=True)
class Doc:
    id: str      # path relative to the corpus root, without ".md"
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str      # f"{doc_id}#{n}"
    doc_id: str
    start: int   # character offsets into the doc's text
    end: int
    text: str    # an optional context prefix (e.g. "Docker Hub > Usage and limits" or a repeated
                 # table header) followed by doc.text[start:end]; always text.endswith(doc.text[start:end])


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Gold:
    """Where the true answer lives: a span of one doc's text."""
    doc_id: str
    start: int
    end: int
    quote: str


@dataclass
class Question:
    id: str
    text: str
    answer: str            # short reference answer
    gold: Gold | None      # None means the docs don't contain the answer


@dataclass
class Judgement:
    label: str             # one of LABELS
    reason: str


@dataclass
class Trace:
    """Everything one run of the pipeline produced for one question."""
    question_id: str
    exact: list[Hit]       # top 50 by exact search with the bot's embedder (instrumentation only)
    candidates: list[Hit]  # what the bot's index returned
    reranked: list[Hit]    # candidates after the reranker
    packed: list[Chunk]    # what went into the prompt, in prompt order
    prompt: str
    answer: str


@dataclass
class Experiment:
    swap: str              # e.g. "oracle", "chunking", "index"
    description: str       # plain English, shown in the report
    answer: str
    judgement: Judgement


@dataclass
class Verdict:
    cause: str                     # one of CAUSES
    stages: list[str] = field(default_factory=list)   # stages involved (several when cause == "multiple")
    loss_point: str | None = None  # earliest stage where the gold passage was lost, from the trace alone
    experiments: list[Experiment] = field(default_factory=list)
    explanation: str = ""
    fix: str = ""
