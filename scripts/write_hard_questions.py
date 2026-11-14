"""Write the harder question set (data/questions_hard.jsonl).

Same writer as the default set, with a stricter brief: questions that hinge on one specific detail
(a value in a table row, a limit, a condition, a platform difference), asked the way a user who doesn't
know the docs' vocabulary would ask. These are what real support tickets look like, and what a RAG bot
most often gets wrong. Reported separately from the default set.
"""
from pathlib import Path

import ragblame  # noqa: F401  (sets OpenMP env before torch/faiss)
from ragblame import questions
from ragblame.corpus import load_docs

HARD_RULES = """
- Make each question hinge on ONE specific detail: a value in a table row, a number or limit, a condition \
("only if…", "except…"), a default setting, or a difference between platforms or plans. Prefer details \
that sit deep inside long pages, tables and lists rather than in the opening paragraph.
- Write the question as a user who does not know Docker's own terms would: describe the situation or the \
symptom rather than naming the setting or feature.
"""

questions.QUESTION_PROMPT = questions.QUESTION_PROMPT.replace(
    "- Do not ask about the page itself", HARD_RULES.strip("\n") + "\n- Do not ask about the page itself")
qs, rejected = questions.write_questions(load_docs(Path("data/docker")), n=40, seed=11)
qs = [type(q)(id=q.id.replace("q", "h"), text=q.text, answer=q.answer, gold=q.gold) for q in qs]
questions.save_questions(qs, Path("data/questions_hard.jsonl"))
print(f"kept {len(qs)} rejected {rejected}")
