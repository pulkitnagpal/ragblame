"""Named, realistic fixes: small changes to the bot's Config that a real team could ship.

A diagnosis says which stage lost the answer ("ranking", "index", ...). A *fix* is the practical
change you'd try for that stage. The perfect swaps in diagnose.py cheat (they know where the gold
passage is); these don't, so running one shows how much of the problem a real change recovers.
"""
from __future__ import annotations

from ragblame.pipeline import DEFAULT_PROMPT

# Asking the model to quote its evidence first makes it look at the right line before answering,
# which helps when the answer was in the prompt but the model latched onto another passage.
QUOTE_FIRST_PROMPT = (DEFAULT_PROMPT + " First quote the line from the context that answers the question, "
                      "then give the answer.")

# Each fix: a plain-English description, and the Config fields it changes.
FIXES: dict[str, dict] = {
    "heading-chunks": {
        "description": "Cut the docs at their headings, so a section's facts stay together in one chunk.",
        "changes": {"chunker": "heading"},
    },
    "strong-embedder": {
        "description": "Use a stronger embedding model, so questions and their answers land closer together.",
        "changes": {"embedder": "Qwen/Qwen3-Embedding-0.6B"},
    },
    "ef-64": {
        "description": "Let the approximate index explore more of its graph per search (efSearch 16 to 64).",
        "changes": {"ef_search": 64},
    },
    "more-candidates": {
        "description": "Fetch 40 candidates from search instead of 20, giving the reranker more to choose from.",
        "changes": {"k_candidates": 40},
    },
    "better-reranker": {
        "description": "Use a stronger reranker to put the most relevant chunks first.",
        "changes": {"reranker": "cross-encoder/ettin-reranker-150m-v1"},
    },
    "bigger-budget": {
        "description": "Allow 1,800 words of context in the prompt instead of 900.",
        "changes": {"budget_words": 1800},
    },
    "quote-first-prompt": {
        "description": "Ask the model to quote the supporting line from the context before answering.",
        "changes": {"system_prompt": QUOTE_FIRST_PROMPT},
    },
}

# The fix to suggest for each verdict cause. None means no config change can help:
#   docs_gap   - the docs themselves lack the answer, so the fix is to write it;
#   unresolved - no swap fixed it, so there is nothing to point at;
#   multiple   - several stages; the caller uses the fix for the first stage in verdict.stages.
FIX_FOR_CAUSE: dict[str, str | None] = {
    "docs_gap": None,
    "generation": "quote-first-prompt",
    "chunking": "heading-chunks",
    "embedding": "strong-embedder",
    "index": "ef-64",
    "ranking": "better-reranker",
    "packing": "bigger-budget",
    "multiple": None,
    "distracted": "quote-first-prompt",
    "unresolved": None,
}
