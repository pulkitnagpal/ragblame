# 6. Packing: what actually reaches the prompt

The LLM can only read what's in its prompt, and prompts have a budget. The demo bot packs the top
reranked chunks in order until it hits 900 words. With 220-word chunks, that's **4 chunks**: anything
ranked 5th or lower is invisible to the model, however relevant.

Order matters too. LLMs pay the most attention to the start and end of their context and the least to
the middle (the "lost in the middle" effect, Liu et al. 2023). A correct passage that makes it into the
prompt but sits buried in the middle can still be ignored.

ragblame's packing swap puts the correct chunk first in the prompt and re-asks. If that flips the
answer to correct, the cause was **packing**. Suggested fix: a bigger budget, or better ordering.
