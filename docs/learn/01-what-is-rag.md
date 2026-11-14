# 1. What RAG is, and why it exists

An LLM only knows what it saw in training. Ask it about your product and it either says it doesn't
know or, worse, makes something up.

**Real example from this repo** (the demo bot's own model, Qwen3.5 4B, running locally):

> **Question:** How many image pulls does a free Docker account get?
>
> **With no docs:** "Docker does not limit the number of image pulls for free accounts. You can pull as
> many images as you need. ... Pushes: limited ... up to 1000 images per month."
>
> All of that is invented.

RAG (retrieval-augmented generation) is an open-book exam: before the model answers, look up the
relevant pages and paste them into the prompt.

> **With the docs (the demo bot, `ragblame ask`):** "A free Docker account (unauthenticated users)
> gets 100 pulls per IPv4 address or IPv6 /64 subnet within a 6-hour period. Authenticated Personal
> users get 200 pulls per 6 hours."
>
> That's right, and it came from `docker-hub/usage/pulls`.

The "look up the relevant pages" part is a chain of stages, and each one can break:

| Stage | What it does | Lesson |
|---|---|---|
| Chunking | cut docs into small pieces | [2](02-chunking.md) |
| Embedding | turn each piece into a vector of numbers that captures its meaning | 3 |
| Index | find the nearest vectors fast (approximately) | 4 |
| Reranking | a smarter model re-reads the best candidates and reorders them | 5 |
| Packing | choose which pieces fit in the prompt, in what order | 6 |
| Generation | the LLM writes the answer from those pieces | 7 |

When the final answer is wrong, any one of these could be the culprit. ragblame finds out which.
