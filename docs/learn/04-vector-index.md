# 4. The vector index: fast but approximate

To answer a question, the bot embeds it and looks for the nearest chunk vectors. Checking all 1,219
chunks is fine here, but real systems have millions, so they use an **approximate nearest neighbour
(ANN) index**. The demo bot uses HNSW (in FAISS): a graph where each vector links to a few neighbours,
and search hops greedily towards the question. It's fast, but it can stop early and miss a true
neighbour.

Measured on this repo's 60 questions, comparing the default HNSW settings with exact search:

| Setting | Recall@20 (how many of the true top 20 the index returned) |
|---|---|
| default (`efSearch=16`) | **0.937** |
| `efSearch=64` (searches more of the graph) | **0.992** |

So about 6% of the time, the shortcut drops something exact search would have found.

**A real failure it caused (q016):** "My autobuilds email me every time a build finishes... which
setting gets emails only when a build breaks?" The answer ("Only failures") ranked **first** under
exact search, but the approximate index never returned it, so the bot said it didn't know. ragblame
re-ran it with exact search and the answer came out right: an **index** failure. Suggested fix:
`ef-64`.
