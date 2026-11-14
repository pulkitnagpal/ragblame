# 3. Embeddings: meaning as numbers

To search by meaning rather than by matching words, every chunk is turned into a list of numbers, a
**vector**. The demo bot's embedder (`bge-small-en-v1.5`) gives 384 numbers per chunk. Texts that mean
similar things get vectors that point in similar directions, and we measure that with **cosine
similarity** (1 = same direction, 0 = unrelated).

Real numbers from this repo:

| Sentence A | Sentence B | Similarity |
|---|---|---|
| "Personal accounts get 200 pulls per 6 hours." | "A free signed-in user can download images 200 times every six hours." | **0.69** |
| "Personal accounts get 200 pulls per 6 hours." | "Docker Desktop uses VirtioFS for file sharing on Mac." | **0.48** |

The first pair shares almost no words, yet scores much higher: the model learned that "pulls" ≈
"download images" and "6 hours" ≈ "six hours". That's what lets a user's own wording find the docs.

**Where it breaks:** a small embedder can place a question far from the passage that answers it,
especially with product jargon. Then even a perfect search won't find it. ragblame calls this an
**embedding** failure: the correct passage isn't in the top results even with exact search.
