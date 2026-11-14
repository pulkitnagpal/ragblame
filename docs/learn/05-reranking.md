# 5. Reranking: a second, smarter opinion

Vector search is quick but shallow: it compares two vectors computed separately. A **reranker**
(a cross-encoder) reads the question and each candidate **together** and scores how well that passage
answers that question. It's slower, so it only re-sorts the top 20 candidates. The demo bot uses
`ms-marco-MiniLM-L6-v2`, a popular small default.

Usually that helps. **A real case where it hurt (q011):** "Where do I need to put the registry.json file
on Ubuntu laptops?" Search ranked the passage with the Linux path **3rd of 20**. The reranker pushed it
down to **13th**, out of the prompt, and the bot said the docs don't give a location. ragblame moved the
correct chunk back to the top (a "perfect reranker") and the answer came out right: a **ranking**
failure. Suggested fix: a stronger reranker.
