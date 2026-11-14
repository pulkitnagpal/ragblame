# The launch film

```bash
cd film && npm install          # opentype.js, for the logo's wordmark
cd ..
node film/logo.mjs              # docs/img/logo.svg, logo.png (+ logo-dark, logo-mark)
node film/stage.mjs             # dist/film: the stage and data.js, read from the real run files
node film/render.mjs video dist/ragblame-1.0.mp4 16
node film/render.mjs gif docs/img/hero.gif 23.0 36.4 1.45 960      # act 2, the x-ray, at 1.45×
node film/boards.mjs            # media/boards/*.png
uv run --no-sync ragblame report runs/2026-11-14T20-01-57 && node film/report-shot.mjs   # docs/img/report.png
```

The film is one 1920×1080 HTML stage (`film.html`, `film.js`, `film.css`), the same approach as the unremote film.
Everything on it is a function of the time, so headless Brave (or Chrome; set `RAGBLAME_CHROME` to use another
Chromium) steps it frame by frame over the DevTools protocol (`cdp.mjs`) and the frames are piped to ffmpeg: H.264
High, yuv420p, BT.709, 60 fps, no audio. The boards and the GIF are the same stage, stopped or sampled.

`stage.mjs` reads the run files at build time, so re-running it picks up new results. It checks what it shows and
stops if anything disagrees (the ranks against the verdict's own explanation, the run clock against `stats.seconds`,
the fix's before/after against the per-question labels). What it draws from, all real:

- `runs/2026-11-14T19-52-36/results.json`, the default 60-question run: q021 (the hook: the question, the bot's first
  sentence, the "wrong" label and the gold quote from the Docker docs), q011 (the x-ray: the top-20 search order, the
  reranked order, the gold chunk, the exact-search rank, how many chunks were packed, the two experiments and the
  verdict), the bot's config and the number of doc pages.
- `runs/2026-11-14T20-01-57/results.json`, the harder 40-question run: the `ragblame run` log lines (rebuilt in the
  runner's own format from the summary and stats; only the start and end lines carry a clock time, because only those
  two are recorded), the 13 causes, the calibration, and `fix_simulation` (better-reranker: before/after counts and
  which questions it fixed and broke).
- `ragblame/fixes.py`, read through Python: the config diffs in the guessing montage (each one a real alternative to
  the bot's real config) and the reranker swap in act 3.

If `fix_simulation` is missing, act 3 is left out of the captions and the grid is not drawn.

Stills for checking a change: `node film/render.mjs stills 5.4,27.2,35.4,53.2` (written to `dist/film/stills`).

| s | Beat | Caption |
|---|---|---|
| 0–6.6 | **Confidently wrong.** A support chat: the real q021 and the bot's real first sentence, stamped wrong, beside what the Docker docs say. | Your RAG bot is confidently wrong. |
| 6.6–17.4 | **The guessing loop.** `config.yaml` diffs (chunker, embedder, efSearch, candidates, budget, prompt), each redeployed and left at "still wrong?", the week ticking by. | So you guess. Chunk size? Embeddings? Index? Prompt? · Redeploy. Hope. |
| 17.4–23 | **`ragblame run`** on the harder set, with the runner's log lines and the counts. | ragblame re-runs each failure with one stage made perfect. |
| 23–36.6 | **The x-ray.** q011 travels the six stages; the camera moves in as the reranker drops the right passage from 3rd to 13th, under the prompt cut-off; the two experiments tick; the verdict stamps **ranking**. | Search found the right passage 3rd. The reranker pushed it to 13th. · Below the prompt cut-off, the bot never saw it. · The swap that fixes it is the culprit. With proof. |
| 36.6–42.6 | **Every failure.** The 13 hard-set failures sort themselves into their causes. | 13 failures on the hard set. Each one traced to a stage. |
| 42.6–54 | **The fix, measured.** `ragblame fix better-reranker`: the 40 questions as tiles; 7 flip right, 3 flip wrong; 27 → 31; then the fixes by blamed stage. | Then it tests the fix on every question. · Including what it breaks. · Ranking and packing: 6 of 7 fixed. Index, embedding, generation: 0 of 3. |
| 54–60 | **End card.** Logo, "git blame for wrong RAG answers", "Day 5/5 · open source", the repo. | |

No Docker logo anywhere: "Docker docs" is plain text, and the docs are Apache-2.0 (see `data/docker/NOTICE`).
