// Gathers what the film draws from into dist/film: the stage (film.html/js/css + fonts) and data.js, built from the
// real run files. Re-run after a run changes (a fix simulation or the benchmark lands) and the film picks it up.
//   node film/stage.mjs
//
// Sources:
//   runs/2026-11-14T19-52-36/results.json   the default 60-question run: q021 (the hook) and q011 (the proof)
//   runs/2026-11-14T20-01-57/results.json   the harder 40-question run: the terminal, the cause bar, calibration, fix
import { cpSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { markBody } from './logo.mjs';

const HERE = dirname(fileURLToPath(import.meta.url)), ROOT = join(HERE, '..');
const OUT = join(ROOT, 'dist', 'film');
const DEFAULT_RUN = process.env.RAGBLAME_DEFAULT_RUN || '2026-11-14T19-52-36';
const HARD_RUN = process.env.RAGBLAME_HARD_RUN || '2026-11-14T20-01-57';
const HARD_QUESTIONS = 'data/questions_hard.jsonl';

mkdirSync(OUT, { recursive: true });
for (const f of ['film.html', 'film.js', 'film.css']) cpSync(join(HERE, f), join(OUT, f === 'film.html' ? 'index.html' : f));
cpSync(join(HERE, 'fonts'), join(OUT, 'fonts'), { recursive: true });

/** Reads a results file; a run that is writing it at this moment can leave it half-written, so retry a few times. */
function load(id) {
  const f = join(ROOT, 'runs', id, 'results.json');
  for (let i = 0; ; i++) {
    try { return JSON.parse(readFileSync(f, 'utf8')); } catch (err) {
      if (i >= 5) throw new Error(`can't read ${f}: ${err.message}`);
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 1000);
    }
  }
}
const fail = (msg) => { throw new Error('stage: ' + msg); };
const check = (ok, msg) => { if (!ok) fail(msg); };

const def = load(DEFAULT_RUN), hard = load(HARD_RUN);
const byId = (r, id) => r.questions.find((e) => e.question.id === id) || fail(`${id} not in ${r.id}`);
const firstSentence = (s) => s.trim().split('\n')[0].match(/^.*?[.!?](?=\s|$)/)?.[0] ?? s.trim().split('\n')[0];
const goldIn = (gold) => (c) => c.doc_id === gold.doc_id && c.start <= gold.start && c.end >= gold.end;
const data = {};

// ---- 1. the hook: q021, the bot's real wrong answer ----
{
  const e = byId(def, 'q021');
  check(e.judgement?.label === 'wrong', 'q021 is no longer judged wrong');
  data.hook = {
    id: 'q021', run: def.id, question: e.question.text,
    first: firstSentence(e.trace.answer),          // "Yes, you can install the new RPM over your current one."
    label: e.judgement.label, quote: e.question.gold.quote, cause: e.verdict?.cause,
  };
}

// ---- 2. the chain: the demo bot's real config ----
{
  const c = def.config, short = (m) => String(m).split('/').pop();
  data.chain = {
    docs: Object.keys(def.docs || {}).length,
    stages: [
      ['chunks', `${c.chunk_words} words, ${c.overlap_words} overlap`],
      ['embeddings', short(c.embedder)],
      ['index', `${c.index.toUpperCase()} · efSearch ${c.ef_search}`],
      ['reranker', short(c.reranker)],
      ['prompt', `${c.budget_words}-word budget`],
      ['LLM', short(c.generator).replace(/-MLX-4bit$/, '')],
    ],
  };
}

// ---- 3. `ragblame run` on the harder set: the runner's own log lines (ragblame/runner.py), rebuilt from results.json ----
{
  const s = hard.summary, n = s.total;
  const failed = hard.questions.filter((e) => e.judgement && e.judgement.label !== 'correct').length;
  check(n === hard.questions.length, 'summary.total does not match the question count');
  const py = (o) => '{' + Object.entries(o).map(([k, v]) => `'${k}': ${typeof v === 'object' ? py(v) : v}`).join(', ') + '}';
  const judged = { ...s, causes: {} };   // summarize() before diagnosis has no causes yet
  // The run's id is its start time and `created` its end; only those two clock times are known, so only those
  // two lines carry one (the runner stamps every line, the others are left blank rather than guessed).
  const hhmmss = (iso) => iso.slice(11, 19).replace(/-/g, ':');
  const start = hhmmss(hard.id), end = hhmmss(hard.created);
  const secs = (t) => t.split(':').reduce((a, x) => a * 60 + Number(x), 0);
  check(secs(end) - secs(start) === hard.stats.seconds, `run clock ${start}→${end} does not match stats.seconds ${hard.stats.seconds}`);
  const answered = []; for (let i = 10; i <= n; i += 10) answered.push(i); if (answered.at(-1) !== n) answered.push(n);
  data.run = {
    id: hard.id, cmd: `ragblame run --questions ${HARD_QUESTIONS}`,
    total: n, correct: s.correct, partial: s.partial, wrong: s.wrong, diagnosed: Object.values(s.causes).reduce((a, b) => a + b, 0),
    failed, seconds: hard.stats.seconds, claude: hard.stats.claude_calls,
    lines: [
      [start, `run ${hard.id}: ${n} questions`],
      ...answered.map((i) => ['', `answered ${i}/${n}`]),
      ['', `judged: ${py(judged)}`],
      ['', `diagnosing ${failed} failures`],
      [end, `done in ${hard.stats.seconds}s with ${hard.stats.claude_calls} Claude calls: ${py(s)}`],
      [null, `report: runs/${hard.id}/report.html`],
    ],
  };
}

// ---- 4. the proof: q011, ranks from the trace, checked against the verdict's own explanation ----
{
  const e = byId(def, 'q011'), tr = e.trace, v = e.verdict || fail('q011 has no verdict');
  const isGold = goldIn(e.question.gold);
  const cand = tr.candidates.map((x) => x.chunk), rer = tr.reranked.map((x) => x.chunk);
  const searchRank = cand.findIndex(isGold) + 1, rerankRank = rer.findIndex(isGold) + 1, packed = tr.packed.length;
  check(searchRank > 0 && rerankRank > 0, 'q011 gold chunk not in the candidates or the reranked list');
  check(!tr.packed.some(isGold), 'q011 gold chunk is in the prompt after all');
  const ex = v.explanation;
  const m1 = ex.match(/rank (\d+) of (\d+)/), m2 = ex.match(/down to rank (\d+)/), m3 = ex.match(/top (\d+) results/);
  check(m1 && +m1[1] === searchRank && +m1[2] === cand.length, `explanation's search rank (${m1?.[0]}) ≠ trace (${searchRank} of ${cand.length})`);
  check(m2 && +m2[1] === rerankRank, `explanation's reranked rank (${m2?.[0]}) ≠ trace (${rerankRank})`);
  check(m3 && +m3[1] === packed, `explanation's prompt size (${m3?.[0]}) ≠ trace (${packed} packed)`);
  const SWAP = { oracle: 'Hand the LLM only the right passage', ranking: 'Put it back on top', packing: 'Put it first in the prompt',
    index: 'Use exact search', embedding: 'Use a perfect embedder', chunking: 'Re-chunk so the answer sits whole' };
  const WORD = { correct: 'right', partial: 'partly right', wrong: 'still wrong' };
  // the bot's real answer, cut to the sentence that shows it went wrong
  const said = tr.answer.split(/(?<=\.)\s+/).find((x) => /does not specify/.test(x)) || firstSentence(tr.answer);
  data.proof = {
    id: 'q011', run: def.id, question: e.question.text, said: said.replace(/^However, /, '').replace(/^the/, 'The'),
    label: e.judgement.label, quote: e.question.gold.quote, gold: rer[rerankRank - 1].id,
    search: cand.map((c) => c.id), reranked: rer.map((c) => c.id),
    searchRank, rerankRank, total: cand.length, packed,
    exactRank: tr.exact.map((x) => x.chunk).findIndex(isGold) + 1, whole: cand.some(isGold) || tr.exact.some((x) => isGold(x.chunk)),
    experiments: v.experiments.map((x) => [SWAP[x.swap] || x.description, WORD[x.judgement.label] || x.judgement.label, x.judgement.label]),
    cause: v.cause,
  };
}

// ---- 1b. the guessing montage: the bot's real config, and the real alternatives in ragblame/fixes.py ----
{
  // FIXES is Python; ask Python for it rather than parsing the source
  const fixes = JSON.parse(execFileSync('uv', ['run', '--no-sync', 'python', '-c',
    'import json; from ragblame.fixes import FIXES; print(json.dumps(FIXES))'], { cwd: ROOT, encoding: 'utf8' }));
  const c = def.config, short = (m) => String(m).split('/').pop();
  // the stages an engineer would poke at first; the reranker is left for act 3, where its fix is measured
  const order = ['heading-chunks', 'strong-embedder', 'ef-64', 'more-candidates', 'bigger-budget', 'quote-first-prompt'];
  data.guesses = order.filter((k) => fixes[k]).map((k) => {
    const [key, to] = Object.entries(fixes[k].changes)[0];
    if (key === 'system_prompt') return { key, from: '"… say you don\'t know."', to: '"… say you don\'t know. ' + to.slice(c.system_prompt.length).trim() + '"' };
    const fmt = (v) => (typeof v === 'string' ? short(v) : String(v));
    return { key, from: fmt(c[key]), to: fmt(to) };
  });
  check(data.guesses.every((g) => g.from !== 'undefined'), 'a guessed config key is not in the run config');
  const fx = hard.fix_simulation && fixes[hard.fix_simulation.fix];
  if (fx) { const [key, to] = Object.entries(fx.changes)[0]; data.fixDiff = { key, from: short(hard.config[key]), to: short(to) }; }
}

// ---- 5. the breakdown of the harder run, plus calibration and the fix simulation when they are there ----
{
  const s = hard.summary;
  data.breakdown = { total: s.total, correct: s.correct, partial: s.partial, wrong: s.wrong,
    causes: Object.entries(s.causes).sort((a, b) => b[1] - a[1]) };
  if (hard.calibration) data.breakdown.calibration = { n: hard.calibration.n, agreement: hard.calibration.agreement,
    agree: hard.calibration.pairs.filter((p) => p.judge === p.reference).length };
  const f = hard.fix_simulation;
  if (f) data.breakdown.fix = { name: f.fix, description: f.description, before: f.before, after: f.after,
    fixed: f.fixed_ids.length, broken: f.broken_ids.length };
  // every hard question: its label and cause before, and what the fix did to it
  data.hard = hard.questions.map((e) => ({ id: e.question.id, label: e.judgement?.label ?? 'error', cause: e.verdict?.cause ?? null,
    fixed: !!f?.fixed_ids.includes(e.question.id), broken: !!f?.broken_ids.includes(e.question.id) }));
  if (f) {
    const right = data.hard.filter((q) => q.label === 'correct').length;
    check(right === f.before.correct, `fix 'before' (${f.before.correct}) ≠ the run's correct count (${right})`);
    check(right + f.fixed_ids.length - f.broken_ids.length === f.after.correct, 'fixed/broken ids do not add up to the after count');
    check(data.hard.filter((q) => q.broken).every((q) => q.label === 'correct'), 'a broken question was not right before');
  }
}

writeFileSync(join(OUT, 'data.js'),
  `window.RAGBLAME_MARK = ${JSON.stringify(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">${markBody()}</svg>`)};\n` +
  `window.RAGBLAME_DATA = ${JSON.stringify(data, null, 1)};\n`);
console.log(`staged ${OUT}`);
console.log(`  hook q021 (${def.id}): "${data.hook.first}" → ${data.hook.label}`);
console.log(`  run ${hard.id}: ${data.run.total} questions, ${data.run.correct} right, ${data.run.diagnosed} diagnosed, ${data.run.claude} Claude calls, ${data.run.seconds} s`);
console.log(`  proof q011: search rank ${data.proof.searchRank} of ${data.proof.total} → reranked ${data.proof.rerankRank}, prompt top ${data.proof.packed}, cause ${data.proof.cause}`);
console.log(`  causes: ${data.breakdown.causes.map(([c, n]) => c + ' ' + n).join(', ')}`);
console.log(`  calibration: ${data.breakdown.calibration ? data.breakdown.calibration.agreement + ' over ' + data.breakdown.calibration.n : 'absent'}`);
console.log(`  fix_simulation: ${data.breakdown.fix ? `${data.breakdown.fix.name} ${data.breakdown.fix.before.correct} → ${data.breakdown.fix.after.correct} correct` : 'absent (beat 5 shows the causes only)'}`);
