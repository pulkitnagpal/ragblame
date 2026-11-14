// The ragblame launch film as one seekable stage. Everything on screen is a function of the time t (seconds): no CSS
// transitions, no timers, so headless Brave can step it frame by frame (film/render.mjs) and the boards can freeze it.
//
//   const film = RagblameFilm.mount(rootElement, window.RAGBLAME_DATA);  film.seek(12.5);
//
// The story: BEFORE, a bot is confidently wrong and the engineer guesses at config; WITH ragblame, one real failure is
// x-rayed stage by stage and proven; AFTER, the fix is measured on every question, breakage included.
// Every number, question, answer, chunk id and config value comes from window.RAGBLAME_DATA, which film/stage.mjs
// builds from the real results files (and ragblame/fixes.py). There are no fallback values.
(function () {
  'use strict';

  // ---------- easing and helpers ----------
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const seg = (t, a, b) => (b <= a ? (t >= a ? 1 : 0) : clamp((t - a) / (b - a)));
  const eo = (x) => 1 - Math.pow(1 - clamp(x), 3);
  const eio = (x) => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
  const back = (x, c = 1.4) => { x = clamp(x); return 1 + (c + 1) * Math.pow(x - 1, 3) + c * Math.pow(x - 1, 2); };
  const lerp = (a, b, x) => a + (b - a) * x;
  const win = (t, a, b, fi = 0.3, fo = 0.3) => Math.min(seg(t, a, a + fi), 1 - seg(t, b - fo, b));
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const ord = (n) => n + ({ 1: 'st', 2: 'nd', 3: 'rd' }[(n % 100 >= 11 && n % 100 <= 13) ? 0 : n % 10] || 'th');
  const CHECK = '<svg viewBox="0 0 20 20"><path d="M4 10.5l4 4 8-9" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  /** A stamp landing: big and tilted, then down hard with a small settle. */
  const stampS = (u) => (u <= 0 ? 2.3 : u < 0.55 ? lerp(2.3, 0.94, eio(u / 0.55)) : lerp(0.94, 1, eo((u - 0.55) / 0.45)));

  function h(tag, cls, html, style) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    if (style) Object.assign(e.style, style);
    return e;
  }
  /** Opacity, an offset, a scale (or x/y scales) and a rotation, applied only when they change. */
  function put(e, o) {
    const op = o.o == null ? 1 : clamp(o.o);
    const vis = op > 0.001;
    if (e._vis !== vis) { e.style.visibility = vis ? 'visible' : 'hidden'; e._vis = vis; }
    if (!vis) return;
    const ops = op.toFixed(3);
    if (e._op !== ops) { e.style.opacity = ops; e._op = ops; }
    const s = o.s == null ? 1 : o.s;
    const tf = `translate(${(o.dx || 0).toFixed(2)}px, ${(o.dy || 0).toFixed(2)}px) scale(${s.toFixed(4)}, ${(o.sy == null ? s : o.sy).toFixed(4)}) rotate(${(o.r || 0).toFixed(2)}deg)`;
    if (e._tf !== tf) { e.style.transform = tf; e._tf = tf; }
  }
  function at(parent, e, x, y, origin) {
    e.classList.add('abs');
    e.style.left = x + 'px'; e.style.top = y + 'px';
    if (origin) e.style.transformOrigin = origin;
    parent.appendChild(e);
    return e;
  }
  const rise = (e, t, a, d = 0.5, k = 1) => put(e, { o: eo(seg(t, a, a + d)) * k, dy: lerp(18, 0, eo(seg(t, a, a + d + 0.1))) });
  const cls = (e, c, on) => { if (!e._c) e._c = {}; if (e._c[c] !== on) { e.classList.toggle(c, on); e._c[c] = on; } };

  // ============================================================================================
  function mount(root, D) {
    if (!D) throw new Error('RAGBLAME_DATA is missing: run node film/stage.mjs');
    const stage = h('div', 'rb-stage');
    root.appendChild(stage);
    stage.appendChild(h('div', 'grid'));
    const scene = () => { const s = h('div', 'scene'); stage.appendChild(s); return s; };

    // ---------------- the timeline ----------------
    const T = { guess: 6.6, run: 17.4, xray: 23.0, causes: 36.6, fix: 42.6, end: 54.0 };
    const DUR = 60.0;

    // =========================== ACT 1: BEFORE ===========================
    // ---- the hook: a support chat, the real q021 ----
    const S1 = scene();
    const HK = D.hook;
    const words = HK.first.replace(/[.!]$/, '').split(' ');
    const chat = at(S1, h('div', 'chat', `<div class="tb"><span class="av">?</span>Support<small>· docs assistant</small></div>
      <div class="body">
        <div class="bub me" data-me>${esc(HK.question)}</div>
        <div class="bub bot" data-bot><span class="who">Bot</span><span class="hl" data-hl>${words.map((w) => `<span class="w">${esc(w)}</span>`).join(' ')}</span><span data-rest>…</span>
          <span class="dots abs" data-dots style="left:26px;top:52px"><i></i><i></i><i></i></span></div>
      </div>`), 200, 236, '50% 50%');
    const me = chat.querySelector('[data-me]'), bot = chat.querySelector('[data-bot]'), hl = chat.querySelector('[data-hl]');
    const wEls = [...chat.querySelectorAll('.w')], rest = chat.querySelector('[data-rest]'), dots = chat.querySelector('[data-dots]');
    hl.style.backgroundImage = 'linear-gradient(var(--bad), var(--bad))';
    const stamp = at(S1, h('div', 'stamp', esc(HK.label)), 820, 650, '50% 50%');
    const docCard = at(S1, h('div', 'doc', `<div class="k">Docker docs say</div><q>${esc(HK.quote)}</q>`), 1290, 500);
    docCard.style.borderLeftColor = 'var(--ok)';

    // ---- the guessing loop: real config values → the real alternatives in ragblame/fixes.py ----
    const S2 = scene();
    const G = D.guesses;
    const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Monday'];
    const GD = [2.0, 1.8, 1.6, 1.4, 1.3, 1.3].slice(0, G.length);
    const G0 = T.guess + 0.6;
    const gT = GD.reduce((a, d, i) => (a.push(a[i] + d), a), [G0]);
    const gTitle = at(S2, h('div', 'big', 'So you guess.', { fontSize: '56px' }), 120, 96);
    const ed = at(S2, h('div', 'ed', `<div class="tb"><i></i><i></i><i></i><span>bot/config.yaml</span></div><div class="body"><div data-d></div>
      <div class="st2"><span class="chip" data-dep><span class="spin" data-spin></span>redeploying</span><span class="chip q" data-q>still wrong?</span></div></div>`), 120, 210);
    const edBody = ed.querySelector('[data-d]'), dep = ed.querySelector('[data-dep]'), spin = ed.querySelector('[data-spin]'), qChip = ed.querySelector('[data-q]');
    const diffEls = G.map((g) => {
      const e = h('div', null, `<div class="dl m">- ${esc(g.key)}: ${esc(g.from)}</div><div class="dl p">+ ${esc(g.key)}: <span data-to></span></div>`);
      e.style.position = 'absolute'; e.style.left = '32px'; e.style.right = '32px'; e.style.top = '30px';
      edBody.appendChild(e);
      return { e, to: e.querySelector('[data-to]'), txt: g.to };
    });
    const dayK = at(S2, h('div', 'k', 'This week'), 1300, 214);
    const dayEl = at(S2, h('div', 'day', ''), 1296, 250);
    const tried = at(S2, h('div', 'tried', G.map((g) => `<div class="it"><b>?</b><s>${esc(g.key)} → ${esc(g.to.length > 26 ? g.to.slice(0, 24) + '…' : g.to)}</s></div>`).join('')), 1300, 380);
    const triedEls = [...tried.children];

    // =========================== ACT 2: WITH ragblame ===========================
    // ---- ragblame run on the harder set ----
    const S3 = scene();
    const RN = D.run;
    const term = at(S3, h('div', 'term', `<div class="tb"><i></i><i></i><i></i><span>~/ragblame</span></div><div class="body">
      <div class="ln"><span class="p">$ </span><span class="cmd" data-cmd></span><span class="caret" data-caret></span></div>
      ${RN.lines.map(([ts, txt], i) => `<div class="ln${i === RN.lines.length - 2 ? ' done' : ''}" data-l>${ts === null ? '' : `<span class="ts">${ts ? '[' + ts + ']' : ''}</span>`}${esc(txt)}</div>`).join('')}
    </div>`), 80, 110);
    const cmdEl = term.querySelector('[data-cmd]'), caret = term.querySelector('[data-caret]');
    const lineEls = [...term.querySelectorAll('[data-l]')];
    const tallyItems = [[RN.total, 'questions'], [RN.correct, 'right'], [RN.diagnosed, 'failures diagnosed'], [RN.claude, 'Claude calls']];
    const tally = at(S3, h('div', 'tally', tallyItems.map(([, l], i) => `<div data-t="${i}"><div class="v${i === 2 ? ' acc' : ''}">0</div><div class="l">${l}</div></div>`).join('') +
      `<div class="l" data-secs style="font:500 21px var(--sans);color:var(--muted)">in ${RN.seconds} s</div>`), 1400, 130);
    const tallyEls = [...tally.querySelectorAll('[data-t]')], secsEl = tally.querySelector('[data-secs]');
    const R0 = T.run;
    const L_T = [R0 + 1.9];
    for (let i = 1; i < RN.lines.length; i++) L_T.push(L_T[i - 1] + (i === RN.lines.length - 2 ? 0.75 : i === RN.lines.length - 1 ? 0.35 : 0.2));
    const iJudged = RN.lines.findIndex(([, x]) => x.startsWith('judged')), iDone = RN.lines.length - 2;
    const tallyAt = [[L_T[0], 0.2], [L_T[iJudged], 0.5], [L_T[iDone], 0.5], [L_T[iDone], 0.6]];

    // ---- the x-ray: q011 traced through the six stages ----
    const S4 = scene();
    const cam = h('div', null, null, { position: 'absolute', inset: '0', transformOrigin: '0 0' }); S4.appendChild(cam);
    const PF = D.proof, CH = D.chain;
    const status = [
      PF.whole ? ['ok', 'whole in one chunk'] : ['bad', 'split across chunks'],
      ['ok', `exact search: #${PF.exactRank}`],
      ['ok', `search: #${PF.searchRank} of ${PF.total}`],
      ['bad', `reranked: #${PF.rerankRank}`],
      ['bad', `only the top ${PF.packed} fit`],
      ['bad', 'never saw it'],
    ];
    const PW = 268, PP = 284, PX = (1920 - (5 * PP + PW)) / 2, PY = 60;
    const pills = CH.stages.map(([name], i) => at(cam, h('div', 'pl', `<div class="t">${esc(name)}</div><div class="s"><span data-s>·</span></div>`), PX + i * PP, PY, '50% 50%'));
    const pillS = pills.map((p) => p.querySelector('[data-s]'));
    const tracer = at(cam, h('div', null, null, { width: '16px', height: '16px', borderRadius: '50%', background: 'var(--accent)', boxShadow: '0 0 0 7px rgba(245,165,36,.2)' }), 0, PY - 22, '50% 50%');
    const qcard = at(cam, h('div', 'qcard', `<div class="qq">${esc(PF.question)}</div><span class="tag" style="background:var(--bad);color:#1A0606">${esc(PF.label)}</span>
      <div class="aa">Bot: <em>“${esc(PF.said).replace(/`([^`]+)`/g, '<code>$1</code>')}”</em></div><div></div>`), 80, 176);
    const ROW = 27, LY = 318;
    const rank = at(cam, h('div', 'rank', `<div class="hd"><div class="h" style="position:relative;height:30px;width:600px"><span data-h0 class="abs">Search results</span><span data-h1 class="abs">After the reranker</span></div><div class="k">top ${PF.total}</div></div>
      <div class="list" data-list style="height:${PF.total * ROW}px"></div>`), 80, LY);
    const list = rank.querySelector('[data-list]'), h0 = rank.querySelector('[data-h0]'), h1 = rank.querySelector('[data-h1]');
    const inprompt = h('div', 'inprompt', null, { height: (PF.packed * ROW - 1) + 'px' }); list.appendChild(inprompt);
    const goldbar = h('div', 'goldbar', null, { height: ROW + 'px' }); list.appendChild(goldbar);
    const cut = h('div', 'cut', `<span>prompt cut-off · top ${PF.packed}</span>`, { top: (PF.packed * ROW - 1) + 'px' }); list.appendChild(cut);
    for (let i = 0; i < PF.total; i++) list.appendChild(h('div', 'row', `<span class="r">${i + 1}</span>`, { top: i * ROW + 'px', height: ROW + 'px' }));
    const ids = [...new Set([...PF.search, ...PF.reranked])];
    const rows = ids.map((id) => {
      const e = h('div', 'row', `<span class="r"></span>${esc(id)}`, { height: ROW + 'px' });
      list.appendChild(e);
      return { e, id, a: PF.search.indexOf(id), b: PF.reranked.indexOf(id) };
    });
    const goldRow = rows.find((r) => r.id === PF.gold);
    const passage = at(cam, h('div', 'side', `<div class="k">The right passage · <span data-rk></span></div><div class="quote">${esc(PF.quote)}</div>`), 1140, LY);
    const rk = passage.querySelector('[data-rk]');
    const exps = at(cam, h('div', 'side', `<div class="k" style="margin-bottom:6px">Re-run with one stage made perfect</div>` +
      PF.experiments.map(([label, word]) => `<div class="exp"><span class="ck">${CHECK}</span><span>${esc(label)}</span><span class="res">→ ${esc(word)}</span></div>`).join('')), 1140, 468);
    const expEls = [...exps.querySelectorAll('.exp')];
    const vstamp = at(cam, h('div', 'vstamp', `<small>Cause</small>${esc(PF.cause)}`), 1150, 726, '50% 50%');
    const vwhy = at(cam, h('div', 'note', `${ord(PF.searchRank)} in search → ${ord(PF.rerankRank)} after the reranker.<br>Put back on top, the answer is right.`), 1150, 838);

    // ---- every hard-set failure, sorted into causes ----
    const S5 = scene();
    const BD = D.breakdown;
    const fails = D.hard.filter((q) => q.cause);
    const order = BD.causes.map(([c]) => c);
    const ALPHA = [1, 0.74, 0.58, 0.46, 0.38, 0.32, 0.28, 0.24];
    const cHead = at(S5, h('div', null, `<div class="k">Harder set · ${BD.total} questions</div><div class="big" style="font-size:52px;margin-top:14px">${BD.correct} right · ${BD.partial} partial · ${BD.wrong} wrong</div>`), 84, 150);
    const CW = 120, IN = 6, BETW = 32;
    const totalW = fails.length * CW + (fails.length - order.length) * IN + (order.length - 1) * BETW;
    const BX = (1920 - totalW) / 2, CY = 400;
    let cx = BX; const groupX = {};
    order.forEach((c) => { groupX[c] = cx; const n = fails.filter((q) => q.cause === c).length; cx += n * CW + (n - 1) * IN + BETW; });
    const placed = {};
    const ctiles = fails.map((q, i) => {
      const k = placed[q.cause] = (placed[q.cause] || 0) + 1;
      const a = BX + i * (CW + (totalW - fails.length * CW) / (fails.length - 1));
      const b = groupX[q.cause] + (k - 1) * (CW + IN);
      const e = at(S5, h('div', 'ct', esc(q.id), { background: `rgba(245,165,36,${ALPHA[Math.min(order.indexOf(q.cause), ALPHA.length - 1)]})` }), 0, CY, '50% 50%');
      return { e, a, b, q };
    });
    const clabels = order.map((c) => { const n = fails.filter((q) => q.cause === c).length;
      return at(S5, h('div', null, `<div style="font:700 44px var(--sans);letter-spacing:-0.03em">${n}</div><div style="font:500 23px var(--sans);color:var(--body);white-space:nowrap">${esc(c.replace('_', ' '))}</div>`), groupX[c] + 2, CY + 96); });
    const cal = BD.calibration;
    const calNote = cal ? at(S5, h('div', 'note', `Judge check: a stricter grader agrees on <b>${cal.agree} of ${cal.n}</b> labels.`), BX, CY + 250) : null;

    // =========================== ACT 3: AFTER ===========================
    const S6 = scene();
    const FX = BD.fix, FD = D.fixDiff;
    const fixCmd = `ragblame fix ${FX ? FX.name : ''} runs/${RN.id}`;
    const fterm = at(S6, h('div', 'term', `<div class="body" style="padding:20px 28px"><div class="ln" style="padding-left:0;text-indent:0"><span class="p">$ </span><span class="cmd" data-cmd></span><span class="caret" data-caret></span></div>
      ${FD ? `<div class="ln" data-diff style="padding-left:0;text-indent:0;color:var(--muted)">${esc(FD.key)}: <span style="color:var(--bad)">${esc(FD.from)}</span> → <span style="color:var(--ok)">${esc(FD.to)}</span> · re-answering all ${BD.total} questions</div>` : ''}</div>`, { width: '1760px', height: 'auto' }), 80, 56);
    const fCmd = fterm.querySelector('[data-cmd]'), fCaret = fterm.querySelector('[data-caret]'), fDiff = fterm.querySelector('[data-diff]');
    const TW = 156, TH = 100, TG = 14, GX = 80, GY = 236, COLS = 8;
    const tiles = D.hard.map((q, i) => {
      const e = at(S6, h('div', 'tile', `<div class="i">${esc(q.id)}</div><div class="c">${q.cause ? esc(q.cause.replace('_', ' ')) : q.label === 'correct' ? 'right' : esc(q.label)}</div>`), GX + (i % COLS) * (TW + TG), GY + Math.floor(i / COLS) * (TH + TG), '50% 50%');
      return { e, q, c: e.querySelector('.c') };
    });
    const fixedT = tiles.filter((x) => x.q.fixed), brokeT = tiles.filter((x) => x.q.broken);
    const RX = 1470;
    const counter = at(S6, h('div', null, `<div class="k">Right answers</div><div style="display:flex;align-items:baseline;gap:22px;margin-top:10px"><span data-v style="display:inline-block;transform-origin:0 100%;font:700 120px/1 var(--sans);letter-spacing:-0.05em;font-variant-numeric:tabular-nums">0</span><span style="font:500 30px var(--sans);color:var(--muted)">of ${BD.total}</span></div>
      <div class="note" style="margin-top:14px;font-size:24px">fixed <b data-nf style="color:var(--ok)">0</b> · broke <b data-nb style="color:var(--bad)">0</b></div>`), RX, GY);
    const cv = counter.querySelector('[data-v]'), nfEl = counter.querySelector('[data-nf]'), nbEl = counter.querySelector('[data-nb]');
    // per cause: how many of its failures the fix turned right
    const per = order.map((c) => { const qs = D.hard.filter((q) => q.cause === c); return [c, qs.filter((q) => q.fixed).length, qs.length]; });
    const perEl = at(S6, h('div', null, `<div class="k" style="margin-bottom:10px">Fixed, by blamed stage</div>` +
      per.map(([c, f, n]) => `<div class="pc${f ? ' yes' : ''}"><span>${esc(c.replace('_', ' '))}</span><b>${f} of ${n}</b></div>`).join(''), { width: '370px' }), RX, GY + 268);
    const perRows = [...perEl.querySelectorAll('.pc')];
    const sum = (cs) => per.filter(([c]) => cs.includes(c)).reduce((a, [, f, n]) => [a[0] + f, a[1] + n], [0, 0]);
    const rp = sum(['ranking', 'packing']), rest3 = sum(['index', 'embedding', 'generation']);

    // =========================== end card ===========================
    const S7 = scene();
    const endMark = at(S7, h('div', null, window.RAGBLAME_MARK || '', { width: '150px', height: '150px', borderRadius: '33px', boxShadow: '0 0 0 2px #343944' }), 960 - 75, 220, '50% 50%');
    const endName = at(S7, h('div', 'end-name', 'ragblame', { width: '1920px', textAlign: 'center' }), 0, 410, '50% 50%');
    const endTag = at(S7, h('div', 'end-tag', 'git blame for wrong RAG answers', { width: '1920px' }), 0, 616, '50% 50%');
    const endFoot = at(S7, h('div', null, '<span class="pill">Day 5/5 · open source</span><span class="end-url">github.com/pulkitnagpal/ragblame</span>', { display: 'flex', gap: '28px', alignItems: 'center', justifyContent: 'center', width: '1920px' }), 0, 760, '50% 50%');

    // ---------------- captions ----------------
    const CAPS = [
      [0.8, 6.3, 'Your RAG bot is confidently wrong.'],
      [7.0, 11.9, 'So you guess. Chunk size? Embeddings? Index? Prompt?'],
      [12.2, 17.1, 'Redeploy. Hope.'],
      [17.8, 22.7, '<b>ragblame</b> re-runs each failure with one stage made perfect.'],
      [23.4, 27.9, `Search found the right passage ${ord(PF.searchRank)}. The reranker pushed it to ${ord(PF.rerankRank)}.`],
      [28.2, 31.5, 'Below the prompt cut-off, the bot never saw it.'],
      [31.8, 36.3, 'The swap that fixes it is the culprit. <b>With proof.</b>'],
      [37.0, 42.3, `${fails.length} failures on the hard set. Each one traced to a stage.`],
    ];
    if (FX) CAPS.push(
      [43.0, 47.6, 'Then it tests the fix on every question.'],
      [47.9, 50.5, '<b>Including what it breaks.</b>'],
      [50.8, 53.8, `Ranking and packing: ${rp[0]} of ${rp[1]} fixed. Index, embedding, generation: ${rest3[0]} of ${rest3[1]}.`]);
    const capEls = CAPS.map(([, , html]) => { const e = h('div', 'cap', html); e.style.visibility = 'hidden'; stage.appendChild(e); return e; });
    let showCaptions = true;

    function seek(t) {
      const sc = [[S1, -1, T.guess], [S2, T.guess, T.run], [S3, T.run, T.xray], [S4, T.xray, T.causes], [S5, T.causes, T.fix], [S6, T.fix, T.end], [S7, T.end, DUR + 1]];
      sc.forEach(([s, a, b]) => { const o = win(t, a, b, 0.45, 0.45); s.style.display = o > 0.001 ? 'block' : 'none'; s.style.opacity = o.toFixed(3); });

      // ----- the hook -----
      if (t < T.guess + 0.1) {
        const slide = lerp(240, 0, eio(seg(t, 3.5, 4.2)));
        put(chat, { dx: slide, o: eo(seg(t, 0, 0.35)) });
        rise(me, t, 0.15, 0.4);
        put(bot, { o: eo(seg(t, 0.85, 1.15)), dy: lerp(14, 0, eo(seg(t, 0.85, 1.25))) });
        const stream = seg(t, 1.6, 2.5), nw = Math.ceil(words.length * stream);
        wEls.forEach((w, i) => { w.style.opacity = i < nw ? 1 : 0; });
        rest.style.opacity = stream >= 1 ? 1 : 0;
        put(dots, { o: t < 1.6 ? 1 : 0 });
        [...dots.children].forEach((d, i) => { d.style.opacity = (0.35 + 0.65 * Math.max(0, Math.sin(t * 6 - i * 0.9))).toFixed(2); });
        hl.style.backgroundSize = `${(eio(seg(t, 2.7, 3.15)) * 100).toFixed(1)}% 3px`;
        const su = seg(t, 3.0, 3.35);
        const shake = t > 3.2 && t < 3.5 ? Math.sin((t - 3.2) * 70) * 3 * (1 - seg(t, 3.2, 3.5)) : 0;
        put(stamp, { o: eo(seg(t, 3.0, 3.1)), s: stampS(su), r: lerp(-14, -6, eo(su)), dx: slide + shake });
        rise(docCard, t, 4.1, 0.5);
      }

      // ----- the guessing loop -----
      if (t > T.guess - 0.1 && t < T.run + 0.1) {
        rise(gTitle, t, T.guess + 0.1, 0.45);
        rise(ed, t, T.guess + 0.25, 0.5);
        let k = G.findIndex((_, i) => t < gT[i + 1]); if (k < 0) k = G.length - 1;
        const u = (t - gT[k]) / GD[k];
        diffEls.forEach((x, i) => {
          if (i !== k) { x.e.style.display = 'none'; return; }
          x.e.style.display = 'block';
          x.e.style.opacity = (eo(seg(u, 0, 0.12)) * (1 - seg(u, 0.88, 1))).toFixed(3);
          x.e.style.transform = `translateY(${lerp(0, -16, seg(u, 0.88, 1)).toFixed(1)}px)`;
          x.to.textContent = x.txt.slice(0, Math.round(x.txt.length * seg(u, 0.1, 0.38)));
          x.e.children[1].style.textDecoration = u > 0.8 ? 'line-through' : 'none';
        });
        put(dep, { o: t >= G0 ? win(u, 0.4, 0.66, 0.04, 0.04) : 0 });
        spin.style.transform = `rotate(${(t * 540) % 360}deg)`;
        put(qChip, { o: t >= G0 ? win(u, 0.64, 0.96, 0.04, 0.06) : 0, s: lerp(0.8, 1, back(seg(u, 0.64, 0.74))) });
        put(dayK, { o: eo(seg(t, G0, G0 + 0.3)) });
        dayEl.textContent = DAYS[Math.min(k, DAYS.length - 1)];
        put(dayEl, { o: eo(seg(t, G0, G0 + 0.3)), dy: lerp(-10, 0, eo(seg(u, 0, 0.1))) });
        triedEls.forEach((e, i) => { const v = seg(t, gT[i] + GD[i] * 0.86, gT[i] + GD[i] * 0.86 + 0.2); put(e, { o: eo(v), dx: lerp(-14, 0, eo(v)) }); });
      }

      // ----- ragblame run -----
      if (t > T.run - 0.1 && t < T.xray + 0.1) {
        put(term, { o: eo(seg(t, R0 + 0.1, R0 + 0.55)), dy: lerp(20, 0, eo(seg(t, R0 + 0.1, R0 + 0.65))) });
        cmdEl.textContent = RN.cmd.slice(0, Math.round(RN.cmd.length * seg(t, R0 + 0.5, R0 + 1.6)));
        caret.style.display = t < L_T[0] ? 'inline-block' : 'none';
        caret.style.opacity = t > R0 + 0.5 && t < R0 + 1.6 ? 1 : (Math.floor(t * 2.5) % 2 ? 1 : 0.15);
        lineEls.forEach((e, i) => { e.style.display = t >= L_T[i] ? 'block' : 'none'; e.style.opacity = eo(seg(t, L_T[i], L_T[i] + 0.15)).toFixed(3); });
        tallyEls.forEach((e, i) => {
          const [a, d] = tallyAt[i];
          put(e, { o: eo(seg(t, a, a + 0.25)), dy: lerp(14, 0, eo(seg(t, a, a + 0.35))) });
          e.firstChild.textContent = Math.round(tallyItems[i][0] * eo(seg(t, a, a + d)));
        });
        rise(secsEl, t, tallyAt[3][0] + 0.3, 0.4);
      }

      // ----- the x-ray -----
      if (t > T.xray - 0.1 && t < T.causes + 0.1) {
        const x0 = T.xray;
        // when the passage reaches each stage: the first three as it travels, then the reranker, the prompt and the LLM
        const reach = [x0 + 1.0, x0 + 1.35, x0 + 1.7, x0 + 4.0, x0 + 4.6, x0 + 5.1];
        const RS = x0 + 2.6, RD = 1.5;
        const V0 = x0 + 6.1 + 0.55 + expEls.length * 0.95 + 0.25;
        pills.forEach((p, i) => {
          const a = x0 + 0.1 + i * 0.07, blam = i === 3 && t >= V0 + 0.25;
          put(p, { o: eo(seg(t, a, a + 0.35)), dy: lerp(-12, 0, eo(seg(t, a, a + 0.4))), s: blam ? lerp(1.1, 1, eo(seg(t, V0 + 0.25, V0 + 0.6))) : 1 });
        });
        rise(qcard, t, x0 + 0.35, 0.45);
        rise(rank, t, x0 + 0.7, 0.45);
        status.forEach(([kind, txt], i) => {
          const on = t >= reach[i];
          pillS[i].textContent = on ? (kind === 'ok' ? '✓ ' : '✗ ') + txt : '·';
          cls(pills[i], 'ok', on && kind === 'ok'); cls(pills[i], 'bad', on && kind === 'bad');
        });
        // the passage travels to the reranker and drops out there
        const tp = t < reach[2] ? 2 * seg(t, reach[0], reach[2]) : 2 + seg(t, reach[3] - 0.5, reach[3]);
        tracer.style.left = (PX + PW / 2 + tp * PP - 8).toFixed(1) + 'px';
        put(tracer, { o: win(t, reach[0] - 0.2, reach[3] + 0.6, 0.2, 0.4), dy: lerp(0, 44, eio(seg(t, reach[3], reach[3] + 0.6))) });
        // reorder: search order → reranked order, with the camera in close
        const u = eio(seg(t, RS, RS + RD));
        put(h0, { o: 1 - seg(t, RS, RS + 0.3) }); put(h1, { o: seg(t, RS + 0.15, RS + 0.45) });
        rows.forEach((r) => {
          const a = r.a < 0 ? PF.total : r.a, b = r.b < 0 ? PF.total : r.b;
          r.e.style.top = (lerp(a, b, u) * ROW).toFixed(1) + 'px';
          r.e.style.opacity = (r.a < 0 ? u : r.b < 0 ? 1 - u : 1).toFixed(3);
        });
        goldbar.style.top = (lerp(goldRow.a, goldRow.b, u) * ROW).toFixed(1) + 'px';
        put(goldbar, { o: eo(seg(t, x0 + 1.5, x0 + 1.8)) });
        cls(goldRow.e, 'gold', t >= x0 + 1.6);
        goldbar.style.background = lerp(goldRow.a, goldRow.b, u) >= PF.packed ? 'var(--bad)' : '';
        put(cut, { o: eo(seg(t, x0 + 1.3, x0 + 1.6)) }); put(inprompt, { o: eo(seg(t, x0 + 1.3, x0 + 1.6)) });
        rise(passage, t, x0 + 1.7, 0.45);
        rk.textContent = u < 0.5 ? `search #${PF.searchRank}` : `reranked #${PF.rerankRank}`;
        rk.style.color = u >= 0.5 ? 'var(--bad)' : 'var(--ok)';
        const z = eio(seg(t, RS - 0.7, RS)) * (1 - eio(seg(t, x0 + 5.0, x0 + 5.8)));
        const S = lerp(1, 1.34, z), fy = LY + 60 + 11 * ROW, fx = 580;
        cam.style.transform = `translate(${lerp(0, 960 - fx * S, z).toFixed(1)}px, ${lerp(0, 480 - fy * S, z).toFixed(1)}px) scale(${S.toFixed(4)})`;
        // the experiments, then the verdict stamp
        const E0 = x0 + 6.1;
        rise(exps, t, E0, 0.45);
        expEls.forEach((e, i) => {
          const a = E0 + 0.55 + i * 0.95, done = t >= a + 0.3;
          e.style.opacity = (0.25 + 0.75 * eo(seg(t, a, a + 0.3))).toFixed(3);
          const ck = e.querySelector('.ck');
          ck.style.background = done ? 'var(--ok)' : 'transparent'; ck.style.borderColor = done ? 'var(--ok)' : ''; ck.style.color = done ? '#06140C' : 'transparent';
          const res = e.querySelector('.res'); res.style.color = 'var(--ok)'; res.style.opacity = eo(seg(t, a + 0.3, a + 0.55)).toFixed(3);
        });
        const vu = seg(t, V0, V0 + 0.42);
        const vshake = t > V0 + 0.23 && t < V0 + 0.5 ? Math.sin((t - V0) * 80) * 3 * (1 - seg(t, V0 + 0.23, V0 + 0.5)) : 0;
        put(vstamp, { o: eo(seg(t, V0, V0 + 0.08)), s: stampS(vu), r: lerp(-12, -4, eo(vu)), dx: vshake });
        rise(vwhy, t, V0 + 0.5, 0.4);
        cls(pills[3], 'blamed', t >= V0 + 0.25);
      }

      // ----- the causes -----
      if (t > T.causes - 0.1 && t < T.fix + 0.1) {
        const c0 = T.causes;
        rise(cHead, t, c0 + 0.1, 0.45);
        const gu = eio(seg(t, c0 + 2.0, c0 + 3.1));
        ctiles.forEach((x, i) => {
          const a = c0 + 0.5 + i * 0.08, v = seg(t, a, a + 0.35);
          x.e.style.left = lerp(x.a, x.b, gu).toFixed(1) + 'px';
          put(x.e, { o: eo(v), s: lerp(0.4, 1, back(v)), dy: Math.sin(gu * Math.PI) * (i % 2 ? -28 : 28) });
        });
        clabels.forEach((e, i) => rise(e, t, c0 + 3.0 + i * 0.1, 0.4));
        if (calNote) rise(calNote, t, c0 + 3.8, 0.4);
      }

      // ----- the fix -----
      if (FX && t > T.fix - 0.1 && t < T.end + 0.1) {
        const f0 = T.fix, HI = f0 + 8.2;
        put(fterm, { o: eo(seg(t, f0 + 0.1, f0 + 0.5)) });
        fCmd.textContent = fixCmd.slice(0, Math.round(fixCmd.length * seg(t, f0 + 0.4, f0 + 1.3)));
        fCaret.style.display = t < f0 + 1.5 ? 'inline-block' : 'none';
        if (fDiff) put(fDiff, { o: eo(seg(t, f0 + 1.5, f0 + 1.8)) });
        const flipAt = (x) => { const fi = fixedT.indexOf(x), bi = brokeT.indexOf(x); return fi >= 0 ? f0 + 2.8 + fi * 0.32 : bi >= 0 ? f0 + 5.4 + bi * 0.35 : Infinity; };
        tiles.forEach((x, i) => {
          const a = f0 + 1.2 + (i % COLS) * 0.03 + Math.floor(i / COLS) * 0.06;
          const ft = flipAt(x), fu = seg(t, ft, ft + 0.36), flipped = fu >= 0.5;
          const ok = flipped ? x.q.fixed : x.q.label === 'correct';
          cls(x.e, 'ok', ok); cls(x.e, 'bad', !ok);
          if (ft < Infinity) x.c.textContent = flipped ? (x.q.fixed ? 'fixed' : 'broke') : (x.q.cause ? x.q.cause.replace('_', ' ') : x.q.label === 'correct' ? 'right' : x.q.label);
          // the closing beat: the ranking and packing failures are ringed; the rest of the grid steps back
          const ring = t >= HI && ['ranking', 'packing'].includes(x.q.cause);
          cls(x.e, 'hi', ring);
          const keep = ring || ['index', 'embedding', 'generation'].includes(x.q.cause);
          put(x.e, { o: eo(seg(t, a, a + 0.3)) * (keep ? 1 : lerp(1, 0.32, seg(t, HI, HI + 0.4))), s: 1, sy: Math.abs(Math.cos(fu * Math.PI)) });
        });
        const nowFixed = fixedT.filter((x) => t >= flipAt(x) + 0.18).length, nowBroke = brokeT.filter((x) => t >= flipAt(x) + 0.18).length;
        rise(counter, t, f0 + 1.6, 0.45);
        nfEl.textContent = nowFixed; nbEl.textContent = nowBroke;
        const roll = seg(t, f0 + 6.7, f0 + 7.5);
        cv.textContent = Math.round(lerp(FX.before.correct, FX.after.correct, roll));
        cv.style.color = roll > 0 ? 'var(--ok)' : '';
        put(cv, { s: roll > 0 && roll < 1 ? 1.06 : lerp(1.06, 1, eo(seg(t, f0 + 7.5, f0 + 7.8))) });
        rise(perEl, t, HI - 0.2, 0.45);
        perRows.forEach((r, i) => { r.style.opacity = eo(seg(t, HI + i * 0.12, HI + 0.3 + i * 0.12)).toFixed(3); });
      }

      // ----- end card -----
      if (t > T.end - 0.1) {
        const E = T.end + 0.3;
        put(endMark, { o: eo(seg(t, E, E + 0.5)), s: lerp(0.7, 1, back(seg(t, E, E + 0.7))) });
        put(endName, { o: eo(seg(t, E + 0.25, E + 0.75)), dy: lerp(20, 0, eo(seg(t, E + 0.25, E + 0.85))) });
        put(endTag, { o: eo(seg(t, E + 0.8, E + 1.3)), dy: lerp(14, 0, eo(seg(t, E + 0.8, E + 1.4))) });
        put(endFoot, { o: eo(seg(t, E + 1.5, E + 2.0)), dy: lerp(14, 0, eo(seg(t, E + 1.5, E + 2.1))) });
      }

      CAPS.forEach(([a, b], i) => { const u = showCaptions ? win(t, a, b, 0.3, 0.3) : 0; put(capEls[i], { o: u, dy: lerp(8, 0, eo(seg(t, a, a + 0.4))) }); });
      return Promise.resolve();
    }

    return { seek, duration: DUR, stage, captions(on) { showCaptions = on; } };
  }

  window.RagblameFilm = { mount };
})();
