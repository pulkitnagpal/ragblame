// Screenshots the top of a real ragblame report for the README: the header, the metrics, the cause bar (and the fix
// section when the run has one) down to the end of the first failure card. 1600 px wide, light theme.
//   uv run --no-sync ragblame report runs/2026-11-14T20-01-57
//   node film/report-shot.mjs [runs/2026-11-14T20-01-57] [docs/img/report.png]
import { writeFileSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { withBrowser } from './cdp.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const run = resolve(ROOT, process.argv[2] || 'runs/2026-11-14T20-01-57');
const out = resolve(ROOT, process.argv[3] || 'docs/img/report.png');
const W = 1600;

await withBrowser(async (open) => {
  const tab = await open(pathToFileURL(join(run, 'report.html')).href, { width: W, height: 1200 });
  await tab.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'light' }] });
  const bottom = await tab.eval(`(() => { const c = document.querySelector('.failure-card'); if (!c) throw new Error('no failure card');
    return Math.ceil(c.getBoundingClientRect().bottom + window.scrollY + 12); })()`);
  await tab.resize(W, bottom);
  await new Promise((r) => setTimeout(r, 300));
  writeFileSync(out, await tab.shot({ clip: { x: 0, y: 0, width: W, height: bottom } }));
  console.log(`wrote ${out} (${W}×${bottom})`);
});
