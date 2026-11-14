// Renders the design boards for the launch thread: key frames of the film, each a 1920×1080 still of the same stage.
//   node film/boards.mjs     → media/boards/NN-name.png
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { withBrowser } from './cdp.mjs';

const HERE = dirname(fileURLToPath(import.meta.url)), ROOT = join(HERE, '..');
export const BOARDS = [
  { t: 5.4, slug: 'confidently-wrong', title: 'Before: confidently wrong' },
  { t: 27.2, slug: 'xray-rank-3-to-13', title: 'With ragblame: the x-ray' },
  { t: 35.4, slug: 'verdict-ranking', title: 'With ragblame: the proof' },
  { t: 53.2, slug: 'fix-27-to-31', title: 'After: the fix, measured' },
];

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const out = join(ROOT, 'media', 'boards');
  mkdirSync(out, { recursive: true });
  await withBrowser(async (open) => {
    const tab = await open(pathToFileURL(join(ROOT, 'dist', 'film', 'index.html')).href, { width: 1920, height: 1080 });
    await tab.eval('window.ready');
    for (const [i, b] of BOARDS.entries()) {
      await tab.eval(`film.seek(${b.t})`);
      const f = join(out, `${String(i + 1).padStart(2, '0')}-${b.slug}.png`);
      writeFileSync(f, await tab.shot());
      console.log(f);
    }
  });
}
