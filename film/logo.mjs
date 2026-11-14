// Writes the ragblame logo files to docs/img: logo.svg / logo.png (mark + wordmark, dark text for light pages),
// logo-dark.svg / logo-dark.png (light text, for dark pages) and logo-mark.svg / logo-mark.png (square).
//   node film/logo.mjs
import { writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import opentype from 'opentype.js';
import { withBrowser } from './cdp.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, '..', 'docs', 'img');
export const INK = '#14151A', ACCENT = '#F5A524', LIGHT = '#ECEBE6';

/** The mark in a 512 box: a dark tile with five lines of a document; one line is blamed (accent, with a gutter tick). */
export function markBody() {
  const lines = [[150, 232], [210, 176], [270, 212], [330, 248], [390, 132]];
  return `<rect width="512" height="512" rx="112" fill="${INK}"/>
<g stroke-linecap="round" stroke-width="30" fill="none">
${lines.map(([y, w], i) => `  <path d="M166 ${y}h${w}" stroke="${i === 2 ? ACCENT : '#4B4F59'}"/>`).join('\n')}
</g>
<rect x="102" y="248" width="20" height="44" rx="10" fill="${ACCENT}"/>`;
}
const markSvg = (size = 512) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="${size}" height="${size}"><title>ragblame</title>${markBody()}</svg>`;

function wordmark() {
  const font = opentype.loadSync(join(HERE, 'fonts', 'Inter-800.ttf'));
  const size = 168, tracking = -0.035 * size;
  let x = 0; const parts = [];
  const glyphs = font.stringToGlyphs('ragblame');
  glyphs.forEach((g, i) => {
    parts.push(g.getPath(x, 0, size).toPathData(2));
    x += g.advanceWidth * size / font.unitsPerEm + tracking;
    if (i < glyphs.length - 1) x += font.getKerningValue(g, glyphs[i + 1]) * size / font.unitsPerEm;
  });
  // x-height centres a lowercase wordmark on the mark better than the cap height does
  return { d: parts.join(''), width: x - tracking, xH: font.tables.os2.sxHeight * size / font.unitsPerEm };
}

export function logoSvg(text = INK) {
  const w = wordmark();
  const H = 248, mark = 200, gap = 52, pad = 24;
  const W = Math.ceil(pad + mark + gap + w.width + pad);
  const baseline = H / 2 + w.xH / 2 + 4;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><title>ragblame</title>
<g transform="translate(${pad} ${(H - mark) / 2}) scale(${mark / 512})">${markBody()}</g>
<path fill="${text}" transform="translate(${pad + mark + gap} ${baseline.toFixed(2)})" d="${w.d}"/></svg>`;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const files = { 'logo.svg': logoSvg(INK), 'logo-dark.svg': logoSvg(LIGHT), 'logo-mark.svg': markSvg() };
  for (const [f, s] of Object.entries(files)) writeFileSync(join(OUT, f), s + '\n');
  const [, W, H] = files['logo.svg'].match(/viewBox="0 0 (\d+) (\d+)"/).map(Number);
  await withBrowser(async (open) => {
    const shot = async (file, w, h, scale) => (await open(pathToFileURL(join(OUT, file)).href, { width: w, height: h, scale, transparent: true })).shot();
    writeFileSync(join(OUT, 'logo.png'), await shot('logo.svg', W, H, 2));
    writeFileSync(join(OUT, 'logo-dark.png'), await shot('logo-dark.svg', W, H, 2));
    writeFileSync(join(OUT, 'logo-mark.png'), await shot('logo-mark.svg', 512, 512, 2));
  });
  console.log('wrote docs/img/logo.svg, logo.png, logo-dark.svg, logo-dark.png, logo-mark.svg, logo-mark.png');
}
