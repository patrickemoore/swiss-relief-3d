// Records a flight in the demo page as an animated WebP for the README.
// Usage: node scripts/flight.mjs <url> <out.webp>
// Needs the dev server running.
import { chromium } from 'playwright-core';
import sharp from 'sharp';

const [url, out] = process.argv.slice(2);
const WIDTH = 1040;
const HEIGHT = 585;
const FPS = 12;
const SECONDS = 13;
// What happens, and when, in seconds from the first frame.
const SCRIPT = [
  { at: 0.5, run: "swissRelief.flyTo('lake_lucerne')" },
  { at: 7.0, run: "swissRelief.flyTo('matterhorn')" },
];

const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-gpu'],
});
const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
await page.goto(url);
await page.waitForTimeout(7000);

// Frames are taken as they come and stacked; sharp cuts the stack into pages.
const frames = Math.round(SECONDS * FPS);
const shots = [];
const start = Date.now();
let next = 0;
for (let i = 0; i < frames; i++) {
  const t = (Date.now() - start) / 1000;
  while (next < SCRIPT.length && SCRIPT[next].at <= t) await page.evaluate(SCRIPT[next++].run);
  shots.push(await page.screenshot({ type: 'png' }));
  const due = start + ((i + 1) * 1000) / FPS;
  const wait = due - Date.now();
  if (wait > 0) await page.waitForTimeout(wait);
}
const elapsed = Date.now() - start;
await browser.close();

// Screenshots take what they take; the pages play back at the pace they
// were taken, so the flights run in real time.
const raw = Buffer.concat(await Promise.all(shots.map((png) => sharp(png).raw().toBuffer())));
await sharp(raw, { raw: { width: WIDTH, height: HEIGHT * frames, channels: 3, pageHeight: HEIGHT } })
  .webp({ quality: 74, effort: 4, loop: 0, delay: Math.round(elapsed / frames) })
  .toFile(out);
console.log(`${frames} frames -> ${out}`);
