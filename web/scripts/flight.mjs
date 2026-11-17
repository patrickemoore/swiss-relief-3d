// Records a flight in the demo page as an animated WebP for the README.
// Usage: node scripts/flight.mjs <url> <out.webp>
// Needs the dev server running.
//
// The page's clock is driven from here: between two captured frames the
// map advances by exactly one frame's worth of time, however long the
// screenshot took, and the tiles it streams get real time to arrive. The
// result plays at the pace a visitor sees, with the detail they see.
import { chromium } from 'playwright-core';
import sharp from 'sharp';

const [url, out] = process.argv.slice(2);
const WIDTH = 960;
const HEIGHT = 540;
const FPS = 10;
const SECONDS = 15;
// What happens, and when, in seconds from the first frame.
const SCRIPT = [
  { at: 0.3, run: "swissRelief.flyTo('lake_lucerne')" },
  { at: 7.0, run: "swissRelief.flyTo('matterhorn')" },
];
// Real milliseconds given to downloads between frames.
const SETTLE_MS = 40;

const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-gpu'],
});
const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
await page.clock.install();
await page.goto(url);

// Let the map load and settle: real time for the downloads, page time for
// the frames that follow them.
for (let i = 0; i < 50; i++) {
  await page.waitForTimeout(100);
  await page.clock.runFor(100);
}

const frames = Math.round(SECONDS * FPS);
const shots = [];
let next = 0;
for (let i = 0; i < frames; i++) {
  const t = i / FPS;
  while (next < SCRIPT.length && SCRIPT[next].at <= t) await page.evaluate(SCRIPT[next++].run);
  await page.clock.runFor(1000 / FPS);
  await page.waitForTimeout(SETTLE_MS);
  shots.push(await page.screenshot({ type: 'png' }));
}
await browser.close();

// The frames are stacked; sharp cuts the stack into pages.
const raw = Buffer.concat(await Promise.all(shots.map((png) => sharp(png).raw().toBuffer())));
await sharp(raw, { raw: { width: WIDTH, height: HEIGHT * frames, channels: 3, pageHeight: HEIGHT } })
  .webp({ quality: 86, smartSubsample: true, effort: 5, loop: 0, delay: Math.round(1000 / FPS) })
  .toFile(out);
console.log(`${frames} frames -> ${out}`);
