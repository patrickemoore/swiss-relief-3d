// Records a flight in the demo page as an animated GIF for the README.
// Usage: node flight-gif.mjs <url> <out.gif>
// Needs the dev server running; see README.
import { writeFileSync } from 'node:fs';
import gifenc from 'gifenc';
import { chromium } from 'playwright-core';
import pngjs from 'pngjs';

const { GIFEncoder, applyPalette, quantize } = gifenc;
const { PNG } = pngjs;

const [url, out] = process.argv.slice(2);
const WIDTH = 640;
const HEIGHT = 360;
const FPS = 8;
const SCRIPT = [
  { at: 0.0, run: "swissRelief.flyTo('lake_lucerne')" },
  { at: 6.5, run: "swissRelief.flyTo('matterhorn')" },
];
const SECONDS = 12.5;

const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-gpu'],
});
const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
await page.goto(url);
await page.waitForTimeout(7000);

const gif = GIFEncoder();
const frames = Math.round(SECONDS * FPS);
const start = Date.now();
let next = 0;
for (let i = 0; i < frames; i++) {
  const t = (Date.now() - start) / 1000;
  while (next < SCRIPT.length && SCRIPT[next].at <= t) {
    if (SCRIPT[next].run) await page.evaluate(SCRIPT[next].run);
    next++;
  }
  const png = PNG.sync.read(await page.screenshot({ type: 'png' }));
  const rgba = new Uint8Array(png.data.buffer, png.data.byteOffset, png.data.length);
  const palette = quantize(rgba, 256, { format: 'rgb444' });
  const index = applyPalette(rgba, palette, 'rgb444');
  gif.writeFrame(index, WIDTH, HEIGHT, { palette, delay: 1000 / FPS });
  const due = start + ((i + 1) * 1000) / FPS;
  const wait = due - Date.now();
  if (wait > 0) await page.waitForTimeout(wait);
}
gif.finish();
writeFileSync(out, gif.bytes());
console.log(`${frames} frames -> ${out} (${(gif.bytes().length / 1e6).toFixed(1)} MB)`);
await browser.close();
