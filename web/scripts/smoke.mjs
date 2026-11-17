// Loads a page in the installed Chrome and saves screenshots.
// Usage: node scripts/smoke.mjs <url> <out.png> [steps...]
// A step is either a number of seconds to wait or JavaScript to evaluate;
// a step of the form "shot:<file>" saves a screenshot along the way.
import { chromium } from 'playwright-core';

const [url, out, ...steps] = process.argv.slice(2);
const browser = await chromium.launch({
  channel: 'chrome',
  headless: !process.env.HEADED,
  args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-gpu'],
});
const [width, height] = (process.env.VIEWPORT ?? '1280x720').split('x').map(Number);
const page = await browser.newPage({
  colorScheme: process.env.SCHEME ?? "light",
  viewport: { width, height },
  deviceScaleFactor: Number(process.env.DPR ?? 1),
  hasTouch: !!process.env.TOUCH,
});
page.on('console', (message) => {
  if (message.type() !== 'debug') console.log(`[${message.type()}]`, message.text());
});
page.on("pageerror", (error) => console.log("[pageerror]", error.message));
if (process.env.NOGPU) await page.addInitScript(() => delete Navigator.prototype.gpu);
// Hides the filtering of 32-bit floats, as a phone would not have it.
if (process.env.PLAIN)
  await page.addInitScript(() => {
    const features = Object.getOwnPropertyDescriptor(GPUAdapter.prototype, "features").get;
    Object.defineProperty(GPUAdapter.prototype, "features", {
      get() {
        const offered = new Set(features.call(this));
        offered.delete("float32-filterable");
        return offered;
      },
    });
  });
await page.goto(url);
for (const step of steps) {
  if (/^[\d.]+$/.test(step)) await page.waitForTimeout(1000 * Number(step));
  else if (step.startsWith("shot:")) await page.screenshot({ path: step.slice(5) });
  else if (step.startsWith("click:")) await page.click(step.slice(6));
  else if (step.startsWith("swipe:") || step.startsWith("pinch:")) {
    // swipe:<x0>,<y0>,<x1>,<y1> moves one finger; pinch:<x>,<y>,<span0>,<span1>
    // moves two apart or together around a point.
    const [a, b, c, d] = step.slice(6).split(",").map(Number);
    const fingers = step.startsWith("swipe:")
      ? (t) => [{ x: a + (c - a) * t, y: b + (d - b) * t, id: 0 }]
      : (t) => [-1, 1].map((side, id) => ({ x: a + side * 0.5 * (c + (d - c) * t), y: b, id }));
    const touch = await page.context().newCDPSession(page);
    await touch.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: fingers(0) });
    for (let i = 1; i <= 15; i++) {
      await touch.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: fingers(i / 15) });
      await page.waitForTimeout(16);
    }
    await touch.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  }
  else if (step.startsWith("wheel:")) {
    // wheel:<x>,<y>,<deltaY>[,ctrl]
    const [x, y, delta, ctrl] = step.slice(6).split(",");
    await page.mouse.move(Number(x), Number(y));
    if (ctrl) await page.keyboard.down("Control");
    await page.mouse.wheel(0, Number(delta));
    if (ctrl) await page.keyboard.up("Control");
  } else if (step.startsWith("drag:")) {
    // drag:<x0>,<y0>,<x1>,<y1>[,right]
    const [x0, y0, x1, y1, button] = step.slice(5).split(",");
    await page.mouse.move(Number(x0), Number(y0));
    await page.mouse.down({ button: button || "left" });
    await page.mouse.move(Number(x1), Number(y1), { steps: 12 });
    await page.mouse.up({ button: button || "left" });
  }
  else console.log('=>', JSON.stringify(await page.evaluate(step)));
}
await page.screenshot({ path: out });
await browser.close();
