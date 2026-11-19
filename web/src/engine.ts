// Everything between the page and the WebAssembly viewer: loading, sizing,
// the frame loop and input. Nothing here knows about React.

import type { PlacedLabel } from './labels';
import init, { Viewer } from './wasm/relief_web.js';

export interface Place {
  id: string;
  name: string;
}

/** What the viewer reports about its view after each frame it draws. */
export interface ViewState {
  /** Compass direction the camera looks in, radians clockwise from north. */
  heading: number;
  /** Radians below the horizon; pi/2 looks straight down. */
  pitch: number;
  /** Metres from the camera to the middle of the view. */
  distance: number;
  /** Ground metres per CSS pixel at the middle of the view. */
  metresPerPixel: number;
  longitude: number;
  latitude: number;
  /** Terrain height at the middle of the view, in metres. */
  elevation: number;
  /** The place the camera was last sent to, until it is steered away. */
  place: string | null;
  flying: boolean;
  touring: boolean;
  /** Terrain tiles still on their way. */
  loading: number;
}

export interface Insets {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/** How smoothly the map has been drawing; see `Engine.frameStats`. */
export interface FrameStats {
  /** Frames drawn. */
  frames: number;
  /** Median and longest time between frames drawn in a row, in milliseconds. */
  median: number;
  longest: number;
  /** Device pixels per CSS pixel the map is drawn at. */
  pixelRatio: number;
}

/** The two renderings a viewer holds: 0 is shown, 1 is what it is compared with. */
export type Look = 0 | 1;
export type LookName = 'relief' | 'realistic' | 'draped' | 'world-fixed' | 'headlight';

export interface EngineOptions {
  /** Where the packed terrain is served from. */
  dataUrl: string;
  /** Where the WebAssembly module is served from, if not next to the script. */
  wasmUrl?: string | URL;
  /** Upper limit on device pixels per CSS pixel. */
  maxPixelRatio: number;
  /**
   * Whether the map shares the page with other content right now. If so the
   * wheel only zooms with Ctrl or Cmd held and one finger scrolls the page.
   */
  cooperative: () => boolean;
  /**
   * How much of the map the page's controls cover at each edge, in CSS
   * pixels. The view of the whole country keeps clear of them.
   */
  insets?: () => Insets;
  /** Progress of the initial download, 0 to 1. */
  onProgress?: (fraction: number) => void;
  /** Called after every frame that was drawn. */
  onFrame?: (state: ViewState) => void;
  /** The names to show over the frame, in device pixels, with the pixel ratio. */
  onLabels?: (placed: PlacedLabel[], ratio: number) => void;
  /** The first frames are out: the map is worth showing. */
  onShown?: () => void;
  /** The visitor did something that steers the map. */
  onInteract?: () => void;
  /** A wheel or one-finger gesture was left to the page. */
  onGestureLeftToPage?: (gesture: 'wheel' | 'touch') => void;
}

/** One wheel notch in the pixels a browser reports for it. */
const PIXELS_PER_WHEEL_STEP = 100;
const LINES_PER_WHEEL_STEP = 3;
/** A pinch on a trackpad arrives as small wheel deltas with Ctrl set. */
const PINCH_PIXELS_PER_WHEEL_STEP = 12;
/** Wheel steps a double click zooms in by: a factor of two. */
const DOUBLE_CLICK_STEPS = 3.85;
/** Factor the zoom keys and buttons zoom by. */
export const ZOOM_FACTOR = 1.6;
/** Frames drawn before the map is shown. */
const FRAMES_BEFORE_SHOWING = 3;
/** Largest canvas side in device pixels, whatever the screen. */
const MAX_CANVAS_SIDE = 3840;

// Quality follows the frame rate: the canvas resolution is lowered when
// frames take too long and raised again when there is time to spare.
const SLOW_FRAME_MS = 24;
const FAST_FRAME_MS = 12.5;
const QUALITY_STEP = 0.85;
const MIN_QUALITY = 0.5;
const FRAMES_PER_QUALITY_CHECK = 40;
const MS_BETWEEN_QUALITY_CHANGES = 2500;

let wasm: Promise<unknown> | undefined;

/** Loads the WebAssembly module once per page. */
function loadWasm(url: string | URL | undefined): Promise<unknown> {
  // A bundler that sees this resolves the file next to the script.
  const source = url ?? new URL(/* @vite-ignore */ './wasm/relief_web_bg.wasm', import.meta.url);
  wasm ??= init({ module_or_path: source }).catch((error: unknown) => {
    wasm = undefined;
    throw error;
  });
  return wasm;
}

/** Fetches a file, reporting how much of it has arrived. */
async function download(url: string, onProgress: (fraction: number) => void): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());

  const bytes = new Uint8Array(total);
  const reader = response.body.getReader();
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // A server that compresses on the fly sends more than it announced.
    if (received + value.length > bytes.length) return concat(bytes.subarray(0, received), value, reader);
    bytes.set(value, received);
    received += value.length;
    onProgress(received / total);
  }
  return bytes.subarray(0, received);
}

async function concat(
  head: Uint8Array,
  next: Uint8Array,
  reader: ReadableStreamDefaultReader<Uint8Array>,
): Promise<Uint8Array> {
  const chunks = [head, next];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }
  const whole = new Uint8Array(chunks.reduce((length, chunk) => length + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) {
    whole.set(chunk, offset);
    offset += chunk.length;
  }
  return whole;
}

/** Whether this browser can run the viewer at all. */
export function supported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

interface TouchPoint {
  x: number;
  y: number;
}

export class Engine {
  readonly places: Place[];
  private readonly listeners = new AbortController();
  private readonly resizes: ResizeObserver;
  private readonly sightings: IntersectionObserver;
  private animation = 0;
  private inView = true;
  private destroyed = false;

  /** Device pixels per CSS pixel the canvas is drawn at. */
  private ratio = 1;
  private quality = 1;
  private lastDrawn = 0;
  private frameTimes: number[] = [];
  private qualityChanged = 0;
  private measured: number[] = [];
  /** Frames drawn so far, up to the point where the map is shown. */
  private drawn = 0;

  private pointer: { id: number; x: number; y: number; rotate: boolean } | undefined;
  private touches: TouchPoint[] = [];
  private readonly keys = new Set<string>();

  private constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly viewer: Viewer,
    private readonly options: EngineOptions,
  ) {
    this.places = JSON.parse(viewer.places()) as Place[];
    this.resizes = new ResizeObserver(() => this.resize());
    this.resizes.observe(canvas);
    this.sightings = new IntersectionObserver(([entry]) => {
      this.inView = entry.isIntersecting;
      this.run();
    });
    this.sightings.observe(canvas);
    this.listen();
    this.resize();
    this.applyInsets();
    viewer.go_home(false);
    this.run();
  }

  /** Downloads the terrain and opens a viewer on the canvas. */
  static async create(canvas: HTMLCanvasElement, options: EngineOptions): Promise<Engine> {
    if (!supported()) throw new Error('This browser does not support WebGPU.');
    const dataUrl = options.dataUrl.replace(/\/+$/, '');
    const progress = options.onProgress ?? (() => {});
    const [, meta, base] = await Promise.all([
      loadWasm(options.wasmUrl),
      fetch(`${dataUrl}/meta.json`).then((response) => {
        if (!response.ok) throw new Error(`${dataUrl}/meta.json: HTTP ${response.status}`);
        return response.text();
      }),
      download(`${dataUrl}/base.bin`, progress),
    ]);
    progress(1);
    // A terrain with hiking trails has an index of their tiles next to it.
    let trailsIndex: Uint8Array | undefined;
    if ((JSON.parse(meta) as { trails?: unknown }).trails) {
      const response = await fetch(`${dataUrl}/trails/index.bin`);
      if (response.ok) trailsIndex = new Uint8Array(await response.arrayBuffer());
      else console.warn(`[swiss-relief] ${dataUrl}/trails/index.bin: HTTP ${response.status}; no trails`);
    }

    // The viewer reads the canvas's size when it opens.
    const ratio = Math.min(window.devicePixelRatio || 1, options.maxPixelRatio);
    const box = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(box.width * ratio));
    canvas.height = Math.max(1, Math.round(box.height * ratio));
    const layoutSide = Math.round(Math.max(box.width, box.height));
    const viewer = await Viewer.create(canvas, new URL(dataUrl, document.baseURI).href, meta, base, layoutSide, trailsIndex);
    return new Engine(canvas, viewer, options);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    cancelAnimationFrame(this.animation);
    this.listeners.abort();
    this.resizes.disconnect();
    this.sightings.disconnect();
    this.viewer.free();
  }

  // --- Commands ------------------------------------------------------------

  flyTo(place: string): void {
    this.viewer.fly_to(place);
  }

  /** Flies back to the view of the whole country. */
  flyHome(): void {
    this.applyInsets();
    this.viewer.go_home(true);
  }

  zoomBy(factor: number): void {
    this.viewer.zoom_by(factor);
  }

  faceNorth(): void {
    this.viewer.face_north();
  }

  toggleTilt(): void {
    this.viewer.toggle_tilt();
  }

  setTour(on: boolean): void {
    this.viewer.set_tour(on);
  }

  setAttract(on: boolean): void {
    this.viewer.set_attract(on);
  }

  /** Shows look 1 on the right from this share of the width, or not at all. */
  setCompare(split: number | undefined): void {
    this.viewer.set_compare(split);
  }

  setLook(look: Look, name: LookName): void {
    this.viewer.set_look(look, name);
  }

  /** Changes some parameters of a look; see the viewer's presets for names. */
  patchLook(look: Look, patch: Record<string, unknown>): void {
    this.viewer.patch_look(look, JSON.stringify(patch));
  }

  /** Whether the terrain came with hiking trails. */
  get hasTrails(): boolean {
    return this.viewer.has_trails();
  }

  /** Shows or hides the hiking trails. */
  setTrails(on: boolean): void {
    this.viewer.set_trails(on);
  }

  /** Megabytes of terrain tiles downloaded so far. */
  get downloadedMegabytes(): number {
    return this.viewer.fetched_megabytes();
  }

  /** How smoothly the map has drawn since this was last asked. */
  frameStats(): FrameStats {
    const times = this.measured.sort((a, b) => a - b);
    this.measured = [];
    return {
      frames: times.length,
      median: times[times.length >> 1] ?? 0,
      longest: times[times.length - 1] ?? 0,
      pixelRatio: this.ratio,
    };
  }

  private applyInsets(): void {
    const insets = this.options.insets?.() ?? { top: 0, right: 0, bottom: 0, left: 0 };
    const scaled = [insets.top, insets.right, insets.bottom, insets.left].map((inset) => inset * this.ratio);
    this.viewer.set_insets(scaled[0], scaled[1], scaled[2], scaled[3]);
  }

  // --- The frame loop ------------------------------------------------------

  /** Starts the loop if the map can be seen, and lets it end if not. */
  private run(): void {
    cancelAnimationFrame(this.animation);
    if (this.destroyed || !this.inView || document.hidden) {
      this.lastDrawn = 0;
      return;
    }
    this.animation = requestAnimationFrame((time) => this.frame(time));
  }

  private frame(time: number): void {
    let drew = false;
    try {
      drew = this.viewer.frame(time);
    } catch (error) {
      console.error('[swiss-relief]', error);
      return;
    }
    if (drew) {
      const state = JSON.parse(this.viewer.state()) as ViewState;
      state.metresPerPixel *= this.ratio;
      this.options.onFrame?.(state);
      this.options.onLabels?.(JSON.parse(this.viewer.labels()) as PlacedLabel[], this.ratio);
      // The very first frames build pipelines and stall; the page shows the
      // map once they are behind it.
      if (this.drawn < FRAMES_BEFORE_SHOWING && ++this.drawn === FRAMES_BEFORE_SHOWING) this.options.onShown?.();
      this.adapt(time);
    } else {
      this.lastDrawn = 0;
    }
    this.run();
  }

  /** Follows the frame rate with the resolution. */
  private adapt(time: number): void {
    if (this.lastDrawn) {
      this.frameTimes.push(time - this.lastDrawn);
      if (this.measured.length < 100_000) this.measured.push(time - this.lastDrawn);
    }
    this.lastDrawn = time;
    if (this.frameTimes.length < FRAMES_PER_QUALITY_CHECK) return;
    const sorted = this.frameTimes.sort((a, b) => a - b);
    const median = sorted[sorted.length >> 1];
    this.frameTimes = [];
    if (time - this.qualityChanged < MS_BETWEEN_QUALITY_CHANGES) return;

    let quality = this.quality;
    if (median > SLOW_FRAME_MS) quality = Math.max(MIN_QUALITY, quality * QUALITY_STEP);
    else if (median < FAST_FRAME_MS) quality = Math.min(1, quality / QUALITY_STEP);
    if (quality !== this.quality) {
      this.quality = quality;
      this.qualityChanged = time;
      this.resize();
    }
  }

  private resize(): void {
    const box = this.canvas.getBoundingClientRect();
    if (box.width < 1 || box.height < 1) return;
    const wanted = Math.min(window.devicePixelRatio || 1, this.options.maxPixelRatio) * this.quality;
    this.ratio = Math.min(wanted, MAX_CANVAS_SIDE / Math.max(box.width, box.height));
    this.viewer.set_pixel_ratio(this.ratio);
    const width = Math.max(1, Math.round(box.width * this.ratio));
    const height = Math.max(1, Math.round(box.height * this.ratio));
    if (width === this.canvas.width && height === this.canvas.height) return;
    this.canvas.width = width;
    this.canvas.height = height;
    // The insets are in device pixels, which may just have changed size.
    this.applyInsets();
    this.viewer.resize(width, height);
  }

  // --- Input ---------------------------------------------------------------

  private listen(): void {
    const { canvas } = this;
    const options = { signal: this.listeners.signal };
    const active = { signal: this.listeners.signal, passive: false };

    canvas.addEventListener('pointerdown', (event) => this.pointerDown(event), options);
    canvas.addEventListener('pointermove', (event) => this.pointerMove(event), options);
    canvas.addEventListener('pointerup', (event) => this.pointerUp(event), options);
    canvas.addEventListener('pointercancel', (event) => this.pointerUp(event), options);
    canvas.addEventListener('pointerleave', () => this.viewer.pointer_left(), options);
    canvas.addEventListener('contextmenu', (event) => event.preventDefault(), options);
    canvas.addEventListener('dblclick', (event) => this.doubleClick(event), options);
    canvas.addEventListener('wheel', (event) => this.wheel(event), active);

    canvas.addEventListener('touchstart', (event) => this.touch(event), active);
    canvas.addEventListener('touchmove', (event) => this.touch(event), active);
    canvas.addEventListener('touchend', (event) => this.touch(event), active);
    canvas.addEventListener('touchcancel', (event) => this.touch(event), active);

    canvas.addEventListener('keydown', (event) => this.key(event, true), options);
    canvas.addEventListener('keyup', (event) => this.key(event, false), options);
    canvas.addEventListener('blur', () => this.releaseKeys(), options);
    document.addEventListener('visibilitychange', () => this.run(), options);
  }

  /** A position on the canvas in the device pixels the viewer counts in. */
  private place(event: { clientX: number; clientY: number }): TouchPoint {
    const box = this.canvas.getBoundingClientRect();
    return { x: (event.clientX - box.left) * this.ratio, y: (event.clientY - box.top) * this.ratio };
  }

  private pointerDown(event: PointerEvent): void {
    // Fingers are handled as touches, where the page may want them instead.
    if (event.pointerType === 'touch' || this.pointer) return;
    const rotate = event.button !== 0 || event.ctrlKey || event.shiftKey;
    this.pointer = { id: event.pointerId, x: event.clientX, y: event.clientY, rotate };
    this.canvas.setPointerCapture(event.pointerId);
    this.canvas.focus({ preventScroll: true });
    this.viewer.set_dragging(true);
    this.options.onInteract?.();
  }

  private pointerMove(event: PointerEvent): void {
    if (event.pointerType === 'touch') return;
    const at = this.place(event);
    this.viewer.pointer_at(at.x, at.y);
    const pointer = this.pointer;
    if (pointer?.id !== event.pointerId) return;
    const [dx, dy] = [event.clientX - pointer.x, event.clientY - pointer.y];
    this.viewer.drag(dx * this.ratio, dy * this.ratio, pointer.rotate);
    pointer.x = event.clientX;
    pointer.y = event.clientY;
  }

  private pointerUp(event: PointerEvent): void {
    if (this.pointer?.id !== event.pointerId) return;
    this.pointer = undefined;
    this.viewer.set_dragging(false);
  }

  private doubleClick(event: MouseEvent): void {
    const at = this.place(event);
    this.viewer.pointer_at(at.x, at.y);
    this.viewer.wheel(event.shiftKey ? -DOUBLE_CLICK_STEPS : DOUBLE_CLICK_STEPS);
    this.options.onInteract?.();
  }

  private wheel(event: WheelEvent): void {
    const zooms = event.ctrlKey || event.metaKey;
    if (this.options.cooperative() && !zooms) {
      this.options.onGestureLeftToPage?.('wheel');
      return;
    }
    event.preventDefault();
    const at = this.place(event);
    this.viewer.pointer_at(at.x, at.y);

    const pinching = event.ctrlKey && event.deltaMode === 0 && Math.abs(event.deltaY) < 30;
    const perStep =
      event.deltaMode === 1 ? LINES_PER_WHEEL_STEP : pinching ? PINCH_PIXELS_PER_WHEEL_STEP : PIXELS_PER_WHEEL_STEP;
    // A flick of a free-spinning wheel should not cross the country.
    this.viewer.wheel(Math.max(-4, Math.min(4, -event.deltaY / perStep)));
    this.options.onInteract?.();
  }

  /**
   * One finger moves the map and two pinch, twist and move it. While the map
   * shares the page, one finger is left to the page for scrolling.
   */
  private touch(event: TouchEvent): void {
    const now = Array.from(event.touches, (touch) => this.place(touch));
    const before = this.touches;
    this.touches = now;
    const cooperative = this.options.cooperative();
    if (now.length === 1 && cooperative) {
      if (event.type === 'touchmove') this.options.onGestureLeftToPage?.('touch');
      this.viewer.set_dragging(false);
      return;
    }
    if (event.cancelable) event.preventDefault();
    this.viewer.set_dragging(now.length > 0);
    if (event.type === 'touchstart') this.options.onInteract?.();
    // A finger landing or lifting changes what the gesture is measured from.
    if (event.type !== 'touchmove' || now.length !== before.length || now.length === 0) return;

    const centre = (points: TouchPoint[]) => ({
      x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
      y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
    });
    const [from, to] = [centre(before), centre(now)];
    this.viewer.pointer_at(to.x, to.y);
    this.viewer.drag(to.x - from.x, to.y - from.y, false);
    if (now.length < 2) return;

    const span = (points: TouchPoint[]) => Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y);
    const angle = (points: TouchPoint[]) => Math.atan2(points[1].y - points[0].y, points[1].x - points[0].x);
    if (span(before) > 0) this.viewer.pinch(span(now) / span(before));
    // The viewer turns by drag distance: 200 pixels to the radian.
    let twist = angle(now) - angle(before);
    twist = Math.atan2(Math.sin(twist), Math.cos(twist));
    this.viewer.drag(-twist * 200, 0, true);
  }

  private key(event: KeyboardEvent, down: boolean): void {
    const moves = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE'];
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (down && !event.repeat) {
      if (event.key === '+' || event.key === '=') this.zoomBy(ZOOM_FACTOR);
      else if (event.key === '-' || event.key === '_') this.zoomBy(1 / ZOOM_FACTOR);
      else if (event.code === 'KeyN') this.faceNorth();
      else if (event.code === 'Home' || event.key === '0') this.flyHome();
    }
    if (!moves.includes(event.code)) return;
    event.preventDefault();
    if (down) this.keys.add(event.code);
    else this.keys.delete(event.code);
    if (down && !event.repeat) this.options.onInteract?.();

    // Arrows move the map; with Shift they turn and tilt it.
    const axis = (negative: string, positive: string) =>
      Number(this.keys.has(positive)) - Number(this.keys.has(negative));
    const [across, along] = [axis('ArrowLeft', 'ArrowRight'), axis('ArrowDown', 'ArrowUp')];
    const turning = event.shiftKey;
    this.viewer.set_keys(
      axis('KeyA', 'KeyD') + (turning ? 0 : across),
      axis('KeyS', 'KeyW') + (turning ? 0 : along),
      axis('KeyQ', 'KeyE') + (turning ? across : 0),
      turning ? -along : 0,
      false,
    );
  }

  private releaseKeys(): void {
    this.keys.clear();
    this.viewer.set_keys(0, 0, 0, 0, false);
  }
}
