# swiss-relief (React component)

The relief map as a drop-in React component. It renders live in the browser
with WebGPU and streams the terrain tiles it needs from wherever you host
them.

```tsx
import { SwissRelief } from 'swiss-relief';
import 'swiss-relief/style.css';

export function Map() {
  return (
    <div style={{ height: '70vh', borderRadius: 20, overflow: 'hidden' }}>
      <SwissRelief dataUrl="/terrain" />
    </div>
  );
}
```

The component fills its parent, so give the parent a size.

This package is not on npm yet. Build it with `npm run build` and point your
project at `web/` (`npm install ../swiss-relief-3d/web`, or `npm link`).

## What gets served

| | Size | Served from |
| --- | --- | --- |
| Component JS + CSS | 22 KB gz + 3 KB | your bundle |
| `wasm/relief_web_bg.wasm` | 1.2 MB (300 KB compressed) | fetched once per page |
| Terrain | 2.6 MB up front, then tiles on demand | static files at `dataUrl` |
| Hiking trails (optional) | 17 MB in all, a few KB per tile on demand | `dataUrl/trails` |

The terrain directory is produced by `relief-prep web` (see the repository
README) and lands in `demo/public/terrain`. Put it on any static host or CDN and
pass its URL as `dataUrl`. With every level it is 520 MB; a close-up view
pulls down 8–15 MB of it, the whole-country view nothing at all. If that is
too much to host, drop the finest level:

```sh
cargo run --release -p relief-prep -- web --finest 1   # 20 m cells, 200 MB
cargo run --release -p relief-prep -- web --finest 2   # 40 m cells, 60 MB
```

Tiles never change, so give them a long `Cache-Control`. They are already
deflated; there is no point compressing them again on the wire. If the
terrain lives on another origin it needs CORS.

Bundlers pick up the wasm file through `new URL('./wasm/relief_web_bg.wasm',
import.meta.url)` (Vite, webpack 5 and Next.js all handle this). If yours does
not, copy it from `dist/wasm/` to your static folder and pass `wasmUrl`.

### Next.js

The map only exists in the browser, so skip SSR:

```tsx
'use client';
import dynamic from 'next/dynamic';
import 'swiss-relief/style.css';

const SwissRelief = dynamic(() => import('swiss-relief').then((m) => m.SwissRelief), { ssr: false });
```

## Props

| Prop | Default | |
| --- | --- | --- |
| `dataUrl` | required | Where the packed terrain is served from. |
| `wasmUrl` | found by the bundler | Where the wasm module is served from. |
| `initialPlace` | | Fly here once loaded: `lake_lucerne`, `jungfrau`, `lauterbrunnen`, `aletsch`, `matterhorn`, `gotthard`, `locarno`, `bernina`, `chur`, `saentis`, `lake_geneva`, `creux_du_van`. |
| `autoTour` | `false` | Cycle through the places until the visitor takes over. |
| `attract` | `true` | Slowly turn a tilted view that is left alone. |
| `compare` | `false` | Open with the split against conventional lighting. |
| `controls` | `true` | Show the on-map controls. Mouse, touch and keyboard work regardless. |
| `cooperativeGestures` | `true` | Wheel zooms only with Ctrl/⌘, one finger scrolls the page. Off in fullscreen. |
| `labels` | `true` | Names of peaks, lakes, glaciers and towns. |
| `trails` | `true` | The hiking trails, where the terrain was packed with them; the visitor can switch them in the layers menu. |
| `readout` | `false` | Coordinates and elevation under the cursor, bottom left. |
| `hints` | `true` | The one-line "drag to move" hint when the map opens. |
| `poster` | | Image shown while loading, and instead of the map where WebGPU is missing. |
| `maxPixelRatio` | `2` | Cap on device pixels per CSS pixel. |
| `theme` | `'auto'` | `'light'`, `'dark'` or follow the system. |
| `attribution` | `'Terrain © swisstopo'`, or `'Terrain and trails © swisstopo'` with trails | swisstopo's terms require the credit. |
| `onReady`, `onError` | | Get a handle to the map, or the reason it failed. |

A `poster` is worth having; the demo uses `demo/public/poster.jpg`, rendered with
`relief-viewer snapshot --preset presets/showcase.json`.

The ref handle (also passed to `onReady`) has `places`, `flyTo(id)`,
`flyHome()`, `setCompare(on)` and `setTour(on)`:

```tsx
const map = useRef<SwissReliefHandle>(null);
<button onClick={() => map.current?.flyTo('matterhorn')}>Matterhorn</button>
<SwissRelief ref={map} dataUrl="/terrain" controls={false} />
```

## Behaviour on a page

- Frames are drawn only when something changed, and not at all while the
  map is scrolled out of view or the tab is hidden.
- If frames run long the canvas resolution drops, and comes back when there
  is headroom.
- `prefers-reduced-motion` disables the tour and the idle turning.
- WebGPU is required: current Chrome, Edge and Safari, Firefox on Windows.
  Elsewhere the component shows the poster and a short note. `supported()`
  tells you in advance.
- Phones: the layout adapts, small canvases load less terrain and skip the
  10 m level, and GPUs without `float32-filterable` get a shader that
  filters by hand. Tested mostly in a desktop browser emulating a phone;
  reports from real devices welcome.

Mouse: drag to move, wheel to zoom at the pointer, right-drag or Shift-drag
to turn and tilt, double-click to zoom in. Touch: drag, pinch, twist.
Keyboard with the map focused: arrows move, Shift+arrows turn and tilt,
`+`/`-` zoom, `N` faces north, `0` shows the whole country.

## Styling

Controls inherit the page font. Colours and metrics are custom properties on
`.sr`:

```css
.sr {
  --sr-accent: #0b63ce; /* switches, compass needle, focus rings */
  --sr-radius: 10px;
  --sr-gap: 16px; /* distance of the controls from the edge */
}
```

## Development

```sh
npm install
npm run dev          # demo page at http://localhost:5173
npm run build        # the library, into dist/
npm run build:demo   # the demo page, into dist-demo/
```

Demo page query options: `?tour`, `?compare`, `?dark`, `?light`, `?readout`,
`?embed` (no wordmark).

`src/` is the component and everything in `demo/` is the page around it:
`demo/main.tsx`, `demo/demo.css` and the static files in `demo/public`,
which is also where `relief-prep web` puts the terrain. The built demo does
not include the terrain; serve a copy next to it or build with
`VITE_TERRAIN_URL` set.

The renderer is Rust, in `crates/web`. After changing it:

```sh
cargo build --release -p relief-web --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/src/wasm target/wasm32-unknown-unknown/release/relief_web.wasm
```

`wasm-bindgen` must match the version in `Cargo.lock`. The generated bindings
in `src/wasm/` are checked in so the package builds without a Rust toolchain.

### Scripts

These drive the demo page; none of them is part of the package. (The
gazetteer the labels come from is built by `tools/labels.mjs` at the
repository root, since it feeds the renderer, not the page.)

- `scripts/smoke.mjs` drives the page in headless Chrome and takes
  screenshots. Steps are seconds, JavaScript, or gestures:

  ```sh
  node scripts/smoke.mjs "http://localhost:5173/" out.png 4 "swissRelief.flyTo('matterhorn')" 7
  PLAIN=1 TOUCH=1 VIEWPORT=390x780 DPR=2 node scripts/smoke.mjs http://localhost:5173/ phone.png 5 "pinch:195,450,80,300" 2
  ```

  `PLAIN` hides float32 filtering as a phone GPU would, `NOGPU` hides WebGPU
  entirely, `SCHEME=dark` for dark mode. `swissRelief` is on `window` in dev
  builds.
- `scripts/flight.mjs` records the flight animation in the README.
