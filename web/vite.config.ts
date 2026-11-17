import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vite';

const WASM = 'wasm/relief_web_bg.wasm';

/** The module's name with a digest of its contents, as a site can cache for ever. */
function hashedName(): string {
  const digest = createHash('sha256').update(readFileSync(resolve('src', WASM))).digest('hex').slice(0, 8);
  return WASM.replace('.wasm', `.${digest}.wasm`);
}

/**
 * Puts the WebAssembly module next to the script that loads it. The script
 * finds it with `new URL('./wasm/...', import.meta.url)`, which Vite is told
 * to leave alone: in library mode it would inline the megabyte as base64.
 *
 * In the library the instruction to leave it alone is then taken out again,
 * so that the bundler of the site using the library does pick the file up.
 * The demo instead gets the file under a name with a digest in it, so that
 * the site can tell browsers to keep it for ever.
 */
function shipWasm(scriptDir: string, forBundlers: boolean): Plugin {
  const shipped = forBundlers ? WASM : hashedName();
  return {
    name: 'ship-wasm',
    apply: 'build',
    enforce: 'pre',
    // The generated bindings carry a default address of their own, which is
    // never used: the engine always says where the module is.
    transform: (code, id) =>
      id.endsWith('relief_web.js') ? code.replace("new URL('relief_web_bg.wasm', import.meta.url)", 'undefined') : null,
    renderChunk: (code) =>
      forBundlers ? code.replace(/\/\*\s*@vite-ignore\s*\*\/\s*/g, '') : code.replaceAll(`./${WASM}`, `./${shipped}`),
    closeBundle() {
      const target = resolve(scriptDir, shipped);
      mkdirSync(dirname(target), { recursive: true });
      copyFileSync(resolve('src', WASM), target);
    },
  };
}

// `vite` serves the demo page and `vite build` bundles it. `vite build
// --mode library` bundles the component for use in another site, with React
// left to that site.
export default defineConfig(({ mode }) => {
  const library = mode === 'library';
  return {
    plugins: [react(), shipWasm(library ? 'dist' : 'dist-demo/assets', library)],
    build: library
      ? {
          lib: { entry: 'src/index.ts', formats: ['es'], fileName: 'swiss-relief' },
          rollupOptions: { external: ['react', 'react-dom', 'react/jsx-runtime'] },
          copyPublicDir: false,
        }
      : // The terrain in demo/public is half a gigabyte; the built demo is
        // served next to a copy of it, or told where else it is with
        // VITE_TERRAIN_URL.
        { outDir: 'dist-demo', copyPublicDir: false },
    // Static files belong to the demo page, not to the component.
    publicDir: 'demo/public',
    server: { port: 5173, strictPort: true },
    preview: { port: 5174, strictPort: true },
  };
});
