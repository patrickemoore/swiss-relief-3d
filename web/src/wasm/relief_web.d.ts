/* tslint:disable */
/* eslint-disable */

/**
 * A terrain viewer drawing into a canvas.
 */
export class Viewer {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * The graphics adapter in use, for diagnostics.
     */
    adapter(): string;
    /**
     * Opens a viewer on `canvas`, whose width and height are its size in
     * device pixels; `layout_side` is its longer side in CSS pixels, which
     * tells a phone from a desktop. `meta` and `base` are the contents of the packed
     * terrain's `meta.json` and `base.bin`, and `data_url` is where its
     * tiles are served from.
     */
    static create(canvas: HTMLCanvasElement, data_url: string, meta: string, base: Uint8Array, layout_side: number): Promise<Viewer>;
    /**
     * Pointer motion in device pixels that moves the ground, or with
     * `rotate` turns and tilts the view.
     */
    drag(dx: number, dy: number, rotate: boolean): void;
    /**
     * Turns the view so that north is up.
     */
    face_north(): void;
    /**
     * Megabytes of tiles fetched so far.
     */
    fetched_megabytes(): number;
    /**
     * Flies to a place by its id. False if there is no such place.
     */
    fly_to(id: string): boolean;
    /**
     * Advances to `time`, in milliseconds as an animation frame gives it,
     * and draws if anything has changed. Returns whether it drew.
     */
    frame(time: number): boolean;
    /**
     * Goes back to the view of the whole terrain: in a flight, or at once.
     */
    go_home(fly: boolean): void;
    /**
     * The names placed over the last frame, as JSON: see `labels::Placed`.
     */
    labels(): string;
    /**
     * Changes some of a look's parameters, given as a JSON object with the
     * names the presets use, for instance `{"color": false}`.
     */
    patch_look(look: number, patch: string): void;
    /**
     * A pinch by `factor` since the last call; above 1 zooms in.
     */
    pinch(factor: number): void;
    /**
     * The places to fly to, as JSON: `[{ "id": ..., "name": ... }]`.
     */
    places(): string;
    /**
     * Where the pointer is, in device pixels from the canvas's top-left
     * corner. Zooming holds the ground there in place.
     */
    pointer_at(x: number, y: number): void;
    pointer_left(): void;
    /**
     * Call when the canvas has been given a new size in device pixels.
     */
    resize(width: number, height: number): void;
    /**
     * Whether a view left alone starts to turn slowly.
     */
    set_attract(on: boolean): void;
    /**
     * Shows the second look on the right from this share of the width, or
     * only the first with `undefined`.
     */
    set_compare(split?: number | null): void;
    /**
     * A button or a finger went down or came up. Coasting starts when the
     * last one lifts.
     */
    set_dragging(dragging: boolean): void;
    /**
     * Tells the viewer how much of the canvas the page's controls cover at
     * each edge, in device pixels. The view of the whole terrain keeps
     * clear of them.
     */
    set_insets(top: number, right: number, bottom: number, left: number): void;
    /**
     * The keyboard's intent, each in -1..1: move right and forward, turn
     * clockwise and tilt down.
     */
    set_keys(right: number, forward: number, turn: number, tilt: number, fast: boolean): void;
    /**
     * Sets a look (0 or 1): `relief` is the method as the viewer shows it,
     * and `plan`, `realistic`, `draped`, `world-fixed` and `headlight` are
     * the conditions of the evaluation.
     */
    set_look(look: number, name: string): void;
    /**
     * Call when the page's device pixels per CSS pixel change: the names
     * keep their size on the page.
     */
    set_pixel_ratio(ratio: number): void;
    /**
     * Starts or stops a round of the places. Steering stops it too.
     */
    set_tour(on: boolean): void;
    /**
     * The state of the view as JSON, for a compass, a scale bar and such.
     */
    state(): string;
    /**
     * Switches between looking straight down and a tilted view.
     */
    toggle_tilt(): void;
    /**
     * Wheel steps; positive zooms in.
     */
    wheel(steps: number): void;
    /**
     * Zooms by a factor, eased; above 1 zooms in.
     */
    zoom_by(factor: number): void;
}

export function start(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_viewer_free: (a: number, b: number) => void;
    readonly start: () => void;
    readonly viewer_adapter: (a: number) => [number, number];
    readonly viewer_create: (a: any, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => any;
    readonly viewer_drag: (a: number, b: number, c: number, d: number) => void;
    readonly viewer_face_north: (a: number) => void;
    readonly viewer_fetched_megabytes: (a: number) => number;
    readonly viewer_fly_to: (a: number, b: number, c: number) => number;
    readonly viewer_frame: (a: number, b: number) => [number, number, number];
    readonly viewer_go_home: (a: number, b: number) => void;
    readonly viewer_labels: (a: number) => [number, number];
    readonly viewer_patch_look: (a: number, b: number, c: number, d: number) => [number, number];
    readonly viewer_pinch: (a: number, b: number) => void;
    readonly viewer_places: (a: number) => [number, number];
    readonly viewer_pointer_at: (a: number, b: number, c: number) => void;
    readonly viewer_pointer_left: (a: number) => void;
    readonly viewer_resize: (a: number, b: number, c: number) => void;
    readonly viewer_set_attract: (a: number, b: number) => void;
    readonly viewer_set_compare: (a: number, b: number) => void;
    readonly viewer_set_dragging: (a: number, b: number) => void;
    readonly viewer_set_insets: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly viewer_set_keys: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly viewer_set_look: (a: number, b: number, c: number, d: number) => [number, number];
    readonly viewer_set_pixel_ratio: (a: number, b: number) => void;
    readonly viewer_set_tour: (a: number, b: number) => void;
    readonly viewer_state: (a: number) => [number, number];
    readonly viewer_toggle_tilt: (a: number) => void;
    readonly viewer_wheel: (a: number, b: number) => void;
    readonly viewer_zoom_by: (a: number, b: number) => void;
    readonly wasm_bindgen_b07697265e4f617b___convert__closures_____invoke___js_sys_9093bffe05d08519___Function_fn_wasm_bindgen_b07697265e4f617b___JsValue_____wasm_bindgen_b07697265e4f617b___sys__Undefined___js_sys_9093bffe05d08519___Function_fn_wasm_bindgen_b07697265e4f617b___JsValue_____wasm_bindgen_b07697265e4f617b___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_b07697265e4f617b___convert__closures_____invoke___wasm_bindgen_b07697265e4f617b___JsValue__core_608f92abc48d28da___result__Result_____wasm_bindgen_b07697265e4f617b___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
