import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';

import { CompareDivider, PlaceRail, ScaleBar, TopBar, ViewControls, type LookOptions } from './controls';
import { Engine, ZOOM_FACTOR, supported, type Place, type ViewState } from './engine';
import { LabelLayer } from './labels';
import './swiss-relief.css';

export interface SwissReliefProps {
  /** Where the packed terrain is served from, for example `/terrain`. */
  dataUrl: string;
  /** Where the WebAssembly module is served from, if your bundler does not find it. */
  wasmUrl?: string | URL;
  className?: string;
  style?: CSSProperties;
  /** A place to fly to once the map has loaded; see `places` on the handle. */
  initialPlace?: string;
  /** Tour the places until the visitor takes over. Off for visitors who ask for reduced motion. */
  autoTour?: boolean;
  /** Let a tilted view that is left alone turn slowly. Default true; off with reduced motion. */
  attract?: boolean;
  /** Open with the comparison against conventional lighting. */
  compare?: boolean;
  /** Show the controls. Default true. */
  controls?: boolean;
  /**
   * Share the page with what surrounds the map: the wheel zooms only with
   * Ctrl or Cmd held, and one finger scrolls the page. Default true. It is
   * always off in full screen.
   */
  cooperativeGestures?: boolean;
  /** An image shown until the map is ready, and instead of it where it cannot run. */
  poster?: string;
  /** Upper limit on device pixels per CSS pixel. Default 2. */
  maxPixelRatio?: number;
  /** Colours of the controls. `auto` follows the visitor's system. */
  theme?: 'light' | 'dark' | 'auto';
  /** The credit for the data. swisstopo asks to be named. */
  attribution?: ReactNode;
  /** Show the coordinates and elevation under the cursor. Default false. */
  readout?: boolean;
  /** Show the short hint on how to move the map when it opens. Default true. */
  hints?: boolean;
  /** Name peaks, lakes, glaciers and towns on the map. Default true. */
  labels?: boolean;
  onReady?: (map: SwissReliefHandle) => void;
  onError?: (error: Error) => void;
}

/** What a ref to the component offers. */
export interface SwissReliefHandle {
  /** The places the map can fly to. */
  places: Place[];
  flyTo(place: string): void;
  flyHome(): void;
  setCompare(on: boolean): void;
  setTour(on: boolean): void;
}

type Status =
  | { kind: 'loading'; progress: number }
  // The engine runs; `shown` once its first frames are out.
  | { kind: 'ready'; shown: boolean }
  | { kind: 'unsupported' }
  | { kind: 'failed'; message: string };

type Hint = 'intro' | 'wheel' | 'touch' | null;

const DEFAULT_OPTIONS: LookOptions = { color: true, fog: true, water: true, followsView: true, height: 1.2 };
/** A view steeper than this, in radians below the horizon, counts as a map. */
const MAP_PITCH = 1.45;
/** How often the interface catches up with the view, in milliseconds. */
const STATE_INTERVAL_MS = 120;
/** What the controls cover of the map's edges, in CSS pixels. */
const CONTROL_INSETS = { top: 62, right: 62, bottom: 92, left: 12 };
const BARE_INSETS = { top: 12, right: 12, bottom: 12, left: 12 };
const INTRO_HINT_MS = 9000;
const GESTURE_HINT_MS = 1800;

function prefersReducedMotion(): boolean {
  return typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function usesTouch(): boolean {
  return typeof matchMedia !== 'undefined' && matchMedia('(pointer: coarse)').matches;
}

function hintText(hint: Hint, cooperative: boolean): string | null {
  const mac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform);
  const zoomKey = mac ? '⌘' : 'Ctrl';
  switch (hint) {
    case 'intro':
      if (usesTouch()) return cooperative ? 'Use two fingers to move, pinch and turn the map' : 'Drag to move · pinch to zoom · twist to turn';
      return `Drag to move · ${cooperative ? `${zoomKey} + scroll` : 'scroll'} to zoom · right-drag to turn`;
    case 'wheel':
      return `Hold ${zoomKey} and scroll to zoom the map`;
    case 'touch':
      return 'Use two fingers to move the map';
    default:
      return null;
  }
}

function describe(view: ViewState): string {
  const degrees = (value: number, positive: string, negative: string) =>
    `${Math.abs(value).toFixed(3)}° ${value >= 0 ? positive : negative}`;
  const elevation = Math.round(view.elevation).toLocaleString('en-US');
  return `${degrees(view.latitude, 'N', 'S')}  ${degrees(view.longitude, 'E', 'W')}  ·  ${elevation} m`;
}

/**
 * An interactive 3D relief map of Switzerland, shaded in the Swiss manner.
 * It fills its parent, so give that a size.
 */
export const SwissRelief = forwardRef<SwissReliefHandle, SwissReliefProps>(function SwissRelief(props, ref) {
  const {
    dataUrl,
    wasmUrl,
    className,
    style,
    initialPlace,
    autoTour = false,
    attract = true,
    compare: startComparing = false,
    controls = true,
    cooperativeGestures = true,
    poster,
    maxPixelRatio = 2,
    theme = 'auto',
    attribution = 'Terrain © swisstopo',
    readout = false,
    hints = true,
    labels = true,
    onReady,
    onError,
  } = props;

  const frame = useRef<HTMLDivElement>(null);
  const host = useRef<HTMLDivElement>(null);
  const labelHost = useRef<HTMLDivElement>(null);
  const labelLayer = useRef<LabelLayer | null>(null);
  const engine = useRef<Engine | null>(null);
  const needle = useRef<HTMLSpanElement | null>(null);
  const heading = useRef(0);

  const [status, setStatus] = useState<Status>({ kind: 'loading', progress: 0 });
  const [places, setPlaces] = useState<Place[]>([]);
  const [view, setView] = useState<ViewState | null>(null);
  const [atHome, setAtHome] = useState(true);
  const [comparing, setComparing] = useState(startComparing);
  const [split, setSplit] = useState(0.5);
  const [menuOpen, setMenuOpen] = useState(false);
  const [options, setOptions] = useState(DEFAULT_OPTIONS);
  const [fullscreen, setFullscreen] = useState(false);
  const [hint, setHint] = useState<Hint>(null);

  // The engine asks about these at any time, so they live in refs.
  const cooperative = useRef(cooperativeGestures);
  cooperative.current = cooperativeGestures && !fullscreen;
  const showLabels = useRef(labels);
  showLabels.current = labels;
  const callbacks = useRef({ onReady, onError });
  callbacks.current = { onReady, onError };
  const insets = useRef(CONTROL_INSETS);
  insets.current = controls ? CONTROL_INSETS : BARE_INSETS;
  const opening = useRef({ initialPlace, autoTour });

  const turnNeedle = useCallback((element: HTMLSpanElement | null) => {
    needle.current = element;
    if (element) element.style.transform = `rotate(${-heading.current}rad)`;
  }, []);

  // The engine and its canvas live as long as the data address does.
  useEffect(() => {
    const parent = host.current;
    if (!parent) return;
    if (!supported()) {
      setStatus({ kind: 'unsupported' });
      return;
    }
    // A canvas of its own for every engine: a WebGPU canvas cannot be
    // handed from one device to the next.
    const canvas = document.createElement('canvas');
    canvas.tabIndex = 0;
    canvas.setAttribute('role', 'application');
    canvas.setAttribute('aria-label', 'Relief map of Switzerland. Arrow keys move, plus and minus zoom.');
    parent.append(canvas);

    let cancelled = false;
    let created: Engine | undefined;
    let pending: ViewState | undefined;
    let timer = 0;
    const publish = () => {
      timer = 0;
      if (pending && !cancelled) setView(pending);
      pending = undefined;
    };

    setStatus({ kind: 'loading', progress: 0 });
    Engine.create(canvas, {
      dataUrl,
      wasmUrl,
      maxPixelRatio,
      cooperative: () => cooperative.current,
      insets: () => insets.current,
      onProgress: (progress) => {
        if (!cancelled) setStatus((now) => (now.kind === 'loading' ? { kind: 'loading', progress } : now));
      },
      onFrame: (state) => {
        // The compass turns with every frame; the rest can lag a little.
        heading.current = state.heading;
        if (needle.current) needle.current.style.transform = `rotate(${-state.heading}rad)`;
        pending = state;
        timer ||= window.setTimeout(publish, STATE_INTERVAL_MS);
      },
      onLabels: (placed, ratio) => {
        if (!labelHost.current) return;
        labelLayer.current ||= new LabelLayer(labelHost.current);
        labelLayer.current.update(showLabels.current ? placed : [], ratio);
      },
      onShown: () => {
        if (!cancelled) setStatus((now) => (now.kind === 'ready' ? { kind: 'ready', shown: true } : now));
      },
      onInteract: () => {
        setAtHome(false);
        setHint((now) => (now === 'intro' ? null : now));
      },
      onGestureLeftToPage: (gesture) => setHint(gesture),
    })
      .then((made) => {
        if (cancelled) {
          made.destroy();
          return;
        }
        created = made;
        engine.current = made;
        // For looking at a running map from the console.
        if (import.meta.env?.DEV) Object.assign(window, { swissRelief: made });
        setPlaces(made.places);
        setStatus({ kind: 'ready', shown: false });

        const calm = prefersReducedMotion();
        if (opening.current.initialPlace) {
          made.flyTo(opening.current.initialPlace);
          setAtHome(false);
        }
        if (opening.current.autoTour && !calm) made.setTour(true);
        callbacks.current.onReady?.(handleOf(made, setComparing));
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const problem = error instanceof Error ? error : new Error(String(error));
        console.error('[swiss-relief]', problem);
        setStatus({ kind: 'failed', message: problem.message });
        callbacks.current.onError?.(problem);
      });

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      created?.destroy();
      engine.current = null;
      canvas.remove();
    };
  }, [dataUrl, wasmUrl, maxPixelRatio]);

  const ready = status.kind === 'ready';
  const shown = ready && status.shown;

  // The opening hint arrives with the map.
  useEffect(() => {
    if (shown && hints) setHint('intro');
  }, [shown, hints]);

  useEffect(() => {
    if (ready) engine.current?.setCompare(comparing ? split : undefined);
  }, [ready, comparing, split]);

  useEffect(() => {
    if (ready) engine.current?.setAttract(attract && !prefersReducedMotion());
  }, [ready, attract]);

  useEffect(() => {
    if (!ready || !engine.current) return;
    engine.current.patchLook(0, {
      color: options.color,
      fog: options.fog,
      water: options.water,
      light_mode: options.followsView ? 'CameraRelative' : 'WorldFixed',
      geometry_exaggeration: options.height,
    });
    // Both looks stand on the same ground, or the divider would show a step.
    engine.current.patchLook(1, {
      water: options.water,
      geometry_exaggeration: options.height,
      shading_exaggeration: options.height,
    });
  }, [ready, options]);

  // Hints go away by themselves.
  useEffect(() => {
    if (!hint) return;
    const timer = window.setTimeout(() => setHint(null), hint === 'intro' ? INTRO_HINT_MS : GESTURE_HINT_MS);
    return () => window.clearTimeout(timer);
  }, [hint]);

  useEffect(() => {
    const changed = () => setFullscreen(document.fullscreenElement === frame.current);
    document.addEventListener('fullscreenchange', changed);
    return () => document.removeEventListener('fullscreenchange', changed);
  }, []);

  useImperativeHandle(
    ref,
    () => ({
      places,
      flyTo: (place) => engine.current?.flyTo(place),
      flyHome: () => engine.current?.flyHome(),
      setCompare: setComparing,
      setTour: (on) => engine.current?.setTour(on),
    }),
    [places],
  );

  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void frame.current?.requestFullscreen();
  };

  const canFullscreen = typeof document !== 'undefined' && document.fullscreenEnabled;
  const classes = ['sr', `sr-theme-${theme}`, fullscreen ? 'sr-is-fullscreen' : '', className ?? ''].filter(Boolean).join(' ');
  const message = hintText(hint, cooperative.current);

  return (
    <div ref={frame} className={classes} style={style}>
      {poster && <img className="sr-poster" src={poster} alt="" data-hidden={shown} />}
      <div ref={host} className="sr-canvas" data-ready={shown} />
      <div ref={labelHost} className="sr-labels" data-ready={shown} aria-hidden="true" />

      {(status.kind === 'loading' || (ready && !shown)) && (
        <div className="sr-loading" role="status">
          <span className="sr-spinner" aria-hidden="true" />
          Loading terrain
          <span className="sr-progress" aria-hidden="true">
            <span style={{ transform: `scaleX(${status.kind === 'loading' ? Math.max(0.04, status.progress) : 1})` }} />
          </span>
        </div>
      )}

      {(status.kind === 'unsupported' || status.kind === 'failed') && (
        <div className="sr-notice" role="alert">
          <strong>{status.kind === 'unsupported' ? 'This map needs WebGPU' : 'The map could not start'}</strong>
          <span>
            {status.kind === 'unsupported'
              ? 'It renders the terrain live on your graphics card. A current version of Chrome, Edge or Safari can show it.'
              : status.message}
          </span>
        </div>
      )}

      {shown && controls && (
        <>
          <TopBar
            comparing={comparing}
            menuOpen={menuOpen}
            options={options}
            onCompare={setComparing}
            onMenu={setMenuOpen}
            onOptions={setOptions}
          />
          <ViewControls
            needle={turnNeedle}
            tilted={!!view && view.pitch < MAP_PITCH}
            fullscreen={fullscreen}
            canFullscreen={canFullscreen}
            onZoom={(direction) => {
              setAtHome(false);
              engine.current?.zoomBy(direction > 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR);
            }}
            onNorth={() => engine.current?.faceNorth()}
            onTilt={() => {
              setAtHome(false);
              engine.current?.toggleTilt();
            }}
            onFullscreen={toggleFullscreen}
          />
          {comparing && (
            <CompareDivider split={split} left="Swiss relief" right="Conventional lighting" onSplit={setSplit} />
          )}
          <PlaceRail
            places={places}
            current={view?.place ?? null}
            home={atHome && !view?.place && !view?.touring}
            touring={!!view?.touring}
            onPlace={(id) => {
              setAtHome(false);
              engine.current?.flyTo(id);
            }}
            onHome={() => {
              setAtHome(true);
              engine.current?.flyHome();
            }}
            onTour={(on) => {
              setAtHome(false);
              engine.current?.setTour(on);
            }}
          />
          <div className="sr-foot">
            {view && <ScaleBar metresPerPixel={view.metresPerPixel} />}
            {view && readout && <span className="sr-readout">{describe(view)}</span>}
            <span className="sr-credit" data-streaming={!!view && view.loading > 0}>
              {attribution}
            </span>
          </div>
          {message && (
            <div className="sr-hint" role="status" data-kind={hint}>
              {message}
            </div>
          )}
        </>
      )}
    </div>
  );
});

function handleOf(engine: Engine, setComparing: (on: boolean) => void): SwissReliefHandle {
  return {
    places: engine.places,
    flyTo: (place) => engine.flyTo(place),
    flyHome: () => engine.flyHome(),
    setCompare: setComparing,
    setTour: (on) => engine.setTour(on),
  };
}
