// The controls that sit on top of the map. Each is a plain function of
// what it shows; the component that owns the engine wires them up.

import { useEffect, useRef, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react';

import type { Place } from './engine';
import {
  CompareIcon,
  ExpandIcon,
  FlatIcon,
  GripIcon,
  HomeIcon,
  LayersIcon,
  MinusIcon,
  NeedleIcon,
  PauseIcon,
  PlayIcon,
  PlusIcon,
  ShrinkIcon,
  TiltIcon,
} from './icons';

interface ButtonProps {
  label: string;
  onClick: () => void;
  pressed?: boolean;
  children: ReactNode;
}

/** A square button with an icon; its label is its tooltip and its name. */
export function IconButton({ label, onClick, pressed, children }: ButtonProps) {
  return (
    <button type="button" className="sr-button" title={label} aria-label={label} aria-pressed={pressed} onClick={onClick}>
      {children}
    </button>
  );
}

interface ViewControlsProps {
  /** Registers the element that is turned to point north. */
  needle: (element: HTMLSpanElement | null) => void;
  tilted: boolean;
  fullscreen: boolean;
  canFullscreen: boolean;
  onZoom: (direction: 1 | -1) => void;
  onNorth: () => void;
  onTilt: () => void;
  onFullscreen: () => void;
}

export function ViewControls(props: ViewControlsProps) {
  return (
    <div className="sr-stack" role="group" aria-label="View">
      <div className="sr-panel sr-column">
        <IconButton label="Zoom in" onClick={() => props.onZoom(1)}>
          <PlusIcon />
        </IconButton>
        <IconButton label="Zoom out" onClick={() => props.onZoom(-1)}>
          <MinusIcon />
        </IconButton>
      </div>
      <div className="sr-panel sr-column">
        <IconButton label="North up" onClick={props.onNorth}>
          <span className="sr-needle" ref={props.needle}>
            <NeedleIcon />
          </span>
        </IconButton>
        <IconButton label={props.tilted ? 'Look straight down' : 'Tilt the view'} onClick={props.onTilt}>
          {props.tilted ? <FlatIcon /> : <TiltIcon />}
        </IconButton>
      </div>
      {props.canFullscreen && (
        <div className="sr-panel sr-column">
          <IconButton label={props.fullscreen ? 'Leave full screen' : 'Full screen'} onClick={props.onFullscreen}>
            {props.fullscreen ? <ShrinkIcon /> : <ExpandIcon />}
          </IconButton>
        </div>
      )}
    </div>
  );
}

interface PlaceRailProps {
  places: Place[];
  /** The place the view is at, if any. */
  current: string | null;
  /** Whether the whole terrain is in view. */
  home: boolean;
  touring: boolean;
  onPlace: (id: string) => void;
  onHome: () => void;
  onTour: (on: boolean) => void;
}

/** The places to fly to, in a row that scrolls sideways when it must. */
export function PlaceRail({ places, current, home, touring, onPlace, onHome, onTour }: PlaceRailProps) {
  const rail = useRef<HTMLDivElement>(null);
  // Keeps the current place in sight while a tour moves along the row. Only
  // the row scrolls: `scrollIntoView` would move the page under the visitor.
  useEffect(() => {
    const row = rail.current;
    const chip = row?.querySelector<HTMLElement>('[aria-current="true"]');
    if (!row || !chip) return;
    const left = chip.offsetLeft - row.offsetLeft - (row.clientWidth - chip.offsetWidth) / 2;
    row.scrollTo({ left, behavior: 'smooth' });
  }, [current]);

  return (
    <nav className="sr-panel sr-rail" aria-label="Places">
      <IconButton label={touring ? 'Stop the tour' : 'Take a tour'} pressed={touring} onClick={() => onTour(!touring)}>
        {touring ? <PauseIcon /> : <PlayIcon />}
      </IconButton>
      <span className="sr-rule" />
      <div className="sr-chips" ref={rail}>
        <button type="button" className="sr-chip sr-chip-home" aria-current={home} onClick={onHome}>
          <HomeIcon />
          Whole country
        </button>
        {places.map((place) => (
          <button
            key={place.id}
            type="button"
            className="sr-chip"
            aria-current={place.id === current}
            onClick={() => onPlace(place.id)}
          >
            {place.name}
          </button>
        ))}
      </div>
    </nav>
  );
}

/** The look's options, as the names the viewer knows them by. */
export interface LookOptions {
  color: boolean;
  fog: boolean;
  water: boolean;
  /** Whether the light keeps to the upper left of the screen as the view turns. */
  followsView: boolean;
  /** Vertical exaggeration of the terrain. */
  height: number;
}

interface TopBarProps {
  comparing: boolean;
  menuOpen: boolean;
  options: LookOptions;
  onCompare: (on: boolean) => void;
  onMenu: (open: boolean) => void;
  onOptions: (options: LookOptions) => void;
}

function Switch({ label, hint, on, onChange }: { label: string; hint?: string; on: boolean; onChange: (on: boolean) => void }) {
  return (
    <label className="sr-switch">
      <span className="sr-switch-text">
        {label}
        {hint && <small>{hint}</small>}
      </span>
      <input type="checkbox" role="switch" checked={on} onChange={(event) => onChange(event.target.checked)} />
      <span className="sr-switch-track" aria-hidden="true" />
    </label>
  );
}

export function TopBar({ comparing, menuOpen, options, onCompare, onMenu, onOptions }: TopBarProps) {
  const menu = useRef<HTMLDivElement>(null);
  // A click anywhere else, or Escape, closes the menu.
  useEffect(() => {
    if (!menuOpen) return;
    const outside = (event: Event) => {
      if (!menu.current?.contains(event.target as Node)) onMenu(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onMenu(false);
    };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', outside);
      document.removeEventListener('keydown', escape);
    };
  }, [menuOpen, onMenu]);

  const set = (change: Partial<LookOptions>) => onOptions({ ...options, ...change });
  return (
    <div className="sr-top" ref={menu}>
      <div className="sr-panel sr-row">
        <button type="button" className="sr-text-button" aria-pressed={comparing} onClick={() => onCompare(!comparing)}>
          <CompareIcon />
          Compare
        </button>
        <span className="sr-rule" />
        <IconButton label="Appearance" pressed={menuOpen} onClick={() => onMenu(!menuOpen)}>
          <LayersIcon />
        </IconButton>
      </div>
      {menuOpen && (
        <div className="sr-panel sr-menu" role="dialog" aria-label="Appearance">
          <Switch label="Colour" hint="Tints by elevation" on={options.color} onChange={(color) => set({ color })} />
          <Switch label="Haze" hint="Fades the lowlands and the distance" on={options.fog} onChange={(fog) => set({ fog })} />
          <Switch label="Lakes" on={options.water} onChange={(water) => set({ water })} />
          <Switch
            label="Light follows the view"
            hint="Always from the upper left, as on a map"
            on={options.followsView}
            onChange={(followsView) => set({ followsView })}
          />
          <label className="sr-slider">
            <span className="sr-switch-text">
              Relief height
              <small>{options.height.toFixed(1)}×</small>
            </span>
            <input
              type="range"
              min="1"
              max="2.4"
              step="0.1"
              value={options.height}
              onChange={(event) => set({ height: Number(event.target.value) })}
            />
          </label>
        </div>
      )}
    </div>
  );
}

interface DividerProps {
  /** Where the divider stands, as a share of the width. */
  split: number;
  left: string;
  right: string;
  onSplit: (split: number) => void;
}

/** The line between the two looks, with a grip to drag it by. */
export function CompareDivider({ split, left, right, onSplit }: DividerProps) {
  const drag = (event: ReactPointerEvent<HTMLDivElement>) => {
    const frame = event.currentTarget.parentElement?.getBoundingClientRect();
    if (!frame || !event.currentTarget.hasPointerCapture(event.pointerId)) return;
    onSplit(Math.min(0.95, Math.max(0.05, (event.clientX - frame.left) / frame.width)));
  };
  const nudge = (step: number) => onSplit(Math.min(0.95, Math.max(0.05, split + step)));
  return (
    <>
      <span className="sr-tag sr-tag-left" style={{ right: `calc(${100 - split * 100}% + 14px)` }}>
        {left}
      </span>
      <span className="sr-tag sr-tag-right" style={{ left: `calc(${split * 100}% + 14px)` }}>
        {right}
      </span>
      <div
        className="sr-divider"
        style={{ left: `${split * 100}%` }}
        role="slider"
        tabIndex={0}
        aria-label="Comparison divider"
        aria-orientation="horizontal"
        aria-valuemin={5}
        aria-valuemax={95}
        aria-valuenow={Math.round(split * 100)}
        onPointerDown={(event) => event.currentTarget.setPointerCapture(event.pointerId)}
        onPointerMove={drag}
        onKeyDown={(event) => {
          if (event.key === 'ArrowLeft') nudge(-0.02);
          if (event.key === 'ArrowRight') nudge(0.02);
        }}
      >
        <span className="sr-grip">
          <GripIcon />
        </span>
      </div>
    </>
  );
}

/** A round length close to `metres`: 1, 2 or 5 times a power of ten. */
function roundLength(metres: number): number {
  const magnitude = 10 ** Math.floor(Math.log10(metres));
  const lead = metres / magnitude;
  return (lead >= 5 ? 5 : lead >= 2 ? 2 : 1) * magnitude;
}

/** A scale bar for the middle of the view. */
export function ScaleBar({ metresPerPixel }: { metresPerPixel: number }) {
  if (!(metresPerPixel > 0)) return null;
  const length = roundLength(96 * metresPerPixel);
  const label = length >= 1000 ? `${length / 1000} km` : `${length} m`;
  return (
    <span className="sr-scale" style={{ width: length / metresPerPixel }}>
      {label}
    </span>
  );
}
