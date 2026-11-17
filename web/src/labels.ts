/**
 * The names on the map, as HTML over the canvas. The renderer decides each
 * frame which names show and where (see `labels.rs`); this keeps one element
 * per shown name and moves it, so that the text is real text: crisp at any
 * pixel ratio, in the page's font, selectable by nothing and in the way of
 * nothing.
 */

export type LabelKind = 'peak' | 'lake' | 'town' | 'glacier';

/** A name placed by the renderer, in device pixels. */
export interface PlacedLabel {
  id: number;
  kind: LabelKind;
  name: string;
  elevation?: number;
  /** The place itself. */
  x: number;
  y: number;
  /** Left, top, width and height of the text's box. */
  rect: [number, number, number, number];
  /** Font size in device pixels. */
  font: number;
  opacity: number;
}

export class LabelLayer {
  private readonly nodes = new Map<number, HTMLElement>();

  constructor(private readonly container: HTMLElement) {}

  /** Shows exactly `placed`, scaled from device to CSS pixels by `ratio`. */
  update(placed: PlacedLabel[], ratio: number): void {
    const seen = new Set<number>();
    for (const label of placed) {
      seen.add(label.id);
      let node = this.nodes.get(label.id);
      if (!node) {
        node = make(label);
        this.container.appendChild(node);
        this.nodes.set(label.id, node);
      }
      const [left, top, width, height] = label.rect;
      const style = node.style;
      style.transform = `translate(${(left / ratio).toFixed(1)}px, ${(top / ratio).toFixed(1)}px)`;
      style.width = `${Math.ceil(width / ratio)}px`;
      style.height = `${Math.ceil(height / ratio)}px`;
      style.fontSize = `${(label.font / ratio).toFixed(2)}px`;
      style.opacity = label.opacity.toFixed(3);
      const mark = node.firstElementChild as HTMLElement | null;
      if (mark?.classList.contains('sr-mark')) {
        // The marker is at the place; the box holds only the text.
        mark.style.transform = `translate(${((label.x - left) / ratio).toFixed(1)}px, ${((label.y - top) / ratio).toFixed(1)}px)`;
      }
    }
    for (const [id, node] of this.nodes) {
      if (!seen.has(id)) {
        node.remove();
        this.nodes.delete(id);
      }
    }
  }

  clear(): void {
    for (const node of this.nodes.values()) node.remove();
    this.nodes.clear();
  }
}

function make(label: PlacedLabel): HTMLElement {
  const node = document.createElement('div');
  node.className = `sr-label sr-label-${label.kind}`;
  if (label.kind === 'peak' || label.kind === 'town') {
    const mark = document.createElement('span');
    mark.className = 'sr-mark';
    node.appendChild(mark);
  }
  const name = document.createElement('span');
  name.className = 'sr-label-name';
  name.textContent = label.name;
  node.appendChild(name);
  if (label.kind === 'peak' && label.elevation !== undefined) {
    const elevation = document.createElement('span');
    elevation.className = 'sr-label-elevation';
    elevation.textContent = String(label.elevation);
    node.appendChild(elevation);
  }
  return node;
}
