// The interface's icons, drawn on a 20 px grid with a 1.5 px stroke.

import type { ReactNode, SVGProps } from 'react';

function Icon({ children, ...props }: SVGProps<SVGSVGElement> & { children: ReactNode }) {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  );
}

export const PlusIcon = () => (
  <Icon>
    <path d="M10 4.5v11M4.5 10h11" />
  </Icon>
);

export const MinusIcon = () => (
  <Icon>
    <path d="M4.5 10h11" />
  </Icon>
);

/** A compass needle pointing up; the caller turns it toward north. */
export const NeedleIcon = () => (
  <Icon strokeWidth="0">
    <path d="M10 1.5 13.6 10H6.4z" className="sr-needle-north" />
    <path d="M10 18.5 6.4 10h7.2z" className="sr-needle-south" />
  </Icon>
);

/** A tilted plane: the map seen at an angle. */
export const TiltIcon = () => (
  <Icon>
    <path d="M3 13.5 7.2 6h5.6l4.2 7.5z" />
    <path d="M5.1 9.75h9.8" />
  </Icon>
);

/** A square: the map seen from straight above. */
export const FlatIcon = () => (
  <Icon>
    <rect x="4" y="4" width="12" height="12" rx="1.5" />
    <path d="M4 10h12M10 4v12" />
  </Icon>
);

export const ExpandIcon = () => (
  <Icon>
    <path d="M7.5 3.5h-4v4M12.5 3.5h4v4M16.5 12.5v4h-4M3.5 12.5v4h4" />
  </Icon>
);

export const ShrinkIcon = () => (
  <Icon>
    <path d="M3.5 7.5h4v-4M16.5 7.5h-4v-4M12.5 16.5v-4h4M7.5 16.5v-4h-4" />
  </Icon>
);

/** Two halves of a picture with a divider between them. */
export const CompareIcon = () => (
  <Icon>
    <rect x="3" y="4.5" width="14" height="11" rx="2" />
    <path d="M10 2.5v15" />
    <path d="M6.2 8.5 4.8 10l1.4 1.5M13.8 8.5l1.4 1.5-1.4 1.5" />
  </Icon>
);

export const LayersIcon = () => (
  <Icon>
    <path d="m10 3.5 7 3.75-7 3.75-7-3.75z" />
    <path d="m3.4 10.6 6.6 3.5 6.6-3.5M3.4 13.4l6.6 3.5 6.6-3.5" />
  </Icon>
);

export const HomeIcon = () => (
  <Icon>
    <path d="M3.5 9.2 10 3.6l6.5 5.6" />
    <path d="M5.2 8.2v7.6h9.6V8.2" />
  </Icon>
);

export const PlayIcon = () => (
  <Icon strokeWidth="0">
    <path d="M7 4.6v10.8a.6.6 0 0 0 .92.5l8.1-5.4a.6.6 0 0 0 0-1L7.92 4.1a.6.6 0 0 0-.92.5z" fill="currentColor" />
  </Icon>
);

export const PauseIcon = () => (
  <Icon strokeWidth="0">
    <rect x="5.5" y="4.5" width="3.2" height="11" rx="1" fill="currentColor" />
    <rect x="11.3" y="4.5" width="3.2" height="11" rx="1" fill="currentColor" />
  </Icon>
);

/** The grip of the comparison divider. */
export const GripIcon = () => (
  <Icon>
    <path d="M7.5 6.5 4 10l3.5 3.5M12.5 6.5 16 10l-3.5 3.5" />
  </Icon>
);
