/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Where the demo finds the packed terrain; `/terrain` if unset. */
  readonly VITE_TERRAIN_URL?: string;
}
