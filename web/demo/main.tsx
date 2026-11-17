import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { SwissRelief } from '../src';
import './demo.css';

// Where the packed terrain is served from.
const terrain = import.meta.env.VITE_TERRAIN_URL ?? '/terrain';
// The map fills the page. `?tour` starts the tour, `?compare` the comparison,
// `?light` and `?dark` fix the colours of the controls, `?readout` shows the
// coordinates under the cursor, and `?embed` leaves out the wordmark.
const query = new URLSearchParams(location.search);
const theme = query.has('dark') ? 'dark' : query.has('light') ? 'light' : 'auto';

function Demo() {
  return (
    <>
      <SwissRelief
        dataUrl={terrain}
        theme={theme}
        autoTour={query.has('tour')}
        compare={query.has('compare')}
        readout={query.has('readout')}
        hints={false}
        cooperativeGestures={false}
        poster="/poster.jpg"
      />
      {!query.has('embed') && (
        <div className="demo-mark" data-theme={theme}>
          <img src="/icon.svg" alt="" width="22" height="22" />
          Swiss relief
        </div>
      )}
    </>
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Demo />
  </StrictMode>,
);
