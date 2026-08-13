# Frontend

React 18 + TypeScript + Vite + Tailwind + Recharts + TanStack Query + Framer Motion.

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api and /ws to :8000
npm run build
npm run typecheck
```

The API must be running (`make serve` from the repo root). In dev, Vite proxies `/api`,
`/health` and `/ws` to `localhost:8000`, so everything is same-origin and the CORS
allowlist is not involved. For a deployed frontend, set `VITE_API_BASE`.

## The four views

1. **Explorer** — the hero. One sample, three corruption sliders, live updates over the
   WebSocket, and every architecture responding to the *same* corrupted input. The ghost
   marker on each dial shows the clean prediction so drift is visible rather than
   inferred.
2. **Results** — retention curves faceted by corruption axis, with ±1 std bands over
   seeds. Architectures toggle on and off.
3. **Reliance** — the MRS matrix and the 7-subset removal grid.
4. **Pareto** — clean performance against AUDC, with the frontier ringed and
   modality-dropout variants linked to their controls.

## Conventions

- **Nothing encodes meaning in hue alone.** Series carry a colour *and* a dash pattern;
  heatmap cells carry intensity *and* the printed number. Everything survives greyscale
  and any colour-vision type.
- **Skeletons, never spinners.** Loading states mirror the shape of the content.
- **Numbers are monospace with tabular figures**, so a value changing does not shift the
  layout around it.
- `prefers-reduced-motion` disables every transition.
- Chart views are lazy-loaded: Recharts is ~550 kB and none of it is needed for the first
  interaction a visitor sees.
- **Provenance is always visible.** A synthetic-data or untrained-model badge sits in the
  header whenever either is true, so a screenshot can never misrepresent what is shown.
