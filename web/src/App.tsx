import { lazy, Suspense, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from './api/client';
import { Chip, SkeletonPanel } from './components/Primitives';
import { ExplorerView } from './views/ExplorerView';

// The hero view loads eagerly; the three chart views are split out because Recharts is
// ~550 kB and none of it is needed to render the interaction a visitor sees first.
const DashboardView = lazy(() =>
  import('./views/DashboardView').then((m) => ({ default: m.DashboardView })),
);
const RelianceView = lazy(() =>
  import('./views/RelianceView').then((m) => ({ default: m.RelianceView })),
);
const ParetoView = lazy(() =>
  import('./views/ParetoView').then((m) => ({ default: m.ParetoView })),
);

const VIEWS = [
  { id: 'explorer', label: 'Explorer', hint: 'Live degradation' },
  { id: 'dashboard', label: 'Results', hint: 'Retention curves' },
  { id: 'reliance', label: 'Reliance', hint: 'Which modality matters' },
  { id: 'pareto', label: 'Pareto', hint: 'Accuracy vs robustness' },
] as const;

type ViewId = (typeof VIEWS)[number]['id'];

export default function App() {
  const [view, setView] = useState<ViewId>('explorer');
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, retry: 0 });

  return (
    <div className="min-h-screen flex flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2
                   focus:z-50 focus:panel focus:px-3 focus:py-2 focus:text-sm"
      >
        Skip to content
      </a>

      <header className="border-b border-ink-800 bg-ink-900/60 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="mr-auto">
            <h1 className="text-base font-medium text-chalk-100 tracking-tight">
              When Fusion Breaks
            </h1>
            <p className="text-2xs text-chalk-400">
              Graceful degradation in multimodal emotion recognition
            </p>
          </div>

          <nav aria-label="Views" className="flex gap-1 order-3 sm:order-none w-full sm:w-auto">
            {VIEWS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setView(item.id)}
                aria-current={view === item.id ? 'page' : undefined}
                title={item.hint}
                className={`tab flex-1 sm:flex-none ${view === item.id ? 'tab-active' : ''}`}
              >
                {item.label}
              </button>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            {health.data && (
              <>
                <Chip tone={health.data.status === 'ok' ? 'accent' : 'warn'}>
                  {health.data.dataset}
                </Chip>
                {health.data.dataset_source === 'synthetic' && (
                  <Chip tone="warn" title="Synthetic features — pipeline validation only">
                    synthetic data
                  </Chip>
                )}
                {health.data.trained_models === 0 && (
                  <Chip tone="warn" title="No checkpoints found; models are randomly initialised">
                    untrained
                  </Chip>
                )}
              </>
            )}
            {health.isError && <Chip tone="warn">API offline</Chip>}
          </div>
        </div>
      </header>

      <main id="main" className="flex-1 max-w-7xl w-full mx-auto px-4 py-5">
        {view === 'explorer' && <ExplorerView />}
        <Suspense fallback={<SkeletonPanel rows={8} title="Loading view" />}>
          {view === 'dashboard' && <DashboardView />}
          {view === 'reliance' && <RelianceView />}
          {view === 'pareto' && <ParetoView />}
        </Suspense>
      </main>

      <footer className="border-t border-ink-800 mt-6">
        <div
          className="max-w-7xl mx-auto px-4 py-4 text-2xs text-chalk-400 flex flex-wrap
                     justify-between gap-3"
        >
          <span>
            H1: more sophisticated fusion is more brittle. Retention is measured as skill
            above chance.
          </span>
          <span className="tabular">
            {health.data ? `v${health.data.version} · cache: ${health.data.cache}` : '—'}
          </span>
        </div>
      </footer>
    </div>
  );
}
