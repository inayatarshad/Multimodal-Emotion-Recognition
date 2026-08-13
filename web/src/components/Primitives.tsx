import type { ReactNode } from 'react';

export function Panel({
  title,
  subtitle,
  right,
  children,
  className = '',
}: {
  title?: string;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel p-4 ${className}`}>
      {(title || right) && (
        <header className="flex items-start justify-between gap-3 mb-3">
          <div>
            {title && <h2 className="panel-title">{title}</h2>}
            {subtitle && <p className="text-xs text-chalk-400 mt-1">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = 'default',
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: 'default' | 'accent' | 'warn' | 'danger';
}) {
  const toneClass = {
    default: 'text-chalk-100',
    accent: 'text-accent',
    warn: 'text-warn',
    danger: 'text-danger',
  }[tone];
  return (
    <div>
      <div className="panel-title">{label}</div>
      <div className={`tabular text-xl mt-1 ${toneClass}`}>{value}</div>
      {hint && <div className="text-2xs text-chalk-400 mt-0.5">{hint}</div>}
    </div>
  );
}

/** Skeleton loaders, never a spinner on a blank page. */
export function Skeleton({ className = 'h-24' }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}

export function SkeletonPanel({ rows = 3, title }: { rows?: number; title?: string }) {
  return (
    <Panel title={title}>
      <div className="space-y-2" role="status" aria-label="Loading">
        {Array.from({ length: rows }).map((_, index) => (
          <Skeleton key={index} className={`h-4 ${index === 0 ? 'w-2/3' : 'w-full'}`} />
        ))}
      </div>
    </Panel>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="panel p-8 text-center">
      <p className="text-chalk-200 font-medium">{title}</p>
      {children && <div className="text-sm text-chalk-400 mt-2 max-w-lg mx-auto">{children}</div>}
    </div>
  );
}

export function ErrorState({ error, retry }: { error: unknown; retry?: () => void }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="panel p-6 border-danger/40">
      <p className="text-danger font-medium text-sm">Could not reach the API</p>
      <p className="text-xs text-chalk-400 mt-2 font-mono break-all">{message}</p>
      <p className="text-xs text-chalk-400 mt-3">
        Start it with <code className="text-accent">make serve</code> (or{' '}
        <code className="text-accent">./tasks.ps1 serve</code> on Windows).
      </p>
      {retry && (
        <button type="button" onClick={retry} className="tab mt-3 border border-ink-600">
          Retry
        </button>
      )}
    </div>
  );
}

export function Chip({
  children,
  tone,
  title,
}: {
  children: ReactNode;
  tone?: 'warn' | 'accent';
  title?: string;
}) {
  const toneClass =
    tone === 'warn'
      ? 'border-warn/40 text-warn'
      : tone === 'accent'
        ? 'border-accent/40 text-accent'
        : '';
  return (
    <span className={`chip ${toneClass}`} title={title}>
      {children}
    </span>
  );
}
