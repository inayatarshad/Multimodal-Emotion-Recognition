import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../api/client';
import type { DegradationCurve } from '../api/types';
import { Chip, EmptyState, ErrorState, Panel, SkeletonPanel } from '../components/Primitives';
import { axisLabel, fixed } from '../lib/format';
import { modelLabel, seriesStyle, sortModels } from '../lib/palette';

/**
 * Results dashboard: retention curves faceted by corruption axis, with seed bands.
 *
 * Architectures can be toggled off, which matters when eight series overlap. Each panel
 * exports to SVG for the paper — the same numbers appear in both, so a figure can never
 * disagree with the table it accompanies.
 */
export function DashboardView() {
  const query = useQuery({ queryKey: ['degradation'], queryFn: api.degradation });
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const grouped = useMemo(() => {
    const curves = (query.data?.curves ?? []).filter((c) => !c.axis.startsWith('remove.'));
    const byAxis = new Map<string, DegradationCurve[]>();
    for (const curve of curves) {
      const list = byAxis.get(curve.axis) ?? [];
      list.push(curve);
      byAxis.set(curve.axis, list);
    }
    return [...byAxis.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [query.data]);

  const allModels = useMemo(
    () => sortModels([...new Set((query.data?.curves ?? []).map((c) => c.model))], (m) => m),
    [query.data],
  );

  if (query.isError) return <ErrorState error={query.error} retry={() => query.refetch()} />;
  if (query.isLoading) {
    return (
      <div className="grid gap-4 lg:grid-cols-2">
        {[0, 1, 2, 3].map((i) => (
          <SkeletonPanel key={i} rows={6} />
        ))}
      </div>
    );
  }
  if (grouped.length === 0) {
    return (
      <EmptyState title="No sweep results yet">
        Run <code className="text-accent">make experiments PRESET=dev</code> to populate{' '}
        <code>experiments/results/</code>, then reload.
      </EmptyState>
    );
  }

  const toggle = (model: string) =>
    setHidden((current) => {
      const next = new Set(current);
      if (next.has(model)) next.delete(model);
      else next.add(model);
      return next;
    });

  const brittleness = query.data?.brittleness ?? {};
  const spearman = brittleness.spearman;

  return (
    <div className="space-y-4">
      <Panel
        title="Degradation curves"
        subtitle={`Metric: ${query.data?.metric ?? '—'} · retention of skill above chance · bands are ±1 std over seeds`}
        right={
          <div className="flex flex-wrap gap-2 justify-end">
            <Chip tone={query.data?.provenance === 'synthetic' ? 'warn' : undefined}>
              provenance: {query.data?.provenance}
            </Chip>
            {spearman !== undefined && Number.isFinite(spearman) && (
              <Chip tone={spearman < -0.5 ? 'accent' : 'warn'}>
                brittleness ρ = {fixed(spearman, 2)}
              </Chip>
            )}
          </div>
        }
      >
        <div className="flex flex-wrap gap-2">
          {allModels.map((model) => {
            const style = seriesStyle(model);
            const isHidden = hidden.has(model);
            return (
              <button
                key={model}
                type="button"
                onClick={() => toggle(model)}
                aria-pressed={!isHidden}
                className={`chip transition-opacity ${isHidden ? 'opacity-35' : ''}`}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: style.color }}
                  aria-hidden="true"
                />
                {modelLabel(model)}
              </button>
            );
          })}
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        {grouped.map(([axis, curves]) => (
          <AxisPanel key={axis} axis={axis} curves={curves} hidden={hidden} />
        ))}
      </div>
    </div>
  );
}

function AxisPanel({
  axis,
  curves,
  hidden,
}: {
  axis: string;
  curves: DegradationCurve[];
  hidden: Set<string>;
}) {
  const visible = sortModels(
    curves.filter((c) => !hidden.has(c.model)),
    (c) => c.model,
  );

  // Recharts wants row-per-x; the curves are column-per-series.
  const rows = useMemo(() => {
    const severities = curves[0]?.severities ?? [];
    return severities.map((severity, index) => {
      const row: Record<string, number | null> = { severity };
      for (const curve of curves) {
        row[curve.model] = curve.retention[index] ?? null;
        const std = curve.retention_std[index] ?? 0;
        const value = curve.retention[index] ?? 0;
        // Recharts draws a band from an [low, high] tuple; store it as a pair.
        row[`${curve.model}__band`] = std;
        row[`${curve.model}__lo`] = value - std;
        row[`${curve.model}__hi`] = value + std;
      }
      return row;
    });
  }, [curves]);

  const seeds = Math.max(...curves.map((c) => c.seeds), 1);

  return (
    <Panel
      title={axisLabel(axis)}
      subtitle={curves[0] ? `AUDC ${fixed(curves[0].audc, 2)} · ${seeds} seed(s)` : undefined}
      right={
        seeds < 2 ? (
          <Chip tone="warn" title="Single-run numbers are not reportable">
            1 seed
          </Chip>
        ) : undefined
      }
    >
      <div className="h-56" role="img" aria-label={`Retention curves for ${axisLabel(axis)}`}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 6, right: 8, bottom: 4, left: -18 }}>
            <CartesianGrid stroke="#232834" strokeDasharray="2 4" />
            <XAxis
              dataKey="severity"
              stroke="#7e8799"
              tick={{ fontSize: 10, fill: '#7e8799' }}
              tickFormatter={(v: number) => v.toFixed(1)}
            />
            <YAxis
              domain={[0, 1.15]}
              stroke="#7e8799"
              tick={{ fontSize: 10, fill: '#7e8799' }}
              tickFormatter={(v: number) => v.toFixed(1)}
            />
            <Tooltip
              contentStyle={{
                background: '#12151c',
                border: '1px solid #333a4a',
                borderRadius: 6,
                fontSize: 11,
              }}
              labelFormatter={(v) => `severity ${Number(v).toFixed(2)}`}
              formatter={(value: number, name: string) => [value?.toFixed(3), modelLabel(name)]}
            />
            <ReferenceLine
              y={0.9}
              stroke="#4a5265"
              strokeDasharray="3 3"
              label={{ value: '0.9', fill: '#7e8799', fontSize: 9, position: 'right' }}
            />
            {visible.map((curve) => {
              const style = seriesStyle(curve.model);
              return [
                <Area
                  key={`${curve.model}-band`}
                  type="monotone"
                  dataKey={`${curve.model}__hi`}
                  stroke="none"
                  fill={style.color}
                  fillOpacity={0.1}
                  isAnimationActive={false}
                  legendType="none"
                  tooltipType="none"
                />,
                <Line
                  key={curve.model}
                  type="monotone"
                  dataKey={curve.model}
                  stroke={style.color}
                  strokeWidth={2}
                  strokeDasharray={style.dash || undefined}
                  dot={{ r: 2.5, fill: style.color }}
                  activeDot={{ r: 4 }}
                  isAnimationActive={false}
                  connectNulls
                />,
              ];
            })}
            <Legend
              wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
              formatter={(value: string) => modelLabel(value)}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}
