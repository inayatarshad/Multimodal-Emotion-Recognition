import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import { api } from '../api/client';
import type { ParetoPoint } from '../api/types';
import { Chip, EmptyState, ErrorState, Panel, SkeletonPanel, Stat } from '../components/Primitives';
import { compactNumber, fixed, signed } from '../lib/format';
import { modelLabel, seriesStyle, sortModels } from '../lib/palette';

/**
 * Robustness Pareto: clean performance against AUDC.
 *
 * Modality-dropout variants are linked to their p=0 control so the trade-off reads as a
 * movement — accuracy given up on one axis, robustness gained on the other — which is the
 * form a practitioner can actually make a decision from.
 */
export function ParetoView() {
  const pareto = useQuery({ queryKey: ['pareto'], queryFn: api.pareto });
  const degradation = useQuery({ queryKey: ['degradation'], queryFn: api.degradation });

  const points = useMemo(
    () => sortModels(pareto.data?.points ?? [], (p) => p.label),
    [pareto.data],
  );

  const links = useMemo(() => {
    const controls = new Map(
      points.filter((p) => p.modality_dropout === 0).map((p) => [p.base_model, p]),
    );
    return points
      .filter((p) => p.modality_dropout > 0)
      .map((variant) => ({ variant, control: controls.get(variant.base_model) }))
      .filter((pair): pair is { variant: ParetoPoint; control: ParetoPoint } => !!pair.control);
  }, [points]);

  if (pareto.isError) return <ErrorState error={pareto.error} retry={() => pareto.refetch()} />;
  if (pareto.isLoading) return <SkeletonPanel rows={8} title="Robustness Pareto" />;
  if (points.length === 0) {
    return (
      <EmptyState title="No Pareto data yet">
        Run <code className="text-accent">make experiments PRESET=mitigation</code> for the
        full trade-off, or any preset for the controls alone.
      </EmptyState>
    );
  }

  const brittleness = degradation.data?.brittleness ?? {};
  const spearman = brittleness.spearman;
  // Mirrors `_h1_verdict` in src/wfb/reporting/tables.py — including its refusal to read
  // anything into the trend on synthetic features, where it is circular by construction:
  // the generator plants a text x audio interaction that the sophisticated architectures
  // exploit for their clean-data lead, and that interaction is the first thing corruption
  // destroys. This view is the most-seen artifact in the project, so it is the last place
  // that should imply the hypothesis has support it does not have.
  const isSynthetic = (degradation.data?.provenance ?? '').includes('synthetic');
  const verdict =
    spearman === undefined || !Number.isFinite(spearman) || (brittleness.n ?? 0) < 3
      ? 'Too few architectures to read a trend.'
      : isSynthetic
        ? 'This says nothing about H1 — on synthetic features the trend is circular by construction. It shows the measurement chain works; it is not evidence.'
        : spearman <= -0.5
          ? 'Consistent with H1: the strongest clean models degrade fastest.'
          : spearman >= 0.5
            ? 'H1 is disconfirmed here — the strongest models are also the most robust.'
            : 'No clear monotone relationship; H1 unsupported either way.';

  const chartData = points.map((point) => ({
    ...point,
    x: point.clean_score,
    y: point.mean_audc,
    z: Math.log10(Math.max(point.parameters, 10)),
  }));

  return (
    <div className="space-y-4">
      <Panel
        title="Robustness Pareto frontier"
        subtitle={`Clean ${pareto.data?.metric ?? 'score'} against mean AUDC. Point size is
                   parameter count; ringed points are on the frontier.`}
        right={
          spearman !== undefined && Number.isFinite(spearman) ? (
            <Chip tone={spearman < -0.5 ? 'accent' : 'warn'}>
              brittleness ρ = {fixed(spearman, 2)}
            </Chip>
          ) : undefined
        }
      >
        <div className="h-80" role="img" aria-label="Clean performance versus robustness">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 12, right: 20, bottom: 16, left: -8 }}>
              <CartesianGrid stroke="#232834" strokeDasharray="2 4" />
              <XAxis
                type="number"
                dataKey="x"
                name="clean"
                stroke="#7e8799"
                tick={{ fontSize: 10, fill: '#7e8799' }}
                domain={['dataMin - 0.02', 'dataMax + 0.02']}
                tickFormatter={(v: number) => v.toFixed(2)}
                label={{
                  value: `clean ${pareto.data?.metric ?? ''}`,
                  position: 'insideBottom',
                  offset: -8,
                  fill: '#7e8799',
                  fontSize: 10,
                }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="AUDC"
                stroke="#7e8799"
                tick={{ fontSize: 10, fill: '#7e8799' }}
                domain={['dataMin - 0.02', 'dataMax + 0.02']}
                tickFormatter={(v: number) => v.toFixed(2)}
              />
              <ZAxis type="number" dataKey="z" range={[60, 380]} />
              <Tooltip
                cursor={{ strokeDasharray: '3 3', stroke: '#4a5265' }}
                contentStyle={{
                  background: '#12151c',
                  border: '1px solid #333a4a',
                  borderRadius: 6,
                  fontSize: 11,
                }}
                content={({ payload }) => {
                  const point = payload?.[0]?.payload as ParetoPoint | undefined;
                  if (!point) return null;
                  return (
                    <div className="bg-ink-850 border border-ink-600 rounded p-2 text-xs">
                      <div className="text-chalk-100">{modelLabel(point.label)}</div>
                      <div className="tabular text-chalk-400 mt-1">
                        clean {fixed(point.clean_score)} · AUDC {fixed(point.mean_audc)}
                      </div>
                      <div className="tabular text-chalk-400">
                        {compactNumber(point.parameters)} params
                        {point.modality_dropout > 0 && ` · p=${point.modality_dropout}`}
                      </div>
                    </div>
                  );
                }}
              />
              <Scatter data={chartData} isAnimationActive={false}>
                {chartData.map((point) => (
                  <Cell
                    key={point.label}
                    fill={seriesStyle(point.label).color}
                    stroke={point.on_frontier ? '#f2f4f8' : 'none'}
                    strokeWidth={point.on_frontier ? 2 : 0}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-chalk-300 mt-2">{verdict}</p>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Points">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left">
                  <th scope="col" className="panel-title py-2 pr-3">
                    Model
                  </th>
                  <th scope="col" className="panel-title py-2 px-2 text-right">
                    Clean
                  </th>
                  <th scope="col" className="panel-title py-2 px-2 text-right">
                    AUDC
                  </th>
                  <th scope="col" className="panel-title py-2 px-2 text-right">
                    Params
                  </th>
                </tr>
              </thead>
              <tbody>
                {points.map((point) => (
                  <tr key={point.label} className="border-t border-ink-800">
                    <th scope="row" className="py-2 pr-3 font-normal text-left text-chalk-100">
                      <span className="flex items-center gap-2">
                        <span
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: seriesStyle(point.label).color }}
                          aria-hidden="true"
                        />
                        {modelLabel(point.label)}
                        {point.on_frontier && <span className="text-accent text-2xs">★</span>}
                      </span>
                    </th>
                    <td className="py-2 px-2 text-right tabular">{fixed(point.clean_score)}</td>
                    <td className="py-2 px-2 text-right tabular">{fixed(point.mean_audc)}</td>
                    <td className="py-2 px-2 text-right tabular text-chalk-400">
                      {compactNumber(point.parameters)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel
          title="Modality-dropout trade-off"
          subtitle="What robustness costs in clean accuracy, per architecture."
        >
          {links.length === 0 ? (
            <p className="text-sm text-chalk-400">
              The mitigation arm has not been run. Try{' '}
              <code className="text-accent">make experiments PRESET=mitigation</code>.
            </p>
          ) : (
            <ul className="space-y-3">
              {links.map(({ variant, control }) => {
                const cleanDelta = variant.clean_score - control.clean_score;
                const audcDelta = variant.mean_audc - control.mean_audc;
                return (
                  <li
                    key={variant.label}
                    className="flex items-center justify-between gap-4 border-b border-ink-800 pb-2"
                  >
                    <div>
                      <div className="text-sm text-chalk-100">{modelLabel(variant.base_model)}</div>
                      <div className="text-2xs text-chalk-400">
                        p(drop) = {variant.modality_dropout}
                      </div>
                    </div>
                    <div className="flex gap-6">
                      <Stat
                        label="clean"
                        value={<span className="text-sm">{signed(cleanDelta)}</span>}
                        tone={cleanDelta < -0.02 ? 'warn' : 'default'}
                      />
                      <Stat
                        label="AUDC"
                        value={<span className="text-sm">{signed(audcDelta)}</span>}
                        tone={audcDelta > 0 ? 'accent' : 'danger'}
                      />
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}
