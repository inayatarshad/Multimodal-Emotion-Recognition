import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import type { RelianceEntry } from '../api/types';
import { Chip, EmptyState, ErrorState, Panel, SkeletonPanel } from '../components/Primitives';
import { fixed } from '../lib/format';
import { modelLabel, sortModels } from '../lib/palette';

const SUBSETS = ['T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'] as const;
const MODALITIES = ['text', 'audio', 'visual'] as const;

/**
 * Modality Reliance Matrix — where Q2's text-dominance finding lands.
 *
 * Colour uses a single-hue sequential ramp with luminance doing the work, plus the number
 * printed in every cell, so nothing depends on hue discrimination.
 */
export function RelianceView() {
  const query = useQuery({ queryKey: ['reliance'], queryFn: api.reliance });

  if (query.isError) return <ErrorState error={query.error} retry={() => query.refetch()} />;
  if (query.isLoading) return <SkeletonPanel rows={8} title="Modality reliance" />;

  const entries = sortModels(query.data?.entries ?? [], (e) => e.model);
  if (entries.length === 0) {
    return (
      <EmptyState title="No reliance results yet">
        Run <code className="text-accent">make experiments PRESET=dev</code> first.
      </EmptyState>
    );
  }

  return (
    <div className="space-y-4">
      <Panel
        title="Modality Reliance Score"
        subtitle="MRS = 1 − retention when that modality is removed. 1.0 means the model is
                  useless without it; 0.0 means it never used it."
        right={
          <Chip tone={query.data?.provenance === 'synthetic' ? 'warn' : undefined}>
            provenance: {query.data?.provenance}
          </Chip>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <caption className="sr-only">Modality Reliance Score by architecture</caption>
            <thead>
              <tr className="text-left">
                <th scope="col" className="panel-title py-2 pr-4">
                  Architecture
                </th>
                {MODALITIES.map((modality) => (
                  <th key={modality} scope="col" className="panel-title py-2 px-3 text-center">
                    {modality}
                  </th>
                ))}
                <th scope="col" className="panel-title py-2 px-3 text-left">
                  Share
                </th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.model} className="border-t border-ink-800">
                  <th scope="row" className="py-2 pr-4 font-normal text-chalk-100 text-left">
                    {modelLabel(entry.model)}
                  </th>
                  {MODALITIES.map((modality) => (
                    <td key={modality} className="py-2 px-1.5">
                      <HeatCell value={entry.mrs[modality]} />
                    </td>
                  ))}
                  <td className="py-2 px-3">
                    <ShareBar entry={entry} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel
        title="The 7-subset removal grid"
        subtitle="Retention when each subset of modalities is removed. If −AV exceeds −T, the
                  model is text-dominated."
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <caption className="sr-only">Retention by removed modality subset</caption>
            <thead>
              <tr>
                <th scope="col" className="panel-title py-2 pr-4 text-left">
                  Architecture
                </th>
                {SUBSETS.map((subset) => (
                  <th key={subset} scope="col" className="panel-title py-2 px-1.5 text-center">
                    −{subset}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.model} className="border-t border-ink-800">
                  <th scope="row" className="py-2 pr-4 font-normal text-chalk-100 text-left">
                    {modelLabel(entry.model)}
                  </th>
                  {SUBSETS.map((subset) => (
                    <td key={subset} className="py-2 px-1.5">
                      <HeatCell value={entry.subset_retention[subset]} invert />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-2xs text-chalk-400 mt-3">
          Values are retention of clean skill above chance. −TAV is the floor: every modality
          gone.
        </p>
      </Panel>
    </div>
  );
}

function HeatCell({ value, invert = false }: { value: number | undefined; invert?: boolean }) {
  if (value === undefined || !Number.isFinite(value)) {
    return <div className="text-center text-chalk-400 tabular text-xs">—</div>;
  }
  const clamped = Math.max(0, Math.min(1, value));
  // Intensity, not hue, carries the magnitude — readable in greyscale and to any
  // colour-vision type. The printed number is the ground truth either way.
  const intensity = invert ? 1 - clamped : clamped;
  return (
    <div
      className="rounded text-center tabular text-xs py-1.5 border border-ink-700"
      style={{
        backgroundColor: `rgba(77, 212, 196, ${0.06 + intensity * 0.5})`,
        color: intensity > 0.6 ? '#08090c' : '#d8dde7',
      }}
      title={`${fixed(value, 3)}`}
    >
      {fixed(value, 2)}
    </div>
  );
}

function ShareBar({ entry }: { entry: RelianceEntry }) {
  const colors: Record<string, string> = {
    text: '#4dd4c4',
    audio: '#e69f00',
    visual: '#56b4e9',
  };
  return (
    <div
      className="flex h-3 rounded overflow-hidden border border-ink-700 min-w-[7rem]"
      role="img"
      aria-label={MODALITIES.map(
        (m) => `${m} ${Math.round((entry.mrs_normalized[m] ?? 0) * 100)}%`,
      ).join(', ')}
    >
      {MODALITIES.map((modality) => {
        const share = entry.mrs_normalized[modality] ?? 0;
        return (
          <div
            key={modality}
            style={{ width: `${share * 100}%`, backgroundColor: colors[modality] }}
            title={`${modality}: ${(share * 100).toFixed(0)}%`}
          />
        );
      })}
    </div>
  );
}
