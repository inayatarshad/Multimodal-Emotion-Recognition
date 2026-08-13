import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { api, CLEAN_CORRUPTION } from '../api/client';
import type { CorruptionRequest, ModalityName, PredictResponse } from '../api/types';
import { CorruptionSlider } from '../components/CorruptionSlider';
import { PredictionDial } from '../components/PredictionDial';
import { Chip, EmptyState, ErrorState, Panel, SkeletonPanel, Stat } from '../components/Primitives';
import { fixed, percent, signed } from '../lib/format';
import { modelLabel, seriesStyle, sortModels } from '../lib/palette';
import { useLiveCompare } from '../hooks/useLiveCompare';

const DEFAULT_OPERATORS: Record<ModalityName, string> = {
  text: 'asr_error',
  audio: 'gaussian_noise',
  visual: 'occlusion',
};

/**
 * The hero view.
 *
 * One sample, three corruption sliders, and every architecture responding to the *same*
 * corrupted input simultaneously. Watching one model fall off a cliff while another holds
 * steady — under provably identical inputs — is the entire hypothesis in a single
 * interaction.
 */
export function ExplorerView() {
  const samples = useQuery({ queryKey: ['samples'], queryFn: () => api.samples(24) });
  const models = useQuery({ queryKey: ['models'], queryFn: api.models });
  const operators = useQuery({ queryKey: ['corruptions'], queryFn: api.corruptions });

  const [sampleId, setSampleId] = useState<string | null>(null);
  const [corruption, setCorruption] = useState<CorruptionRequest>(CLEAN_CORRUPTION);

  const activeSample = sampleId ?? samples.data?.[0]?.id ?? null;
  const modelNames = useMemo(
    () => sortModels(models.data ?? [], (m) => m.name).map((m) => m.name),
    [models.data],
  );

  const { data, status, busy } = useLiveCompare(activeSample, corruption, modelNames);

  const setModality = (modality: ModalityName, patch: Partial<{ type: string; severity: number }>) =>
    setCorruption((current) => ({
      ...current,
      [modality]: { ...current[modality], ...patch },
    }));

  if (samples.isError) return <ErrorState error={samples.error} retry={() => samples.refetch()} />;
  if (samples.isLoading || models.isLoading) {
    return (
      <div className="grid gap-4 lg:grid-cols-3">
        <SkeletonPanel title="Sample" rows={5} />
        <SkeletonPanel title="Corruption" rows={5} />
        <SkeletonPanel title="Prediction" rows={5} />
      </div>
    );
  }

  const sample = samples.data?.find((s) => s.id === activeSample);
  const results = data?.results ?? [];
  const anyCorruption = Object.values(corruption).some(
    (c) => c.type !== 'none' && c.severity > 0,
  );

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
        {/* ---------------------------------------------------------- sample + dial */}
        <Panel
          title="Live degradation explorer"
          subtitle="Every architecture sees the identical corrupted input."
          right={
            <div className="flex items-center gap-2">
              <Chip tone={status === 'live' ? 'accent' : undefined}>
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    status === 'live'
                      ? 'bg-accent'
                      : status === 'error'
                        ? 'bg-danger'
                        : 'bg-warn'
                  }`}
                />
                {status === 'live' ? 'websocket' : status}
              </Chip>
              {busy && <Chip>updating…</Chip>}
            </div>
          }
        >
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <label htmlFor="sample-picker" className="panel-title">
                Sample
              </label>
              <select
                id="sample-picker"
                value={activeSample ?? ''}
                onChange={(event) => setSampleId(event.target.value)}
                className="bg-ink-800 border border-ink-700 rounded px-2 py-1 text-xs
                           text-chalk-200 font-mono"
              >
                {samples.data?.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.id} ({item.sentiment})
                  </option>
                ))}
              </select>
              {sample && (
                <Chip>
                  ground truth <span className="tabular">{fixed(sample.label, 2)}</span>
                </Chip>
              )}
              <Chip>{data?.corruption_description ?? 'clean'}</Chip>
            </div>

            {/* Placeholder for the clip. The API exposes media_url; until curated media
                is attached, showing an honest placeholder beats an empty box. */}
            <div
              className="aspect-video w-full rounded-md bg-ink-850 border border-ink-700
                         grid place-items-center text-chalk-400 text-xs"
            >
              {sample?.media_url ? (
                <video src={sample.media_url} controls className="w-full h-full rounded-md" />
              ) : (
                <div className="text-center px-6">
                  <p>No media attached to this sample.</p>
                  <p className="mt-1 text-2xs">
                    Predictions below are computed from the cached aligned features.
                  </p>
                </div>
              )}
            </div>

            {results[0] && (
              <PredictionDial
                prediction={results[0].prediction}
                cleanPrediction={results[0].clean_prediction}
                label={sample?.label ?? null}
              />
            )}
          </div>
        </Panel>

        {/* ---------------------------------------------------------------- controls */}
        <Panel title="Corruption" subtitle="Severity is normalised to [0, 1].">
          <div className="space-y-5">
            {(['text', 'audio', 'visual'] as ModalityName[]).map((modality) => (
              <CorruptionSlider
                key={modality}
                modality={modality}
                operators={operators.data ?? []}
                type={corruption[modality].type}
                severity={corruption[modality].severity}
                onTypeChange={(type) =>
                  setModality(modality, {
                    type,
                    severity:
                      corruption[modality].severity === 0 && type !== 'none'
                        ? 0.4
                        : corruption[modality].severity,
                  })
                }
                onSeverityChange={(severity) => setModality(modality, { severity })}
              />
            ))}

            <div className="flex gap-2 pt-1">
              <button
                type="button"
                onClick={() => setCorruption(CLEAN_CORRUPTION)}
                className="tab border border-ink-600 flex-1"
                disabled={!anyCorruption}
              >
                Reset
              </button>
              <button
                type="button"
                onClick={() =>
                  setCorruption({
                    text: { type: DEFAULT_OPERATORS.text, severity: 0.6 },
                    audio: { type: DEFAULT_OPERATORS.audio, severity: 0.6 },
                    visual: { type: DEFAULT_OPERATORS.visual, severity: 0.6 },
                  })
                }
                className="tab border border-ink-600 flex-1"
              >
                Break it
              </button>
            </div>

            {results[0] && (
              <div className="pt-2 border-t border-ink-700 space-y-2">
                <div className="panel-title">Modality contribution</div>
                {results[0].contributions.map((contribution) => (
                  <div key={contribution.modality} className="space-y-1">
                    <div className="flex justify-between text-2xs">
                      <span className="text-chalk-300">{contribution.modality}</span>
                      <span className="tabular text-chalk-400">
                        {percent(contribution.relative)}
                      </span>
                    </div>
                    <div className="h-1 rounded-full bg-ink-700 overflow-hidden">
                      <motion.div
                        className="h-full bg-accent"
                        animate={{ width: `${contribution.relative * 100}%` }}
                        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Panel>
      </div>

      {/* ---------------------------------------------------- small multiples per model */}
      <Panel
        title="All architectures, same input"
        subtitle="If H1 holds, the sophisticated models drift furthest for the same corruption."
      >
        {results.length === 0 ? (
          <EmptyState title="Waiting for the first prediction" />
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {sortModels(results, (r) => r.model).map((result) => (
              <ModelCard key={result.model} result={result} />
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

function ModelCard({ result }: { result: PredictResponse }) {
  const style = seriesStyle(result.model);
  const drift = Math.abs(result.delta);
  const tone = drift > 0.6 ? 'danger' : drift > 0.25 ? 'warn' : 'default';

  return (
    <div className="rounded-md border border-ink-700 bg-ink-850 p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-2 text-sm text-chalk-100">
          <span
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: style.color }}
            aria-hidden="true"
          />
          {modelLabel(result.model)}
        </span>
        <span
          className={`tabular text-2xs ${
            tone === 'danger' ? 'text-danger' : tone === 'warn' ? 'text-warn' : 'text-chalk-400'
          }`}
        >
          {signed(result.delta, 2)}
        </span>
      </div>

      <PredictionDial
        prediction={result.prediction}
        cleanPrediction={result.clean_prediction}
        compact
      />

      <div className="flex justify-between text-2xs text-chalk-400">
        <Stat label="" value={<span className="text-xs">{result.sentiment}</span>} />
        <span className="tabular self-end">
          {result.cached ? 'cached' : `${result.latency_ms.toFixed(0)} ms`}
        </span>
      </div>
    </div>
  );
}
