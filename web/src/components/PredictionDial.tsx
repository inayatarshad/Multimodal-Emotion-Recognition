import { motion } from 'framer-motion';
import { fixed, sentimentColor, sentimentPosition, signed } from '../lib/format';

/**
 * The sentiment readout: current prediction, a ghost marker for the clean prediction, and
 * the drift between them.
 *
 * The ghost marker is the important part. Without a persistent reference the current
 * value is just a number; with it, the *distance the model has moved* is visible at a
 * glance, which is the thing the whole study is about.
 */
export function PredictionDial({
  prediction,
  cleanPrediction,
  label,
  compact = false,
}: {
  prediction: number;
  cleanPrediction: number;
  label?: number | null;
  compact?: boolean;
}) {
  const position = sentimentPosition(prediction);
  const cleanPosition = sentimentPosition(cleanPrediction);
  const labelPosition = label === null || label === undefined ? null : sentimentPosition(label);
  const drift = prediction - cleanPrediction;

  return (
    <div className={compact ? 'space-y-1.5' : 'space-y-3'}>
      {!compact && (
        <div className="flex items-baseline justify-between">
          <span
            className="tabular text-3xl"
            style={{ color: sentimentColor(prediction) }}
            aria-label={`prediction ${fixed(prediction, 2)}`}
          >
            {fixed(prediction, 2)}
          </span>
          <span
            className={`tabular text-sm ${Math.abs(drift) > 0.3 ? 'text-warn' : 'text-chalk-400'}`}
          >
            {signed(drift, 2)} vs clean
          </span>
        </div>
      )}

      <div
        className="relative h-8 rounded bg-ink-800 border border-ink-700 overflow-hidden"
        role="img"
        aria-label={`Sentiment ${fixed(prediction, 2)} on a scale from -3 to 3, clean baseline ${fixed(cleanPrediction, 2)}`}
      >
        {/* Neutral band, so "how far from zero" is readable without counting pixels. */}
        <div className="absolute inset-y-0 left-1/2 w-px bg-ink-600" />

        {labelPosition !== null && (
          <div
            className="absolute inset-y-0 w-0.5 bg-chalk-400/50"
            style={{ left: `${labelPosition}%` }}
            title={`ground truth ${fixed(label, 2)}`}
          />
        )}

        {/* Ghost marker: where the model was before corruption. */}
        <div
          className="absolute inset-y-1 w-0.5 bg-chalk-300/40"
          style={{ left: `${cleanPosition}%` }}
          title={`clean prediction ${fixed(cleanPrediction, 2)}`}
        />

        <motion.div
          className="absolute inset-y-1 w-1 rounded-full"
          style={{ backgroundColor: sentimentColor(prediction) }}
          animate={{ left: `${position}%` }}
          transition={{ type: 'spring', stiffness: 240, damping: 26 }}
        />

        {/* Drift bar between ghost and current — the "confidence collapse" made visual. */}
        <motion.div
          className="absolute inset-y-3 bg-warn/25"
          animate={{
            left: `${Math.min(position, cleanPosition)}%`,
            width: `${Math.abs(position - cleanPosition)}%`,
          }}
          transition={{ type: 'spring', stiffness: 240, damping: 26 }}
        />
      </div>

      <div className="flex justify-between text-2xs text-chalk-400">
        <span>−3 negative</span>
        {compact && (
          <span className="tabular" style={{ color: sentimentColor(prediction) }}>
            {fixed(prediction, 2)}
            <span className="text-chalk-400"> ({signed(drift, 2)})</span>
          </span>
        )}
        <span>positive +3</span>
      </div>
    </div>
  );
}
