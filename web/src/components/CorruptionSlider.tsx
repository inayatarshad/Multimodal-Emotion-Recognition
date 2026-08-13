import type { CorruptionInfo, ModalityName } from '../api/types';
import { MODALITY_COLORS } from '../lib/palette';

/**
 * One modality's corruption control: an operator picker plus a severity slider.
 *
 * The physical interpretation of the current severity (SNR in dB, WER, blur sigma) is
 * shown live next to the value, because "0.4" means nothing on its own and the whole
 * point of the view is to make degradation legible.
 */
export function CorruptionSlider({
  modality,
  operators,
  type,
  severity,
  onTypeChange,
  onSeverityChange,
  disabled = false,
}: {
  modality: ModalityName;
  operators: CorruptionInfo[];
  type: string;
  severity: number;
  onTypeChange: (type: string) => void;
  onSeverityChange: (severity: number) => void;
  disabled?: boolean;
}) {
  const available = operators.filter((op) => op.applies_to.includes(modality));
  const current = available.find((op) => op.name === type);
  const color = MODALITY_COLORS[modality] ?? '#4dd4c4';
  const sliderId = `severity-${modality}`;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={sliderId} className="flex items-center gap-2 panel-title">
          <span
            className="w-2 h-2 rounded-full shrink-0"
            style={{ backgroundColor: color }}
            aria-hidden="true"
          />
          {modality}
        </label>
        <select
          value={type}
          onChange={(event) => onTypeChange(event.target.value)}
          disabled={disabled}
          aria-label={`${modality} corruption type`}
          className="bg-ink-800 border border-ink-700 rounded px-2 py-1 text-xs
                     text-chalk-200 max-w-[10.5rem] disabled:opacity-40"
        >
          <option value="none">none</option>
          {available.map((op) => (
            <option key={op.name} value={op.name}>
              {op.name.replace(/_/g, ' ')}
            </option>
          ))}
        </select>
      </div>

      <input
        id={sliderId}
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={severity}
        disabled={disabled || type === 'none'}
        onChange={(event) => onSeverityChange(Number(event.target.value))}
        className="slider"
        style={{ accentColor: color }}
        aria-valuetext={`${Math.round(severity * 100)} percent severity`}
      />

      <div className="flex items-center justify-between text-2xs">
        <span className="text-chalk-400">
          {type === 'none' ? 'clean' : (current?.unit ?? 'severity')}
        </span>
        <span className="tabular text-chalk-200">
          {type === 'none' ? '—' : `${(severity * 100).toFixed(0)}%`}
        </span>
      </div>
    </div>
  );
}
