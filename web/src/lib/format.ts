export function fixed(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

export function signed(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

export function percent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function compactNumber(value: number): string {
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(0)}k`;
  return String(value);
}

/** Map a sentiment score in [-3, 3] to a 0–100 position for the prediction dial. */
export function sentimentPosition(value: number): number {
  return ((Math.max(-3, Math.min(3, value)) + 3) / 6) * 100;
}

export function sentimentColor(value: number): string {
  if (value > 0.35) return '#4dd4c4';
  if (value < -0.35) return '#e05c5c';
  return '#7e8799';
}

/** Human label for a corruption axis id such as `audio.gaussian_noise`. */
export function axisLabel(axis: string): string {
  const [modality, kind] = axis.split('.');
  if (!kind) return axis;
  const pretty = kind.replace(/_/g, ' ');
  return `${(modality ?? '').replace(/^\w/, (c) => c.toUpperCase())} · ${pretty}`;
}

export function useReducedMotion(): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
