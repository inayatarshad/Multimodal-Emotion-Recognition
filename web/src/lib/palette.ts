/**
 * Okabe–Ito, the standard colourblind-safe qualitative palette.
 *
 * Every chart series is also given a distinct dash pattern and marker, so nothing in the
 * UI encodes meaning in hue alone — a hard requirement from the spec and the reason these
 * charts stay readable in greyscale.
 */
export const SERIES_COLORS = [
  '#4dd4c4', // accent (teal)
  '#e69f00', // orange
  '#56b4e9', // sky
  '#cc79a7', // pink
  '#009e73', // green
  '#d55e00', // vermillion
  '#f0e442', // yellow
  '#b3b9c7', // grey
] as const;

export const DASH_PATTERNS = ['', '6 3', '2 3', '8 3 2 3', '4 2', '10 4', '1 3', '6 2 1 2'];

/** Canonical ordering along the fusion-sophistication axis — H1's independent variable. */
export const MODEL_ORDER = [
  'text_only',
  'audio_only',
  'visual_only',
  'late',
  'early',
  'lmf',
  'tfn',
  'mult',
];

export const MODEL_LABELS: Record<string, string> = {
  text_only: 'Text only',
  audio_only: 'Audio only',
  visual_only: 'Visual only',
  late: 'Late fusion',
  early: 'Early fusion',
  lmf: 'LMF',
  tfn: 'TFN',
  mult: 'MulT',
};

export function modelLabel(name: string): string {
  const [base, suffix] = name.split('+');
  const label = MODEL_LABELS[base ?? name] ?? base ?? name;
  return suffix ? `${label} +${suffix}` : label;
}

export function modelIndex(name: string): number {
  const base = name.split('+')[0] ?? name;
  const index = MODEL_ORDER.indexOf(base);
  return index === -1 ? MODEL_ORDER.length : index;
}

export function seriesStyle(name: string): { color: string; dash: string } {
  const index = modelIndex(name);
  return {
    color: SERIES_COLORS[index % SERIES_COLORS.length] ?? SERIES_COLORS[0],
    dash: DASH_PATTERNS[index % DASH_PATTERNS.length] ?? '',
  };
}

export function sortModels<T>(items: T[], key: (item: T) => string): T[] {
  return [...items].sort((a, b) => {
    const delta = modelIndex(key(a)) - modelIndex(key(b));
    return delta !== 0 ? delta : key(a).localeCompare(key(b));
  });
}

export const MODALITY_COLORS: Record<string, string> = {
  text: '#4dd4c4',
  audio: '#e69f00',
  visual: '#56b4e9',
};
