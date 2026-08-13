import type {
  CompareResponse,
  CorruptionInfo,
  CorruptionRequest,
  DegradationResponse,
  HealthResponse,
  ModelInfo,
  ParetoResponse,
  PredictResponse,
  RelianceResponse,
  SampleInfo,
} from './types';

/**
 * In dev the Vite proxy forwards /api and /ws to the backend, so the default empty base
 * keeps everything same-origin. Set VITE_API_BASE when the API lives elsewhere.
 */
const BASE = import.meta.env.VITE_API_BASE ?? '';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* non-JSON error body; the status text is the best we have */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>('/health'),
  models: () => request<ModelInfo[]>('/api/models'),
  corruptions: () => request<CorruptionInfo[]>('/api/corruptions'),
  samples: (limit = 24) => request<SampleInfo[]>(`/api/samples?limit=${limit}`),

  predict: (body: {
    sample_id: string;
    model: string;
    corruption: CorruptionRequest;
    return_attention?: boolean;
  }) => request<PredictResponse>('/api/predict', { method: 'POST', body: JSON.stringify(body) }),

  compare: (body: {
    sample_id: string;
    models?: string[];
    corruption: CorruptionRequest;
    return_attention?: boolean;
  }) => request<CompareResponse>('/api/compare', { method: 'POST', body: JSON.stringify(body) }),

  degradation: () => request<DegradationResponse>('/api/results/degradation'),
  reliance: () => request<RelianceResponse>('/api/results/reliance'),
  pareto: () => request<ParetoResponse>('/api/results/pareto'),
};

export function websocketUrl(path = '/ws/live'): string {
  if (BASE) {
    return `${BASE.replace(/^http/, 'ws')}${path}`;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
}

export const CLEAN_CORRUPTION: CorruptionRequest = {
  text: { type: 'none', severity: 0 },
  audio: { type: 'none', severity: 0 },
  visual: { type: 'none', severity: 0 },
};
