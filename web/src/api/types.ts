/** Mirrors `src/wfb/serving/schemas.py`. Keep the two in step. */

export type ModalityName = 'text' | 'audio' | 'visual';

export interface CorruptionSetting {
  type: string;
  severity: number;
  params?: Record<string, number>;
}

export type CorruptionRequest = Record<ModalityName, CorruptionSetting>;

export interface ModalityContribution {
  modality: ModalityName;
  contribution: number;
  relative: number;
}

export interface PredictResponse {
  sample_id: string;
  model: string;
  prediction: number;
  label: number | null;
  clean_prediction: number;
  delta: number;
  confidences: Record<string, number>;
  sentiment: 'negative' | 'neutral' | 'positive';
  contributions: ModalityContribution[];
  attention: Record<string, number[][]>;
  corruption_description: string;
  corruption_hash: string;
  latency_ms: number;
  cached: boolean;
}

export interface CompareResponse {
  sample_id: string;
  corruption_description: string;
  results: PredictResponse[];
  latency_ms: number;
}

export interface ModelInfo {
  name: string;
  architecture: string;
  modalities: ModalityName[];
  parameters: number;
  trained: boolean;
  checkpoint: string | null;
  clean_metrics: Record<string, number>;
  fusion_rank: number | null;
}

export interface SampleInfo {
  id: string;
  dataset: string;
  split: string;
  label: number;
  sentiment: string;
  media_url: string | null;
  transcript: string | null;
}

export interface CorruptionInfo {
  name: string;
  applies_to: ModalityName[];
  unit: string;
  doc: string;
}

export interface HealthResponse {
  status: 'ok' | 'degraded';
  version: string;
  models_loaded: number;
  trained_models: number;
  dataset: string;
  dataset_source: string;
  cache: 'redis' | 'memory' | 'disabled';
  uptime_seconds: number;
}

export interface DegradationCurve {
  model: string;
  axis: string;
  metric: string;
  severities: number[];
  retention: number[];
  retention_std: number[];
  values: number[];
  audc: number;
  audc_std: number;
  critical: number | null;
  seeds: number;
}

export interface DegradationResponse {
  dataset: string;
  metric: string;
  provenance: string;
  curves: DegradationCurve[];
  brittleness: Record<string, number>;
}

export interface RelianceEntry {
  model: string;
  mrs: Record<string, number>;
  mrs_normalized: Record<string, number>;
  subset_retention: Record<string, number>;
}

export interface RelianceResponse {
  dataset: string;
  metric: string;
  provenance: string;
  entries: RelianceEntry[];
}

export interface ParetoPoint {
  label: string;
  base_model: string;
  modality_dropout: number;
  clean_score: number;
  mean_audc: number;
  parameters: number;
  on_frontier: boolean;
}

export interface ParetoResponse {
  dataset: string;
  metric: string;
  points: ParetoPoint[];
}
