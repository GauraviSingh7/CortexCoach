/**
 * Wire types, mirroring the FastAPI backend exactly.
 *
 * Sources of truth:
 *   backend/core/broadcaster.py   WebSocket payloads
 *   backend/schemas/data_models.py  SessionReport
 *   backend/models/model_status.py  ModelStatus
 */

/** Where a reported signal actually came from. */
export type Source = "model" | "heuristic" | "unavailable";

/** What a model is actually doing, not merely whether it constructed. */
export type ModelState = "trained" | "heuristic" | "unavailable";

export interface ModelStatusEntry {
  name: string;
  state: ModelState;
  detail: string;
  weights_loaded: string | null;
  artifacts_found: string[];
  artifacts_missing: string[];
  blocking_reason: string | null;
}

export interface ModelStatusPayload {
  models: Record<string, ModelStatusEntry>;
  all_trained: boolean;
  degraded: string[];
  trained_count: number;
  total_count: number;
}

export type Speaker = "coach" | "coachee";

export interface GrowPhase {
  phase: string;
  confidence: number;
  reasoning: string;
  /** True when this turn continued the phase already in progress. */
  inherited: boolean;
}

/** Live text for an utterance still being spoken. */
export interface PartialMessage {
  type: "partial";
  speaker: Speaker;
  speaker_id: string | null;
  transcript: string;
  timestamp: number;
}

/** A completed turn with its full analysis. */
export interface FinalMessage {
  type: "final";
  timestamp: number;
  speaker: Speaker;
  speaker_id: string | null;
  transcript: string;
  grow_phase: GrowPhase;
  emotion_trend: Record<string, number>;
  engagement_score: number;
  coaching_quality: Record<string, number | string[]>;
  suggestions: string[];
  learning_style: string;
  vak_visual: number;
  vak_auditory: number;
  vak_kinesthetic: number;
  vak_confidence: number;
  digression_level: number;
  digression_detected: boolean;
  sarcasm_detected: boolean;
  sarcasm_score: number;
  sarcasm_type: string;
  sources: Record<string, Source>;
}

export type FeedbackMessage = PartialMessage | FinalMessage;

export const isFinal = (m: FeedbackMessage): m is FinalMessage =>
  m.type === "final";

export interface GrowPhaseRow {
  phase: string;
  turns: number;
  percentage: number;
  avg_confidence: number;
}

export interface GrowCoverage {
  total_turns: number;
  classified_turns: number;
  unclassified_turns: number;
  coverage_pct: number;
  phases_observed: string[];
  phases_missing: string[];
}

export interface EmotionPoint {
  timestamp: number;
  emotion: string;
  confidence: number;
}

export interface SarcasmMoment {
  speaker: Speaker;
  text: string;
  score: number;
  type: string;
}

export interface SarcasmSummary {
  count_detected?: number;
  total_evaluated?: number;
  average_score?: number;
  max_score?: number;
  by_type?: Record<string, number>;
  moments?: SarcasmMoment[];
  source?: string;
}

export interface DigressionMoment {
  speaker: Speaker;
  text: string;
  score: number;
  reason: string;
}

export interface DigressionSummary {
  average_score?: number;
  max_score?: number;
  off_topic_moments?: number;
  total_evaluated?: number;
  moments?: DigressionMoment[];
}

export interface SessionReport {
  session_id: string;
  duration_minutes: number;
  participants: Record<string, Record<string, number>>;
  grow_phases: GrowPhaseRow[];
  grow_coverage: GrowCoverage | Record<string, never>;
  emotional_journey: Record<string, EmotionPoint[]>;
  learning_style_analysis: Record<string, number>;
  key_insights: string[];
  coaching_effectiveness: Record<string, number>;
  recommendations: string[];
  transcript_summary: string;
  sarcasm_summary: SarcasmSummary;
  digression_summary: DigressionSummary;
  model_status: ModelStatusPayload | Record<string, never>;
  analysis_sources: Record<string, Source>;
}

export interface StopSessionResponse {
  status: string;
  report: SessionReport;
  report_file: string | null;
  cached?: boolean;
}

export interface SessionStatus {
  active: boolean;
  session_id: string | null;
  chunks_processed: number;
}

export interface AudioDevice {
  index: number;
  name: string;
  channels: number;
  sample_rate: number;
}
