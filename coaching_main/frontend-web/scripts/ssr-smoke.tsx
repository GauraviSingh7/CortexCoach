/**
 * Headless render check.
 *
 * Renders the whole component tree to a string. Effects do not run under
 * renderToString, so no network or WebSocket is touched - but every
 * component body, prop access and conditional branch on the initial path
 * executes, which catches the runtime errors a typecheck cannot.
 */
import { renderToString } from "react-dom/server";
import { createElement } from "react";
import App from "../src/App";
import { SessionReport } from "../src/components/SessionReport";
import { ModelStatusPanel } from "../src/components/ModelStatusPanel";
import { GrowTimeline } from "../src/components/GrowTimeline";
import { EmotionChart } from "../src/components/EmotionChart";
import { LiveTranscript } from "../src/components/LiveTranscript";
import { StatsBanner } from "../src/components/StatsBanner";
import type { FinalMessage, ModelStatusPayload, SessionReport as Report } from "../src/types";

const turn = (over: Partial<FinalMessage> = {}): FinalMessage => ({
  type: "final",
  timestamp: 1700000000,
  speaker: "coach",
  speaker_id: "B",
  transcript: "What would you like to get out of our time today?",
  grow_phase: { phase: "Goal", confidence: 0.9, reasoning: "6 indicators", inherited: false },
  emotion_trend: { hopeful: 0.7, excited: 0.3 },
  engagement_score: 0.62,
  coaching_quality: { overall: 0.88, questioning: 0.9, listening: 0.85 },
  suggestions: ["Ask what else"],
  learning_style: "Visual (51%)",
  vak_visual: 0.51, vak_auditory: 0.22, vak_kinesthetic: 0.27, vak_confidence: 0.51,
  digression_level: 0.1, digression_detected: false,
  sarcasm_detected: false, sarcasm_score: 0.0, sarcasm_type: "none",
  sources: { emotion: "heuristic", engagement: "heuristic", sarcasm: "heuristic" },
  ...over,
});

const status: ModelStatusPayload = {
  models: {
    vak_inference: {
      name: "vak_inference", state: "heuristic",
      detail: "using keyword heuristic", weights_loaded: null,
      artifacts_found: ["config.json"], artifacts_missing: ["model.safetensors"],
      blocking_reason: "weights file is missing",
    },
  },
  all_trained: false, degraded: ["vak_inference"], trained_count: 0, total_count: 4,
};

const report: Report = {
  session_id: "3f9ee139-aaaa-bbbb-cccc-ddddeeeeffff",
  duration_minutes: 4.2,
  participants: { coach: { total_turns: 20, avg_words: 21, engagement_avg: 0.6 },
                  coachee: { total_turns: 20, avg_words: 30, engagement_avg: 0.6 } },
  grow_phases: [
    { phase: "Reality", turns: 14, percentage: 35, avg_confidence: 0.47 },
    { phase: "Way Forward", turns: 12, percentage: 30, avg_confidence: 0.54 },
    { phase: "Options", turns: 8, percentage: 20, avg_confidence: 0.53 },
    { phase: "Goal", turns: 6, percentage: 15, avg_confidence: 0.58 },
  ],
  grow_coverage: { total_turns: 40, classified_turns: 40, unclassified_turns: 0,
                   coverage_pct: 100, phases_observed: [], phases_missing: [] },
  emotional_journey: { coach: [], coachee: [] },
  learning_style_analysis: { visual: 0.51, auditory: 0.22, kinesthetic: 0.27 },
  key_insights: ["Coach asked 18 questions"],
  coaching_effectiveness: { overall: 0.88, questioning: 0.89, listening: 1.0, engagement_management: 0.6 },
  recommendations: ["Keep building on this"],
  transcript_summary: "The coach explored career...",
  sarcasm_summary: { count_detected: 2, total_evaluated: 40,
    moments: [{ speaker: "coachee", text: "Clearly that worked out great for me.", score: 0.6, type: "mock_enthusiasm" }] },
  digression_summary: { off_topic_moments: 1, total_evaluated: 40 },
  model_status: status,
  analysis_sources: { emotion: "heuristic", sarcasm: "heuristic" },
};

const cases: Array<[string, () => string]> = [
  ["App (idle)", () => renderToString(createElement(App))],
  ["SessionReport (full)", () => renderToString(createElement(SessionReport, { report }))],
  ["SessionReport (null)", () => renderToString(createElement(SessionReport, { report: null }))],
  ["ModelStatusPanel", () => renderToString(createElement(ModelStatusPanel, { status }))],
  ["ModelStatusPanel (null)", () => renderToString(createElement(ModelStatusPanel, { status: null }))],
  ["GrowTimeline (empty)", () => renderToString(createElement(GrowTimeline, { turns: [] }))],
  ["GrowTimeline (data)", () => renderToString(createElement(GrowTimeline, { turns: [turn(), turn({ grow_phase: { phase: "Reality", confidence: 0.6, reasoning: "", inherited: true } })] }))],
  ["EmotionChart (no signal)", () => renderToString(createElement(EmotionChart, { turns: [turn({ emotion_trend: {} })] }))],
  ["LiveTranscript (empty)", () => renderToString(createElement(LiveTranscript, { turns: [], partials: { coach: "", coachee: "" } }))],
  ["LiveTranscript (turns)", () => renderToString(createElement(LiveTranscript, { turns: [turn(), turn({ speaker: "coachee", sarcasm_detected: true, sarcasm_type: "mock_enthusiasm", emotion_trend: {} })], partials: { coach: "still speaking", coachee: "" } }))],
  ["StatsBanner", () => renderToString(createElement(StatsBanner, { stats: {
      turnCount: 2, coachTurns: 1, coacheeTurns: 1, latest: turn(), growPhase: "Goal",
      engagement: 0.62, learningStyle: "Visual (51%)", digression: 0.1,
      sarcasmCount: 0, digressionCount: 0, suggestions: [], sources: { engagement: "heuristic" } } }))],
];

let failures = 0;
for (const [name, run] of cases) {
  try {
    const html = run();
    if (!html || html.length < 10) throw new Error("rendered empty output");
    console.log(`  PASS  ${name}  (${html.length} chars)`);
  } catch (error) {
    failures += 1;
    console.log(`  FAIL  ${name}: ${(error as Error).message}`);
  }
}
console.log(failures === 0 ? `\nall ${cases.length} render checks passed` : `\n${failures} render check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
