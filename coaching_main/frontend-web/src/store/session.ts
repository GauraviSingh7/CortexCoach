/**
 * Live session state.
 *
 * A single reducer holds everything the dashboard renders. Final turns are
 * appended to an immutable list; partials live in a separate map keyed by
 * speaker so an in-progress utterance never pollutes the transcript.
 */

import { useCallback, useMemo, useReducer } from "react";
import type {
  FeedbackMessage,
  FinalMessage,
  ModelStatusPayload,
  SessionReport,
  Source,
  Speaker,
} from "../types";
import { isFinal } from "../types";

export interface SessionState {
  active: boolean;
  sessionId: string | null;
  /** Completed turns, in order. */
  turns: FinalMessage[];
  /** In-progress text per speaker; cleared when that speaker's turn lands. */
  partials: Record<Speaker, string>;
  report: SessionReport | null;
  modelStatus: ModelStatusPayload | null;
  error: string | null;
  busy: boolean;
}

const EMPTY_PARTIALS: Record<Speaker, string> = { coach: "", coachee: "" };

export const initialState: SessionState = {
  active: false,
  sessionId: null,
  turns: [],
  partials: EMPTY_PARTIALS,
  report: null,
  modelStatus: null,
  error: null,
  busy: false,
};

export type Action =
  | { type: "session/starting" }
  | { type: "session/started"; sessionId: string }
  | { type: "session/stopped"; report: SessionReport | null }
  | { type: "session/failed"; error: string }
  | { type: "feedback"; message: FeedbackMessage }
  | { type: "modelStatus"; status: ModelStatusPayload }
  | { type: "error/clear" };

export function reducer(state: SessionState, action: Action): SessionState {
  switch (action.type) {
    case "session/starting":
      return { ...state, busy: true, error: null };

    case "session/started":
      // A new session must not inherit anything from the previous one.
      return {
        ...initialState,
        modelStatus: state.modelStatus,
        active: true,
        sessionId: action.sessionId,
      };

    case "session/stopped":
      return {
        ...state,
        active: false,
        busy: false,
        partials: EMPTY_PARTIALS,
        report: action.report,
      };

    case "session/failed":
      return { ...state, busy: false, active: false, error: action.error };

    case "feedback": {
      const message = action.message;
      if (!isFinal(message)) {
        return {
          ...state,
          partials: { ...state.partials, [message.speaker]: message.transcript },
        };
      }
      return {
        ...state,
        turns: [...state.turns, message],
        partials: { ...state.partials, [message.speaker]: "" },
      };
    }

    case "modelStatus":
      return { ...state, modelStatus: action.status };

    case "error/clear":
      return { ...state, error: null };

    default:
      return state;
  }
}

/** Values derived from the turn list, recomputed only when turns change. */
export interface LiveStats {
  turnCount: number;
  coachTurns: number;
  coacheeTurns: number;
  latest: FinalMessage | null;
  growPhase: string;
  engagement: number;
  learningStyle: string;
  digression: number;
  sarcasmCount: number;
  digressionCount: number;
  suggestions: string[];
  sources: Record<string, Source>;
}

export function useLiveStats(turns: FinalMessage[]): LiveStats {
  return useMemo(() => {
    const latest = turns.length ? turns[turns.length - 1] : null;
    return {
      turnCount: turns.length,
      coachTurns: turns.filter((t) => t.speaker === "coach").length,
      coacheeTurns: turns.filter((t) => t.speaker === "coachee").length,
      latest,
      growPhase: latest?.grow_phase.phase ?? "Not started",
      engagement: latest?.engagement_score ?? 0,
      learningStyle: latest?.learning_style ?? "Insufficient Data",
      digression: latest?.digression_level ?? 0,
      sarcasmCount: turns.filter((t) => t.sarcasm_detected).length,
      digressionCount: turns.filter((t) => t.digression_detected).length,
      suggestions: latest?.suggestions ?? [],
      sources: latest?.sources ?? {},
    };
  }, [turns]);
}

export function useSession() {
  const [state, dispatch] = useReducer(reducer, initialState);

  const onFeedback = useCallback(
    (message: FeedbackMessage) => dispatch({ type: "feedback", message }),
    [],
  );

  return { state, dispatch, onFeedback };
}
